"""Pure plan checks and small CPU-tensor synthetic-cache invariants."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from fp8_decode_sweep import FOUR_NODE_METHODS, dbo_layout, trial_for
from e2e_matrix import collect_trial_result, validate_matrix
from e2e_server import validate_sweep_control


@pytest.mark.parametrize("method", ("dp-tp8", "tknp-tp8", "tp8-pp2", "tp", "dcp"))
def test_full_two_node_budget_and_capacity_waves(method):
    caps = [6, 34] if method == "tknp-tp8" else [6, 6] if method == "dp-tp8" else [42]
    trial = trial_for(method, 32768, 1024, 48, caps)
    validate_matrix(dict(trials=[trial]), 16)
    assert sum(len(role["gpus"]) for role in trial["servers"]) == 16
    if method == "dp-tp8":
        assert len(trial["servers"]) == 2
        assert [r["max_num_seqs"] for r in trial["servers"]] == [6, 6]
    elif method == "tknp-tp8":
        assert trial["servers"][0]["owner_counts"] == [6, 34]
    elif method == "tp8-pp2":
        assert "pp_decode_microbatch" not in trial["servers"][0]
        assert trial["servers"][0]["graph_batch_size"] == 42
    elif method == "tp":
        assert trial["servers"][0]["tp"] == 16
    else:
        assert trial["servers"][0]["dcp"] == 2
        assert trial["servers"][0]["max_num_seqs"] == 42
        assert trial["servers"][0]["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] == "1"


def test_preinitialized_kv_uses_real_distinct_storage_and_no_timed_fill(monkeypatch):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
        DecodeBenchConnectorWorker, DecodeBenchConnectorMetadata)
    options = dict(preinitialize_pool=True, fill_mean=0.0, fill_std=0.02, fill_seed=123)
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda k, d: options.get(k, d)))
    worker = DecodeBenchConnectorWorker(config)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cache = torch.zeros(8, 2, 16, 1, 128, dtype=torch.bfloat16)
    worker.register_kv_caches({"layer0": cache})
    assert worker.preinitialized_bytes == cache.numel() * cache.element_size()
    assert torch.isfinite(cache).all() and cache.float().std() > 0.01
    assert not torch.equal(cache[0], cache[1])
    snapshot = cache.clone()
    monkeypatch.setattr(worker, "_fill_blocks", lambda *args: pytest.fail("Timed KV fill"))
    worker.start_fill_kv(DecodeBenchConnectorMetadata(reqs_to_fill={"r": (([0, 1],), 32)}))
    torch.testing.assert_close(cache, snapshot, rtol=0, atol=0)
    assert worker.per_request_fill_calls == 0
    # An inter-point reset must not change graph-captured KV addresses or
    # accumulate byte/layer counts. Restore the same random physical pool.
    pointer = cache.data_ptr()
    cache.fill_(42)
    worker.preinitialized_bytes = worker.preinitialized_layers = 0
    worker.register_kv_caches(worker.kv_caches)
    assert cache.data_ptr() == pointer
    assert worker.preinitialized_layers == 1
    torch.testing.assert_close(cache, snapshot, rtol=0, atol=0)


def test_loader_option_is_explicit_and_does_not_change_default():
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    base = dict(load_format="dummy")
    assert not DummyModelLoader(SimpleNamespace(**base, model_loader_extra_config={})).fbgemm_fp8_dummy
    assert DummyModelLoader(SimpleNamespace(**base,
        model_loader_extra_config={"fbgemm_fp8_dummy": True})).fbgemm_fp8_dummy
    with pytest.raises(ValueError):
        DummyModelLoader(SimpleNamespace(**base, model_loader_extra_config={"unknown": True}))


def test_persistent_sweep_controls_reject_layout_changes_and_invalid_pp_sizes():
    base = dict(tknp=1, pp=2, graph_maximum=21, preinitialized=True)
    validate_sweep_control(dict(key="b48", action="prepare", pp_decode_microbatch=21,
                                reset_random_kv=True), **base)
    validate_sweep_control(dict(key="b48", action="checkpoint"), **base)
    for change in (dict(tknp=2), dict(pp=1), dict(graph_maximum=16), dict(preinitialized=False)):
        with pytest.raises(ValueError):
            validate_sweep_control(dict(key="b48", action="prepare", pp_decode_microbatch=21,
                                       reset_random_kv=True), **(base | change))
    for body in (dict(key="../escape", action="prepare"), dict(key="b4", action="unknown"),
                 dict(key="b4", action="prepare", pp_decode_microbatch=0)):
        with pytest.raises(ValueError):
            validate_sweep_control(body, **base)


def test_persistent_trial_metrics_only_include_its_measured_requests(tmp_path):
    client = tmp_path / "client"
    server = tmp_path / "server0"
    client.mkdir()
    server.mkdir()
    workload = tmp_path / "workload.jsonl"
    workload.write_text("{}\n")
    (tmp_path / "launches.json").write_text("[]")
    (client / "requests.jsonl").write_text(json.dumps(dict(server_request_id="ours")) + "\n")
    (server / "execution.json").write_text("[]")
    events = [dict(event="admit", request_id="ours", time=1),
        dict(event="first_compute", request_id="ours", time=2)]
    for index, (request_id, waiting, seconds) in enumerate((("other", 99, 99), ("ours", 3, 0.1))):
        events.append(dict(event="step", step=index, scheduled_request_ids=[request_id],
            waiting=waiting, waiting_capacity=waiting, preemptions=0, preemptions_total=0,
            kv_free_blocks=5, phase="decode", seconds=seconds))
    events.append(dict(event="shutdown", remaining_requests=0, all_kv_free=True))
    (server / "rank0.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    result = collect_trial_result(dict(name="test"), {}, workload, tmp_path)
    assert result["telemetry"]["server0/rank0"]["max_waiting"] == 3
    assert result["engine_stage_latency"]["decode"]["steps"] == 1
    assert result["engine_stage_latency"]["decode"]["p50_s"] == 0.1


@pytest.mark.parametrize("method", FOUR_NODE_METHODS)
@pytest.mark.parametrize("batch", [4, 8, 16, 32, 64, 128])
def test_full_four_node_production_grid(method, batch):
    caps = ([6, 33, 33, 33] if method.startswith("tknp") else
            [6] * 4 if method == "dp-tp8" else [112])
    trial = trial_for(method, 32768, 1024, batch, caps, 32)
    validate_matrix(dict(trials=[trial]), 32)
    assert sorted(g for r in trial["servers"] for g in r["gpus"]) == list(range(32))
    for role in trial["servers"]:
        assert role["cuda_graph"] and not role["disable_combo_kernels"]
        if method == "tp8-pp4":
            assert role["tp"] == 8 and role["pp"] == 4
            assert "pp_decode_microbatch" not in role
            assert role["graph_batch_size"] == min(batch, 112)
        if method == "dcp":
            assert role["tp"] == 32 and role["dcp"] == 4
        if method == "tknp-tp8-dbo":
            assert role["enable_dbo"]
            assert role["env"]["VLLM_TKNP_DBO_ZERO_ROOT_C1"] == "1"
            assert role["env"]["VLLM_TKNP_DBO_LOOKAHEAD"] == "0"
            assert role["graph_batch_size"] >= 16


@pytest.mark.parametrize("degree", [2, 4])
def test_production_dbo_layouts_fit_validated_cohort_shapes(monkeypatch, degree):
    from vllm.forward_context import build_tknp_dbo_cohort_rank_order
    from vllm.v1.worker.tknp_serving_slots import validate_multi_layout_dbo_shapes
    monkeypatch.setenv("VLLM_TKNP_DBO_ZERO_ROOT_C1", "1")
    monkeypatch.setenv("VLLM_TKNP_DBO_C0_FRACTION", "0.5")
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "6")
    cohorts = {}
    for batch in [4, 8, 16, 32, 64, 128]:
        size, counts = dbo_layout(batch, [6] + [33] * (degree - 1))
        order = build_tknp_dbo_cohort_rank_order(counts)
        cohorts[size] = [(order[:size // 2].count(i), order[size // 2:].count(i)) for i in range(degree)]
        assert cohorts[size][0][1] == 0
        assert all(n <= cap for n, cap in zip(counts, [6] + [33] * (degree - 1)))
    validate_multi_layout_dbo_shapes(cohorts)


def test_snapshot_and_completion_are_fail_closed(tmp_path):
    import hashlib
    from eos_fp8_four_node_suite import BATCHES, finish, verify_snapshot
    source = tmp_path / "source.py"
    source.write_text("# frozen\n")
    manifest = dict(commit="test", source_sha256={"source.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    (tmp_path / ".research-source.json").write_text(json.dumps(manifest))
    assert verify_snapshot(tmp_path) == manifest
    source.write_text("# modified\n")
    with pytest.raises(RuntimeError, match="Frozen source changed"):
        verify_snapshot(tmp_path)
    (tmp_path / "sweep").mkdir()
    state = dict(finished_time=1, failures=[], trials=[dict(batch=b, output_tokens=b * 1024,
        full_graph_all_ranks=True, measured_preemptions=0, client=dict(failed=0)) for b in BATCHES])
    summary = tmp_path / "sweep/summary.json"
    summary.write_text(json.dumps(state))
    assert finish(tmp_path, 0) == 0
    state["trials"].pop()
    summary.write_text(json.dumps(state))
    assert finish(tmp_path, 0) == 1
    assert finish(tmp_path, 143) == 1


def test_independent_batch_elements_own_four_nodes_and_exit():
    from eos_fp8_four_node_suite import HERE
    script = (HERE / "jobs/eos_fp8_four_node_sweep.sbatch").read_text()
    assert "#SBATCH --nodes=4" in script and "#SBATCH --exclusive" in script
    assert "#SBATCH --array=0-5" in script and "%1" not in script
    assert "trap tknp_finish EXIT" in script
    assert "--batches 4 8 16 32 64 128" in script
    assert "--smoke-tokens 32" in script


def test_live_smoke_gate_requires_each_rank_and_actual_replay(tmp_path):
    from e2e_matrix import validate_live_segment
    key = "smoke-b2"
    client = tmp_path / (key + "-client")
    server = tmp_path / "server0"
    client.mkdir()
    server.mkdir()
    (client / "summary.json").write_text(json.dumps(dict(completed=2, failed=0, output_tokens=6)))
    segment = dict(key=key, batch=2, graph_size=8, smoke_output_tokens=3)
    identity = dict(pid=123, model=1, kv=2, graphs={"8": 3})
    resources = [dict(rank=i, graph_reuse_identity=identity) for i in range(2)]
    (server / "manifest.json").write_text(json.dumps(dict(worker_resources=resources)))
    records = [dict(rank=i, counts={"graph:FULL_REPLAY": 3, "decode:FULL": 3},
        graph_reuse_identity=identity, active_graph_size=8, per_request_kv_fill_calls=0) for i in range(2)]
    execution = server / (key + "-execution.json")
    execution.write_text(json.dumps(records))
    for i in range(2):
        (server / f"rank{i}.jsonl").write_text(json.dumps(dict(event="sweep_control", key=key,
            action="checkpoint", all_kv_free=True, remaining_requests=0)) + "\n")
    trial = dict(servers=[dict(gpus=[0, 1], cuda_graph=True, graph_layouts={8: [2, 6]})])
    validate_live_segment(trial, segment, tmp_path)
    records[1]["counts"]["graph:FULL_REPLAY"] = 0
    execution.write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match="Smoke graph"):
        validate_live_segment(trial, segment, tmp_path)
    execution.write_text(json.dumps(records[:1]))
    with pytest.raises(RuntimeError, match="assigned ranks"):
        validate_live_segment(trial, segment, tmp_path)
