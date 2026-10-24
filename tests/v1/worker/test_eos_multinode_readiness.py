"""CPU-only topology/runtime guards for EOS collection."""
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from eos_multinode_readiness import configuration, METHODS, check_resume_plan, stage_trials
from e2e_slurm import launch_role, client_command
from e1_70b_suite import build_trial, topology, validate_paper_plan
from eos_crossover_readiness import crossover_batch
from e2e_server import warm_sampling_collectives


def test_eos_decode_budget_and_runtime_are_not_lyris_defaults():
    config = configuration("123", ["a", "b", "c"], "8b", 512, 33, METHODS, "decode")
    assert config["gpus_per_node"] == 8
    for trial in config["trials"]:
        assert sum(len(r["gpus"]) for r in trial["servers"]) == 16
        assert all(r["role"] == "decode" for r in trial["servers"])
    role = config["trials"][0]["servers"][0]
    command, url = launch_role(config, role, 0, ["python", "-m", "torch.distributed.run",
        "--standalone", "--nproc-per-node=16", "/repo/e2e_server.py"], {})
    assert "--nodelist=a,b" in command
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in command
    assert any("vllm-tknp-v1.sqsh" in v for v in command)
    assert "UCX_TLS=^gdr_copy" in command
    assert not any("cu13" in v or "arm64" in v for v in command)
    assert "VLLM_RESEARCH_GIT_HEAD=unknown" not in command
    assert url == "http://a:8600"
    client = client_command(config, ["python", "client.py"])
    assert config["runtime"]["python"] in client


def test_real_handoff_has_dedicated_tp8_prefill_and_disjoint_decode_nodes():
    config = configuration("123", ["p", "d0", "d1"], "8b", 512, 33,
                           ["tknp", "tknp-dbo", "dp", "pp"], "e2e")
    for trial in config["trials"]:
        assert trial["servers"][0]["gpus"] == list(range(8))
        assert trial["servers"][0]["tp"] == 8
        assert sorted(g for r in trial["servers"][1:] for g in r["gpus"]) == list(range(8, 24))
    off, on = [t["servers"][1] for t in config["trials"][:2]]
    assert off["owner_counts"] == on["owner_counts"] == [2, 6]


def test_dcp_real_handoff_and_missing_prefill_node_fail_closed():
    with pytest.raises(ValueError):
        configuration("1", ["a", "b", "c"], "8b", 512, 33, ["dcp"], "e2e")
    with pytest.raises(ValueError):
        configuration("1", ["a", "b"], "8b", 512, 33, ["tknp"], "e2e")


def test_capacity_suite_supports_tp8_prefill_and_tp8_microbatch():
    assert topology("tknp-tp8-dbo", 16) == (8, 1, 2, 1, 1)
    trial = build_trial("tknp-tp8-dbo", 32768, 33, 16,
        list(range(8, 24)), [4, 12], "e2e", list(range(8)), prefill_tp=8)
    assert trial["servers"][0]["tp"] == 8
    assert trial["servers"][0]["gpus"] == list(range(8))
    assert trial["servers"][1]["enable_dbo"]
    assert trial["servers"][1]["owner_counts"] == [4, 12]
    assert trial["servers"][1]["env"]["VLLM_TKNP_DBO_LOOKAHEAD"] == "1"


def test_paper_controller_rejects_historical_search_space():
    validate_paper_plan(["tknp-tp8", "dp-tp8", "tp8-pp2", "dcp"], 16, "decode", 8)
    for method in ["tknp", "dp-native", "pp", "tp4-pp4", "tp"]:
        with pytest.raises(ValueError):
            validate_paper_plan([method], 16, "decode", 8)
    with pytest.raises(ValueError):
        validate_paper_plan(["dcp"], 16, "e2e", 8)
    with pytest.raises(ValueError):
        validate_paper_plan(["dp-tp8"], 16, "e2e", 1)


def test_resume_never_relabels_changed_workload_or_staged_measurement():
    import copy
    original = configuration("1", ["a", "b", "c"], "8b", 512, 33, ["tknp"], "e2e")
    updated = copy.deepcopy(original)
    updated["port_base"] = 8700
    updated["slurm_job_id"] = "2"
    check_resume_plan(original, updated)
    for key, value in [("model", "70b"), ("max_model_len", 32768), ("gpu_memory_utilization", .8)]:
        changed = copy.deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError):
            check_resume_plan(original, changed)
    changed = copy.deepcopy(original)
    changed["trials"][0]["staged_decode"] = True
    with pytest.raises(ValueError):
        check_resume_plan(original, changed)


def test_capacity_control_uses_observed_interval_and_never_forces_a_win():
    assert crossover_batch(100, 86) == 94
    assert crossover_batch(86, 86) is None
    assert crossover_batch(80, 86) is None
    assert crossover_batch(89, 88) is None


def test_resident_pp_requires_explicit_ordinary_measurement_and_stays_unstaged():
    config = configuration("1", ["p", "a", "b"], "8b", 512, 33, ["tknp", "dp", "pp"], "e2e")
    with pytest.raises(ValueError, match="PP resident"):
        stage_trials(config, 8, 8)
    stage_trials(config, 8, 8, skip_pp=True)
    assert [t.get("staged_decode", False) for t in config["trials"]] == [True, True, False]
    with pytest.raises(ValueError, match="fit"):
        stage_trials(config, 16, 8, skip_pp=True)


@pytest.mark.parametrize("tp", [1, 8])
def test_sampling_warmup_releases_unused_cache_before_native_collective(monkeypatch, tp):
    from types import SimpleNamespace
    events = []
    cuda = SimpleNamespace(
        mem_get_info=lambda device: (100, 200), memory_allocated=lambda device: 64,
        memory_reserved=lambda device: 128,
        empty_cache=lambda: events.append("empty_cache"),
        synchronize=lambda device: events.append("synchronize"))
    fake_torch = SimpleNamespace(cuda=cuda, bfloat16="bf16",
        ones=lambda *args, **kwargs: "probe", ones_like=lambda value: value,
        testing=SimpleNamespace(assert_close=lambda *args, **kwargs: events.append("assert_close")))
    group = SimpleNamespace(world_size=tp,
        all_gather=lambda *args, **kwargs: events.append("all_gather") or "gathered")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", SimpleNamespace(get_tp_group=lambda: group))
    worker = SimpleNamespace(device="cuda:0")
    evidence = warm_sampling_collectives(worker)
    assert events == (["empty_cache", "all_gather", "assert_close", "synchronize"] if tp > 1
                      else ["empty_cache", "synchronize"])
    assert evidence["tp_world_size"] == tp
    assert evidence["before"]["torch_reserved_bytes"] == 128


def test_eager_and_capture_cohort_threads_bind_device_before_stream_context():
    import ast
    module = ast.parse((HERE.parents[2] / "vllm/v1/worker/gpu_ubatch_wrapper.py").read_text())
    functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
                 and n.name in {"_ubatch_thread", "_capture_ubatch_thread"}}
    assert len(functions) == 2
    for function in functions.values():
        assert ast.unparse(function.body[0]) == "torch.cuda.set_device(self.device)"
