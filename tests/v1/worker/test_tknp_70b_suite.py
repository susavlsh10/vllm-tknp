import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from e1_70b_suite import build_trial, quotas, topology, best_tknp_capacity, measurement_order, retryable_port_startup, measurement_passes, tknp_capacity_rebalance_needed
from e2e_matrix import validate_matrix
from e2e_slurm import launch_role
from e2e_server import aligned_owner_counts, owner_graphs_consistent


def test_quotas_are_capacity_limited():
    assert quotas([3, 15], 18) == [3, 15]
    assert quotas([3, 15], 6) == [3, 3]
    assert quotas([3, 15], 24) == [3, 15]


def test_native_dp_does_not_inherit_lifetime_slot_cap():
    trial = build_trial("dp-native", 32768, 1025, 18, [1, 2], [3, 3], "e2e")
    validate_matrix({"trials": [trial]}, 4)
    assert all(r["max_num_seqs"] == 9 for r in trial["servers"][1:])
    assert all(r["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] == "0" for r in trial["servers"][1:])


def test_lifetime_dp_is_separate_baseline():
    trial = build_trial("dp-lifetime", 32768, 1025, 18, [1, 2], [3, 3], "e2e")
    assert all(r["max_num_seqs"] == 3 for r in trial["servers"][1:])


def test_synthetic_dcp_never_has_prefill_producer():
    trial = build_trial("dcp", 32768, 1025, 8, list(range(16)), [12], "decode")
    validate_matrix({"trials": [trial]}, 16)
    assert len(trial["servers"]) == 1
    assert trial["servers"][0]["connector"] == "synthetic"


def test_distributed_launch_uses_actual_nodes_and_only_assigned_devices():
    role = dict(gpus=list(range(4, 12)), tknp=8)
    command, url = launch_role({"slurm_job_id": "123", "slurm_nodes": ["p", "d0", "d1"]},
                               role, 1, ["python", "-m", "torch.distributed.run", "--standalone",
                                         "--nproc-per-node=8", "/repo/e2e_server.py", "--port", "8201"], {})
    assert "--nodelist=d0,d1" in command
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command
    assert url == "http://d0:8201"
    assert command[-2:] == ["--host", "0.0.0.0"]


def test_all_main_topologies_use_eight_decode_gpus():
    for method in ["tknp", "tknp-tp2", "dp-native", "dp-lifetime", "dp-tp2", "tp", "pp", "tp4-pp2"]:
        tp, pp, tk, dcp, replicas = topology(method, 8)
        assert tp * pp * tk * replicas == 8
        assert tp % dcp == 0


def test_dcp_is_not_faked_below_the_kv_head_replication_threshold():
    assert topology("dcp", 16) == (16, 1, 1, 2, 1)
    for size in [2, 4, 8]:
        try:
            topology("dcp", size)
        except ValueError as exc:
            assert "minimum TP16" in str(exc)
        else:
            raise AssertionError("Unsupported GQA DCP topology was admitted")


def test_graph_alignment_preserves_physical_owner_capacities():
    assert aligned_owner_counts([3, 16], 2) == [3, 15]
    assert aligned_owner_counts([3, 16, 16, 16], 4) == [3, 15, 15, 15]
    trial = build_trial("tknp", 32768, 1025, 19, [1, 2], [3, 16], "e2e")
    assert trial["servers"][1]["graph_batch_size"] == 18
    trial = build_trial("tknp-tp2", 32768, 1025, 51, [1, 2, 5, 6], [20, 33], "e2e")
    assert trial["servers"][1]["graph_batch_size"] == 52


def test_prefill_pool_and_pipeline_batches_use_disjoint_complete_budget():
    trial = build_trial("pp-pipelined", 32768, 4097, 48, [1, 2, 5, 6], [49], "e2e", [0, 3, 4, 7])
    validate_matrix({"trials": [trial]}, 8)
    decoder = trial["servers"][-1]
    assert decoder["max_num_seqs"] == 48
    assert decoder["graph_batch_size"] == decoder["pp_decode_microbatch"] == 12
    assert "max_num_batched_tokens" not in decoder
    assert sum(len(r["gpus"]) for r in trial["servers"]) == 8


def test_crossover_uses_best_hybrid_not_weakest_tknp_graph_capacity():
    capacities = {"tknp": [1] + [5] * 15, "tknp-tp8": [41, 45], "dp-tp8": [40, 40]}
    assert best_tknp_capacity(capacities, 16) == 86


def test_report_keeps_prefill_and_post_prefill_stage_boundaries_separate():
    from e1_70b_report import stage_latencies
    result = stage_latencies([dict(prefill_s=3, ttft_from_dispatch_s=5,
                                    e2e_from_dispatch_s=15, tpot_s=.1)], True)
    assert result["percentiles"]["prefill_queue_and_compute_s"]["p50"] == 3
    assert result["percentiles"]["decode_admission_handoff_to_first_token_s"]["p50"] == 2
    assert result["percentiles"]["decode_admission_handoff_and_generation_s"]["p50"] == 12
    synthetic = stage_latencies([dict(prefill_s=None, ttft_from_dispatch_s=2,
                                      e2e_from_dispatch_s=12, tpot_s=.1)], False)
    assert synthetic["percentiles"]["decode_admission_handoff_and_generation_s"]["p50"] == 12


def test_measurement_order_preserves_all_baselines():
    source = {m: [] for m in ["tknp", "dp-native", "dp-tp2", "tp", "pp", "dp-lifetime"]}
    assert measurement_order(source) == ["tknp", "dp-tp2", "tp", "pp", "dp-native", "dp-lifetime"]
    assert set(measurement_order(source)) == set(source)


def test_only_unmeasured_port_startup_failure_can_retry():
    error = "RuntimeError: Server exited during initialization"
    assert retryable_port_startup(error, ["TCPStore EADDRINUSE"])
    assert not retryable_port_startup("Client failed", ["TCPStore EADDRINUSE"])
    assert not retryable_port_startup(error, ["CUDA out of memory"])


def test_frontier_control_prioritizes_families_without_dropping_configurations():
    source = dict.fromkeys(["dcp", "tp", "tknp", "tknp-tp8", "dp-native", "dp-tp8", "tp4-pp4", "pp"])
    ordered = measurement_order(source, frontier_control=True)
    assert ordered[:5] == ["dcp", "tp", "tknp-tp8", "dp-tp8", "tp4-pp4"]
    assert set(ordered) == set(source)


def test_cross_context_order_keeps_paired_repeats_together():
    assert measurement_passes([32, 64, 96], 1) == [
        (32, False), (64, False), (96, False), (96, True), (64, True), (32, True)]
    assert measurement_passes([32], 3) == [(32, None)]


def test_failure_resolution_does_not_merge_independent_repeats():
    from e1_70b_report import canonical_attempt
    assert canonical_attempt("point-tknp-repeat1-retry42") == "point-tknp-repeat1"
    assert canonical_attempt("point-tknp-retry12-capacity-recalibrated") == "point-tknp"


def test_strong_hybrid_pipeline_baseline_uses_all_sixteen_gpus():
    assert topology("tp8-pp2", 16) == (8, 2, 1, 1, 1)
    trial = build_trial("tp8-pp2", 98304, 257, 86, list(range(16)), [88], "decode")
    validate_matrix({"trials": [trial]}, 16)
    assert trial["servers"][0]["pp_decode_microbatch"] == 43


def test_runtime_owner_rebalance_removes_avoidable_tail_wave():
    trial = build_trial("tknp-tp2", 32768, 4097, 48, [1, 2, 5, 6], [19, 30], "e2e")
    assert trial["servers"][-1]["owner_counts"] == [19, 29]
    assert tknp_capacity_rebalance_needed(trial, [18, 30], 48)
    assert not tknp_capacity_rebalance_needed(trial, [18, 29], 48)
    assert not tknp_capacity_rebalance_needed(trial, [19, 30], 48)
    calibrated = build_trial("tknp-tp2", 32768, 4097, 48, [1, 2, 5, 6], [18, 30], "e2e")
    assert calibrated["servers"][-1]["owner_counts"] == [18, 30]


def test_startup_capture_rebalance_keeps_exact_lanes_and_capacity_safety():
    path = HERE.parents[2] / "vllm/v1/worker/tknp_serving_slots.py"
    spec = importlib.util.spec_from_file_location("standalone_serving_slots", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile("from __future__ import annotations\n" + path.read_text(), str(path), "exec"), module.__dict__)
    assert module.fit_capture_split([28, 34, 33, 33], 128, [271, 341, 341, 341], 10) == [27, 34, 34, 33]
    assert module.fit_capture_split([3, 15], 18, [31, 151], 10) == [3, 15]
    try:
        module.fit_capture_split([3, 15], 18, [21, 151], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Capture rebalancing overcommitted actual KV pools")


def test_graph_owner_layout_must_agree_across_tensor_parallel_shards():
    good = [{"graph_splits": {128: [27, 34, 34, 33]}}] * 16
    assert owner_graphs_consistent(good)
    assert not owner_graphs_consistent(good + [{"graph_splits": {128: [28, 34, 33, 33]}}])
