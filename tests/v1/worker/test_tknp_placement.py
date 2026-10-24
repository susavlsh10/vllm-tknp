# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[3]
placement = load("tknp_placement_pure", ROOT / "vllm/v1/core/sched/tknp_placement.py")
client = load("tknp_e2e_client_pure", ROOT / "examples/offline_inference/TKNP/e2e_workload.py")
slots = load("tknp_graph_slots_pure", ROOT / "vllm/v1/worker/tknp_serving_slots.py")
matrix = load("tknp_matrix_pure", ROOT / "examples/offline_inference/TKNP/e2e_matrix.py")
server = load("tknp_server_pure", ROOT / "examples/offline_inference/TKNP/e2e_server.py")
numerics = load("tknp_numerics_pure", ROOT / "examples/offline_inference/TKNP/diagnose_e2e_tokens.py")
Request = placement.PlacementRequest


def test_balanced_window_keeps_fifo_prefix_and_input_limits():
    requests = [Request(str(i), n, 1) for i, n in enumerate([20, 80, 40, 100])]
    free, limits, loads = [100, 100], [1, 1], [0, 0]
    assigned = placement.place_balanced_window(
        requests, free, 16, loads, request_limits=limits)
    assert list(assigned) == ["0", "1"]
    assert free == [100, 100] and limits == [1, 1] and loads == [0, 0]
    assert [r.request_id for r in requests] == ["0", "1", "2", "3"]


def test_balanced_window_does_not_skip_nonfitting_oldest_request():
    assert placement.place_balanced_window(
        [Request("old", 1000, 1), Request("new", 1, 1)], [1, 1], 16) == {}


def test_balanced_window_root_bias_and_lifetime_reservations():
    requests = [Request(str(i), n, 1) for i, n in enumerate([100, 80, 40, 20])]
    assigned = placement.place_balanced_window(
        requests, [20]*4, 16, request_limits=[1]*4, root_bias_tokens=100)
    assert assigned["3"] == 0
    for owner in range(4):
        assert sum(r.blocks(16) for r in requests
                   if assigned[r.request_id] == owner) <= 20


def test_lifetime_blocks_and_fifo_capacity_queue():
    requests = [Request("large", 100, 29), Request("small", 1, 1)]
    assert requests[0].blocks(128) == 2
    assert placement.place_fifo(requests, [1, 1], 128) == {}
    assert placement.place_fifo(requests, [1, 2], 128) == {"large": 1, "small": 0}


def test_future_growth_counts_inflight_receive_not_unadmitted_waiting():
    # Two admitted requests each need 4 blocks eventually, currently holding 2.
    # A third queued request holding zero blocks must not deadlock admission.
    assert placement.unreserved_lifetime_blocks(10, 16, [(32, 17, 2), (32, 17, 2), (32, 17, 0)]) == 6


def test_finished_growth_and_zero_capacity_are_safe():
    assert placement.unreserved_lifetime_blocks(2, 16, [(32, 17, 2)]) == 0
    assert placement.unreserved_lifetime_blocks(10, 16, [(32, 17, 4)]) == 10
    with pytest.raises(ValueError):
        placement.unreserved_lifetime_blocks(-1, 16, [])


def test_cost_placement_balances_long_and_short():
    requests = [Request(str(i), n, 1) for i, n in enumerate([1000, 100, 100, 100])]
    owners = placement.place_fifo(requests, [100, 100], 128)
    assert owners == {"0": 0, "1": 1, "2": 1, "3": 1}


def test_existing_load_and_deterministic_ties():
    req = [Request("new", 10, 10)]
    assert placement.place_fifo(req, [10, 10], 128, [100, 0]) == {"new": 1}
    assert placement.place_fifo(req, [10, 10], 128) == {"new": 0}


def test_invalid_policy_and_capacity():
    with pytest.raises(ValueError):
        placement.place_fifo([], [-1], 128)
    with pytest.raises(ValueError):
        placement.place_fifo([], [10], 128, policy="unknown")


def test_workload_is_reproducible_and_open_loop():
    a = client.make_workload(10, [128, 1024], [4, 16], 10, 7)
    assert a == client.make_workload(10, [128, 1024], [4, 16], 10, 7)
    assert a[0]["arrival_s"] == 0
    assert a[-1]["arrival_s"] > a[1]["arrival_s"]
    assert [len(r["prompt_token_ids"]) for r in a[:2]] == [128, 1024]


def test_failed_requests_are_not_goodput():
    rows = [{"requested_tokens": 4, "output_tokens": 4, "ttft_s": 0.1,
             "tpot_s": 0.01, "e2e_s": 0.2},
            {"requested_tokens": 4, "output_tokens": 2, "error": "timeout"}]
    summary = client.summarize(rows, 2, 1, 0.1)
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["system_output_tps"] == 2
    assert summary["request_goodput_rps"] == 0.5


def test_graph_lane_admission_limit_is_independent_of_kv_capacity():
    requests = [Request(str(i), 10, 10) for i in range(5)]
    owners = placement.place_fifo(requests, [100, 100], 128, request_limits=[1, 2])
    assert owners == {"0": 0, "1": 1, "2": 1}


def test_slot_permutation_masks_empty_owner_and_cohort():
    layout = [0, 0, 1, 1, 1, 1, 1, 1]
    plan = slots.plan_slots([1, 0, 1], layout, 1)
    assert plan.request_slots == [2, 0, 3]
    assert plan.local_request_indices == [0, 2, None, None, None, None]
    assert slots.plan_slots([1], layout, 0).local_request_indices == [None, None]
    with pytest.raises(ValueError):
        slots.plan_slots([0, 0, 0], layout, 0)


def test_explicit_capture_split_capacity_and_shape():
    assert slots.validate_capture_split([2, 6], 8, [100, 100], 16) == [2, 6]
    for counts, total, free in [([2, 6], 7, [100, 100]), ([2, 6], 8, [100, 90]),
                                 ([2, -6], -4, [100, 100]), ([8], 8, [100, 100])]:
        with pytest.raises(ValueError):
            slots.validate_capture_split(counts, total, free, 16)


def test_matrix_rejects_overlapping_gpus_and_dcp_multiplication():
    with pytest.raises(ValueError):
        matrix.validate_matrix({"trials": [{"name": "bad", "servers": [
            {"gpus": [0]}, {"gpus": [0]}]}]}, 4)
    matrix.validate_matrix({"trials": [{"name": "valid", "servers": [
        {"role": "decode", "gpus": [0, 1], "tp": 2, "dcp": 2}]}]}, 4)
    with pytest.raises(ValueError, match="DCP interleaved KV"):
        matrix.validate_matrix({"trials": [{"name": "unsafe-transport", "servers": [
            {"role": "decode", "gpus": [0, 1], "tp": 2, "dcp": 2, "connector": "nixl"}]}]}, 4)


def test_async_receive_abort_waits_for_destination_write_completion():
    req = SimpleNamespace(request_id="r", status=SimpleNamespace(name="WAITING_FOR_REMOTE_KVS"))
    assert server.defer_kv_abort(req, set())
    assert not server.defer_kv_abort(req, {"r"})
    assert not server.defer_kv_abort(None, set())
    req.status.name = "RUNNING"
    assert not server.defer_kv_abort(req, set())


def test_repetitions_have_fresh_paths_and_alternate_baseline_order(tmp_path, monkeypatch):
    config = {"trials": [{"name": name, "servers": [{"role": "decode", "gpus": [0]}]}
                         for name in ("off", "on")]}
    config_path, workload, output = tmp_path / "matrix.json", tmp_path / "workload.jsonl", tmp_path / "results"
    config_path.write_text(json.dumps(config))
    workload.write_text("{}\n")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4)))
    monkeypatch.setattr(sys, "argv", ["matrix", "--matrix", str(config_path), "--workload", str(workload),
                                     "--output", str(output), "--repetitions", "2", "--alternate-order"])
    paths = []
    def run(trial, config, workload, path):
        paths.append(path.name)
        return {"trial": trial, "completed": 1}
    monkeypatch.setattr(matrix, "run_trial", run)
    matrix.main()
    assert paths == ["rep00-off", "rep00-on", "rep01-on", "rep01-off"]
    results = json.loads((output / "validation.json").read_text())
    assert [r["repetition"] for r in results] == [0, 0, 1, 1]
    assert [r["execution_index"] for r in results] == [0, 1, 2, 3]


def test_near_tie_diagnostic_does_not_hide_greedy_divergence():
    a = {"r": {"token_ids": [1, 2], "server_request_id": "a", "prompt_tokens": 128}}
    b = {"r": {"token_ids": [1, 3], "server_request_id": "b", "prompt_tokens": 128}}
    la = {("a", 1): {"ids": [2, 3], "values": [14.5625, 14.5]}}
    lb = {("b", 1): {"ids": [2, 3], "values": [14.5625, 14.5625]}}
    report = numerics.compare(a, b, la, lb)
    assert report["exact_requests"] == 0
    assert report["divergences"][0]["consistent_with_bf16_near_tie"]
    lb[("b", 1)]["values"] = [14.0, 15.0]
    assert not numerics.compare(a, b, la, lb)["divergences"][0]["consistent_with_bf16_near_tie"]


def test_staged_residency_rejects_queueing_and_honors_owner_lanes(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.tknp_placement", placement)
    health = [{"config": {"placement": "cost"}, "reservation_block_tokens": 16,
               "resident_limits": [{"capacity_tokens": 1000, "request_slots": 1},
                                   {"capacity_tokens": 1000, "request_slots": 2}]}]
    workload = client.make_workload(3, [32], [4], 0, 1)
    client.validate_staged_residency(workload, health, "least-work")
    with pytest.raises(ValueError, match="cannot all fit resident"):
        client.validate_staged_residency(client.make_workload(4, [32], [4], 0, 1), health, "least-work")
    health[0]["resident_limits"][1]["capacity_tokens"] = 16
    with pytest.raises(ValueError, match="cannot all fit resident"):
        client.validate_staged_residency(workload, health, "least-work")


def test_synthetic_matrix_cannot_claim_real_handoff():
    matrix.validate_matrix({"trials": [{"name": "dummy", "servers": [
        {"role": "decode", "gpus": [0, 1], "tknp": 2, "connector": "synthetic"}]}]}, 4)
    with pytest.raises(ValueError, match="Synthetic KV"):
        matrix.validate_matrix({"trials": [{"name": "bad", "servers": [
            {"role": "prefill", "gpus": [0]},
            {"role": "decode", "gpus": [1, 2], "tknp": 2, "connector": "synthetic"}]}]}, 4)


def test_controlled_generator_uses_model_bos():
    assert client.make_workload(1, [32], [4], 0, 1, bos_token_id=1)[0]["prompt_token_ids"][0] == 1


def test_backed_singleton_cohort_shapes_fail_before_capture():
    server.reject_varying_singleton_positions([2, 2], [0, 4])
    server.reject_varying_singleton_positions([0, 1], [0, 1])
    with pytest.raises(ValueError, match="singleton changes size"):
        server.reject_varying_singleton_positions([1, 1], [0, 2])


def test_resident_pause_rejects_unvalidated_pp_batch_queue(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.tknp_placement", placement)
    with pytest.raises(ValueError, match="asynchronous PP"):
        client.validate_staged_residency(client.make_workload(1, [32], [4], 0, 1),
                                        [{"config": {"pp": 2}}], "least-work")


def test_arrivals_are_paired_across_context_distributions():
    uniform = client.make_workload(8, [128], [4], 10, 7)
    mixed = client.make_workload(8, [64, 256], [4, 8], 10, 7)
    assert [r["arrival_s"] for r in uniform] == [r["arrival_s"] for r in mixed]
