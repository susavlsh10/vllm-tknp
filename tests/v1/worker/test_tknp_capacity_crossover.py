# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location("capacity_pure", HERE / "e1_capacity_crossover.py")
    capacity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capacity)
finally:
    sys.path.pop(0)


def test_request_lifetime_cap_reserves_null_and_rounds_partial_pages():
    assert capacity.lifetime_caps([{"rank": 0, "kv_blocks": 10, "block_tokens": 16}], 33) == [3]
    assert capacity.lifetime_caps([{"rank": 1, "kv_blocks": 4, "block_tokens": 16},
                                   {"rank": 0, "kv_blocks": 10, "block_tokens": 16}], 33) == [3, 1]


def test_comparison_has_equal_gpu_budget_and_no_microbatching():
    for method, caps in (("tknp", [2, 16]), ("dp2", [2, 2])):
        roles = capacity.roles(method, caps, 36865)
        capacity.validate_matrix({"trials": [{"name": method, "servers": roles}]}, 4)
        assert [gpu for role in roles for gpu in role["gpus"]] == [0, 1, 2]
        assert all(role["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] == "1" for role in roles)
        assert all(role["cuda_graph"] for role in roles[1:])
        assert all(not role.get("enable_dbo", False) for role in roles)
        assert roles[0]["kv_cache_memory_bytes"] == 32 * 1024 ** 3


def test_tknp_quota_does_not_force_equal_owners():
    decoder = capacity.roles("tknp", [2, 16], 36865)[1]
    assert decoder["owner_counts"] == [2, 16]
    assert decoder["graph_batch_size"] == decoder["max_num_seqs"] == 18
    assert decoder["env"]["VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS"] == "36865"


def test_dp_replicas_use_their_individual_measured_capacity():
    decoders = capacity.roles("dp2", [2, 3], 36865)[1:]
    assert [role["max_num_seqs"] for role in decoders] == [2, 3]
    assert [role["graph_batch_size"] for role in decoders] == [2, 3]


def test_unpaired_data_is_not_reported_as_a_comparison():
    tk = {"method": "tknp", "offered_requests": 16,
          "client": {"system_output_tps": 200, "duration_s": 50}}
    dp = {"method": "dp2", "offered_requests": 16,
          "client": {"system_output_tps": 100, "duration_s": 100}}
    assert capacity.compare([tk]) == []
    assert capacity.compare([tk, dp])[0]["tknp_vs_dp_output_tps_ratio"] == 2
