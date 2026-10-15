# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location("pilot_pure", HERE / "e1_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
finally:
    sys.path.pop(0)


def test_on_off_have_identical_memory_gpu_ownership_and_graph_settings():
    for batch in (8, 16, 32, 64):
        off = pilot.decoder(32768, batch, 129, False)
        on = pilot.decoder(32768, batch, 129, True)
        assert on.pop("enable_dbo")
        assert not off.pop("enable_dbo")
        assert on == off
        config = pilot.common_config(32768, batch, 129)
        assert config["gpu_memory_utilization"] == 0.8
        assert "kv_cache_memory_bytes" not in config
        assert config["require_bf16"]


def test_large_batches_use_balanced_total_owners_and_non_singleton_peer_cohorts():
    for batch in (16, 32, 64):
        role = pilot.decoder(32768, batch, 129, True)
        root, peer = role["owner_counts"]
        assert root == peer == batch // 2
        split = int(float(role["env"]["VLLM_TKNP_DBO_C0_FRACTION"]) * batch)
        assert split - root >= 2
        assert batch - split >= 2


def test_capacity_reserves_null_block_and_rounds_lifetime_up():
    assert pilot.capacity([{"kv_blocks": 10, "block_tokens": 16}], 32, 1) == [3]
    assert pilot.capacity([{"kv_blocks": 10, "block_tokens": 16}], 32, 17) == [2]


def test_resident_plan_uses_aggregate_capacity_not_equal_root_quota():
    assert pilot.fit_owner_plan([32, 32], [31, 35]) == [31, 33]
    assert pilot.fit_owner_plan([16, 48], [31, 35]) == [29, 35]
    assert pilot.fit_owner_plan([64, 64], [31, 35]) is None


def test_prefill_exports_use_the_same_lifetime_backpressure_as_decode():
    assert "env" not in pilot.producer()
    role = pilot.producer(True)
    assert role["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] == "1"
    assert role["kv_cache_memory_bytes"] == pilot.producer()["kv_cache_memory_bytes"]
