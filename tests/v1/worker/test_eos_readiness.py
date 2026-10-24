"""CPU-only checks for the bounded one-node EOS smoke matrix."""
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from eos_readiness import METHODS, configuration, audit_point
from e2e_matrix import validate_matrix


def test_all_methods_fit_one_node_without_overlapping_roles():
    config = configuration("llama8b", 512, 33, 2, METHODS)
    validate_matrix(config, 8)
    assert len(config["trials"]) == 9
    assert all(t["servers"][0]["gpus"] == [0] for t in config["trials"])


def test_dbo_preserves_fixed_owner_lanes_and_zero_c1_root():
    config = configuration("llama8b", 512, 33, 2, ["tknp2", "tknp2-dbo"])
    off, on = [t["servers"][1] for t in config["trials"]]
    assert off["owner_counts"] == on["owner_counts"] == [2, 6]
    assert not off.get("enable_dbo") and on["enable_dbo"]
    assert on["env"]["VLLM_TKNP_DBO_ZERO_ROOT_C1"] == "1"
    assert on["env"]["VLLM_TKNP_DBO_INDEPENDENT_COMM"] == "1"


def test_dp_is_real_replication_and_pp_has_native_concurrency():
    config = configuration("llama8b", 512, 33, 2, ["tp2-dp2", "tp2-pp2"])
    dp, pp = config["trials"]
    assert [s["tp"] for s in dp["servers"][1:]] == [2, 2]
    assert pp["servers"][1]["pp"] == 2
    assert pp["servers"][1]["pp_decode_microbatch"] == 4


def test_all_roles_use_same_explicit_bf16_pool_policy():
    config = configuration("llama8b", 32768, 33, 32, METHODS)
    assert config["require_bf16"] and config["kv_cache_dtype"] == "auto"
    assert config["kv_cache_memory_bytes"] == 32 * 1024 ** 3
    assert config["load_format"] == "dummy"


@pytest.mark.parametrize("corruption", ["execution", "dtype", "tokens"])
def test_evidence_audit_rejects_incomplete_or_mismatched_evidence(monkeypatch, corruption):
    import e1_pilot_report
    resource = dict(rank=0, model_dtype="torch.bfloat16", quantization=None,
                    kv_storage_dtypes=["torch.bfloat16"])
    evidence = dict(all_decode_workers_full_graph=True, all_servers_zero_measured_preemptions=True,
                    all_servers_kv_drained=True, resources={"server0": [resource]},
                    execution={"server0": [dict(rank=0)]},
                    client=dict(completed=8, output_tokens=264))
    trial = dict(servers=[dict(role="decode", gpus=[0])])
    monkeypatch.setattr(e1_pilot_report, "analyze", lambda _: evidence)
    assert audit_point(Path("unused"), trial, 8, 33) is evidence
    if corruption == "execution":
        evidence["execution"]["server0"] = []
    elif corruption == "dtype":
        resource["kv_storage_dtypes"] = ["torch.float8_e4m3fn"]
    else:
        evidence["client"]["output_tokens"] = 263
    with pytest.raises(RuntimeError):
        audit_point(Path("unused"), trial, 8, 33)
