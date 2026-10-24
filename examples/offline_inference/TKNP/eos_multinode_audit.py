#!/usr/bin/env python3
"""CPU-only fail-closed readback of the bounded EOS readiness campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from eos_readiness import audit_point
from e1_70b_report import canonical_attempt

FAMILIES = {
    "decode8b": ([512], ["tknp", "tknp-dbo", "dp", "pp", "dcp"]),
    "e2e8b": ([512], ["tknp", "tknp-dbo", "dp", "pp"]),
    "decode70b": ([32768, 65536, 98304], ["tknp", "tknp-dbo", "dp", "pp", "dcp"]),
    "e2e70b": ([32768], ["tknp", "tknp-dbo", "dp", "pp"]),
    "resident_synthetic70b": ([32768], ["tknp", "tknp-dbo", "dp", "pp"]),
    "mixed70b": ([98304], ["tknp", "tknp-dbo"]),
    "e2e70b96k": ([98304], ["tknp-dbo", "pp"]),
}


def tokens(path):
    return {r["request_id"]: r["token_ids"] for r in map(json.loads,
        (path / "client/requests.jsonl").read_text().splitlines())}


def audit(root):
    aliases = {name: name + "_85pct" for name in
        ["decode70b", "e2e70b", "resident_synthetic70b", "mixed70b", "e2e70b96k", "pair70b", "capacity70b", "crossover70b"]}
    aliases.update({name: name + "_supported_85pct" for name in
                    ["e2e70b", "resident_synthetic70b", "e2e70b96k"]})
    result = dict(scope="EOS multi-node readiness only; dummy BF16 weights, BF16 KV; not paper measurements or trained-model correctness",
        queue_metric_scope="Legacy max_waiting_capacity counts scheduler WAITING status, not a reason-coded KV stall. It can include token/sequence limits and PP pipeline scheduling. DP's lifetime slot limits are derived from physical KV pools; do not infer PP memory exhaustion from WAITING alone.",
        selected=[], retained_failures=[], token_pairs=[], resident_comparison=[], capacities=[],
        completed_requests=0, output_tokens=0,
        limitations=["Real NIXL KV handoff into DCP is not implemented; DCP uses labeled synthetic KV",
                     "PP asynchronous batch queues do not support resident pause/resume; PP is verified in ordinary E2E mode and excluded from the staged real-versus-synthetic comparison",
                     "70B is a hand-authored architecture proxy with the cached Llama 3.1 tokenizer, not a downloaded checkpoint",
                     "No 405B BF16 TP8 validation on H100; that root weight shard does not fit 80GB",
                     "Readiness pilots do not establish paper-quality performance or a universal microbatch gain"])
    evidence_by_key = {}
    for family, (contexts, methods) in FAMILIES.items():
        directory = root / aliases.get(family, family)
        state = json.loads((directory / "readiness.json").read_text())
        result["retained_failures"].extend(dict(family=family, **row) for row in state["attempts"] if row["status"] != "passed")
        for context in contexts:
            config = json.loads((directory / f"matrix-c{context}.json").read_text())
            assert len(config["trials"]) == len(methods) and {t["name"] for t in config["trials"]} == set(methods)
            if "70b" in family:
                assert config["gpu_memory_utilization"] == 0.85
            workload_path = directory / f"workload-c{context}.jsonl"
            workload = list(map(json.loads, workload_path.read_text().splitlines()))
            count, outputs = len(workload), workload[0]["max_tokens"]
            lengths = config.get("workload_spec", {}).get("lengths", [context])
            assert all(len(r["prompt_token_ids"]) == lengths[i % len(lengths)] and r["max_tokens"] == outputs for i,r in enumerate(workload))
            selected = {}
            for trial in config["trials"]:
                key = f"c{context}-{trial['name']}"
                rows = [r for r in state["attempts"] if r["key"] == key and r["status"] == "passed"]
                assert len(rows) == 1, f"Missing/ambiguous passing point: {family}/{key}"
                row = rows[0]
                assert hashlib.sha256(workload_path.read_bytes()).hexdigest() == row["workload_sha256"]
                path = directory / Path(row["path"]).name
                evidence = audit_point(path, trial, count, outputs)
                if family in {"e2e70b", "resident_synthetic70b", "e2e70b96k"}:
                    staged = bool(evidence["client"].get("staged_decode"))
                    assert staged == (trial["name"] != "pp")
                selected[trial["name"]] = path
                evidence_by_key[(family, context, trial["name"])] = evidence
                result["completed_requests"] += count
                result["output_tokens"] += count * outputs
                result["selected"].append(dict(family=family, context=context, method=trial["name"],
                    count=count, tokens=outputs, active_gpus=sum(len(s["gpus"]) for s in trial["servers"]),
                    capacities=row["capacities"], client=evidence["client"],
                    all_ranks_graph=True, bf16=True, drained=True, path=str(path)))
            if {"tknp", "tknp-dbo"} <= selected.keys():
                off, on = tokens(selected["tknp"]), tokens(selected["tknp-dbo"])
                assert off == on, f"Dummy token mismatch: {family}/c{context}"
                result["token_pairs"].append(dict(family=family, context=context, exact=True,
                    checked_tokens=sum(len(t) for t in off.values())))
    for method in ["tknp", "tknp-dbo", "dp"]:
        real = evidence_by_key[("e2e70b", 32768, method)]
        synthetic = evidence_by_key[("resident_synthetic70b", 32768, method)]
        assert real["client"]["real_kv_handoff"] and not synthetic["client"]["real_kv_handoff"]
        assert real["client"].get("staged_decode") and synthetic["client"].get("staged_decode")
        result["resident_comparison"].append(dict(method=method,
            real=real["client"]["staged_decode"], synthetic=synthetic["client"]["staged_decode"],
            real_engine=real.get("resident_full_batch_engine"),
            synthetic_engine=synthetic.get("resident_full_batch_engine")))
    unit = ET.parse(root / "unit_final_runtime.xml")
    cases = list(unit.iter("testcase"))
    assert len(cases) >= 95 and all(not any(c.tag in {"error", "failure", "skipped"} for c in t) for t in cases)
    result["unit_tests"] = len(cases)
    assert "TOKENIZER_IDENTICAL_BACKEND_AND_100_ENCODINGS_PASS" in (root / "tokenizer_validation.log").read_text()
    result["copied_tokenizer_identity_verified"] = True
    fix_config = json.loads((root / "thread_device_fix90/matrix-c32768.json").read_text())
    fix_trial = fix_config["trials"][0]
    assert fix_config["gpu_memory_utilization"] == 0.9
    fix_evidence = audit_point(root / "thread_device_fix90/c32768-tknp-dbo", fix_trial, 16, 33)
    result["thread_device_fix90"] = dict(all_ranks_graph=True, bf16=True, drained=True,
        sampling_memory=[r["sampling_collective_warmup"] for r in fix_evidence["resources"]["server0"]])
    comm = (root / "changing_payload.log").read_text()
    assert all(f"INDEPENDENT_COHORT_NCCL_PASS rank={rank} changing_replays=64 layers=2" in comm for rank in range(4))
    result["changing_payload_graph"] = dict(nodes=2, ranks=4, replays=64, layers=2, passed=True)
    pair = json.loads((root / aliases["pair70b"] / "pair.json").read_text())
    assert len(pair["pairs"]) == 1 and pair["pairs"][0]["ranks"] == 16 and pair["pairs"][0]["exact_output_tokens"]
    for mode, dbo in [("off", False), ("pipelined", True)]:
        metrics = list((root / aliases["pair70b"] / mode).glob("steady_rank*.json"))
        assert {m.name for m in metrics} == {f"steady_rank{i}.json" for i in range(16)}
        for m in metrics:
            data = json.loads(m.read_text())
            assert not data["profiled"] and data["enable_dbo"] == dbo
            assert data["tensor_parallel_size"] == 8 and data["token_parallel_size"] == 2
            assert data["batch"] == 64 and data["context"] == 32768
    result["microbatch_pair"] = pair
    result["microbatch_pair_scope"] = "One unprofiled pair, different off/on attention ownership; system strategy pilot, not an overlap-only causal ablation"
    suite = json.loads((root / aliases["capacity70b"] / "suite.json").read_text())
    assert suite.get("finished_time") and not suite.get("paused_reason")
    assert suite["config"]["gpus_per_node"] == 8
    assert suite["config"]["gpu_memory_utilization"] == 0.85
    for method in ["tknp-tp8", "dp-tp8", "tp8-pp2", "dcp"]:
        rows = [r for r in suite["attempts"] if r["status"] == "completed" and canonical_attempt(r["name"]).endswith("-" + method)
                and not r["name"].startswith("probe-")]
        assert rows, "Missing overload measurement: " + method
        row = rows[-1]
        path = root / aliases["capacity70b"] / row["name"]
        trial = json.loads((path / "trial.json").read_text())["trial"]
        evidence = audit_point(path, trial, row["offered"], row["tokens"])
        assert evidence["client"]["workload_sha256"] == row["workload_sha256"]
        result["capacities"].append(dict(method=method, offered=row["offered"],
            measured_capacity=row["capacities"], client=evidence["client"],
            peak_waiting_per_server={name: max(v["max_waiting_capacity"] for k,v in evidence["telemetry"].items() if k.startswith(name+"/"))
                for name in evidence["resources"]},
            zero_preemptions=evidence["all_servers_zero_measured_preemptions"]))
    completed = {canonical_attempt(r["name"]) for r in suite["attempts"] if r["status"] == "completed"}
    result["retained_capacity_failures"] = [r for r in suite["attempts"] if r["status"] != "completed"]
    result["capacity_failures"] = [r for r in result["retained_capacity_failures"] if canonical_attempt(r["name"]) not in completed]
    assert not result["capacity_failures"], "Capacity collection has unresolved failures"
    crossing = json.loads((root / aliases["crossover70b"] / "summary.json").read_text())
    plan = crossing["plan"]
    assert plan["measured_dp_capacity"] < plan["batch"] <= plan["measured_tknp_capacity"]
    assert len(crossing["trials"]) == 4
    matrix = json.loads((root / aliases["crossover70b"] / "matrix.json").read_text())
    workload_hash = hashlib.sha256((root / aliases["crossover70b"] / "workload.jsonl").read_bytes()).hexdigest()
    assert all(row["workload_sha256"] == workload_hash for row in crossing["trials"])
    result["crossover"] = dict(plan=plan, trials=[])
    for trial in matrix["trials"]:
        path = root / aliases["crossover70b"] / trial["name"]
        evidence = audit_point(path, trial, plan["batch"], 33)
        result["crossover"]["trials"].append(dict(name=trial["name"], client=evidence["client"],
            telemetry=evidence["telemetry"], all_ranks_graph=True, bf16=True, drained=True))
    initial = root / "decode70b/readiness.json"
    if initial.exists():
        result["retained_90pct_attempts_not_selected"] = json.loads(initial.read_text())["attempts"]
    old_handoff = root / "e2e70b_85pct/readiness.json"
    if old_handoff.exists():
        result["retained_handoff_attempts_not_selected"] = json.loads(old_handoff.read_text())["attempts"]
    result["selected_directories"] = aliases
    result["all_required_readiness_checks_pass"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["all_required_readiness_checks_pass", "unit_tests", "completed_requests", "output_tokens"]}, indent=2))
