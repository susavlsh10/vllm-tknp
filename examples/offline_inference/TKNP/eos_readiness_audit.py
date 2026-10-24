#!/usr/bin/env python3
"""Recheck preserved EOS evidence without allocating or running GPUs."""
import argparse
import hashlib
import json
from pathlib import Path

from eos_readiness import audit_point


def audit(root):
    result = dict(scope="Single-node H100 8B script readiness, not a paper performance or trained-quality evaluation",
                  weights="dummy BF16", kv="BF16", end_to_end=[], paired_tokens=[],
                  completed_requests=0, output_tokens=0)
    for family in ("short", "long32k"):
        directory = root / family
        config = json.loads((directory / "matrix.json").read_text())
        state = json.loads((directory / "readiness.json").read_text())
        workload = [json.loads(line) for line in (directory / "workload.jsonl").read_text().splitlines()]
        assert hashlib.sha256((directory / "workload.jsonl").read_bytes()).hexdigest() == state["workload_sha256"]
        assert len({w["max_tokens"] for w in workload}) == 1
        for trial in config["trials"]:
            selected = [r for r in state["trials"] if r["name"] == trial["name"] and r["status"] == "passed"]
            if len(selected) != 1:
                raise ValueError("Missing or ambiguous selected trial: " + family + "/" + trial["name"])
            path = directory / Path(selected[0]["path"]).name
            evidence = audit_point(path, trial, len(workload), workload[0]["max_tokens"])
            client = evidence["client"]
            result["completed_requests"] += client["completed"]
            result["output_tokens"] += client["output_tokens"]
            result["end_to_end"].append(dict(family=family, name=trial["name"],
                context=len(workload[0]["prompt_token_ids"]), requests=client["completed"],
                output_tokens=client["output_tokens"], path=str(path.resolve()),
                all_workers_full_graph=True, bf16_verified=True, drained=True,
                active_gpus=sum(len(s["gpus"]) for s in trial["servers"]),
                allocated_gpus=8, client_output_tps=client["system_output_tps"]))
        names = {t["name"] for t in config["trials"]}
        for off, on in (("tknp2", "tknp2-dbo"), ("tp2-tknp2", "tp2-tknp2-dbo")):
            if not {off, on} <= names:
                continue
            def tokens(name):
                selected = next(r for r in state["trials"] if r["name"] == name and r["status"] == "passed")
                path = directory / Path(selected["path"]).name / "client/requests.jsonl"
                return {r["request_id"]: r["token_ids"] for r in map(json.loads, path.read_text().splitlines())}
            a, b = tokens(off), tokens(on)
            result["paired_tokens"].append(dict(family=family, off=off, on=on,
                exact=a == b, checked_tokens=sum(len(v) for v in a.values())))
    pair = json.loads((root / "standalone_pair.json").read_text())
    for name, enabled in (("standalone_off", False), ("standalone_on", True)):
        metrics = list((root / name).glob("steady_rank*.json"))
        assert sorted(p.name for p in metrics) == [f"steady_rank{i}.json" for i in range(4)]
        for path in metrics:
            data = json.loads(path.read_text())
            assert data["enable_dbo"] == enabled and not data["profiled"]
            assert (data["batch"], data["context"]) == (32, 32768)
        log = (root / name / "node0.log").read_text()
        assert log.count("TKNP_CUDAGRAPH_REPLAY padded_size=32") >= 4
        if enabled:
            assert log.count("TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=32") >= 4
    result["standalone"] = pair
    result["standalone_scope"] = "TP2 x TKNP2 on four active GPUs, 32K/B32/128 output; dummy KV, distinct from real P/D; off/on ownership differs, not a causal overlap ablation"
    result["dcp"] = "Not run: valid Llama DCP requires at least TP16+DCP2, beyond this one-node eight-GPU budget"
    result["all_infrastructure_checks_pass"] = True
    result["paired_dummy_token_checks_exact"] = all(p["exact"] for p in result["paired_tokens"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_infrastructure_checks_pass", "completed_requests",
                                            "output_tokens", "paired_dummy_token_checks_exact")}, indent=2))
