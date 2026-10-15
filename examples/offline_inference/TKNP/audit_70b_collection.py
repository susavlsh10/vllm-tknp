#!/usr/bin/env python3
"""CPU-only coverage audit of the selected 70B collection, not GPU evidence."""
import argparse
import json
from pathlib import Path


def audit(report):
    rows = report["trials"]
    eligible = [r for r in rows if r["comparison_eligible"]]
    checks = []

    def require(label, matches, methods, minimum_repeats=1):
        counts = {m: sum(r["method"] == m for r in matches) for m in methods}
        checks.append(dict(label=label, passed=all(n >= minimum_repeats for n in counts.values()),
                           required_trials_per_method=minimum_repeats, observed=counts))

    for context, batch in [(32768, 18), (65536, 8), (98304, 6)]:
        methods = ["tknp", "dp-native", "tp", "pp-pipelined"]
        if context != 98304:
            methods.append("dp-lifetime")
        require(f"one-node real P2+D2 c{context} b{batch}",
                [r for r in eligible if (r["family"], r["producer_gpus"], r["decoder_gpus"],
                 r["context"], r["offered_requests"], r["output_tokens"]) ==
                 ("e2e", 2, 2, context, batch, 4097)], methods)
    require("three matched 32K one-node TKNP/lifetime-DP trials", [r for r in eligible if
            (r["family"], r["producer_gpus"], r["decoder_gpus"], r["context"],
             r["offered_requests"], r["output_tokens"]) == ("e2e", 2, 2, 32768, 18, 4097)],
            ["tknp", "dp-lifetime"], 3)
    require("one-node below-fit control", [r for r in eligible if
            (r["family"], r["producer_gpus"], r["decoder_gpus"], r["context"],
             r["offered_requests"], r["output_tokens"]) == ("e2e", 2, 2, 32768, 6, 4097)],
            ["tknp", "dp-lifetime"])
    for context, batch in [(32768, 48), (65536, 26), (98304, 16)]:
        require(f"two-node real P4+D4 c{context} b{batch}", [r for r in eligible if
                (r["family"], r["producer_gpus"], r["decoder_gpus"], r["context"],
                 r["offered_requests"], r["output_tokens"]) == ("e2e", 4, 4, context, batch, 4097)],
                ["tknp", "tknp-tp2", "dp-native", "dp-tp2", "tp", "pp-pipelined"])
    require("repaired initial P1+D4 HTTP-failed native-DP control", [r for r in eligible if
            (r["family"], r["producer_gpus"], r["decoder_gpus"], r["context"],
             r["offered_requests"], r["output_tokens"]) == ("e2e", 1, 4, 32768, 51, 1025)], ["dp-native"])

    four = [r for r in eligible if (r["family"], r["decoder_gpus"], r["producer_gpus"],
            r["output_tokens"]) == ("synthetic-decode", 16, 0, 257)]
    all_methods = ["dcp", "tp", "tknp", "tknp-tp2", "tknp-tp4", "tknp-tp8", "dp-native",
                   "dp-lifetime", "dp-tp2", "dp-tp4", "dp-tp8", "pp", "tp4-pp4", "tp8-pp2"]
    controls = ["dcp", "tp", "tknp-tp8", "dp-tp8", "tp8-pp2", "dp-native"]
    crossover_batches = {}
    for context in [32768, 65536, 98304]:
        candidates = [r for r in four if r["context"] == context]
        # Fixed from the original measured-pool plans; do not mistake an
        # overload control for a partially completed main group.
        batch = {32768: 260, 65536: 130, 98304: 86}[context]
        crossover_batches[context] = batch
        require(f"four-node synthetic crossover c{context} b{batch}",
                [r for r in candidates if r["offered_requests"] == batch], all_methods)
    for context, direction, label in [(32768, -1, "below-fit"), (98304, 1, "overload")]:
        cross = crossover_batches[context]
        candidates = [r for r in four if r["context"] == context and cross is not None
                      and direction * (r["offered_requests"] - cross) > 0]
        batches = sorted({r["offered_requests"] for r in candidates})
        batch = max(batches, key=lambda b: len({r["method"] for r in candidates
                    if r["offered_requests"] == b}), default=None)
        require(f"four-node synthetic {label} c{context} b{batch}",
                [r for r in candidates if r["offered_requests"] == batch], controls)

    checks.append(dict(label="all selected completed points pass graph/dtype/token/KV-drain gates",
                       passed=len(rows) == len(eligible), selected=len(rows), eligible=len(eligible)))
    checks.append(dict(label="microbatching disabled consistently in this matrix", passed=all(
        c.get("enable_dbo") is False
        for r in rows for c in r["actual_decode_configurations"])))
    worker_failures = []
    checked_roles = 0
    for row in eligible:
        path = Path(row["path"])
        config = json.loads((path / "analysis.json").read_text())["configuration"]
        for index, role in enumerate(config["servers"]):
            manifest = json.loads((path / f"server{index}/manifest.json").read_text())
            ranks = sorted(r["rank"] for r in manifest["worker_resources"])
            checked_roles += 1
            if ranks != list(range(len(role["gpus"]))):
                worker_failures.append(dict(point=row["name"], server=index,
                                            expected_gpus=len(role["gpus"]), observed_ranks=ranks))
    checks.append(dict(label="worker evidence covers every assigned GPU without missing or duplicate ranks",
                       passed=not worker_failures, checked_roles=checked_roles, failures=worker_failures))
    family_checks = []
    for context, batch in crossover_batches.items():
        methods = {r["method"] for r in four if r["context"] == context and r["offered_requests"] == batch}
        family_checks.append(dict(context=context, batch=batch,
            tknp=any(m.startswith("tknp") for m in methods), tp="tp" in methods,
            dp=any(m.startswith("dp") for m in methods), pp=any("pp" in m for m in methods),
            dcp="dcp" in methods))
    return dict(scope="Selected capacity-crossover matrix plus fit/overload controls, not a full batch frontier or arrival-rate study.",
                complete=all(c["passed"] for c in checks), checks=checks,
                four_node_family_coverage=family_checks,
                four_node_all_families_present=all(all(v for k,v in c.items() if k not in {"context", "batch"}) for c in family_checks),
                report_audit=report["audit"],
                limitations=["Dummy BF16 weights and BF16 KV; no trained-checkpoint quality claim.",
                             "Four-node DCP and other 16-GPU points use synthetic KV, not real disaggregated prefill.",
                             "Single trials except the explicitly repeated one-node 32K pair.",
                             "Observed WAITING-state queues are not an exclusive KV-memory cause label."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
