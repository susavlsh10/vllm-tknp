#!/usr/bin/env python3
"""Produce a small, reproducible analysis artifact from the full raw audit."""
import argparse
import json
from pathlib import Path
from statistics import mean


def build(report):
    result = []
    for group in report["leaders"]:
        primary_real = (group["family"] == "e2e" and group["output_tokens"] == 4097
                        and group["producer_gpus"] == group["decoder_gpus"])
        primary_synthetic = group["family"] == "synthetic-decode" and group["decoder_gpus"] == 16
        if not (primary_real or primary_synthetic):
            continue
        rows = [r for r in report["trials"] if r["comparison_eligible"] and all(
            r[k] == group[k] for k in ["family", "decoder_gpus", "producer_gpus", "context",
                                       "output_tokens", "offered_requests"])]
        details = {}
        for method, statistics in group["configuration_statistics"].items():
            trials = [r for r in rows if r["method"] == method]
            details[method] = dict(
                throughput=statistics,
                mean_of_trial_p95_ttft_s=mean(r["client"]["ttft_s"]["p95"] for r in trials),
                mean_of_trial_p95_itl_ms=1000*mean(r["client"]["intertoken_s"]["p95"] for r in trials),
                max_observed_decode_waiting_state=max(r["decode_queue"]["peak_capacity_waiting"] for r in trials),
                max_observed_prefill_waiting_state=max(r["prefill_queue"]["peak_capacity_waiting"] for r in trials),
                max_observed_decode_preemptions=max(r["decoder_preemptions"] for r in trials),
                runtime_lifetime_capacities=[r["runtime_lifetime_capacities"] for r in trials],
                gpu_memory_utilization_settings=sorted({c["gpu_memory_utilization"]
                    for r in trials for c in r["actual_decode_configurations"]}),
                paths=[r["path"] for r in trials])
        result.append(dict(
            **{k:group[k] for k in ["family", "decoder_gpus", "producer_gpus", "context",
                                     "output_tokens", "offered_requests", "best_methods", "best_tps",
                                     "tknp_over_best_dp"]}, methods=details))
    return dict(
        model="Llama 3.1 70B Instruct architecture, dummy BF16 weights, BF16 KV",
        selected_point_audit=report["audit"], groups=result,
        interpretation=[
            "TKNP has a capacity-pressure crossover against unsharded independent TP1 DP replicas, but is not uniformly faster than TP-sharded DP or TP.",
            "One-node 32K repeated results compare three matched TKNP/lifetime-aware DP pairs. Other configurations are exploratory single trials.",
            "Queueing/TTFT and ITL can move in opposite directions: faster burst admission does not imply faster individual decode steps.",
            "DP can still finish sooner while queueing: in the real two-node 32K/48-request case, TP2 x DP2 peaked at 10 decode-WAITING requests versus zero for TP2 x TKNP2, yet had higher total TPS. Capacity alone is not a sufficient throughput criterion.",
            "Preserve negative controls and compare against the best tested TP x DP configuration, not only the weakest DP layout.",
            "Selected valid results support a memory/queueing claim, not a universal throughput win or trained-checkpoint quality claim."],
        measurement=[
            "Real P2+D2 one-node and P4+D4 two-node runs use model-computed KV, actual NIXL transfer and 4097 output tokens; cohosted logical P/D roles.",
            "Four-node runs use all sixteen GPUs for decode, synthetic KV and 257 output tokens. No real four-node disaggregated DCP result is claimed.",
            "System TPS is offered output tokens divided by complete client burst makespan; model startup is excluded. Synthetic KV initialization is included.",
            "Waiting-state snapshot peaks are not an exclusive KV-memory cause label and are not inferred by subtracting a capacity estimate from the offered batch.",
            "All selected points require expected request/token counts, BF16 storage, FULL decode-only graph replay, consistent TKNP owner layouts and drained KV.",
            "Native DP mixed prefill/recompute calls may be eager; they and measured preemptions remain baseline costs.",
            "Microbatching/DBO is disabled throughout this matrix. This collection does not establish a microbatching gain.",
            "Controlled exact-length all-at-once bursts, not RULER, variable-length arrival traffic, or a complete batch frontier."],
        publication_assessment="Do not use this matrix to claim TKNP beats all baselines. Retain the strongest TP/DP/DCP comparisons and resolve performance scope before selecting headline paper plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    brief = build(json.loads(args.report.read_text()))
    args.output.write_text(json.dumps(brief, indent=2) + "\n")
    print("wrote", args.output, "groups", len(brief["groups"]))
