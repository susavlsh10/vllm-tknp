#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Combine the original crossover and two ordered confirmation pairs."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("analysis95.json", "analysis_pair2.json", "analysis_pair3.json")
    pairs = []
    controls = []
    for index, name in enumerate(names, 1):
        data = json.loads((args.base / name).read_text())
        selected = {r["method"]: r for r in data["trials"] if r["offered_requests"] == 18}
        if set(selected) != {"tknp", "dp2"}:
            raise RuntimeError("Missing matched batch-18 pair")
        for row in selected.values():
            if row["client"]["completed"] != 18 or row["client"]["failed"]:
                raise RuntimeError("Incomplete workload")
            if not all(row[key] for key in ("all_decode_workers_full_graph", "all_decode_workers_no_prefill_recomputation",
                                            "all_servers_zero_measured_preemptions", "all_servers_kv_drained")):
                raise RuntimeError("Invalid measured pair")
        if selected["tknp"]["workload_sha256"] != selected["dp2"]["workload_sha256"]:
            raise RuntimeError("Different paired workload")
        agreement = next(r for r in data["comparisons"] if r["batch"] == 18)
        pairs.append({"pair": index, "analysis": str((args.base / name).resolve()),
                      "throughput_ratio": agreement["throughput_ratio_tknp_over_dp"],
                      "exact_token_requests": agreement["exact_token_requests"],
                      "methods": {method: {"duration_s": row["client"]["duration_s"],
                          "output_tps": row["client"]["system_output_tps"],
                          "p95_ttft_s": row["client"]["ttft_s"]["p95"],
                          "p50_itl_ms": row["client"]["intertoken_s"]["p50"] * 1000,
                          "decode_capacity": row["total_decode_capacity"],
                          "kv_gib": row["decode_kv_gib"],
                          "peak_decode_capacity_waiting": row["aggregate_decode_queue"]["peak_capacity_waiting"],
                          "capacity_queue_area_request_seconds": row["aggregate_decode_queue"]["capacity_queue_area_request_seconds"]}
                          for method, row in selected.items()}})
        if index == 1:
            controls = [r for r in data["trials"] if r["offered_requests"] == 6]
    metrics = {}
    for method in ("tknp", "dp2"):
        tps = [p["methods"][method]["output_tps"] for p in pairs]
        metrics[method] = {"mean_output_tps": mean(tps), "sample_stdev_output_tps": stdev(tps),
            "mean_duration_s": mean(p["methods"][method]["duration_s"] for p in pairs),
            "mean_p95_ttft_s": mean(p["methods"][method]["p95_ttft_s"] for p in pairs),
            "mean_p50_itl_ms": mean(p["methods"][method]["p50_itl_ms"] for p in pairs)}
    result = {"scope": data["scope"], "context_tokens": 32768, "output_tokens_per_request": 4097,
              "offered_requests": 18, "paired_repetitions": 3,
              "methods": metrics, "pairs": pairs, "single_pair_batch6_controls": controls,
              "mean_throughput_ratio": metrics["tknp"]["mean_output_tps"] / metrics["dp2"]["mean_output_tps"],
              "min_pair_ratio": min(p["throughput_ratio"] for p in pairs),
              "max_pair_ratio": max(p["throughput_ratio"] for p in pairs),
              "exact_token_requests_total": sum(p["exact_token_requests"] for p in pairs),
              "validity": "all decoder ranks FULL graphs, no decode-side prefill/recomputation, zero P/D preemptions, all KV drained",
              "admission_policy": "TKNP and DP reserve full request-lifetime KV; DP is not using stock preempt/recompute admission",
              "recipe": "one prefill GPU plus two decode GPUs; same 95% automatic decode memory utilization; microbatching OFF",
              "pair_order": ["TKNP then DP", "DP then TKNP", "TKNP then DP"]}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ("pairs", "single_pair_batch6_controls")}, indent=2))


if __name__ == "__main__":
    main()
