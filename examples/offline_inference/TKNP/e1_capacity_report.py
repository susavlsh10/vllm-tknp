#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize capacity-crossover data without executing inference.

Queue areas and aggregate peaks reconstruct timestamped journal snapshots;
they are sampled estimates, not exact per-request capacity-wait durations.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from e1_pilot_report import analyze
from e2e_workload import percentile


def snapshots(paths):
    events = []
    for index, path in enumerate(paths):
        for line in path.open():
            event = json.loads(line)
            if event["event"] == "step":
                events.append((event["time"], index, event.get("waiting_capacity", 0), event["running"]))
    states = {}
    peak = peak_running = 0
    area = 0.0
    previous = None
    for timestamp, index, waiting, running in sorted(events):
        if previous is not None:
            area += (timestamp - previous) * sum(row[0] for row in states.values())
        states[index] = waiting, running
        peak = max(peak, sum(row[0] for row in states.values()))
        peak_running = max(peak_running, sum(row[1] for row in states.values()))
        previous = timestamp
    return {"peak_capacity_waiting": peak, "peak_running": peak_running,
            "capacity_queue_area_request_seconds": area,
            "scope": "timestamp-aligned right-continuous journal snapshots; approximate aggregate peak/area"}


def describe(path):
    row = analyze(path)
    trial = row["configuration"]
    requests = [json.loads(line) for line in (path / "client/requests.jsonl").open()]
    decoders = [f"server{i}" for i, role in enumerate(trial["servers"]) if role["role"] == "decode"]
    producers = [f"server{i}" for i, role in enumerate(trial["servers"]) if role["role"] == "prefill"]
    lifetime = requests[0]["prompt_tokens"] + requests[0]["requested_tokens"]
    capacities = {name: [(resource["kv_blocks"] - 1) // math.ceil(lifetime / resource["block_tokens"])
                         for resource in sorted(row["resources"][name], key=lambda r: r["rank"])]
                  for name in decoders}
    counts = {name: [r["counts"] for r in row["execution"][name]] for name in decoders}
    recompute = any(any(key.startswith("prefill:") and value for key, value in record.items())
                    for records in counts.values() for record in records)
    queues = {name: snapshots([path / name / "rank0.jsonl"]) for name in producers + decoders}
    ready = []
    for name in decoders:
        ready.extend(row["telemetry"][name + "/rank0"]["first_compute_queue_s"].values())
    return {"name": path.name, "path": str(path.resolve()), "client": row["client"],
            "method": "tknp" if len(decoders) == 1 else "dp2",
            "offered_requests": len(requests), "request_lifetime_tokens": lifetime,
            "runtime_lifetime_capacity": capacities,
            "total_decode_capacity": sum(sum(v) for v in capacities.values()),
            "decode_kv_gib": sum(r["kv_allocated_bytes"] for name in decoders
                                  for r in row["resources"][name]) / 2 ** 30,
            "route_counts": dict(Counter(str(r.get("decode_replica")) for r in requests)),
            "queues": queues,
            "aggregate_decode_queue": snapshots([path / name / "rank0.jsonl" for name in decoders]),
            "decode_admit_to_first_compute_s": {"p50": percentile(ready, 0.5), "p95": percentile(ready, 0.95),
                "scope": "includes capacity waiting, transfer/handshake/polling; NOT exclusive queue or wire latency"},
            "all_decode_workers_full_graph": row["all_decode_workers_full_graph"],
            "all_decode_workers_no_prefill_recomputation": not recompute,
            "all_servers_zero_measured_preemptions": row["all_servers_zero_measured_preemptions"],
            "all_servers_kv_drained": row["all_servers_kv_drained"],
            "workload_sha256": row["client"]["workload_sha256"],
            "resources": row["resources"], "execution": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed crossover run directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "crossover.json").read_text())
    # Only chosen results, not probes, superseded or interrupted trials.
    records = [describe(args.run / row["name"]) for row in summary["results"]]
    pairs = {}
    for record in records:
        pairs.setdefault(record["offered_requests"], {})[record["method"]] = record
    comparisons = []
    for batch, methods in sorted(pairs.items()):
        if set(methods) != {"tknp", "dp2"}:
            continue
        tk, dp = methods["tknp"], methods["dp2"]
        if tk["workload_sha256"] != dp["workload_sha256"]:
            raise RuntimeError("Paired workloads differ")
        a = {r["request_id"]: r["token_ids"] for r in map(json.loads, (args.run / tk["name"] / "client/requests.jsonl").read_text().splitlines())}
        b = {r["request_id"]: r["token_ids"] for r in map(json.loads, (args.run / dp["name"] / "client/requests.jsonl").read_text().splitlines())}
        comparisons.append({"batch": batch, "throughput_ratio_tknp_over_dp": tk["client"]["system_output_tps"] / dp["client"]["system_output_tps"],
                            "exact_token_requests": sum(a[k] == b[k] for k in a), "requests": len(a)})
    with args.output.open("x") as handle:
        json.dump({"scope": summary["scope"], "config": summary["config"], "trials": records,
                   "comparisons": comparisons}, handle, indent=2)
    for r in records:
        print(r["name"], "capacity", r["total_decode_capacity"], "tps", round(r["client"]["system_output_tps"], 2),
              "seconds", round(r["client"]["duration_s"], 2), "queue", r["aggregate_decode_queue"]["peak_capacity_waiting"],
              "valid", all(r[key] for key in ("all_decode_workers_full_graph", "all_decode_workers_no_prefill_recomputation",
                                             "all_servers_zero_measured_preemptions", "all_servers_kv_drained")))
    print("COMPARISONS", comparisons)


if __name__ == "__main__":
    main()
