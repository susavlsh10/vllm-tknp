#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze preserved pilot artifacts without running GPUs or changing trials."""
import argparse
import json
from pathlib import Path
from statistics import mean

from e2e_workload import percentile


def analyze(path):
    client = json.loads((path / "client/summary.json").read_text())
    trial = json.loads((path / "trial.json").read_text())
    roles = {}
    manifests = {}
    for file in sorted(path.glob("server*/manifest.json")):
        manifest = json.loads(file.read_text())
        manifests[file.parent.name] = manifest
        roles[file.parent.name] = manifest["arguments"]["connector"]
    decode_names = {name for name, m in manifests.items()
                    if m["arguments"]["enable_dbo"] or m["arguments"]["cuda_graph"]}
    resources = {name: m["worker_resources"] for name, m in manifests.items()}
    execution = {name: values for name, values in trial["worker_execution"].items() if name in decode_names}
    graph_ok = bool(execution) and all(values and all(
        record["counts"].get("decode:FULL", 0) > 0
        and all(not count for key, count in record["counts"].items()
                if key.startswith("decode:") and key != "decode:FULL")
        for record in values) for values in execution.values())
    result = {"name": path.name, "group": path.parent.name,
              "trial_directory": str(path.resolve()), "configuration": trial["trial"],
              "client": client, "resources": resources,
              "all_decode_workers_full_graph": graph_ok,
              "all_servers_zero_measured_preemptions": all(
                  row["preemptions_measured"] == 0 for row in trial["telemetry"].values()),
              "all_servers_kv_drained": all(
                  row["shutdown"]["all_kv_free"] and row["shutdown"]["remaining_requests"] == 0
                  for row in trial["telemetry"].values()),
              "execution": execution, "telemetry": trial["telemetry"]}
    # Trim first/last four post-release full-batch steps. Replicated journals
    # are grouped by step; report slowest rank, never summed across ranks.
    if "staged_decode" in client:
        requests = [json.loads(line) for line in (path / "client/requests.jsonl").read_text().splitlines()]
        count = len(requests)
        grouped = {}
        for name in decode_names:
            for file in sorted((path / name).glob("rank*.jsonl")):
                released = False
                for event in map(json.loads, file.read_text().splitlines()):
                    if event["event"] == "release":
                        released = True
                    elif (released and event["event"] == "step" and event.get("phase") == "decode"
                          and event["scheduled_tokens"] == count):
                        key = (name, event["step"])
                        grouped[key] = max(grouped.get(key, 0), event["seconds"])
        times = [seconds for _, seconds in sorted(grouped.items())][4:-4]
        result["resident_full_batch_engine"] = {
            "scope": "post-release full-batch engine-step wall time, slowest replicated rank; first/last four excluded; NOT GPU-kernel-only time",
            "steps": len(times), "mean_s": mean(times) if times else None,
            "p50_s": percentile(times, 0.5), "p95_s": percentile(times, 0.95),
            "output_tps": count / mean(times) if times else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [analyze(file.parent) for file in sorted(args.root.rglob("trial.json"))]
    result = {"scope": "single-node real 8B BF16 development pilot; retain negative and failed points",
              "trials": records}
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
    for record in records:
        client = record["client"]
        staged = client.get("staged_decode", {})
        print(record["name"], "completed", client["completed"],
              "FULL", record["all_decode_workers_full_graph"],
              "TPS", round(staged.get("output_tps", client["system_output_tps"]), 2),
              "ITL_ms", round(staged.get("steady_intertoken_s", client["intertoken_s"])["p50"] * 1000, 3),
              "engine_TPS", record.get("resident_full_batch_engine", {}).get("output_tps"))


if __name__ == "__main__":
    main()
