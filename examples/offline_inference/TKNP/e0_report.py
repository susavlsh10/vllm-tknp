#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect E0 evidence without turning smoke timings into paper claims."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def trial_summary(directory):
    trial = load(directory / "trial.json")
    client = load(directory / "client/summary.json")
    if trial is None or client is None:
        return {"status": "incomplete", "directory": str(directory)}
    roles = trial["trial"]["servers"]
    decoders = [f"server{i}" for i, role in enumerate(roles) if role["role"] == "decode"]
    graphs = {}
    for name in decoders:
        execution = load(directory / name / "execution.json")
        graphs[name] = execution
    replay = all(records and all(row["counts"].get("decode:FULL", 0) > 0
                                  and row["counts"].get("decode:NONE", 0) == 0
                                  for row in records) for records in graphs.values())
    cleanup = all(row["shutdown"]["all_kv_free"] and not row["shutdown"]["remaining_requests"]
                  for row in trial["telemetry"].values())
    queues = {name: max((row.get("max_waiting_capacity", 0)
                        for key, row in trial["telemetry"].items() if key.startswith(name + "/")), default=0)
              for name in decoders}
    return {"status": "passed" if replay and cleanup and client["failed"] == 0 else "failed",
            "directory": str(directory), "completed": client["completed"],
            "all_decoder_workers_full_graph": replay, "all_kv_drained": cleanup,
            "capacity_queue_peak_per_endpoint": queues,
            "staged_decode": client.get("staged_decode"), "decode_wave": client.get("decode_wave"),
            "engine_stage_latency": trial.get("engine_stage_latency"),
            "worker_execution": graphs}


def report(root):
    suites = {}
    for group in ("staged32-v3", "staged96", "queued32", "queued32-barrier", "tp32", "pp32"):
        trials = root / group / "trials"
        suites[group] = {directory.name: trial_summary(directory)
                         for directory in sorted(trials.glob("*")) if directory.is_dir()}
    unit_path = next((root / name / "unit.xml" for name in
                      ("unit-final3", "unit-final2", "unit-final")
                      if (root / name / "unit.xml").exists()), root / "missing-unit.xml")
    unit = None
    if unit_path.exists():
        tree = ET.parse(unit_path).getroot()
        unit = {name: sum(int(s.get(name, 0)) for s in tree.iter("testsuite"))
                for name in ("tests", "failures", "errors", "skipped")}
    controls = {name: load(root / name / "verdict.json") for name in
                ("long-natural-v3", "short", "short-natural-final", "plain-random", "micro-random-kv", "tp2-tknp2-32")}
    numerical = {name: load(root / name) for name in
                 ("full_logits_long.json", "full_logits_plain_random.json", "full_logits_micro_random.json")}
    return {"scope": "single-node real Llama 3.1 8B E0, BF16 weights/KV; development evidence, NOT paper results or multi-node signoff",
            "suites": suites, "targeted_tests": unit, "correctness_controls": controls,
            "numerical_diagnostics": numerical,
            "measurement_modes": {
                "end_to_end": "actual arrivals, real prefill/handoff, admission, decode and full drain",
                "prefill_barrier": "real exports prepared before decoder burst; handoff/queueing still timed",
                "resident_staged": "one priming token then pause/resume in SAME live engines; priming/pause excluded",
                "synthetic_staged": "same resident decode schedule with constant dummy KV; not model quality"},
            "remaining_gates": ["Multi-node P/D was not run: user authorized one node only",
                                "Real NIXL-to-DCP interleaved layout is not implemented",
                                "Large checkpoints and topology-specific smokes remain before collection",
                                "Varying singleton cohort positions fail early; eight-lane [2,6] layout validated",
                                "Exact greedy and numerical tolerance are separate; random-control mismatches retained"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.results)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({"targeted_tests": result["targeted_tests"],
                      "suites": {name: {trial: row["status"] for trial, row in records.items()}
                                 for name, records in result["suites"].items()}}, indent=2))
