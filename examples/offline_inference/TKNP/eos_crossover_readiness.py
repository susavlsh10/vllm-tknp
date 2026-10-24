#!/usr/bin/env python3
"""One measured-capacity midpoint control; no allocations or capacity guessing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from e1_70b_suite import build_trial, measured_caps
from e2e_matrix import run_trial, validate_matrix
from e2e_slurm import client_command
from eos_readiness import audit_point


def crossover_batch(tknp_capacity, dp_capacity):
    if tknp_capacity <= dp_capacity:
        return None
    batch = math.ceil((tknp_capacity + dp_capacity) / 4) * 2
    return batch if dp_capacity < batch <= tknp_capacity else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads((args.suite / "suite.json").read_text())
    if not suite.get("finished_time") or suite.get("paused_reason"):
        raise ValueError("Finish the capacity discovery first")
    caps = suite["capacities"]["32768"]
    tk_capacity = sum(caps["tknp-tp8"]) // 2 * 2
    dp_capacity = sum(caps["dp-tp8"])
    batch = crossover_batch(tk_capacity, dp_capacity)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(scope="One short synthetic-KV capacity/admission control, not paper throughput",
                measured_tknp_capacity=tk_capacity, measured_dp_capacity=dp_capacity, batch=batch)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if batch is None:
        raise RuntimeError("No measured TKNP capacity interval above DP; do not manufacture a crossover")
    config = dict(suite["config"])
    context, outputs = 32768, 33
    config["max_model_len"] = context + outputs
    # A decode-only continuation may use exactly two nodes, without a reserved
    # prefill node. Preserve the suite's recorded physical placement.
    decoder_gpus = config.get("decoder_gpus", list(range(8, 24)))
    config["trials"] = [build_trial(method, context, outputs, batch, decoder_gpus, caps[method], "decode")
        for method in ["tknp-tp8", "dp-tp8", "tp8-pp2", "dcp"]]
    validate_matrix(config, len(config["slurm_nodes"])*8)
    (args.output / "matrix.json").write_text(json.dumps(config, indent=2)+"\n")
    workload = args.output / "workload.jsonl"
    if not workload.exists():
        command = [sys.executable, str(Path(__file__).with_name("e2e_workload.py")), "generate", "--natural",
            "--tokenizer", config["model"], "--output", str(workload), "--count", str(batch),
            "--lengths", str(context), "--output-lengths", str(outputs)]
        subprocess.run(client_command(config, command), check=True, timeout=180)
    summary_path = args.output / "summary.json"
    rows = json.loads(summary_path.read_text())["trials"] if summary_path.exists() else []
    for trial in config["trials"]:
        if any(r["name"] == trial["name"] for r in rows):
            continue
        print("START_CROSSOVER", trial["name"], flush=True)
        destination = args.output / trial["name"]
        run_trial(trial, config, workload, destination)
        evidence = audit_point(destination, trial, batch, outputs)
        row = dict(name=trial["name"], path=str(destination), client=evidence["client"],
            capacities=measured_caps(destination, context, outputs, trial),
            workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
            telemetry=evidence["telemetry"], full_graph=True, bf16=True, drained=True)
        rows.append(row)
        summary_path.write_text(json.dumps(dict(plan=plan, trials=rows), indent=2)+"\n")
        print("CROSSOVER_PASS", trial["name"], flush=True)
    print("EOS_CROSSOVER_READINESS_PASS", flush=True)


if __name__ == "__main__":
    main()
