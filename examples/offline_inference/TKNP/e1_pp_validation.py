#!/usr/bin/env python3
"""Tiny PP token-budget validation on explicitly assigned otherwise-idle GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from e1_70b_suite import build_trial, MODEL
from e2e_matrix import run_trial, validate_matrix
from e2e_slurm import client_command
from e1_pilot_report import analyze

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--job", required=True)
p.add_argument("--nodes", nargs="+", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
config = dict(model=MODEL, load_format="dummy", require_bf16=True, kv_cache_dtype="auto",
              max_model_len=32833, max_num_seqs=8, max_num_batched_tokens=1024,
              gpu_memory_utilization=0.95, timeout=300, routing="least-work", warmup_requests=0,
              slurm_job_id=a.job, slurm_nodes=a.nodes, port_base=8300, nixl_port_base=20200, rdzv_port_base=24500,
              scope="Pipeline correctness/graph/queue validation; NOT a performance measurement",
              cache_root=str(a.output.parent / "shared_cache"), triton_root=str(a.output.parent / "shared_triton"))
here = Path(__file__).resolve().parent
records = []
for family, count in [("decode", 8), ("e2e", 2)]:
    workload = a.output / f"{family}-workload.jsonl"
    cmd = [sys.executable, str(here / "e2e_workload.py"), "generate", "--natural", "--tokenizer", MODEL,
           "--count", str(count), "--lengths", "32768", "--output-lengths", "65", "--output", str(workload)]
    subprocess.run(client_command(config, cmd), check=True)
    trial = build_trial("pp-pipelined", 32768, 65, count, [3, 7], [8], family, [4])
    if family == "e2e":
        trial["prefill_barrier"] = True  # Two exports fit the 32GiB source pool.
    validate_matrix(dict(config, trials=[trial]), 8)
    output = a.output / family
    run_trial(trial, config, workload, output)
    row = analyze(output)
    depths = [event.get("batch_queue_depth", 0) for event in map(json.loads, (output / f"server{len(trial['servers'])-1}/rank0.jsonl").read_text().splitlines()) if event["event"] == "step"]
    if not row["all_decode_workers_full_graph"] or not row["all_servers_kv_drained"] or max(depths) < 1:
        raise RuntimeError("PP queue/replay/drain gate failed")
    records.append(dict(family=family, full_graph=True, kv_drained=True, max_queue_depth=max(depths), completed=row["client"]["completed"]))
    print("PP_VALIDATED", records[-1], flush=True)
(a.output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
