#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Matched single-node TKNP/DP capacity test; requires an existing allocation.

One real prefill producer and two decode GPUs, BF16 weights/storage, actual
NIXL handoff, and full decode CUDA graphs. Dummy weights are opt-in and labeled.
No allocations, checkpoint downloads, or system configuration changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from e2e_matrix import run_trial, validate_matrix
from e1_pilot_report import analyze


HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3


def lifetime_caps(resources, lifetime):
    return [max(0, (row["kv_blocks"] - 1) // math.ceil(lifetime / row["block_tokens"]))
            for row in sorted(resources, key=lambda row: row["rank"])]


def roles(method, caps, lifetime):
    producer = {"role": "prefill", "gpus": [0], "connector": "nixl",
                "kv_cache_memory_bytes": 32 * GIB,
                "env": {"VLLM_TKNP_LIFETIME_ADMISSION": "1"}}
    common = {"role": "decode", "connector": "nixl", "cuda_graph": True,
              "disable_combo_kernels": True,
              "env": {"VLLM_KV_CACHE_LAYOUT": "HND", "VLLM_TKNP_LIFETIME_ADMISSION": "1"}}
    if method == "tknp":
        decoder = dict(common, gpus=[1, 2], tknp=2, placement="cost",
                       graph_slots=True, graph_batch_size=sum(caps),
                       max_num_seqs=sum(caps), owner_counts=caps, enable_dbo=False)
        decoder["env"] = dict(common["env"], VLLM_TKNP_REQUIRE_CUDAGRAPH="1",
                              VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(lifetime))
        return [producer, decoder]
    return [producer] + [dict(common, gpus=[gpu], max_num_seqs=cap,
                              graph_batch_size=cap, env=dict(common["env"]))
                         for gpu, cap in zip((1, 2), caps)]


def compare(records):
    pairs = {}
    for row in records:
        pairs.setdefault(row["offered_requests"], {})[row["method"]] = row
    return [{"offered_requests": count,
             "tknp_vs_dp_output_tps_ratio": methods["tknp"]["client"]["system_output_tps"] /
                                           methods["dp2"]["client"]["system_output_tps"],
             "tknp_seconds": methods["tknp"]["client"]["duration_s"],
             "dp_seconds": methods["dp2"]["client"]["duration_s"]}
            for count, methods in sorted(pairs.items()) if set(methods) == {"tknp", "dp2"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--tokens", type=int, default=4097)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batches", type=int, nargs="+",
                        help="Otherwise test below both capacities and inside the measured crossover")
    parser.add_argument("--reference", type=Path, help="Reuse completed capacity probes from this run directory")
    parser.add_argument("--dp-reference", type=Path,
                        help="Use a later completed DP trial if its runtime pools differ from the tiny probe")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--reverse-order", action="store_true", help="Reverse paired method order for confirmation runs")
    args = parser.parse_args()
    if args.context < 32768 or args.tokens < 17 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Use 32K+ context, 17+ outputs, and a memory utilization strictly between zero and one")
    if args.batches and min(args.batches) <= 0:
        parser.error("Positive offered batches required")
    import torch
    if torch.cuda.device_count() != 4:
        parser.error("Requires one existing four-GPU allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    lifetime = args.context + args.tokens
    scope = ("single-node 70B architecture throughput/capacity proxy with dummy BF16 weights; "
             "actual BF16 KV computed by dummy model, real NIXL handoff; NOT real-checkpoint validation"
             if args.load_format == "dummy" else
             "single-node real-checkpoint 70B BF16 end-to-end prefill/NIXL/decode comparison")
    config = {"model": args.model, "load_format": args.load_format, "kv_cache_dtype": "auto",
              "require_bf16": True, "max_model_len": lifetime, "max_num_seqs": 64,
              "max_num_batched_tokens": 1024, "gpu_memory_utilization": args.gpu_memory_utilization,
              "warmup_requests": 0, "routing": "least-work", "timeout": 3600, "scope": scope}
    report = {"scope": scope, "arguments": {k: str(v) if isinstance(v, Path) else v
                                            for k, v in vars(args).items()},
              "config": config, "attempts": [], "results": []}
    sources = [HERE / "e1_capacity_crossover.py", HERE / "e2e_matrix.py", HERE / "e2e_server.py",
               HERE.parents[2] / "vllm/v1/core/sched/scheduler.py",
               HERE.parents[2] / "vllm/v1/core/sched/tknp_placement.py"]
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    summary = args.output / "crossover.json"
    def persist():
        summary.write_text(json.dumps(report, indent=2) + "\n")
    persist()
    memory_file = (args.output / "gpu_memory.csv").open("x")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
                                "--format=csv,nounits", "-l", "1"], stdout=memory_file, stderr=subprocess.STDOUT)
    def execute(name, method, caps, count, tokens):
        workload = args.output / f"workload-b{count}-o{tokens}.jsonl"
        if not workload.exists():
            subprocess.run([sys.executable, str(HERE / "e2e_workload.py"), "generate", "--natural",
                            "--tokenizer", args.model, "--output", str(workload), "--count", str(count),
                            "--lengths", str(args.context), "--output-lengths", str(tokens)], check=True)
        trial = {"name": name, "servers": roles(method, caps, lifetime)}
        validate_matrix(dict(config, trials=[trial]), 4)
        (args.output / f"{name}-config.json").write_text(json.dumps(dict(config, trials=[trial]), indent=2) + "\n")
        entry = {"name": name, "method": method, "status": "running",
                 "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest()}
        report["attempts"].append(entry)
        persist()
        print("START", name, flush=True)
        try:
            run_trial(trial, config, workload, args.output / name)
            record = analyze(args.output / name)
            record.update(method=method, offered_requests=count)
            if not (record["all_decode_workers_full_graph"] and
                    record["all_servers_zero_measured_preemptions"] and record["all_servers_kv_drained"]):
                raise RuntimeError("Graph, lifetime admission, or KV-drain validity gate failed")
            entry.update(status="completed", client=record["client"])
            persist()
            print("DONE", name, record["client"]["system_output_tps"], flush=True)
            return record
        except Exception:
            entry.update(status="failed", error=traceback.format_exc())
            persist()
            raise
    try:
        if args.reference:
            probe_base = args.reference
        else:
            for method in ("tknp", "dp2"):
                execute("probe-" + method, method, [1, 1], 2, 17)
            probe_base = args.output
        tk_resources = json.loads((probe_base / "probe-tknp/server1/manifest.json").read_text())["worker_resources"]
        dp_base = args.dp_reference or (probe_base / "probe-dp2")
        dp_resources = [json.loads((dp_base / f"server{i}/manifest.json").read_text())["worker_resources"][0]
                        for i in (1, 2)]
        tk_caps = lifetime_caps(tk_resources, lifetime)
        dp_caps = [lifetime_caps([row], lifetime)[0] for row in dp_resources]
        if min(tk_caps + dp_caps) < 1:
            raise RuntimeError("At least one decoder cannot fit one request lifetime")
        report["capacity"] = {"lifetime_tokens": lifetime, "tknp_owner_requests": tk_caps,
                              "dp_replica_requests": dp_caps, "tknp_total": sum(tk_caps),
                              "dp_total": sum(dp_caps), "probe_base": str(probe_base),
                              "dp_reference": str(dp_base),
                              "tknp_resources": tk_resources, "dp_resources": dp_resources}
        persist()
        print("CAPACITY", report["capacity"], flush=True)
        if args.probe_only:
            return
        if args.batches:
            batches = args.batches
        elif sum(tk_caps) > sum(dp_caps):
            batches = sorted({sum(dp_caps), sum(tk_caps)})
        else:
            raise RuntimeError("No measured capacity interval: do not claim a capacity crossover")
        for index, count in enumerate(batches):
            order = ("dp2", "tknp") if index % 2 == 0 else ("tknp", "dp2")
            if args.reverse_order:
                order = tuple(reversed(order))
            for method in order:
                for calibration in range(3):
                    caps = tk_caps if method == "tknp" else dp_caps
                    suffix = "" if not calibration else f"-calibration{calibration}"
                    row = execute(f"c{args.context}-b{count}-o{args.tokens}-{method}{suffix}",
                                  method, caps, count, args.tokens)
                    actual = (lifetime_caps(row["resources"]["server1"], lifetime) if method == "tknp" else
                              [lifetime_caps(row["resources"][f"server{i}"], lifetime)[0] for i in (1, 2)])
                    row["runtime_lifetime_capacity"] = actual
                    row["configured_admission_capacity"] = list(caps)
                    if actual == caps:
                        break
                    report.setdefault("capacity_adjustments", []).append({"method": method, "batch": count,
                        "configured": caps, "runtime": actual, "trial": row["name"]})
                    if method == "tknp":
                        tk_caps = actual
                    else:
                        dp_caps = actual
                    # Below both limits, unused memory cannot limit this burst.
                    # Above the old limit, rerun rather than crediting TKNP for
                    # a baseline constrained by an outdated probe estimate.
                    if all(a <= c for a, c in zip(actual, caps)) or count <= min(sum(caps), sum(actual)):
                        break
                    report.setdefault("superseded_trials", []).append(row["name"])
                    persist()
                else:
                    raise RuntimeError("Runtime capacities did not stabilize; do not report a crossover")
                report["results"].append(row)
                report["comparisons"] = compare(report["results"])
                persist()
        print("CROSSOVER_COMPLETE", json.dumps(report["comparisons"]), flush=True)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        memory_file.close()


if __name__ == "__main__":
    main()
