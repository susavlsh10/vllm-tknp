#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Matched single-node 8B pilot. Uses an existing allocation; never submits jobs.

Resident runs retain real KV after one prime token. Queue runs include real
prefill/transfer and drain. Equal automatic per-GPU memory utilization, not
equal KV bytes: the latter would remove the weight-replication advantage.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from e2e_matrix import run_trial, validate_matrix


HERE = Path(__file__).resolve().parent


def common_config(context, count, tokens):
    return {"model": "meta-llama/Llama-3.1-8B-Instruct", "load_format": "auto",
            "kv_cache_dtype": "auto", "require_bf16": True,
            "max_model_len": context + tokens, "max_num_seqs": 128,
            "max_num_batched_tokens": 4096, "gpu_memory_utilization": 0.8,
            "warmup_requests": 0, "timeout": 1800,
            "scope": "single-node real 8B BF16 pilot; not large-model paper results",
            "trials": []}


def producer(lifetime_admission=False):
    # The same producer in every real trial. Exported KV drains in waves.
    role = {"role": "prefill", "gpus": [0], "connector": "nixl",
            "kv_cache_memory_bytes": 32 * 1024 ** 3}
    if lifetime_admission:
        role["env"] = {"VLLM_TKNP_LIFETIME_ADMISSION": "1"}
    return role


def decoder(context, count, tokens, enabled, owners=None):
    owners = owners or ([2, 6] if count == 8 else [count // 2, count - count // 2])
    return {"role": "decode", "gpus": [1, 2], "tknp": 2, "connector": "nixl",
            "placement": "cost", "cuda_graph": True, "graph_slots": True,
            "graph_batch_size": count, "owner_counts": owners,
            "enable_dbo": enabled, "disable_combo_kernels": True,
            "env": {"VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS": str(context + tokens),
                    "VLLM_TKNP_REQUIRE_CUDAGRAPH": "1", "VLLM_KV_CACHE_LAYOUT": "HND",
                    "VLLM_TKNP_DBO_ROOT_REQUESTS": str(owners[0]),
                    "VLLM_TKNP_DBO_C0_FRACTION": "0.5" if count == 8 else "0.625",
                    "VLLM_TKNP_DBO_ZERO_ROOT_C1": "1",
                    "VLLM_TKNP_DBO_INDEPENDENT_COMM": "1",
                    "VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE": "1",
                    "VLLM_TKNP_DBO_LOOKAHEAD": "0",
                    "VLLM_TKNP_DBO_LAYOUT": "asymmetric"}}


def generate(path, context, count, tokens):
    subprocess.run([sys.executable, str(HERE / "e2e_workload.py"), "generate", "--natural",
                    "--output", str(path), "--count", str(count), "--lengths", str(context),
                    "--output-lengths", str(tokens)], check=True)


def resources(path):
    role = json.loads((path / "server1/manifest.json").read_text())
    return sorted(role["worker_resources"], key=lambda r: r["rank"])


def capacity(rows, context, tokens):
    return [max(0, (r["kv_blocks"] - 1) // math.ceil((context + tokens) / r["block_tokens"]))
            for r in rows]


def fit_owner_plan(target, caps):
    """Preserve total batch, shifting only the requests needed for capacity."""
    count = sum(target)
    if sum(caps) < count:
        return None
    root = max(count - caps[1], min(target[0], caps[0]))
    return [root, count - root]


def persist(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n")


def execute(trial, config, workload, path, report, summary_path):
    validate_matrix(dict(config, trials=[trial]), 4)
    report["attempts"].append({"name": path.name, "status": "running", "trial": trial,
                               "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest()})
    entry = report["attempts"][-1]
    persist(summary_path, report)
    print("START", path.name, flush=True)
    try:
        result = run_trial(trial, config, workload, path)
        entry.update(status="completed", completed=result["completed"],
                     client=json.loads((path / "client/summary.json").read_text()),
                     telemetry=result["telemetry"], worker_execution=result["worker_execution"])
    except Exception:
        entry.update(status="failed", error=traceback.format_exc())
        persist(summary_path, report)
        raise
    persist(summary_path, report)
    print("DONE", path.name, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", choices=["resident", "proxy", "queue"], required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 98304])
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--layout", choices=["balanced", "quarter-root"], default="balanced",
                        help="Resident sweep owner plan; quarter-root shifts attention away from dense root")
    parser.add_argument("--reference", type=Path,
                        help="Completed resident section: uses runtime capacities for queue/proxy")
    parser.add_argument("--microbatch", action="store_true", help="Queue/proxy choice after the resident sweep")
    parser.add_argument("--lifetime-admission", action="store_true",
                        help="Native baseline reserves running and in-flight KV growth instead of preempting")
    parser.add_argument("--dp-reference", type=Path,
                        help="Completed DP trial; size EACH replica graph from its actual cache capacity")
    args = parser.parse_args()
    if args.tokens < 17 or args.repetitions < 1 or any(b < 8 or b % 8 for b in args.batches):
        parser.error("Use >=17 tokens, positive repetitions, and graph batches divisible by eight")
    import torch
    if torch.cuda.device_count() != 4:
        parser.error("Requires the existing four-GPU single node")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"section": args.section, "arguments": vars(args), "attempts": [], "capacity_stops": []}
    report["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary_path = args.output / "pilot.json"
    monitor_log = (args.output / "gpu_memory.csv").open("x")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
                                "--format=csv,nounits", "-l", "1"], stdout=monitor_log, stderr=subprocess.STDOUT)
    try:
        for context in args.contexts:
            last_resources = None
            if args.section == "resident":
                if args.reference:
                    reference = json.loads((args.reference / "pilot.json").read_text())
                    completed = [r for r in reference["attempts"] if r["status"] == "completed"
                                 and r["name"].startswith(f"c{context}-")]
                    if completed:
                        last_resources = resources(args.reference / completed[-1]["name"])
                for count in args.batches:
                    config = common_config(context, count, args.tokens)
                    owners = ([count // 4, count - count // 4] if args.layout == "quarter-root" else None)
                    off = decoder(context, count, args.tokens, False, owners)
                    if last_resources:
                        caps = capacity(last_resources, context, args.tokens)
                        fitted = fit_owner_plan(off["owner_counts"], caps)
                        if fitted is None:
                            report["capacity_stops"].append({"context": context, "batch": count,
                                "requested_owners": off["owner_counts"], "runtime_capacity_requests": caps,
                                "reason": "next fixed owner plan cannot fit resident under measured pools; no OOM job launched"})
                            persist(summary_path, report)
                            break
                        if fitted != off["owner_counts"]:
                            report.setdefault("capacity_adjustments", []).append({"context": context,
                                "batch": count, "requested_owners": off["owner_counts"],
                                "actual_owners": fitted, "runtime_capacity_requests": caps})
                        owners = fitted
                    workload = args.output / f"workload-c{context}-b{count}.jsonl"
                    generate(workload, context, count, args.tokens)
                    for rep in range(args.repetitions):
                        for enabled in ([False, True] if rep % 2 == 0 else [True, False]):
                            name = f"c{context}-b{count}-rep{rep}-" + ("on" if enabled else "off")
                            trial = {"name": name, "staged_decode": True,
                                     "servers": [producer(), decoder(context, count, args.tokens, enabled, owners)]}
                            config["trials"] = [trial]
                            persist(args.output / (name + "-config.json"), config)
                            execute(trial, config, workload, args.output / name, report, summary_path)
                            last_resources = resources(args.output / name)
                    # Retain exact matched token check separately from timing.
                    for rep in range(args.repetitions):
                        paths = [args.output / f"c{context}-b{count}-rep{rep}-{mode}/client/requests.jsonl" for mode in ("off", "on")]
                        rows = [{r["request_id"]: r["token_ids"] for r in map(json.loads, p.read_text().splitlines())} for p in paths]
                        divergence = [next(i for i, (a, b) in enumerate(zip(rows[0][k], rows[1][k])) if a != b)
                                      for k in rows[0] if rows[0][k] != rows[1][k]]
                        report.setdefault("paired_tokens", []).append({"context": context, "batch": count,
                            "repetition": rep,
                            "exact_requests": sum(rows[0][k] == rows[1][k] for k in rows[0]), "requests": len(rows[0]),
                            "first_divergence_output_indices": sorted(set(divergence))})
                    persist(summary_path, report)
            else:
                if not args.reference:
                    parser.error("Queue/proxy requires completed resident --reference")
                reference = json.loads((args.reference / "pilot.json").read_text())
                completed = [r for r in reference["attempts"] if r["status"] == "completed"
                             and r["name"].startswith(f"c{context}-") and r["name"].endswith("-off")]
                if not completed:
                    raise ValueError("No completed resident reference for this context")
                source = completed[-1]
                source_role = source["trial"]["servers"][1]
                last_resources = resources(args.reference / source["name"])
                if args.section == "proxy":
                    count = source_role["graph_batch_size"]
                    config = common_config(context, count, args.tokens)
                    workload = args.output / f"workload-c{context}-b{count}.jsonl"
                    generate(workload, context, count, args.tokens)
                    for rep in range(args.repetitions):
                        for real in ([True, False] if rep % 2 == 0 else [False, True]):
                            name = f"c{context}-b{count}-rep{rep}-" + ("real" if real else "synthetic")
                            role = dict(source_role, enable_dbo=args.microbatch,
                                        connector="nixl" if real else "synthetic")
                            trial = {"name": name, "staged_decode": True,
                                     "servers": ([producer()] if real else []) + [role]}
                            persist(args.output / (name + "-config.json"), dict(config, trials=[trial]))
                            execute(trial, config, workload, args.output / name, report, summary_path)
                else:
                    caps = capacity(last_resources, context, args.tokens)
                    # Use nearly the entire measured pool; graph quotas must
                    # not artificially impose the earlier power-of-two cap.
                    owners = caps
                    graph_count = sum(owners)
                    count = graph_count * 2 + 8
                    config = common_config(context, count, args.tokens)
                    workload = args.output / f"workload-c{context}-b{count}.jsonl"
                    generate(workload, context, count, args.tokens)
                    tk = decoder(context, graph_count, args.tokens, args.microbatch, owners)
                    if args.lifetime_admission:
                        tk["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] = "1"
                    dp_caps = [caps[0], caps[0]]
                    if args.dp_reference:
                        dp_caps = [capacity(json.loads((args.dp_reference / f"server{index}/manifest.json").read_text())
                                            ["worker_resources"], context, args.tokens)[0] for index in (1, 2)]
                    # Same GPU utilization, real weights, context and workload.
                    dp = [{"role": "decode", "gpus": [gpu], "connector": "nixl",
                           "cuda_graph": True, "disable_combo_kernels": True,
                           "max_num_seqs": dp_caps[index], "graph_batch_size": dp_caps[index],
                           "env": dict({"VLLM_KV_CACHE_LAYOUT": "HND"}, **(
                               {"VLLM_TKNP_LIFETIME_ADMISSION": "1"} if args.lifetime_admission else {}))}
                          for index, gpu in enumerate((1, 2))]
                    for rep in range(args.repetitions):
                        for method in (["tknp", "dp2"] if rep % 2 == 0 else ["dp2", "tknp"]):
                            name = f"c{context}-b{count}-rep{rep}-" + method
                            trial = {"name": name, "servers": [producer(args.lifetime_admission)] + ([tk] if method == "tknp" else dp)}
                            persist(args.output / (name + "-config.json"), dict(config, trials=[trial]))
                            execute(trial, config, workload, args.output / name, report, summary_path)
                    report.setdefault("queue_capacity", []).append({"context": context,
                        "graph_owners": owners, "offered_requests": count,
                        "dp_replica_limits": dp_caps, "native_lifetime_admission": args.lifetime_admission,
                        "scope": "full-lifetime capacities estimated from resident pilot; decoder graph/concurrency limits sized to those capacities; continuous real P/D, producer serves waves; no prefill barrier"})
                    persist(summary_path, report)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_log.close()
        persist(summary_path, report)


if __name__ == "__main__":
    main()
