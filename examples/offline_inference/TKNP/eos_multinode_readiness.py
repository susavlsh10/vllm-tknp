#!/usr/bin/env python3
"""Bounded EOS readiness in an existing allocation; never allocates GPUs.

Dummy BF16 weights. Real NIXL and synthetic KV remain separate families.
Run on the login host: only orchestration/CPU audit happens there.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from e2e_matrix import run_trial, validate_matrix
from e2e_slurm import client_command
from eos_readiness import audit_point
from e1_70b_suite import measured_caps

HERE = Path(__file__).resolve().parent
METHODS = ("tknp", "tknp-dbo", "dp", "pp", "dcp")


def configuration(job, nodes, model, context, tokens, methods, family, lanes=8, kv_gib=None):
    if len(nodes) < 2 or family == "e2e" and len(nodes) < 3:
        raise ValueError("Need two decode nodes, plus a dedicated node for real P/D")
    if lanes < 8 or lanes % 4:
        raise ValueError("Use at least eight graph lanes, divisible by four")
    offset = 8 if family == "e2e" else 0
    decoder = list(range(offset, offset + 16))
    repo = str(HERE.parents[2])
    snapshot = Path(repo) / ".research-source.json"
    commit = (json.loads(snapshot.read_text())["commit"] if snapshot.exists() else
              subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip())
    config = dict(model=model, load_format="dummy", kv_cache_dtype="auto", require_bf16=True,
        max_model_len=context + tokens, max_num_seqs=lanes, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.9, warmup_requests=0, timeout=900, routing="least-work",
        slurm_job_id=job, slurm_nodes=nodes, gpus_per_node=8,
        port_base=8600, nixl_port_base=20200, rdzv_port_base=24500,
        cache_root="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/readiness_roles",
        triton_root="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/triton_roles",
        runtime=dict(repo=repo, python="/home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3",
            image="/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh", source_commit=commit,
            container_name="tknp_ready_" + str(job),
            env=dict(VLLM_ATTENTION_BACKEND="FLASHINFER", OMP_NUM_THREADS="1",
                # Large H100 KV pools exceed gdr_copy's BAR pinning capacity.
                # Keep RDMA/cuda_ipc/cuda_copy eligible; exclude only gdr_copy.
                UCX_TLS="^gdr_copy",
                TORCHINDUCTOR_CACHE_DIR="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/inductor_torch291")),
        scope=f"EOS multi-node readiness; dummy BF16; {family}; not paper performance", trials=[])
    if kv_gib is not None:
        config["kv_cache_memory_bytes"] = kv_gib * 1024**3
    for method in methods:
        if method not in METHODS or method == "dcp" and family == "e2e":
            raise ValueError("Unsupported method/family (DCP NIXL mapping is not implemented)")
        env = dict(VLLM_KV_CACHE_LAYOUT="HND", VLLM_TKNP_LIFETIME_ADMISSION="1")
        role = dict(role="decode", connector="nixl" if family == "e2e" else "synthetic",
            gpus=decoder, tp=8, cuda_graph=True, graph_batch_size=lanes, max_num_seqs=lanes,
            native_graph_shapes=True, env=env)
        if method.startswith("tknp"):
            owners = [lanes // 4, lanes - lanes // 4]
            role.update(tknp=2, graph_slots=True, placement="cost", owner_counts=owners,
                        native_graph_shapes=False)
            env.update(VLLM_TKNP_REQUIRE_CUDAGRAPH="1", VLLM_TKNP_CAPTURE_REBALANCE="1",
                VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(context + tokens))
            if method == "tknp-dbo":
                role["enable_dbo"] = True
                env.update(VLLM_TKNP_DBO_ROOT_REQUESTS=str(owners[0]), VLLM_TKNP_DBO_C0_FRACTION="0.5",
                    VLLM_TKNP_DBO_ZERO_ROOT_C1="1", VLLM_TKNP_DBO_INDEPENDENT_COMM="1",
                    VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE="1", VLLM_TKNP_DBO_INDEPENDENT_TP="1",
                    VLLM_TKNP_DBO_LAYOUT="asymmetric", VLLM_TKNP_DBO_LOOKAHEAD="0",
                    VLLM_TKNP_DBO_SYMM_MEM="0")
            decoders = [role]
        elif method == "dp":
            decoders = [dict(role, gpus=decoder[i*8:(i+1)*8], graph_batch_size=lanes//2,
                             max_num_seqs=lanes//2) for i in range(2)]
        elif method == "pp":
            role.update(pp=2, pp_decode_microbatch=lanes//2, graph_batch_size=lanes//2)
            decoders = [role]
        else:
            role.update(tp=16, dcp=2, env=dict(env, VLLM_TKNP_LIFETIME_ADMISSION="0"))
            decoders = [role]
        producers = [dict(role="prefill", tp=8, gpus=list(range(8)), connector="nixl")]
        config["trials"].append(dict(name=method, servers=(producers if family == "e2e" else []) + decoders))
    validate_matrix(config, len(nodes)*8)
    return config


def check_resume_plan(previous, requested):
    """Permit new allocation/ports, never silently relabel prior measurements."""
    previous = dict(previous)
    spec = requested.get("workload_spec")
    if "workload_spec" not in previous and spec:
        # Earlier readiness manifests were uniform bursts. The workload file
        # is independently checked below; do not infer heterogeneous arrivals.
        if spec["rate"] != 0 or spec["lengths"] != [previous["max_model_len"] - spec["tokens"]]:
            raise ValueError("Cannot relabel an older uniform-burst workload")
        previous["workload_spec"] = spec
    keys = ("model", "load_format", "kv_cache_dtype", "require_bf16", "max_model_len",
            "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
            "kv_cache_memory_bytes", "warmup_requests", "routing", "gpus_per_node", "trials", "workload_spec")
    if any(previous.get(key) != requested.get(key) for key in keys):
        raise ValueError("Resume changes the experiment plan; choose a fresh result directory")


def stage_trials(config, count, lanes, skip_pp=False):
    """Never bypass the resident client's asynchronous-PP safety guard."""
    if count > lanes:
        raise ValueError("A staged resident cohort must fit all decoder graph lanes")
    for trial in config["trials"]:
        pp = any(role.get("pp", 1) > 1 for role in trial["servers"])
        if pp and not skip_pp:
            raise ValueError("PP resident pause/resume is unsupported; explicitly select ordinary PP mode")
        if not pp:
            trial["staged_decode"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[512])
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--kv-gib", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--lookahead", type=int, choices=[0, 1], default=0,
                        help="Enable the pipelined cohort schedule, not just dual-batch execution")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--family", choices=["e2e", "decode"], default="decode")
    parser.add_argument("--staged-decode", action="store_true", help="Pause resident requests after one decode token, then release together")
    parser.add_argument("--staged-skip-pp", action="store_true",
                        help="Explicitly run PP as ordinary E2E; never claim a resident pause/resume comparison for PP")
    parser.add_argument("--mixed-lengths", type=int, nargs="+", help="Bounded heterogeneous-arrival smoke; lengths must fit every selected context")
    parser.add_argument("--rate", type=float, default=0, help="Poisson requests/second; zero is a burst")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "readiness.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(attempts=[])
    for context in args.contexts:
        config = configuration(args.job, args.nodes, args.model, context, args.tokens,
                               args.methods, args.family, args.lanes, args.kv_gib)
        if not 0 < args.gpu_memory_utilization < 1:
            parser.error("GPU memory utilization must be between zero and one")
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
        for trial in config["trials"]:
            for role in trial["servers"]:
                if role.get("enable_dbo"):
                    role["env"]["VLLM_TKNP_DBO_LOOKAHEAD"] = str(args.lookahead)
        lengths = args.mixed_lengths or [context]
        if min(lengths) < 1 or max(lengths) > context or args.rate < 0:
            parser.error("Workload lengths must be positive and fit the context; nonnegative arrival rate")
        config["workload_spec"] = dict(lengths=lengths, count=args.count, tokens=args.tokens, rate=args.rate)
        if args.staged_decode:
            try:
                stage_trials(config, args.count, args.lanes, args.staged_skip_pp)
            except ValueError as exc:
                parser.error(str(exc))
        matrix_path = args.output / f"matrix-c{context}.json"
        if matrix_path.exists():
            check_resume_plan(json.loads(matrix_path.read_text()), config)
        matrix_path.write_text(json.dumps(config, indent=2)+"\n")
        workload = args.output / f"workload-c{context}.jsonl"
        if not workload.exists():
            command = [sys.executable, str(HERE / "e2e_workload.py"), "generate", "--natural",
                "--tokenizer", args.model, "--output", str(workload), "--count", str(args.count),
                "--lengths", *map(str, lengths), "--output-lengths", str(args.tokens), "--rate", str(args.rate)]
            subprocess.run(client_command(config, command), check=True, timeout=180)
        rows = list(map(json.loads, workload.read_text().splitlines()))
        if len(rows) != args.count or any(len(r["prompt_token_ids"]) != lengths[i % len(lengths)] or r["max_tokens"] != args.tokens for i,r in enumerate(rows)):
            raise ValueError("Resume changes workload size/length; choose a fresh result directory")
        for trial in config["trials"]:
            key = f"c{context}-{trial['name']}"
            if any(r["key"] == key and r["status"] == "passed" for r in state["attempts"]):
                continue
            attempt = sum(r["key"] == key for r in state["attempts"])
            destination = args.output / (key + (f"-retry{attempt}" if attempt else ""))
            row = dict(key=key, path=str(destination), start=time.time(),
                workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest())
            print("START", key, flush=True)
            try:
                run_trial(trial, config, workload, destination)
                evidence = audit_point(destination, trial, args.count, args.tokens)
                caps = measured_caps(destination, context, args.tokens, trial)
                row.update(status="passed", client=evidence["client"], capacities=caps,
                    full_graph_all_ranks=True, bf16=True, kv_drained=True)
                (destination / "analysis.json").write_text(json.dumps(evidence, indent=2)+"\n")
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
                print(row["error"], flush=True)
            row["end"] = time.time()
            state["attempts"].append(row)
            state_path.write_text(json.dumps(state, indent=2)+"\n")
            print("FINISH", key, row["status"], flush=True)
            if row["status"] == "failed":
                raise SystemExit(1)
    print("EOS_MULTINODE_READINESS_PASS", flush=True)


if __name__ == "__main__":
    main()
