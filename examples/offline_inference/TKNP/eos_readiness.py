#!/usr/bin/env python3
"""Bounded single-node EOS 8B script/graph checks, not paper performance data.

No allocations or downloads. Real prefill-computed KV is transferred by NIXL;
dummy weights are explicit and do not establish trained-model correctness.
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

METHODS = ("tp2", "dp2", "tp2-dp2", "pp2", "tp2-pp2", "tknp2",
           "tknp2-dbo", "tp2-tknp2", "tp2-tknp2-dbo")


def configuration(model, context, tokens, kv_gib, methods, load_format="dummy"):
    common = dict(model=model, load_format=load_format, kv_cache_dtype="auto",
                  require_bf16=True, max_model_len=context + tokens,
                  max_num_seqs=16, max_num_batched_tokens=2048,
                  kv_cache_memory_bytes=kv_gib * 1024 ** 3,
                  gpu_memory_utilization=0.9, warmup_requests=0,
                  timeout=900, routing="least-work", trials=[],
                  scope="EOS one-node 8B readiness, artificially capped pools/slots; not a paper comparison")
    producer = dict(role="prefill", gpus=[0], connector="nixl")
    for method in methods:
        if method not in METHODS:
            raise ValueError("Unknown readiness method: " + method)
        env = {"VLLM_TKNP_LIFETIME_ADMISSION": "1", "VLLM_KV_CACHE_LAYOUT": "HND"}
        role = dict(role="decode", connector="nixl", cuda_graph=True,
                    graph_batch_size=8, max_num_seqs=16, env=env)
        if "tknp" in method:
            role.update(gpus=[1, 2, 3, 4] if method.startswith("tp2-") else [1, 2],
                        tp=2 if method.startswith("tp2-") else 1, tknp=2,
                        graph_slots=True, placement="cost", owner_counts=[2, 6])
            env.update(VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(context + tokens),
                       VLLM_TKNP_REQUIRE_CUDAGRAPH="1")
            if method.endswith("-dbo"):
                role["enable_dbo"] = True
                env.update(VLLM_TKNP_DBO_ROOT_REQUESTS="2",
                           VLLM_TKNP_DBO_C0_FRACTION="0.5",
                           VLLM_TKNP_DBO_ZERO_ROOT_C1="1",
                           VLLM_TKNP_DBO_INDEPENDENT_COMM="1",
                           VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE="1",
                           VLLM_TKNP_DBO_INDEPENDENT_TP="1",
                           VLLM_TKNP_DBO_LAYOUT="asymmetric",
                           VLLM_TKNP_DBO_LOOKAHEAD="0",
                           VLLM_TKNP_DBO_SYMM_MEM="0")
            decoders = [role]
        elif method in {"dp2", "tp2-dp2"}:
            tp = 2 if method.startswith("tp2-") else 1
            decoders = [dict(role, tp=tp, gpus=list(range(1+i*tp, 1+(i+1)*tp)),
                             max_num_seqs=2, graph_batch_size=2,
                             native_graph_shapes=True) for i in range(2)]
        else:
            tp, pp = (2, 2) if method == "tp2-pp2" else (1, 2) if method == "pp2" else (2, 1)
            role.update(tp=tp, pp=pp, gpus=list(range(1, 1+tp*pp)), native_graph_shapes=True)
            if pp > 1:
                role.update(pp_decode_microbatch=4, graph_batch_size=4)
            decoders = [role]
        common["trials"].append(dict(name=method, servers=[dict(producer)] + decoders))
    return common


def audit_point(path, trial, count, tokens):
    """Require measured graph, precision, completion and all-rank evidence."""
    from e1_pilot_report import analyze
    evidence = analyze(path)
    if not all(evidence[key] for key in (
            "all_decode_workers_full_graph", "all_servers_zero_measured_preemptions",
            "all_servers_kv_drained")):
        raise RuntimeError("Graph, preemption or KV-drain evidence failed")
    for index, role in enumerate(trial["servers"]):
        name = f"server{index}"
        expected = list(range(len(role["gpus"])))
        resources = evidence["resources"][name]
        if sorted(r["rank"] for r in resources) != expected:
            raise RuntimeError("Missing or duplicate assigned worker evidence")
        if any(r["model_dtype"] != "torch.bfloat16" or r["quantization"] is not None
               or r["kv_storage_dtypes"] != ["torch.bfloat16"] for r in resources):
            raise RuntimeError("BF16 model/KV storage evidence failed")
        if role["role"] == "decode":
            executions = evidence["execution"].get(name, [])
            if sorted(r["rank"] for r in executions) != expected:
                raise RuntimeError("Missing or duplicate measured execution ranks")
    client = evidence["client"]
    if client["completed"] != count or client["output_tokens"] != count * tokens:
        raise RuntimeError("Request/output-token count mismatch")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--load-format", choices=["dummy", "auto"], default="dummy")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--kv-gib", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.context, args.tokens, args.count, args.kv_gib) < 1:
        parser.error("Positive workload/pool sizes required")
    config = configuration(args.model, args.context, args.tokens, args.kv_gib, args.methods, args.load_format)
    validate_matrix(config, args.gpu_count)
    if args.validate_only:
        print(json.dumps(config, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "matrix.json").write_text(json.dumps(config, indent=2) + "\n")
    workload = args.output / "workload.jsonl"
    if not workload.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("e2e_workload.py")),
                        "generate", "--natural", "--tokenizer", args.model,
                        "--output", str(workload), "--count", str(args.count),
                        "--lengths", str(args.context), "--output-lengths", str(args.tokens)], check=True)
    status_path = args.output / "readiness.json"
    state = json.loads(status_path.read_text()) if args.resume and status_path.exists() else dict(
        scope=config["scope"], load_format=args.load_format, kv_dtype="BF16",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        dcp="not run: eight KV heads require at least TP16+DCP2, beyond one 8-GPU EOS node",
        trials=[])
    for trial in config["trials"]:
        if any(r["name"] == trial["name"] and r["status"] == "passed" for r in state["trials"]):
            continue
        attempt = sum(r["name"] == trial["name"] for r in state["trials"])
        destination = args.output / (trial["name"] + (f"-retry{attempt}" if attempt else ""))
        row = dict(name=trial["name"], path=str(destination), start=time.time())
        print("START", trial["name"], destination, flush=True)
        try:
            result = run_trial(trial, config, workload, destination)
            evidence = audit_point(destination, trial, args.count, args.tokens)
            client = evidence["client"]
            row.update(status="passed", result=result, all_workers_graph=True,
                       client=client)
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
            print(row["error"], flush=True)
        row["end"] = time.time()
        state["trials"].append(row)
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        print("FINISH", trial["name"], row["status"], flush=True)
        if row["status"] == "failed":
            # Inspect a first environment/graph failure before repeating it in
            # every method or launching competing kernels on a stale server.
            raise SystemExit(1)
    print("EOS_READINESS_PASS", len(args.methods), flush=True)


if __name__ == "__main__":
    main()
