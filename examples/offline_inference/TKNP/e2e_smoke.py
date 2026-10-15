#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Own exactly four GPUs for single-node real-weight P/D correctness smoke."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    parser.add_argument("--graph-dbo", action="store_true")
    parser.add_argument("--continuous-graph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-combo-kernels", action="store_true")
    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--record-full-logits", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--pp-decode", action="store_true", help="Validate PP1 prefill -> PP2 decode instead of TKNP")
    parser.add_argument("--plain-decode", action="store_true",
                        help="Ordinary TP1 NIXL decoder: numerical control without TKNP")
    parser.add_argument("--tp2-shared-gpu-check", action="store_true",
                        help="Correctness ONLY: TP2xTKNP2 on all four GPUs; prefill/reference share GPUs 0/3. Never use its timing as a comparison")
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--abort-check", action="store_true", help="Cancel before/after KV load, then verify normal serving remains correct")
    parser.add_argument("--count", type=int)
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--output-lengths", type=int, nargs="+")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--graph-batch-size", type=int, default=8)
    parser.add_argument("--root-requests", type=int, default=2)
    parser.add_argument("--kv-cache-memory-bytes", type=int,
                        help="Per-GPU pool for each role; default 2 GiB graph / 512 MiB eager")
    args = parser.parse_args()
    if args.continuous_graph:
        args.graph_dbo = True
    if args.pp_decode and args.graph_dbo:
        parser.error("PP correctness smoke does not use TKNP/DBO")
    if args.pp_decode and args.tp2_shared_gpu_check:
        parser.error("Select PP or TP2xTKNP2 correctness, not both")
    if args.plain_decode and (args.graph_dbo or args.pp_decode or args.tp2_shared_gpu_check):
        parser.error("Plain numerical control does not use TKNP/DBO/PP")
    if args.abort_check and args.graph_dbo and not args.continuous_graph:
        parser.error("Cancellation smoke needs continuous serving, not fixed waves")
    lengths = args.lengths or ([128] if args.graph_dbo and not args.continuous_graph else [128, 256, 512, 1024])
    output_lengths = args.output_lengths or ([16] if args.graph_dbo and not args.continuous_graph else [4, 8, 12, 16])
    count = args.count if args.count is not None else (16 if args.graph_dbo else 8)
    rate = args.rate if args.rate is not None else (16 if args.continuous_graph else 0 if args.graph_dbo else 8)
    if (count <= 0 or min(lengths + output_lengths) <= 0 or rate < 0
            or args.graph_batch_size <= 0 or (args.graph_dbo and args.graph_batch_size < 2) or args.root_requests < 0
            or args.root_requests > args.graph_batch_size // 2
            or (args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0)):
        parser.error("Invalid workload, graph split or memory budget")
    if args.graph_dbo and not args.continuous_graph and (count % args.graph_batch_size or len(set(output_lengths)) != 1):
        parser.error("Fixed waves need complete graph-sized waves and uniform output lengths")
    max_model_len = max(2048, max(lengths) + max(max(output_lengths), 64 if args.abort_check else 0))
    max_seqs = max(16, args.graph_batch_size)
    kv_bytes = args.kv_cache_memory_bytes or (2147483648 if args.graph_dbo else 536870912)
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    workers = []
    handles = []

    def launch(name, gpus, port, extra, count=1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus,
                   VLLM_TKNP_SERVE_DEBUG="1",
                   VLLM_ENABLE_V1_MULTIPROCESSING="0",
                   VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1",
                   VLLM_NIXL_SIDE_CHANNEL_PORT=str(port + 11000),
                   VLLM_NIXL_ABORT_REQUEST_TIMEOUT="720",
                   VLLM_CACHE_ROOT=str(args.output / "cache" / name),
                   TRITON_CACHE_DIR=str(args.output / "triton" / name),
                   NCCL_LAUNCH_ORDER_IMPLICIT="1")
        if name == "decode" and args.graph_dbo:
            env.update(VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(max_model_len),
                       VLLM_TKNP_REQUIRE_CUDAGRAPH="1",
                       VLLM_TKNP_DBO_ROOT_REQUESTS=str(args.root_requests),
                       VLLM_TKNP_DBO_ZERO_ROOT_C1="1",
                       VLLM_TKNP_DBO_INDEPENDENT_COMM="1",
                       VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE="1",
                       VLLM_TKNP_DBO_LOOKAHEAD="0",
                       VLLM_TKNP_DBO_LAYOUT="asymmetric")
        command = [sys.executable]
        if count > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={count}"]
        command += [str(here / "e2e_server.py"), "--port", str(port), "--result-dir", str(args.output / name),
                    "--load-format", args.load_format, "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_seqs),
                    "--record-logit-topk",
                    "--kv-cache-dtype", args.kv_cache_dtype,
                    "--max-num-batched-tokens", str(max(2048, args.graph_batch_size)), "--kv-cache-memory-bytes", str(kv_bytes)] + extra
        if args.record_full_logits:
            command += ["--record-full-logits"]
        handle = (args.output / f"{name}.log").open("x")
        handles.append(handle)
        workers.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        return f"http://127.0.0.1:{port}"

    try:
        prefill = launch("prefill", "0", 8100, ["--connector", "nixl"])
        decode_args = (["--connector", "nixl", "--cuda-graph", "--graph-batch-size", str(args.graph_batch_size)] if args.plain_decode else
                       ["--pp", "2", "--connector", "nixl", "--cuda-graph"] if args.pp_decode else
                       ["--tknp", "2", "--connector", "nixl", "--placement", "legacy" if args.graph_dbo and not args.continuous_graph else "cost"])
        if args.plain_decode and args.disable_combo_kernels:
            decode_args += ["--disable-combo-kernels"]
        if args.tp2_shared_gpu_check:
            decode_args += ["--tp", "2"]
        if args.profile_steps:
            decode_args += ["--profile-after-step", "5", "--profile-steps", str(args.profile_steps)]
        if args.graph_dbo:
            decode_args += ["--cuda-graph", "--enable-dbo", "--graph-batch-size", str(args.graph_batch_size)]
            if args.continuous_graph:
                decode_args += ["--owner-counts", str(args.root_requests),
                                str(args.graph_batch_size - args.root_requests)]
            if args.disable_compile:
                decode_args += ["--disable-compile"]
            if args.disable_combo_kernels:
                decode_args += ["--disable-combo-kernels"]
            decode_args += (["--graph-slots"] if args.continuous_graph else
                            ["--fixed-wave", "--fixed-output-tokens", str(output_lengths[0])])
        decode = launch("decode", "0,1,2,3" if args.tp2_shared_gpu_check else "1" if args.plain_decode else "1,2", 8101,
                        decode_args, 1 if args.pp_decode or args.plain_decode else 4 if args.tp2_shared_gpu_check else 2)
        reference = launch("reference", "3", 8102, [])
        deadline = time.monotonic() + 900
        pending = {prefill, decode, reference}
        while pending:
            if any(p.poll() is not None for p in workers):
                raise RuntimeError("A server exited; inspect role logs")
            for url in list(pending):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if json.load(response)["ready"]:
                            pending.remove(url)
                except (OSError, ValueError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Servers not ready after 15 minutes")
            time.sleep(1)
        print("All real-weight P/D/reference servers ready", flush=True)
        workload = args.output / "workload.jsonl"
        workload_args = (["--count", str(count), "--lengths"] + list(map(str, lengths))
                         + ["--output-lengths"] + list(map(str, output_lengths))
                         + ["--rate", str(rate)])
        if args.natural:
            workload_args += ["--natural"]
        subprocess.run([sys.executable, str(here / "e2e_workload.py"), "generate", "--output", str(workload)] + workload_args, check=True)
        if args.abort_check:
            async def check_aborts():
                import aiohttp
                prompt = json.loads(workload.read_text().splitlines()[-1])["prompt_token_ids"]
                async with aiohttp.ClientSession() as session:
                    for mode in ("headers", "during-kv", "first-token"):
                        req_id = "abort-" + mode + "-" + uuid.uuid4().hex
                        transfer = None
                        async with session.post(prefill + "/generate", json={
                            "request_id": req_id, "prompt_token_ids": prompt, "max_tokens": 1,
                            "kv_transfer_params": {"do_remote_decode": True}}) as response:
                            response.raise_for_status()
                            async for line in response.content:
                                transfer = json.loads(line).get("kv_transfer_params") or transfer
                        if not transfer:
                            raise RuntimeError("Cancellation test did not obtain real KV")
                        async with session.post(decode + "/generate", json={
                            "request_id": req_id, "prompt_token_ids": prompt, "max_tokens": 64,
                            "kv_transfer_params": transfer}) as response:
                            response.raise_for_status()
                            if mode == "during-kv":
                                await asyncio.sleep(0.001)
                            elif mode == "first-token":
                                await response.content.readline()
                    deadline = time.monotonic() + 30
                    clear_polls = 0
                    while clear_polls < 3:
                        await asyncio.sleep(0.02)
                        counts = []
                        for url in (prefill, decode):
                            async with session.get(url + "/health") as response:
                                counts.append((await response.json())["outstanding_requests"])
                        clear_polls = clear_polls + 1 if not any(counts) else 0
                        if time.monotonic() > deadline:
                            raise RuntimeError("KV/request lifecycle did not drain after cancellations")
                (args.output / "cancellation.json").write_text(json.dumps({"passed": True, "modes": ["headers", "during-kv", "first-token"]}) + "\n")
            asyncio.run(check_aborts())
        for label, url, purls in [("colocated", reference, []), ("disaggregated", decode, [prefill])]:
            command = [sys.executable, str(here / "e2e_workload.py"), "run", "--workload", str(workload),
                       "--output", str(args.output / label), "--decode-urls", url]
            if purls:
                command += ["--prefill-urls"] + purls
            subprocess.run(command, check=True)
        deadline = time.monotonic() + 30
        while True:
            counts = []
            for url in (prefill, decode, reference):
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    counts.append(json.load(response)["outstanding_requests"])
            if not any(counts):
                break
            if any(p.poll() is not None for p in workers) or time.monotonic() > deadline:
                raise RuntimeError("Request/KV state did not drain after completed workload")
            time.sleep(0.02)
        def tokens(label):
            return {r["request_id"]: r["token_ids"] for r in
                    map(json.loads, (args.output / label / "requests.jsonl").read_text().splitlines())}
        a, b = tokens("colocated"), tokens("disaggregated")
        matches = {req: a[req] == b[req] for req in a}
        from diagnose_e2e_tokens import analyze
        (args.output / "numerics.json").write_text(json.dumps(analyze(args.output), indent=2) + "\n")
        verdict = {"exact_greedy_tokens": matches, "passed": all(matches.values()),
                   "scope": ("single-node ordinary TP1 NIXL numerical control, without TKNP"
                             if args.plain_decode else "single-node real NIXL, TP2xTKNP2 with shared prefill/reference GPUs; correctness only, timings INVALID"
                             if args.tp2_shared_gpu_check else "single-node real NIXL, PP1 prefill -> PP2 decode, real layer-range KV reads"
                             if args.pp_decode else "single-node real NIXL, continuous fixed graph lanes with DBO"
                             if args.continuous_graph else "single-node real NIXL, two fixed graph/DBO waves; NOT continuous batching"
                             if args.graph_dbo else "single-node TP1 prefill -> TP1xTKNP2 real NIXL; mixed arrivals/completions; eager")}
        (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, indent=2), flush=True)
    finally:
        for process in workers:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in workers:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()
    # Inspect every replicated scheduler, not only the HTTP root. This catches
    # owner-local KV leaks even when all HTTP streams completed successfully.
    shutdowns = []
    for journal in args.output.glob("*/rank*.jsonl"):
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        shutdowns += [event for event in events if event["event"] == "shutdown"]
    verdict["request_kv_lifecycle_passed"] = (
        len(shutdowns) == (3 if args.pp_decode or args.plain_decode else 6 if args.tp2_shared_gpu_check else 4)
        and all(event.get("all_kv_free") and not event["remaining_requests"] for event in shutdowns))
    verdict["passed"] &= verdict["request_kv_lifecycle_passed"]
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if not verdict["passed"]:
        raise RuntimeError("Exact greedy-token or request/KV lifecycle check failed; inspect verdict and numerics")


if __name__ == "__main__":
    main()
