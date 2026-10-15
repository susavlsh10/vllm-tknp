#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research NDJSON serving frontend; external-launcher TKNP or ordinary vLLM.

Only rank zero owns HTTP. It broadcasts admission/abort commands to replicated
engines, using identical internal IDs. No GPU work happens in the HTTP thread.
This is deliberately not a production OpenAI API server.
"""
import argparse
import asyncio
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import threading
import time


def aligned_owner_counts(counts, multiple):
    """Retain a supported exact graph size without exceeding any owner quota."""
    result = list(counts)
    target = sum(result) // multiple * multiple
    if target < multiple:
        raise ValueError("Not enough resident lanes for one TKNP graph")
    while sum(result) > target:
        owner = max(range(len(result)), key=lambda i: (result[i], i))
        result[owner] -= 1
    return result


def owner_graphs_consistent(resources):
    return len({json.dumps(r["graph_splits"], sort_keys=True) for r in resources}) <= 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tknp", type=int, default=1)
    parser.add_argument("--dcp", type=int, default=1)
    parser.add_argument("--connector", choices=["none", "nixl", "synthetic"], default="none")
    parser.add_argument("--placement", choices=["legacy", "count", "memory", "cost"], default="cost")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--require-bf16", action="store_true",
                        help="Reject quantized model execution or non-BF16 KV")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--disable-compile", action="store_true",
                        help="Keep full CUDA graphs but skip Inductor compilation (diagnostic)")
    parser.add_argument("--disable-combo-kernels", action="store_true",
                        help="Keep Inductor/CUDA graphs; disable nightly combo-kernel tuning")
    parser.add_argument("--enable-dbo", action="store_true")
    parser.add_argument("--graph-batch-size", type=int)
    parser.add_argument("--native-graph-shapes", action="store_true",
                        help="Capture ordinary vLLM batch-size buckets for non-TKNP baselines")
    parser.add_argument("--pp-decode-microbatch", type=int,
                        help="Cap native PP decode scheduling per batch, retaining full resident request buffers")
    parser.add_argument("--owner-counts", type=int, nargs="+",
                        help="Explicit captured owner quotas, including for matched DBO on/off")
    parser.add_argument("--fixed-wave", action="store_true", help="Fixed-shape correctness path, NOT continuous batching")
    parser.add_argument("--graph-slots", action="store_true", help="Continuous decode in padded fixed graph lanes")
    parser.add_argument("--fixed-output-tokens", type=int)
    parser.add_argument("--record-logit-topk", action="store_true", help="Correctness only; adds synchronization")
    parser.add_argument("--record-full-logits", action="store_true",
                        help="Correctness only; persist request-aligned full logits")
    parser.add_argument("--profile-after-step", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Capture each worker with PyTorch; profiling runs are not performance results")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--idle-kv-poll-ms", type=float, default=1.0,
                        help="Transfer-only polling delay; prevents an idle export from busy-spinning the CPU")
    args = parser.parse_args()
    if args.dcp > 1 and args.connector == "nixl":
        parser.error("Real NIXL handoff into DCP's interleaved KV layout is not implemented; use a colocated DCP baseline, not ordinary TP KV descriptors")
    if args.tknp > 1 and args.connector == "none":
        parser.error("TKNP serving is decode-only: select nixl or synthetic")
    if args.tknp > 1 and args.cuda_graph and args.placement != "legacy" and not args.graph_slots:
        parser.error("Dynamic placement graph signatures are not yet validated; use eager or legacy fixed-batch")
    if args.enable_dbo and (args.tknp == 1 or not args.cuda_graph or not args.graph_batch_size):
        parser.error("DBO needs TKNP, CUDA graphs and an exact graph batch size")
    if args.fixed_wave and (not args.graph_batch_size or not args.fixed_output_tokens or args.placement != "legacy"):
        parser.error("Fixed waves need an exact batch/output length and legacy ownership")
    if args.graph_slots and (args.fixed_wave or not args.cuda_graph or not args.graph_batch_size
                            or args.tknp < 2 or args.placement == "legacy"):
        parser.error("Continuous graph lanes need TKNP, exact graph capacity and per-request placement, not fixed waves")
    if args.profile_after_step < 0 or args.profile_steps < 0:
        parser.error("Nonnegative profiling interval required")
    if args.idle_kv_poll_ms < 0:
        parser.error("Nonnegative idle KV polling delay required")
    if args.owner_counts and (not args.graph_batch_size or len(args.owner_counts) != args.tknp
                              or min(args.owner_counts) < 0 or sum(args.owner_counts) != args.graph_batch_size):
        parser.error("Owner counts must match TKNP size and exact graph capacity")
    if args.graph_batch_size and (args.graph_batch_size > args.max_num_seqs
                                  or args.graph_batch_size > args.max_num_batched_tokens):
        parser.error("Graph lanes must fit max-num-seqs and max-num-batched-tokens")
    if args.graph_slots and args.graph_batch_size % args.tknp:
        # The validated TKNP dispatcher retains only multiples of TKNP size.
        # Normalize older capacity-driven controllers before engine creation;
        # never remove that core safety check or silently fall back to eager.
        if not args.owner_counts:
            parser.error("Unaligned TKNP lanes require explicit owner quotas")
        args.requested_graph_batch_size = args.graph_batch_size
        args.requested_owner_counts = list(args.owner_counts)
        args.owner_counts = aligned_owner_counts(args.owner_counts, args.tknp)
        args.graph_batch_size = sum(args.owner_counts)
        args.max_num_seqs = args.graph_batch_size
        print(f"TKNP_GRAPH_ALIGNMENT requested={args.requested_graph_batch_size} effective={args.graph_batch_size} owners={args.owner_counts}", flush=True)
    return args


def defer_kv_abort(request, completed_receives):
    """Do not recycle a destination block while an async NIXL read can write it."""
    return (request is not None and request.status.name == "WAITING_FOR_REMOTE_KVS"
            and request.request_id not in completed_receives)


def reject_varying_singleton_positions(c0_counts, c1_counts):
    """Fail before capture threads hang on a backed-symbol singleton guard.

    Attention-rank positions are local, unlike the root's global positions.
    The current trace-once compiler specializes size one. Do not silently
    reuse it for the other cohort's larger position vector.
    """
    for owner, (a, b) in enumerate(zip(c0_counts, c1_counts)):
        if owner and min(a, b) == 1 and max(a, b) > 1:
            raise ValueError("Unsupported compiled cohort position shapes: singleton changes size; increase graph lanes or change owner quotas")


def main():
    args = arguments()
    faulthandler.enable()
    # Do not run an asynchronous CPython frame dump during Dynamo/Inductor
    # startup. SIGSEGVs on this ARM Python build coincided with these dumps
    # inspecting compiler threads. Fatal-signal diagnostics remain enabled.
    os.environ["VLLM_TKNP_PLACEMENT_POLICY"] = args.placement
    if args.fixed_wave:
        os.environ["VLLM_TKNP_KV_WAVE_SIZE"] = str(args.graph_batch_size)
    if args.graph_slots:
        os.environ["VLLM_TKNP_GRAPH_SLOTS"] = str(args.graph_batch_size)
    if args.owner_counts:
        os.environ["VLLM_TKNP_CAPTURE_OWNER_COUNTS"] = json.dumps(args.owner_counts)
        if args.enable_dbo and not args.disable_compile:
            from vllm.forward_context import build_tknp_dbo_cohort_rank_order, tknp_cohort_split_point
            order = build_tknp_dbo_cohort_rank_order(args.owner_counts)
            split = tknp_cohort_split_point(args.graph_batch_size)
            reject_varying_singleton_positions(
                [order[:split].count(rank) for rank in range(args.tknp)],
                [order[split:].count(rank) for rank in range(args.tknp)])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    import torch.distributed as dist
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.parallel_state import get_world_group
    from vllm.sampling_params import RequestOutputKind

    external = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    options = dict(model=args.model, dtype="bfloat16", seed=1,
                   tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
                   decode_context_parallel_size=args.dcp,
                   distributed_executor_backend="external_launcher" if external else "mp",
                   enforce_eager=not args.cuda_graph, enable_prefix_caching=False,
                   max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                   max_num_batched_tokens=args.max_num_batched_tokens,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   load_format=args.load_format, kv_cache_dtype=args.kv_cache_dtype,
                   async_scheduling=False)
    if args.kv_cache_memory_bytes:
        options["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.tknp > 1:
        options.update(enable_token_parallel=True, token_parallel_size=args.tknp)
    if args.nnodes > 1:
        if external:
            raise ValueError("Native multinode MP options must not be combined with torchrun")
        options.update(nnodes=args.nnodes, node_rank=args.node_rank,
                       master_addr=args.master_addr, master_port=args.master_port)
    if args.cuda_graph:
        # NIXL chooses HND KV layout. Its FlashInfer path supports full decode
        # graphs, not full mixed prefill graphs. Avoid an implicit downgrade
        # that tries to capture extra NCCL-containing piecewise graphs.
        options["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
        if args.disable_compile:
            options["compilation_config"]["mode"] = 0
        if args.disable_combo_kernels:
            options["compilation_config"]["inductor_compile_config"] = {
                "combo_kernels": False, "benchmark_combo_kernel": False}
        if args.graph_batch_size:
            options["cudagraph_capture_sizes"] = [args.graph_batch_size]
            if args.native_graph_shapes:
                if args.tknp > 1:
                    raise ValueError("Native bucket capture does not replace TKNP ownership signatures")
                maximum = args.graph_batch_size
                options["cudagraph_capture_sizes"] = sorted(
                    {v for v in [1, 2, 4, *range(8, maximum + 1, 8), maximum] if v <= maximum})
    if args.enable_dbo:
        options.update(enable_dbo=True, dbo_decode_token_threshold=2)
    if args.profile_steps:
        options["profiler_config"] = dict(
            profiler="torch", torch_profiler_dir=str((args.result_dir / "traces").resolve()),
            torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
            torch_profiler_dump_cuda_time_total=False)
    if args.connector != "none":
        name = ("TKNPNixlConnector" if args.tknp > 1 else "PPNixlConnector" if args.pp > 1
                else "NixlConnector") if args.connector == "nixl" else "DecodeBenchConnector"
        module = {"TKNPNixlConnector": "tknp_nixl_connector", "PPNixlConnector": "pp_nixl_connector"}.get(name)
        if external and args.tknp == 1 and args.connector == "nixl" and args.tp * args.pp > 1:
            name = "ExternalPPNixlConnector" if args.pp > 1 else "ExternalNixlConnector"
            module = "external_nixl_connector"
        options["kv_transfer_config"] = KVTransferConfig(
            kv_connector=name, kv_role="kv_both", engine_id=f"research-{args.port}",
            kv_connector_module_path=("vllm.distributed.kv_transfer.kv_connector.v1." + module if module else None),
            kv_load_failure_policy="fail" if args.tknp > 1 or args.pp > 1 else "recompute",
            kv_connector_extra_config={"num_threads": 1, "fill_std": 0.0})
    if args.nnodes > 1 and args.node_rank > 0:
        # Match vllm.entrypoints.cli.serve.run_headless: only the leader owns
        # the scheduler/frontend, while remote native workers consume its MQ.
        # Unlike external_launcher, this retains PP's concurrent batch queue.
        from vllm.engine.arg_utils import EngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        vllm_config = EngineArgs(**dict(options, disable_log_stats=True)).create_engine_config(
            usage_context=UsageContext.LLM_CLASS, headless=True)
        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        halt = threading.Event()
        class HeadlessControl(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/shutdown":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                halt.set()
            def log_message(self, *unused):
                pass
        control = ThreadingHTTPServer((args.host, args.port + 1000), HeadlessControl)
        threading.Thread(target=control.serve_forever, daemon=True).start()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: halt.set())
        executor.start_worker_monitor()
        try:
            while not halt.wait(0.1):
                if executor.is_failed:
                    raise RuntimeError("Native remote model worker failed")
        finally:
            executor.shutdown()
            control.shutdown()
            control.server_close()
        return
    llm = LLM(**options)
    if os.environ.get("VLLM_TKNP_TRACEBACK_INTERVAL"):
        faulthandler.dump_traceback_later(
            int(os.environ["VLLM_TKNP_TRACEBACK_INTERVAL"]), repeat=True)
    engine = llm.llm_engine
    if args.require_bf16 and (str(engine.model_config.dtype) != "torch.bfloat16"
                              or engine.model_config.quantization is not None
                              or args.kv_cache_dtype != "auto"):
        raise RuntimeError("BF16-only experiment refuses quantized weights or non-BF16 KV")
    # Clients provide globally unique IDs; use exactly the same ID on every
    # replicated scheduler. vLLM's random per-process suffix breaks KV owner
    # completion propagation even though fixed offline decode tolerated it.
    def stable_id(request):
        request.external_req_id = request.request_id
    engine.input_processor.assign_request_id = stable_id
    core = engine.engine_core.engine_core
    if args.pp_decode_microbatch:
        if args.pp <= 1 or args.connector == "none" or args.pp_decode_microbatch < 1:
            raise ValueError("PP decode microbatching requires PP>1 and cached decode input")
        # Keep standard validated engine buffer sizes (tokens >= resident
        # requests). Only limit the central scheduler's per-step token budget,
        # so PP can enqueue disjoint request microbatches concurrently.
        core.scheduler.max_num_scheduled_tokens = args.pp_decode_microbatch
    initial_free_blocks = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    world_group = get_world_group().cpu_group if external else None
    def resource_snapshot(worker):
        import torch
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        runner = worker.model_runner
        kv = runner.kv_cache_config
        return {"rank": worker.rank, "gpu": torch.cuda.get_device_name(),
                "hostname": __import__("socket").gethostname(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "gpu_total_bytes": torch.cuda.get_device_properties(worker.device).total_memory,
                "model_memory_bytes": getattr(runner, "model_memory_usage", None),
                "kv_blocks": kv.num_blocks, "block_tokens": worker.cache_config.block_size,
                "kv_allocated_bytes": sum(t.size for t in kv.kv_cache_tensors),
                "graph_splits": getattr(runner, "_tknp_graph_splits", {}),
                "graph_mode": worker.compilation_config.cudagraph_mode.name,
                "compilation_mode": worker.compilation_config.mode.name,
                "model_dtype": str(worker.model_config.dtype),
                "quantization": worker.model_config.quantization,
                "kv_layout": get_kv_cache_layout(),
                "kv_storage_dtypes": sorted({str(t.dtype) for t in runner.kv_caches if isinstance(t, torch.Tensor)})}
    resources = llm.collective_rpc(resource_snapshot)
    if args.tknp > 1 and args.graph_slots and not owner_graphs_consistent(resources):
        raise RuntimeError("TKNP graph owner layouts disagree across TP shards")
    if args.require_bf16 and any(r["kv_storage_dtypes"] != ["torch.bfloat16"] for r in resources):
        raise RuntimeError("Runtime KV storage is not exclusively BF16")
    def install_execution_check(worker):
        from collections import Counter
        from vllm.config import CUDAGraphMode
        runner = worker.model_runner
        runner._research_execution_counts = Counter()
        original = runner._determine_batch_execution_and_padding
        def checked(num_tokens, num_reqs, num_scheduled_tokens_np,
                    max_num_scheduled_tokens, *extra, **kwargs):
            result = original(num_tokens, num_reqs, num_scheduled_tokens_np,
                              max_num_scheduled_tokens, *extra, **kwargs)
            phase = "decode" if max_num_scheduled_tokens == 1 else "prefill"
            runner._research_execution_counts[phase + ":" + result[0].name] += 1
            if args.cuda_graph and phase == "decode" and result[0] != CUDAGraphMode.FULL:
                raise RuntimeError("Research CUDA-graph decoder selected a non-FULL execution mode")
            return result
        runner._determine_batch_execution_and_padding = checked
    llm.collective_rpc(install_execution_check)
    if external:
        all_resources = [None] * dist.get_world_size()
        dist.all_gather_object(all_resources, resources, group=world_group)
        resources = [item for group in all_resources for item in group]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    journal = (args.result_dir / f"rank{rank}.jsonl").open("x")
    journal.write(json.dumps({"event": "configuration", **vars(args)}, default=str) + "\n")
    journal.flush()
    if rank == 0:
        import torch
        root = Path(__file__).resolve().parents[3]
        source_files = (subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=root, text=True).splitlines()
                        if shutil.which("git") else
                        [str(path.relative_to(root)) for path in root.rglob("*")
                         if not any(part.startswith(".") or part in {"ubatch_results", "prompt_cache", "__pycache__"}
                                    for part in path.relative_to(root).parts)])
        digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in source_files if Path(name).suffix in {".py", ".sh", ".sbatch"}
                   and (root / name).is_file()}
        manifest = {"arguments": vars(args), "torch": torch.__version__,
                    "runtime_env": {key: value for key, value in os.environ.items()
                                    if key.startswith(("VLLM_TKNP_", "VLLM_NIXL_", "NCCL_"))},
                    "gpu": torch.cuda.get_device_name(), "world_size": dist.get_world_size() if external else args.tp * args.pp,
                    "git_head": (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                                 if shutil.which("git") else os.environ.get("VLLM_RESEARCH_GIT_HEAD", "unavailable-in-container")),
                    "source_sha256": digests, "synthetic_kv": args.connector == "synthetic",
                    "real_weights": args.load_format == "auto",
                    "graph_mode_requested": args.cuda_graph, "dbo_requested": args.enable_dbo,
                    "worker_resources": resources,
                    "performance_valid": not (args.record_logit_topk or args.record_full_logits or args.profile_steps or args.disable_compile
                                              or os.environ.get("VLLM_TKNP_SERVE_DEBUG") == "1"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        manifest["engine_batch_queue_size"] = core.batch_queue_size
        (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    original_finished = core.scheduler.has_finished_requests
    # The lightweight synchronous engine normally stops stepping after the last
    # output. A P server must continue polling NIXL acknowledgements, otherwise
    # its final transferred request remains pinned until another arrival.
    core.scheduler.has_finished_requests = lambda: original_finished() or any(
        req.is_finished() for req in core.scheduler.requests.values())
    original_schedule = core.scheduler.schedule
    first_scheduled = set()
    remote_ready = set()
    scheduled_state = {"tokens": 0, "preemptions_total": 0, "phase": "idle", "request_ids": []}
    original_preempt = core.scheduler._preempt_request
    def observed_preempt(request, timestamp):
        scheduled_state["preemptions_total"] += 1
        journal.write(json.dumps({"event": "preempt", "request_id": request.request_id,
                                  "time": time.time()}) + "\n")
        return original_preempt(request, timestamp)
    core.scheduler._preempt_request = observed_preempt
    def observed_schedule():
        previous = {req_id: (req.num_computed_tokens, req.num_prompt_tokens)
                    for req_id, req in core.scheduler.requests.items()}
        output = original_schedule()
        scheduled_state["tokens"] = output.total_num_scheduled_tokens
        scheduled_state["request_ids"] = list(output.num_scheduled_tokens)
        phases = set()
        for req_id, count in output.num_scheduled_tokens.items():
            computed, prompt = previous[req_id]
            remote = core.scheduler.requests[req_id].num_external_computed_tokens > 0
            phases.add("prefill" if computed < prompt and not remote else "decode")
        scheduled_state["phase"] = next(iter(phases)) if len(phases) == 1 else "mixed" if phases else "transfer_only"
        for req_id in output.num_scheduled_tokens:
            if req_id not in first_scheduled:
                journal.write(json.dumps({"event": "first_compute", "request_id": req_id,
                                          "time": time.time()}) + "\n")
                first_scheduled.add(req_id)
        return output
    core.scheduler.schedule = observed_schedule
    if args.record_logit_topk or args.record_full_logits:
        def install_diagnostic(worker):
            import torch
            runner = worker.model_runner
            original = runner._sample
            diagnostic_rank = worker.rank
            log = (args.result_dir / f"logits-rank{diagnostic_rank}.jsonl").open("x")
            def sample(logits, spec_decode_metadata):
                if logits is not None:
                    vals, ids = logits.float().topk(5, dim=-1)
                    for row, (req, values, token_ids) in enumerate(zip(runner.input_batch.req_ids, vals.cpu().tolist(), ids.cpu().tolist())):
                        step = len(runner.requests[req].output_token_ids)
                        log.write(json.dumps({"request_id": req, "step": len(runner.requests[req].output_token_ids),
                                              "values": values, "ids": token_ids}) + "\n")
                        if args.record_full_logits:
                            directory = args.result_dir / f"full-logits-rank{diagnostic_rank}"
                            directory.mkdir(exist_ok=True)
                            key = hashlib.sha256(req.encode()).hexdigest()
                            torch.save({"request_id": req, "step": step,
                                        "logits": logits[row].detach().cpu()},
                                       directory / f"{key}-{step}.pt")
                    log.flush()
                if args.record_full_logits:
                    # Check actual KV at the beginning, prompt boundary, and
                    # latest computed token, including on attention-only owners.
                    for req in runner.input_batch.req_ids:
                        state = runner.requests[req]
                        blocks = state.block_ids[0]
                        if not blocks:
                            continue
                        step = len(state.output_token_ids)
                        layers = [0, len(runner.kv_caches) - 1]
                        page = runner.kv_caches[0].shape[2]
                        if page != worker.cache_config.block_size:
                            raise RuntimeError("KV diagnostic does not support hybrid allocator/kernel blocks")
                        positions = sorted({0, state.num_prompt_tokens - 1,
                                            state.num_prompt_tokens + step - 1})
                        positions = [p for p in positions if p // page < len(blocks)]
                        values = torch.stack([torch.stack([
                            runner.kv_caches[layer][blocks[p // page], :, p % page]
                            for p in positions]) for layer in layers])
                        directory = args.result_dir / f"kv-samples-rank{diagnostic_rank}"
                        directory.mkdir(exist_ok=True)
                        key = hashlib.sha256(req.encode()).hexdigest()
                        torch.save({"request_id": req, "step": step, "layers": layers,
                                    "positions": positions, "kv": values.detach().cpu()},
                                   directory / f"{key}-{step}.pt")
                sampled = original(logits, spec_decode_metadata)
                # Diagnostic only: every replicated rank receives the same
                # forced sequence, keeping subsequent KV/logits on one prefix.
                for row, req in enumerate(runner.input_batch.req_ids):
                    state = runner.requests[req]
                    forced = (state.sampling_params.extra_args or {}).get("teacher_tokens")
                    step = len(state.output_token_ids)
                    if forced is not None and step < len(forced):
                        sampled.sampled_token_ids[row, 0] = forced[step]
                return sampled
            runner._sample = sample
        llm.collective_rpc(install_diagnostic)
    commands = queue.Queue()
    clients = {}
    seen_ids = set()
    status = {"ready": True, "error": None, "config": vars(args), "outstanding_requests": 0}
    paused = {}
    pause_targets = {}
    emitted_tokens = {}
    status["paused_requests"] = []
    status["reservation_block_tokens"] = core.scheduler.block_size
    eligible = [r for r in resources
                if not args.graph_slots or r["graph_splits"][args.graph_batch_size][r["rank"] // args.tp] > 0]
    capacity_tokens = [(r["kv_blocks"] - 1) * r["block_tokens"] for r in eligible]
    status["admission_max_tokens"] = min(args.max_model_len,
                                        max(capacity_tokens) if args.tknp > 1 else min(capacity_tokens))
    status["resident_limits"] = []
    for owner in range(args.tknp):
        members = [r for r in resources if args.tknp == 1 or r["rank"] // args.tp == owner]
        status["resident_limits"].append({
            "capacity_tokens": min((r["kv_blocks"] - 1) * r["block_tokens"] for r in members),
            "request_slots": (members[0]["graph_splits"][args.graph_batch_size][owner]
                              if args.graph_slots else args.max_num_seqs)})
    stop = threading.Event()
    http_server = None

    if rank == 0:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        app = FastAPI()

        @app.get("/health")
        async def health():
            return json.loads(json.dumps(status, default=str))

        @app.post("/shutdown")
        async def shutdown():
            # Ask rank zero to broadcast a coordinated stop. Signalling srun
            # directly can destroy torchrun workers before journals are saved.
            if status["outstanding_requests"]:
                raise HTTPException(409, "Drain requests before coordinated shutdown")
            stop.set()
            return {"stopping": True}

        @app.post("/release")
        async def release(request: Request):
            body = await request.json()
            ids = body.get("request_ids")
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or any(req not in status["paused_requests"] for req in ids)):
                raise HTTPException(400, "Release only currently paused, resident requests")
            commands.put(("release", ids))
            return {"accepted": ids}

        @app.post("/generate")
        async def generate(request: Request):
            body = await request.json()
            req_id = body.get("request_id")
            tokens = body.get("prompt_token_ids", [])
            outputs = body.get("max_tokens", 0)
            pause_after = body.get("pause_after_tokens", 0)
            if type(pause_after) is not int or pause_after < 0 or pause_after >= outputs:
                raise HTTPException(400, "pause_after_tokens must precede request completion")
            teacher = body.get("teacher_tokens")
            if teacher is not None and (not args.record_full_logits or not isinstance(teacher, list)
                                        or len(teacher) < outputs
                                        or any(type(t) is not int or not 0 <= t < engine.model_config.get_vocab_size() for t in teacher)):
                raise HTTPException(400, "Teacher forcing requires full-logit diagnostics and valid tokens")
            if args.pp > 1 and args.connector == "nixl" and (body.get("kv_transfer_params") or {}).get("do_remote_decode"):
                raise HTTPException(400, "PP NIXL adapter is decode-only; prefill must use PP1")
            if ((args.tknp > 1 or args.pp > 1) and args.connector == "nixl"
                    and not (body.get("kv_transfer_params") or {}).get("do_remote_prefill")):
                raise HTTPException(400, "This decode endpoint requires real remote-prefill KV parameters")
            if (not isinstance(req_id, str) or not req_id or req_id in seen_ids
                    or not isinstance(tokens, list) or not tokens
                    or any(not isinstance(x, int) or x < 0 or x >= engine.model_config.get_vocab_size() for x in tokens)
                    or not isinstance(outputs, int) or outputs <= 0
                    or (args.fixed_wave and outputs != args.fixed_output_tokens)
                    or len(tokens) + outputs > status["admission_max_tokens"]):
                raise HTTPException(400, "Unique request_id, valid tokens and in-range lengths required")
            seen_ids.add(req_id)
            messages = asyncio.Queue()
            clients[req_id] = (asyncio.get_running_loop(), messages)
            commands.put(("add", body))

            async def stream():
                try:
                    while True:
                        message = await messages.get()
                        yield json.dumps(message) + "\n"
                        if message.get("finished") or message.get("error"):
                            break
                finally:
                    commands.put(("abort", req_id))
                    clients.pop(req_id, None)
            return StreamingResponse(stream(), media_type="application/x-ndjson")

        http_server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))
        threading.Thread(target=http_server.run, daemon=True).start()
    # Every external rank must survive supervisor SIGTERM until rank zero
    # broadcasts the stop command; otherwise teardown breaks a CPU collective.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    step_index = 0
    pending_wave = []
    profiling = False
    profile_done = False
    compute_steps = 0
    deferred_aborts = set()
    last_idle_state = None
    last_step_log_time = 0.0
    try:
        while True:
            batch = []
            if rank == 0:
                if stop.is_set():
                    batch = [("stop", None)]
                else:
                    # Small idle wait is HTTP wakeup latency, included in E2E.
                    try:
                        batch.append(commands.get(timeout=0 if core.scheduler.has_requests() or core.batch_queue else 0.002))
                    except queue.Empty:
                        pass
                    while not commands.empty():
                        batch.append(commands.get_nowait())
                    if args.fixed_wave:
                        immediate = []
                        for cmd, body in batch:
                            if cmd == "add":
                                pending_wave.append((cmd, body))
                            else:
                                if cmd == "abort":
                                    pending_wave = [entry for entry in pending_wave if entry[1]["request_id"] != body]
                                immediate.append((cmd, body))
                        batch = immediate
                        if not core.scheduler.has_requests() and len(pending_wave) >= args.graph_batch_size:
                            batch += pending_wave[:args.graph_batch_size]
                            del pending_wave[:args.graph_batch_size]
            if external:
                payload = [batch]
                dist.broadcast_object_list(payload, src=0, group=world_group)
                batch = payload[0]
            if any(cmd == "stop" for cmd, _ in batch):
                break
            for cmd, body in batch:
                if cmd == "add":
                    params = SamplingParams(temperature=0, max_tokens=body["max_tokens"],
                                            ignore_eos=True, output_kind=RequestOutputKind.DELTA,
                                            extra_args={"kv_transfer_params": body.get("kv_transfer_params"),
                                                        "teacher_tokens": body.get("teacher_tokens")})
                    engine.add_request(body["request_id"], {"prompt_token_ids": body["prompt_token_ids"]}, params)
                    if body.get("pause_after_tokens"):
                        pause_targets[body["request_id"]] = body["pause_after_tokens"]
                    journal.write(json.dumps({"event": "admit", "request_id": body["request_id"],
                                              "time": time.time(), "prompt_tokens": len(body["prompt_token_ids"]),
                                              "output_tokens": body["max_tokens"]}) + "\n")
                    journal.flush()
                elif cmd == "abort":
                    pause_targets.pop(body, None)
                    if body in paused:
                        core.scheduler.running.append(paused.pop(body))
                    if defer_kv_abort(core.scheduler.requests.get(body), core.scheduler.finished_recving_kv_req_ids):
                        deferred_aborts.add(body)
                    else:
                        engine.abort_request([body])
                elif cmd == "release":
                    for req_id in body:
                        core.scheduler.running.append(paused.pop(req_id))
                        pause_targets.pop(req_id, None)
                    journal.write(json.dumps({"event": "release", "request_ids": body,
                                              "time": time.time()}) + "\n")
            status["paused_requests"] = list(paused)
            for req_id in list(deferred_aborts):
                if not defer_kv_abort(core.scheduler.requests.get(req_id), core.scheduler.finished_recving_kv_req_ids):
                    engine.abort_request([req_id])
                    core.scheduler.finished_recving_kv_req_ids.discard(req_id)
                    deferred_aborts.remove(req_id)
            status["outstanding_requests"] = len(core.scheduler.requests)
            if not core.scheduler.has_requests() and not core.batch_queue:
                continue
            before = time.perf_counter()
            if args.profile_steps and not profile_done and not profiling and compute_steps >= args.profile_after_step:
                llm.start_profile()
                profiling = True
                profile_start = compute_steps
            outputs = engine.step()
            pause_ids = []
            if rank == 0:
                for output in outputs:
                    req_id = output.request_id
                    emitted_tokens[req_id] = emitted_tokens.get(req_id, 0) + len(output.outputs[0].token_ids)
                    if (req_id in pause_targets and not output.finished
                            and emitted_tokens[req_id] >= pause_targets[req_id]):
                        pause_ids.append(req_id)
            if external and pause_targets:
                pause_payload = [pause_ids]
                dist.broadcast_object_list(pause_payload, src=0, group=world_group)
                pause_ids = pause_payload[0]
            for req_id in pause_ids:
                req = core.scheduler.requests[req_id]
                core.scheduler.running.remove(req)
                paused[req_id] = req
                journal.write(json.dumps({"event": "paused_resident", "request_id": req_id,
                                          "time": time.time(),
                                          "num_computed_tokens": req.num_computed_tokens}) + "\n")
            status["paused_requests"] = list(paused)
            if scheduled_state["tokens"]:
                compute_steps += 1
            if profiling and compute_steps >= profile_start + args.profile_steps:
                llm.stop_profile()
                profiling = False
                profile_done = True
            elapsed = time.perf_counter() - before
            scheduler = core.scheduler
            status["outstanding_requests"] = len(scheduler.requests)
            for req_id in scheduler.finished_recving_kv_req_ids - remote_ready:
                journal.write(json.dumps({"event": "kv_ready", "request_id": req_id,
                                          "time": time.time()}) + "\n")
                remote_ready.add(req_id)
            owners = (scheduler.token_parallel_scheduler.req_to_tknp_rank
                      if args.tknp > 1 else {})
            context_work = [0] * args.tknp
            for req_id, owner in owners.items():
                if req_id in scheduler.requests:
                    context_work[owner] += scheduler.requests[req_id].num_tokens
            event = {"event": "step", "step": step_index, "time": time.time(),
                                      "scheduled_tokens": scheduled_state["tokens"],
                                      "seconds": elapsed, "running": len(scheduler.running),
                                      "waiting": len(scheduler.waiting),
                                      "remote_waiting": sum(r.status.name == "WAITING_FOR_REMOTE_KVS" for r in scheduler.waiting),
                                      "kv_free_blocks": scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                                      "owners": owners,
                                      "owner_context_tokens": context_work,
                                      "preemptions": sum(r.num_preemptions for r in scheduler.requests.values()),
                                      "preemptions_total": scheduled_state["preemptions_total"]}
            event["paused_resident"] = len(paused)
            event["phase"] = scheduled_state["phase"]
            event["scheduled_request_ids"] = scheduled_state["request_ids"]
            event["waiting_capacity"] = sum(r.status.name == "WAITING" for r in scheduler.waiting)
            event["batch_queue_depth"] = len(core.batch_queue) if core.batch_queue is not None else 0
            idle_state = tuple(event[key] for key in ("running", "waiting", "remote_waiting", "kv_free_blocks", "preemptions_total"))
            now = time.monotonic()
            if scheduled_state["tokens"] or idle_state != last_idle_state or now - last_step_log_time >= 0.1:
                # Preserve every compute step and all queue/capacity changes,
                # not thousands of identical transfer-only polling records.
                journal.write(json.dumps(event) + "\n")
                last_idle_state, last_step_log_time = idle_state, now
            step_index += 1
            if rank == 0:
                for output in outputs:
                    if output.request_id not in clients:
                        continue
                    message = {"request_id": output.request_id, "finished": output.finished,
                               "token_ids": list(output.outputs[0].token_ids),
                               "kv_transfer_params": output.kv_transfer_params}
                    loop, messages = clients[output.request_id]
                    loop.call_soon_threadsafe(messages.put_nowait, message)
            if step_index % 32 == 0 or os.environ.get("VLLM_TKNP_SERVE_DEBUG") == "1":
                journal.flush()
            if not scheduled_state["tokens"] and args.idle_kv_poll_ms:
                time.sleep(args.idle_kv_poll_ms / 1000)
    finally:
        if profiling:
            llm.stop_profile()
        status["ready"] = False
        if http_server:
            http_server.should_exit = True
        def execution_counts(worker):
            import torch
            return {"rank": worker.rank, "counts": dict(worker.model_runner._research_execution_counts),
                    "memory": {"peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
                               "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(worker.device),
                               "scope": "process lifetime, including initialization/capture; excludes non-PyTorch allocations"}}
        execution = llm.collective_rpc(execution_counts)
        if external:
            all_execution = [None] * dist.get_world_size()
            dist.all_gather_object(all_execution, execution, group=world_group)
            execution = [record for records in all_execution for record in records]
        if rank == 0:
            (args.result_dir / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        journal.write(json.dumps({"event": "shutdown", "remaining_requests": len(core.scheduler.requests),
                                  "kv_free_blocks": core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                                  "all_kv_free": core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == initial_free_blocks}) + "\n")
        journal.close()
        if args.nnodes > 1:
            import socket
            import urllib.request
            for host in {r["hostname"] for r in resources} - {socket.gethostname()}:
                try:
                    with urllib.request.urlopen(urllib.request.Request(
                            f"http://{host}:{args.port + 1000}/shutdown", data=b"{}", method="POST"), timeout=2):
                        pass
                except OSError:
                    pass
        core.shutdown()  # Also close the scheduler's connector/listener resources.
        if external:
            from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
