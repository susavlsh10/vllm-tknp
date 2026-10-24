#!/usr/bin/env python3
"""Bounded single-node 405B FBGEMM-FP8/BF16-KV systems validation.

Uses dummy weights, never downloads checkpoint tensors. The initialization hook
is local to this executable: production loaders and quantized kernels are not
modified. Public metadata provenance is retained alongside the derived config.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


MODEL_ID = "meta-llama/Llama-3.1-405B-FP8"


def checkpoint_counts(config):
    h, i, v, n = (config[k] for k in (
        "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"))
    kv = config["num_key_value_heads"] * config["head_dim"]
    return {"BF16": 2 * v * h + n * (2 * h * h + 2 * h * kv)
            + 2 * 3 * h * i + (2 * n + 1) * h,
            "F8_E4M3": (n - 2) * 3 * h * i,
            "F32": (n - 2) * (2 * i + h)}


def prepare(directory, metadata_file=None):
    directory.mkdir(parents=True, exist_ok=False)
    if metadata_file is not None:
        metadata = json.loads(metadata_file.read_text())
    else:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{MODEL_ID}", timeout=60) as response:
            metadata = json.load(response)
    quantization = metadata["config"]["quantization_config"]
    assert quantization["quant_method"] == "fbgemm_fp8"
    # Architecture from the standard 405B MP8 config. Verify against the official
    # tensor census below, including the distinction between 8 and 16 KV heads.
    config = dict(architectures=["LlamaForCausalLM"], model_type="llama",
                  hidden_size=16384, intermediate_size=53248,
                  num_hidden_layers=126, num_attention_heads=128,
                  num_key_value_heads=8, head_dim=128, vocab_size=128256,
                  max_position_embeddings=131072, hidden_act="silu",
                  rms_norm_eps=1e-5, rope_theta=500000.0,
                  rope_scaling=dict(factor=8.0, high_freq_factor=4.0,
                                    low_freq_factor=1.0,
                                    original_max_position_embeddings=8192,
                                    rope_type="llama3"),
                  attention_bias=False, attention_dropout=0.0, mlp_bias=False,
                  tie_word_embeddings=False, bos_token_id=128000,
                  eos_token_id=128001, initializer_range=0.02,
                  pretraining_tp=1, torch_dtype="bfloat16", use_cache=True)
    from transformers import FbgemmFp8Config
    defaults = FbgemmFp8Config().to_dict()
    # The public API omits scalar quantizer options. Use the standard Transformers
    # FBGEMM bound explicitly; do not pretend this is a downloaded config.json.
    config["quantization_config"] = dict(
        quantization, activation_scale_ub=defaults["activation_scale_ub"])
    expected = metadata["safetensors"]["parameters"]
    assert checkpoint_counts(config) == expected, (checkpoint_counts(config), expected)
    ignored = set(quantization["modules_to_not_convert"])
    required = {"lm_head"}
    for layer in range(126):
        required.update(f"model.layers.{layer}.self_attn.{proj}_proj"
                        for proj in ("q", "k", "v", "o"))
    for layer in (0, 125):
        required.update(f"model.layers.{layer}.mlp.{proj}_proj"
                        for proj in ("gate", "up", "down"))
    assert ignored == required
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = dict(model_id=MODEL_ID, revision=metadata["sha"],
                      metadata_url=f"https://huggingface.co/api/models/{MODEL_ID}",
                      architecture_source="standard 405B MP8, checked against official tensor census",
                      quantization_source="official public metadata modules_to_not_convert",
                      activation_scale_ub_source="Transformers FbgemmFp8Config default",
                      activation_scale_ub=config["quantization_config"]["activation_scale_ub"],
                      checkpoint_tensor_counts=expected,
                      raw_weight_bytes_per_tp8_gpu=sum(
                          expected[k] * size for k, size in
                          (("BF16", 2), ("F8_E4M3", 1), ("F32", 4))) // 8,
                      dummy_weights=True, checkpoint_downloaded=False)
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (directory / "public_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("FP8_RECIPE_PREPARED " + json.dumps(provenance), flush=True)


def initialize_fp8_dummy_weights(model, low=-1e-3, high=1e-3, seed=1234):
    """Initialize synthetic quantized tensors with valid scales and bounded scratch."""
    import torch
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8LinearMethod
    quantized = {id(module.weight): module for module in model.modules()
                 if isinstance(getattr(module, "quant_method", None), FBGEMMFp8LinearMethod)}
    scalar_overrides = {}
    for module in quantized.values():
        scalar_overrides[id(module.weight_scale)] = 1e-3 / 224.0
        scalar_overrides[id(module.input_scale_ub)] = module.quant_method.quant_config.input_scale_ub
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if not torch.is_floating_point(tensor):
                continue
            if id(tensor) in scalar_overrides:
                tensor.fill_(scalar_overrides[id(tensor)])
                continue
            generator = torch.Generator(device=tensor.device).manual_seed(seed)
            if id(tensor) in quantized:
                flat = tensor.view(-1)
                chunk = 8 * 1024 * 1024
                for begin in range(0, flat.numel(), chunk):
                    count = min(chunk, flat.numel() - begin)
                    scratch = torch.empty(count, dtype=torch.float16, device=tensor.device)
                    scratch.uniform_(-224, 224, generator=generator)
                    flat[begin:begin + count].copy_(scratch.to(tensor.dtype))
                del scratch
            else:
                tensor.uniform_(low, high, generator=generator)
    print("FP8_DUMMY_INITIALIZED " + json.dumps(dict(
        rank=int(os.environ.get("RANK", 0)), quantized_linear_modules=len(quantized),
        positive_scales=True, temporary_fp16_chunk_elements=8 * 1024 * 1024)), flush=True)


def install_checks(worker):
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    runner = worker.model_runner
    worker._fp8_smoke_counts = Counter()
    worker._fp8_smoke_check_finite = True
    original_dispatch = runner._determine_batch_execution_and_padding
    def dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        worker._fp8_smoke_counts["dispatch:" + result[0].name] += 1
        if result[0] != CUDAGraphMode.FULL:
            raise RuntimeError("Smoke test refuses non-FULL graph dispatch")
        return result
    runner._determine_batch_execution_and_padding = dispatch
    original_graph_call = CUDAGraphWrapper.__call__
    def graph_call(wrapper, *args, **kwargs):
        context = get_forward_context()
        entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
        if (context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and wrapper.runtime_mode == CUDAGraphMode.FULL
                and entry is not None and entry.cudagraph is not None):
            worker._fp8_smoke_counts["actual_full_graph_replays"] += 1
        return original_graph_call(wrapper, *args, **kwargs)
    CUDAGraphWrapper.__call__ = graph_call
    original_logits = runner.model.compute_logits
    def logits(*args, **kwargs):
        result = original_logits(*args, **kwargs)
        if worker._fp8_smoke_check_finite and result is not None:
            assert bool(torch.isfinite(result).all()), "Nonfinite logits"
            worker._fp8_smoke_counts["finite_logits_checks"] += 1
        return result
    runner.model.compute_logits = logits


def resources(worker):
    import torch
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8LinearMethod
    runner = worker.model_runner
    storage = Counter()
    modules = []
    for parameter in runner.model.parameters():
        storage[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
    for name, module in runner.model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, FBGEMMFp8LinearMethod):
            assert module.weight.dtype == torch.float8_e4m3fn
            assert bool(torch.isfinite(module.weight_scale).all())
            assert bool((module.weight_scale > 0).all())
            assert float(module.input_scale_ub) == method.quant_config.input_scale_ub
            assert not method.quant_config.use_marlin
            modules.append(dict(name=name, backend=method.fp8_linear.preferred_backend))
    kv = runner.kv_cache_config
    result = dict(rank=worker.rank, device=str(worker.device),
                  gpu=torch.cuda.get_device_name(worker.device),
                  gpu_total_bytes=torch.cuda.get_device_properties(worker.device).total_memory,
                  model_memory_bytes=runner.model_memory_usage,
                  model_storage_bytes=dict(storage), quantized_modules=modules,
                  quantization=worker.model_config.quantization,
                  model_dtype=str(worker.model_config.dtype),
                  kv_storage_dtypes=sorted({str(t.dtype) for t in runner.kv_caches}),
                  kv_allocated_bytes=sum(t.size for t in kv.kv_cache_tensors),
                  kv_blocks=kv.num_blocks, block_tokens=worker.cache_config.block_size,
                  max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(worker.device),
                  max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(worker.device),
                  free_cuda_bytes=torch.cuda.mem_get_info(worker.device)[0],
                  execution=dict(getattr(worker, "_fp8_smoke_counts", {})))
    assert result["kv_storage_dtypes"] == ["torch.bfloat16"]
    assert result["quantization"] == "fbgemm_fp8"
    # Each of the 124 middle layers has a fused gate/up and a down projection.
    assert len(modules) == 248, len(modules)
    return result


def finite_check_mode(worker, enabled):
    worker._fp8_smoke_check_finite = enabled


def run(args):
    import torch
    import torch.distributed as dist
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.model_executor.model_loader import dummy_loader
    from e2e_server import warm_sampling_collectives
    dummy_loader.initialize_dummy_weights = initialize_fp8_dummy_weights
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    llm = LLM(model=str(args.model), tokenizer=args.tokenizer, dtype="bfloat16",
              quantization="fbgemm_fp8", load_format="dummy",
              tensor_parallel_size=8, distributed_executor_backend="external_launcher",
              seed=1, enforce_eager=False, enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max(args.contexts) + args.tokens + 32,
              max_num_seqs=max(args.batches), max_num_batched_tokens=512,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
              cudagraph_capture_sizes=args.batches, kv_cache_dtype="auto",
              kv_transfer_config=KVTransferConfig(
                  kv_connector="DecodeBenchConnector", kv_role="kv_both",
                  kv_connector_extra_config={"fill_mean": 0.015, "fill_std": 0.0}))
    init_seconds = time.perf_counter() - start
    llm.collective_rpc(warm_sampling_collectives)
    llm.collective_rpc(install_checks)
    rank = dist.get_rank()
    resource = llm.collective_rpc(resources)[0]
    (args.output / f"resources_rank{rank}.json").write_text(json.dumps(resource, indent=2) + "\n")
    core = llm.llm_engine.engine_core.engine_core
    preemptions = [0]
    original_preempt = core.scheduler._preempt_request
    def preempt(*positional, **keywords):
        preemptions[0] += 1
        return original_preempt(*positional, **keywords)
    core.scheduler._preempt_request = preempt
    results = []
    for context in args.contexts:
        for batch in args.batches:
            required_blocks = batch * ((context + args.tokens + resource["block_tokens"] - 1)
                                       // resource["block_tokens"])
            available_blocks = core.scheduler.kv_cache_config.num_blocks
            if required_blocks > available_blocks:
                results.append(dict(context=context, batch=batch, status="capacity_skip",
                                    required_blocks=required_blocks, available_blocks=available_blocks))
                continue
            prompts = [{"prompt_token_ids": [128000] + [1000 + index] * (context - 1)}
                       for index in range(batch)]
            llm.collective_rpc(finite_check_mode, args=(True,))
            warm = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=10,
                                                       ignore_eos=True), use_tqdm=False)
            assert all(len(out.outputs[0].token_ids) == 10 for out in warm)
            llm.collective_rpc(finite_check_mode, args=(False,))
            before_preemptions = preemptions[0]
            records = []
            original_step = llm.llm_engine.step
            def step(*positional, **keywords):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = original_step(*positional, **keywords)
                torch.cuda.synchronize()
                records.append((time.perf_counter() - start) * 1000)
                return outputs
            llm.llm_engine.step = step
            dist.barrier()
            try:
                outputs = llm.generate(prompts, SamplingParams(
                    temperature=0, max_tokens=args.tokens, ignore_eos=True), use_tqdm=False)
            finally:
                llm.llm_engine.step = original_step
            assert all(len(out.outputs[0].token_ids) == args.tokens for out in outputs)
            assert preemptions[0] == before_preemptions
            steady = records[4:-4]
            assert steady, records
            local = dict(context=context, batch=batch, status="pass", rank=rank,
                         steps=len(steady), mean_step_ms=statistics.mean(steady),
                         median_step_ms=statistics.median(steady),
                         p95_step_ms=sorted(steady)[min(len(steady)-1, int(len(steady)*0.95))],
                         measured_preemptions=preemptions[0]-before_preemptions,
                         output_tokens=sum(len(out.outputs[0].token_ids) for out in outputs),
                         timing_scope="synchronized full-engine decode steps, first/last four excluded",
                         token_sha256=hashlib.sha256(json.dumps(
                             [list(out.outputs[0].token_ids) for out in outputs]).encode()).hexdigest())
            peers = [None] * 8
            dist.all_gather_object(peers, local)
            assert len({p["token_sha256"] for p in peers}) == 1
            slowest = max(p["mean_step_ms"] for p in peers)
            point = dict(local, all_rank_metrics=peers, slowest_rank_step_ms=slowest,
                         steady_decode_tokens_per_second=batch * 1000 / slowest)
            results.append(point)
            if rank == 0:
                print("FP8_SMOKE_POINT " + json.dumps(point), flush=True)
            (args.output / f"points_rank{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    final = llm.collective_rpc(resources)[0]
    assert final["execution"].get("actual_full_graph_replays", 0) > 0
    assert final["execution"].get("finite_logits_checks", 0) > 0
    (args.output / f"resources_final_rank{rank}.json").write_text(json.dumps(final, indent=2) + "\n")
    if rank == 0:
        summary = dict(status="pass", initialization_seconds=init_seconds,
                       model_id=MODEL_ID, precision="official FBGEMM mixed FP8/BF16 weights; BF16 KV",
                       dummy_weights=True, dummy_kv=True, accuracy_validated=False,
                       paper_performance_valid=False, points=results,
                       note="Functional smoke, synchronized diagnostic timings, not a final benchmark")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("FP8_TP8_SINGLE_NODE_PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--model-dir", type=Path, required=True)
    prep.add_argument("--metadata-json", type=Path)
    fetch = modes.add_parser("fetch-metadata")
    fetch.add_argument("--output", type=Path, required=True)
    probe = modes.add_parser("run")
    probe.add_argument("--model", type=Path, required=True)
    probe.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--contexts", type=int, nargs="+", default=[512, 32768, 65536, 98304])
    probe.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    probe.add_argument("--tokens", type=int, default=33)
    probe.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.model_dir, args.metadata_json)
    elif args.mode == "fetch-metadata":
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{MODEL_ID}", timeout=60) as response:
            metadata = json.load(response)
        with args.output.open("x") as handle:
            json.dump(metadata, handle, indent=2)
        print("OFFICIAL_METADATA_CACHED", metadata["sha"], flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
