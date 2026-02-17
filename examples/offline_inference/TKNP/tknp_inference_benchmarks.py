"""
Inference benchmarking script for vLLM with Token Parallelism (TKNP) support.

Example usage:
Token parallelism: 
torchrun --nproc-per-node=8 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 4 --token-parallel-size 2 --batch-size 32 --seq-length 32768

Tensor parallelism:
torchrun --nproc-per-node=8 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 8 --token-parallel-size 1 --batch-size 32 --seq-length 16384

Pipeline parallelism:
torchrun --nproc-per-node=8 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 4 --pipeline-parallel-size 2 --batch-size 32 --seq-length 32768


Supported models: 

Llama-3:    meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, 
            meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.3-70B-Instruct
Qwen:       Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-4B-Instruct-2507
            Qwen/Qwen3-32B, Qwen/Qwen2.5-72B-Instruct

Ministral:  ministral/Ministral-3b-instruct

"""


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import argparse
import torch.distributed as dist

from vllm import LLM, SamplingParams
from prompt_generator import generate_benchmark_prompts

import math
import torch
import random
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

from vllm.config import AttentionConfig

def parse_args():
    """Parse command line arguments for distributed vLLM inference."""
    parser = argparse.ArgumentParser(description="Distributed vLLM inference with torchrun")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of tensor parallel processes (default: 1)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of pipeline parallel processes (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Number of data parallel processes (default: 1)")
    parser.add_argument("--token-parallel-size", type=int, default=1,
                        help="Number of token parallel processes (default: 1)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name (default: meta-llama/Llama-3.1-8B-Instruct)")
    # parser.add_argument("--max-model-len", type=int, default=131072,
    #                     help="Maximum model length (default: 131072)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    # batch size and seq length for prompts
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for prompts (default: 8)")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Sequence length for prompts (default: 128)")
    parser.add_argument("--print-outputs", action="store_true",
                        help="Print generated outputs")
    parser.add_argument("--decode-tokens", type=int, default=1000,
                        help="Number of tokens to decode during benchmarking (default: 1000)")
    parser.add_argument("--output-dir", type=str, 
                        default="examples/offline_inference/TKNP/tknp_data",
                        help="Directory to save benchmark results (default: examples/offline_inference/TKNP/tknp_data)")
    parser.add_argument("--collect-data", action="store_true",
                        help="Enable systematic data collection across multiple batch sizes and sequence lengths")
    parser.add_argument("--skip-prefill", action="store_true", default=True,
                        help="Skip prefill by pre-filling KV cache with dummy values (for decode-only benchmarking)")
    parser.add_argument("--load-format", type=str, default="dummy",
                        help="Weight loading format. Use 'dummy' for random weights (default, faster for benchmarking) or 'auto' for real weights")

    return parser.parse_args()

def get_gpu_name():
    """Get the GPU name for the current device."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        # print(f"Detected GPU: {gpu_name}")
        # Extract concise GPU name (e.g., "NVIDIA H100 80GB HBM3" -> "H100", "NVIDIA RTX A5000" -> "A5000")
        # Remove common prefixes and clean up
        gpu_name = gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace("RTX", "").strip()
        # Take first significant part (the model number like H100, A100, B200, A5000, etc.)
        parts = gpu_name.split()
        gpu_name = parts[0] if parts else "unknown"
        return gpu_name
    except:
        return "unknown"

def get_model_name(model_path):
    """Extract concise model name from full model path."""
    # Extract model name from path like "meta-llama/Llama-3.2-1B-Instruct" -> "Llama-3.2-1B"
    model_name = model_path.split("/")[-1]  # Get last part after /
    # Remove common suffixes
    for suffix in ["-Instruct", "-Chat", "-Base", "-v1", "-v2"]:
        model_name = model_name.replace(suffix, "")
    return model_name

def save_benchmark_results(args, metrics, output_dir):
    """
    Save benchmark results to CSV file.
    Only called by rank 0.
    
    Args:
        args: Command line arguments
        metrics: Dictionary containing benchmark metrics
        output_dir: Directory to save the CSV file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get concise names
    gpu_name = get_gpu_name()
    model_name = get_model_name(args.model)
    
    # Construct filename
    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size
    tknp_size = args.token_parallel_size
    
    filename = f"{model_name}_{gpu_name}_TP_{tp_size}_PP_{pp_size}_TKNP_{tknp_size}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    # Prepare data row with concise column names (rounded to 4 decimal places)
    data_row = {
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'decode_time_ms': round(metrics['total_decode_time_ms'], 4),
        'decode_tokens': metrics['total_decode_tokens'],
        'sys_decode_tps': round(metrics['system_decode_throughput'], 4),
        'decode_tps_per_gpu': round(metrics['decode_throughput_per_gpu'], 4),
        'avg_decode_latency_ms': round(metrics['average_decode_latency'], 4),
        'decode_tps_per_user': round(metrics['per_user_decode_throughput'], 4),
    }
    
    # Write to CSV (append mode)
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = list(data_row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data_row)
    
    print(f"Results saved to: {filepath}")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def check_kv_cache_capacity(llm, args):
    """
    Check if the current benchmark configuration fits in KV cache.
    
    For token parallelism, each TKNP rank may have different KV cache capacity
    (e.g., due to different GPU memory availability). We sync across the TKNP
    group to compute the true total capacity rather than assuming uniform size.
    
    Returns:
        tuple: (fits: bool, total_kv_cache_tokens: int, required_tokens: int, max_concurrency: float)
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    
    # Access the kv_cache_config through engine_core -> scheduler
    engine_core = llm.llm_engine.engine_core
    if hasattr(engine_core, 'engine_core'):
        # Multiprocess mode
        engine_core = engine_core.engine_core
    
    scheduler = engine_core.scheduler
    kv_cache_config = scheduler.kv_cache_config
    vllm_config = llm.llm_engine.vllm_config
    
    # Calculate KV cache capacity in tokens for this GPU
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )
    gpu_kv_cache_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    
    # Account for context parallelism if enabled
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        gpu_kv_cache_tokens *= pcp_size * dcp_size
    
    # Account for token parallelism - sync across TKNP ranks to get true total
    tknp_size = vllm_config.parallel_config.token_parallel_size
    enable_tknp = vllm_config.parallel_config.enable_token_parallel and tknp_size > 1
    
    if enable_tknp:
        from vllm.distributed.parallel_state import (
            get_tknp_group,
            is_tknp_initialized,
        )
        assert is_tknp_initialized(), (
            "Token parallel is enabled but TKNP group is not initialized"
        )
        tknp_group = get_tknp_group()
        
        # All-gather local KV cache tokens across the TKNP group
        # to get each rank's actual capacity
        local_tokens_tensor = torch.tensor(
            [gpu_kv_cache_tokens], dtype=torch.long, device="cpu"
        )
        gathered_tokens = [
            torch.zeros(1, dtype=torch.long, device="cpu")
            for _ in range(tknp_group.world_size)
        ]
        torch.distributed.all_gather(
            gathered_tokens, local_tokens_tensor, group=tknp_group.cpu_group
        )
        per_rank_kv_tokens = [t.item() for t in gathered_tokens]
        total_kv_cache_tokens = sum(per_rank_kv_tokens)
    else:
        per_rank_kv_tokens = [gpu_kv_cache_tokens]
        total_kv_cache_tokens = gpu_kv_cache_tokens
    
    # Calculate maximum concurrency
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    
    # Calculate required tokens for current benchmark configuration
    # Each request needs: prompt tokens (seq_length) + decode tokens
    tokens_per_request = args.seq_length + args.decode_tokens
    required_tokens = args.batch_size * tokens_per_request
    
    # Check if configuration fits
    fits = required_tokens <= total_kv_cache_tokens
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"KV Cache Capacity Check")
        print(f"{'='*60}")
        print(f"KV cache this GPU: {gpu_kv_cache_tokens:,} tokens")
        if enable_tknp:
            print(f"Token parallel size: {tknp_size}")
            print(f"Per-TKNP-rank KV cache tokens: {per_rank_kv_tokens}")
            all_equal = len(set(per_rank_kv_tokens)) == 1
            if not all_equal:
                print(f"⚠  KV cache is NOT uniform across TKNP ranks!")
            print(f"Total KV cache (summed across TKNP ranks): {total_kv_cache_tokens:,} tokens")
        print(f"Maximum concurrency for {tokens_per_request:,} tokens per request: {max_concurrency:.2f}x")
        print(f"Total required tokens: {required_tokens:,}")
        print(f"Capacity utilization: {required_tokens / total_kv_cache_tokens * 100:.1f}%")
        
        if fits:
            print(f"✓ Configuration FITS in KV cache - can run concurrently")
        else:
            print(f"✗ Configuration EXCEEDS KV cache capacity by {required_tokens - total_kv_cache_tokens:,} tokens")
            print(f"✗ Requests will be QUEUED - benchmark results will not reflect pure concurrent throughput")
        print(f"{'='*60}\n")
    
    return fits, total_kv_cache_tokens, required_tokens, max_concurrency

def inference_benchmark(llm, prompts, args):
    """
    Benchmark decode performance with skip-prefill enabled.
    
    Since skip-prefill is enabled, KV cache is pre-filled with dummy values,
    so we only measure pure decode latency and throughput.
    
    This uses a two-stage measurement approach:
    1. Warmup run with 10 tokens to measure initialization overhead
    2. Main run with args.decode_tokens + 10 tokens
    3. Actual decode time = main_time - warmup_time (subtracts overhead)
    """
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Benchmark Configuration")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Decode tokens: {args.decode_tokens}")
        print(f"TKNP: {args.token_parallel_size}, TP: {args.tensor_parallel_size}, PP: {args.pipeline_parallel_size}")
        print(f"Skip-prefill: {args.skip_prefill}")
        print(f"{'='*60}\n")

    # ===== Stage 1: Warmup Run (10 tokens) =====
    if dist.get_rank() == 0:
        print("Stage 1: Running warmup with 10 tokens...")
    
    warmup_tokens = 10
    warmup_sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=warmup_tokens)
    
    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    warmup_start.record()
    
    _ = llm.generate(prompts, warmup_sampling_params, use_tqdm=False)
    
    warmup_end.record()
    torch.cuda.synchronize()
    
    warmup_time_ms = warmup_start.elapsed_time(warmup_end)
    
    if dist.get_rank() == 0:
        print(f"Warmup time: {warmup_time_ms:.2f} ms\n")

    # ===== Stage 2: Main Benchmark Run (decode_tokens + 10) =====
    if dist.get_rank() == 0:
        print(f"Stage 2: Running main benchmark with {args.decode_tokens + warmup_tokens} tokens...")
    
    decode_tokens = args.decode_tokens
    main_sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=decode_tokens + warmup_tokens)
    
    main_start = torch.cuda.Event(enable_timing=True)
    main_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    main_start.record()
    
    outputs = llm.generate(prompts, main_sampling_params, use_tqdm=(dist.get_rank() == 0))
    
    main_end.record()
    torch.cuda.synchronize()
    
    main_time_ms = main_start.elapsed_time(main_end)
    
    # ===== Stage 3: Calculate Actual Decode Time =====
    actual_decode_time_ms = main_time_ms - warmup_time_ms
    total_decode_tokens = args.batch_size * decode_tokens
    
    # Calculate metrics using actual decode time
    average_decode_latency = actual_decode_time_ms / decode_tokens
    average_decode_system_throughput = total_decode_tokens / (actual_decode_time_ms / 1000)
    average_decode_throughput_per_user = 1000 / average_decode_latency  # tokens/sec/user
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Decode Performance Results")
        print(f"{'='*60}")
        print(f"Main run time: {main_time_ms:.2f} ms")
        print(f"Warmup time: {warmup_time_ms:.2f} ms")
        print(f"Actual decode time : {actual_decode_time_ms:.2f} ms")
        print(f"Total decode tokens: {total_decode_tokens}")
        print(f"\nDecoding Metrics\n")
        print(f"System Decode Throughput: {average_decode_system_throughput:.2f} tokens/sec")
        print(f"Decode Throughput per GPU: {average_decode_system_throughput / dist.get_world_size():.2f} tokens/sec/GPU")
        print(f"Average Decode Latency: {average_decode_latency:.2f} ms/token/user")
        print(f"Per-user Decode Throughput: {average_decode_throughput_per_user:.2f} tokens/sec/user")
        print(f"{'='*60}\n")

        # Save benchmark results
        metrics = {
            'total_decode_time_ms': actual_decode_time_ms,
            'total_decode_tokens': total_decode_tokens,
            'system_decode_throughput': average_decode_system_throughput,
            'decode_throughput_per_gpu': average_decode_system_throughput / dist.get_world_size(),
            'average_decode_latency': average_decode_latency,
            'per_user_decode_throughput': average_decode_throughput_per_user,
        }
        save_benchmark_results(args, metrics, args.output_dir)

    # Print outputs if requested
    if dist.get_rank() == 0 and args.print_outputs:
        print("-" * 50)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt[:128]!r} ....\nGenerated text: {generated_text!r}\n")
            print("-" * 50)


def setup_vllm_model(args):
    """Setup the LLM with given parallelism configurations."""
    
    # Use `distributed_executor_backend="external_launcher"` so that
    # this llm engine/instance only creates one worker.
    # it is important to set an explicit seed to make sure that
    # all ranks have the same random seed, so that sampling can be
    # deterministic across ranks.
    
    # Prepare LLM kwargs - only include token parallel args if enabled
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "distributed_executor_backend": "external_launcher",
        "seed": args.seed,
        "enforce_eager": True,
        "enable_prefix_caching": False,  # Disable prefix caching for benchmarking
        "gpu_memory_utilization": 0.9,  # Max GPU memory utilization
        "max_num_batched_tokens": 32768,  # max number of tokens in a single forward pass
        "load_format": args.load_format,  # Weight loading format
        # "attention_config": AttentionConfig(backend="FLASHINFER"),# FLASH_ATTN, FLASHINFER
        # "max_model_len": 32768,
    }
    
    # Only add token parallel configs if token parallelism is enabled
    if args.token_parallel_size >= 2:
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    # ADD THIS: Configure DecodeBenchConnector to skip prefill
    if args.skip_prefill:
        from vllm.config import KVTransferConfig
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="DecodeBenchConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "fill_mean": 0.015,  # Mean value for filling KV cache
                "fill_std": 0.0,     # 0 = constant, >0 = random sampling
            }
        )

    llm = LLM(**llm_kwargs)
    if dist.get_rank() == 0:
        print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, "
              f"pipeline_parallel_size={args.pipeline_parallel_size}, "
              f"data_parallel_size={args.data_parallel_size}, "
              f"token_parallel_size={args.token_parallel_size}")
        if args.load_format == "dummy":
            print("⚠️  Using dummy weights (randomly initialized) - suitable for benchmarking only")
        else:
            print(f"Loading real model weights with format: {args.load_format}")
        if args.skip_prefill:
            print("DecodeBenchConnector enabled - KV cache will be pre-filled with dummy values")
    return llm

def run_inference_benchmark(args, llm, batch_size, seq_length):
    # Update args for this configuration
    args.batch_size = batch_size
    args.seq_length = seq_length
    
    prompts = None
    if dist.get_rank() == 0:
        prompts = generate_benchmark_prompts(
            batch_size=batch_size,
            seq_length=seq_length,
            tokenizer=None,
            model_name=args.model,
            vocab_style="natural",
            seed=42
        )
    
    # Broadcast prompts to all ranks
    prompts_list = [prompts]
    dist.broadcast_object_list(prompts_list, src=0)
    prompts = prompts_list[0]
    
    # catch RuntimeError
    try:
        inference_benchmark(llm, prompts, args)
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"Error during inference benchmark: {e}")


def get_kv_cache_info(llm):
    """
    Get KV cache information once and cache it.
    
    For token parallelism, syncs across TKNP ranks to get true total capacity.
    
    Returns:
        dict: Contains total_kv_cache_tokens, per_rank_tokens, enable_tknp, tknp_size, etc.
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    
    # Access the kv_cache_config through engine_core -> scheduler
    engine_core = llm.llm_engine.engine_core
    if hasattr(engine_core, 'engine_core'):
        # Multiprocess mode
        engine_core = engine_core.engine_core
    
    scheduler = engine_core.scheduler
    kv_cache_config = scheduler.kv_cache_config
    vllm_config = llm.llm_engine.vllm_config
    
    # Calculate KV cache capacity in tokens for this GPU
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )
    gpu_kv_cache_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    
    # Account for context parallelism if enabled
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        gpu_kv_cache_tokens *= pcp_size * dcp_size
    
    # Account for token parallelism - sync across TKNP ranks to get true total
    tknp_size = vllm_config.parallel_config.token_parallel_size
    enable_tknp = vllm_config.parallel_config.enable_token_parallel and tknp_size > 1
    
    if enable_tknp:
        from vllm.distributed.parallel_state import (
            get_tknp_group,
            is_tknp_initialized,
        )
        assert is_tknp_initialized(), (
            "Token parallel is enabled but TKNP group is not initialized"
        )
        tknp_group = get_tknp_group()
        
        # All-gather local KV cache tokens across the TKNP group
        local_tokens_tensor = torch.tensor(
            [gpu_kv_cache_tokens], dtype=torch.long, device="cpu"
        )
        gathered_tokens = [
            torch.zeros(1, dtype=torch.long, device="cpu")
            for _ in range(tknp_group.world_size)
        ]
        torch.distributed.all_gather(
            gathered_tokens, local_tokens_tensor, group=tknp_group.cpu_group
        )
        per_rank_kv_tokens = [t.item() for t in gathered_tokens]
        total_kv_cache_tokens = sum(per_rank_kv_tokens)
    else:
        per_rank_kv_tokens = [gpu_kv_cache_tokens]
        total_kv_cache_tokens = gpu_kv_cache_tokens
    
    return {
        'total_kv_cache_tokens': total_kv_cache_tokens,
        'gpu_kv_cache_tokens': gpu_kv_cache_tokens,
        'per_rank_kv_tokens': per_rank_kv_tokens,
        'enable_tknp': enable_tknp,
        'tknp_size': tknp_size,
        'vllm_config': vllm_config,
        'kv_cache_config': kv_cache_config,
    }


def check_workload_fits(kv_cache_info, batch_size, seq_length, decode_tokens, verbose=True):
    """
    Check if workload fits in KV cache using cached KV cache info.
    
    Args:
        kv_cache_info: Cached KV cache information from get_kv_cache_info()
        batch_size: Number of requests
        seq_length: Prompt length
        decode_tokens: Number of decode tokens
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (fits: bool, required_tokens: int, max_concurrency: float)
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    
    total_kv_cache_tokens = kv_cache_info['total_kv_cache_tokens']
    gpu_kv_cache_tokens = kv_cache_info['gpu_kv_cache_tokens']
    per_rank_kv_tokens = kv_cache_info['per_rank_kv_tokens']
    enable_tknp = kv_cache_info['enable_tknp']
    tknp_size = kv_cache_info['tknp_size']
    
    # Calculate maximum concurrency
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        kv_cache_info['vllm_config'], kv_cache_info['kv_cache_config']
    )
    
    # Calculate required tokens
    tokens_per_request = seq_length + decode_tokens
    required_tokens = batch_size * tokens_per_request
    
    # Check if configuration fits
    fits = required_tokens <= total_kv_cache_tokens
    
    if verbose and dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"KV Cache Capacity Check")
        print(f"{'='*60}")
        print(f"KV cache this GPU: {gpu_kv_cache_tokens:,} tokens")
        if enable_tknp:
            print(f"Token parallel size: {tknp_size}")
            print(f"Per-TKNP-rank KV cache tokens: {per_rank_kv_tokens}")
            all_equal = len(set(per_rank_kv_tokens)) == 1
            # if not all_equal:
                # print(f"⚠  KV cache is NOT uniform across TKNP ranks!")
            print(f"Total KV cache (summed across TKNP ranks): {total_kv_cache_tokens:,} tokens")
        print(f"Maximum concurrency for {tokens_per_request:,} tokens per request: {max_concurrency:.2f}x")
        print(f"Total required tokens: {required_tokens:,}")
        print(f"Capacity utilization: {required_tokens / total_kv_cache_tokens * 100:.1f}%")
        
        if fits:
            print(f"✓ Configuration FITS in KV cache - can run concurrently")
        else:
            print(f"✗ Configuration EXCEEDS KV cache capacity by {required_tokens - total_kv_cache_tokens:,} tokens")
            print(f"✗ Requests will be QUEUED - benchmark results will not reflect pure concurrent throughput")
        print(f"{'='*60}\n")
    
    return fits, required_tokens, max_concurrency


def run_data_collection(args, llm):
    """
    Run systematic data collection across multiple batch sizes and sequence lengths.
    """
    # Cache KV cache info once at the start
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("Initializing KV cache information...")
        print("="*80)
    
    kv_cache_info = get_kv_cache_info(llm)
    
    if dist.get_rank() == 0:
        print(f"Total KV cache capacity: {kv_cache_info['total_kv_cache_tokens']:,} tokens")
        print("="*80 + "\n")
    
    # Define specific sequence lengths for each batch size
    batch_seq_configs = {
        # 16: [8192, 16384, 32768],

        # Llama 3.3-70B configs for 4 Nodes (32 GPUs) with TP 8 + TKNP/PP 4
        # 32: [32768, 65536, 98304],
        # 64: [32768, 65536], #, 49152]
        # 128: [16384, 32768]

        # Llama 3.3-70B configs for 8 Nodes (64 GPUs) with TP 8 + TKNP/PP 8
        32: [32768, 65536, 98304, 114688],
        64: [32768, 65536, 98304], #, 49152]
        128: [16384, 32768, 65536],
        256: [16384, 32768],
    }
    
    total_runs = sum(len(seq_lengths) for seq_lengths in batch_seq_configs.values())
    current_run = 0
    successful_runs = 0
    skipped_runs = 0
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("STARTING SYSTEMATIC DATA COLLECTION")
        print("="*80)
        print(f"Total configurations to test: {total_runs}")
        print(f"Batch size -> Sequence lengths mapping:")
        for bs, seq_lens in batch_seq_configs.items():
            print(f"  Batch {bs}: {seq_lens}")
        print("="*80 + "\n")
    
    for batch_size, seq_lengths in batch_seq_configs.items():
        for seq_length in seq_lengths:
            current_run += 1
            
            if dist.get_rank() == 0:
                print("\n" + "#"*80)
                print(f"RUN {current_run}/{total_runs}: Batch Size = {batch_size}, Seq Length = {seq_length}")
                print("#"*80 + "\n")
            
            # Quick check if configuration fits using cached info
            fits, _, _ = check_workload_fits(
                kv_cache_info, batch_size, seq_length, args.decode_tokens, verbose=False
            )
            
            if not fits:
                skipped_runs += 1
                if dist.get_rank() == 0:
                    print(f"⚠️  SKIPPING benchmark - configuration exceeds KV cache capacity")
                    print(f"   Required: {batch_size * (seq_length + args.decode_tokens):,} tokens, "
                          f"Available: {kv_cache_info['total_kv_cache_tokens']:,} tokens")
                    print(f"\n⊘ Skipped run {current_run}/{total_runs} (exceeds KV cache)\n")
                continue
            
            try:
                run_inference_benchmark(args, llm, batch_size, seq_length)
                successful_runs += 1
                if dist.get_rank() == 0:
                    print(f"\n✓ Completed run {current_run}/{total_runs}")
                    
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"\n✗ Error in run {current_run}/{total_runs}: {str(e)}")
                    print("Continuing with next configuration...\n")
                continue
            
            torch.cuda.synchronize()
            dist.barrier()
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("DATA COLLECTION COMPLETE")
        print("="*80)
        print(f"Total configurations: {total_runs}")
        print(f"Successful runs: {successful_runs}")
        print(f"Skipped runs: {skipped_runs}")
        print(f"Results saved to: {args.output_dir}")
        print("="*80 + "\n")


def main():
    args = parse_args()

    llm = setup_vllm_model(args)

    if not args.collect_data:
        # For single run, get KV cache info and check with verbose output
        kv_cache_info = get_kv_cache_info(llm)
        fits, _, _ = check_workload_fits(
            kv_cache_info, args.batch_size, args.seq_length, args.decode_tokens, verbose=True
        )
        
        if not fits:
            if dist.get_rank() == 0:
                print("⚠️  ABORTING: Benchmark configuration exceeds KV cache capacity")
                print("   Results would reflect queued behavior, not concurrent throughput")
                print("   Please reduce batch_size or seq_length and try again")
            dist.destroy_process_group()
            return
        
        # Run single benchmark
        run_inference_benchmark(args, llm, args.batch_size, args.seq_length)
    else:
        # Run systematic data collection (caching happens inside)
        run_data_collection(args, llm)
    
    # destroy the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()