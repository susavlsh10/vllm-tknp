"""
Inference benchmarking script for vLLM with Token Parallelism (TKNP) support.

Example usage:
Token parallelism: 
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --token-parallel-size 2 --batch-size 16 --seq-length 8192

Tensor parallelism:
torchrun --nproc-per-node=8 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 8 --token-parallel-size 1 --batch-size 32 --seq-length 16384

Pipeline parallelism:
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --pipeline-parallel-size 2 --batch-size 16 --seq-length 8192


Supported models: 

Llama-3:    meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, 
            meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.3-70B-Instruct
Qwen:       Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-4B-Instruct-2507

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
        print(f"Detected GPU: {gpu_name}")
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
    
    Returns:
        tuple: (fits: bool, gpu_kv_cache_tokens: int, required_tokens: int, max_concurrency: float)
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
    
    # Calculate KV cache capacity in tokens
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
    
    # Calculate maximum concurrency
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    
    # Calculate required tokens for current benchmark configuration
    # Each request needs: prompt tokens (seq_length) + decode tokens
    tokens_per_request = args.seq_length + args.decode_tokens
    required_tokens = args.batch_size * tokens_per_request
    
    # Check if configuration fits
    fits = required_tokens <= gpu_kv_cache_tokens
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"KV Cache Capacity Check")
        print(f"{'='*60}")
        print(f"GPU KV cache size: {gpu_kv_cache_tokens:,} tokens")
        print(f"Maximum concurrency for {tokens_per_request:,} tokens per request: {max_concurrency:.2f}x")
        print(f"Total required tokens: {required_tokens:,}")
        print(f"Capacity utilization: {required_tokens / gpu_kv_cache_tokens * 100:.1f}%")
        
        if fits:
            print(f"✓ Configuration FITS in KV cache - can run concurrently")
        else:
            print(f"✗ Configuration EXCEEDS KV cache capacity by {required_tokens - gpu_kv_cache_tokens:,} tokens")
            print(f"✗ Requests will be QUEUED - benchmark results will not reflect pure concurrent throughput")
        print(f"{'='*60}\n")
    
    return fits, gpu_kv_cache_tokens, required_tokens, max_concurrency

def inference_benchmark(llm, prompts, args):
    """
    Benchmark decode performance with skip-prefill enabled.
    
    Since skip-prefill is enabled, KV cache is pre-filled with dummy values,
    so we only measure pure decode latency and throughput.
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

    # ===== Decode-Only Measurement =====
    decode_tokens = args.decode_tokens
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=decode_tokens)
    
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    decode_start.record()
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=(dist.get_rank() == 0))
    
    decode_end.record()
    torch.cuda.synchronize()
    
    total_time_ms = decode_start.elapsed_time(decode_end)
    total_decode_tokens = args.batch_size * decode_tokens
    
    # Calculate metrics
    average_decode_latency = total_time_ms / decode_tokens
    average_decode_system_throughput = total_decode_tokens / (total_time_ms / 1000)
    average_decode_throughput_per_user = 1000 / average_decode_latency  # tokens/sec/user
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Decode Performance Results")
        print(f"{'='*60}")
        print(f"Total decode time: {total_time_ms:.2f} ms")
        print(f"Total decode tokens: {total_decode_tokens}")
        print(f"\nDecoding Metrics\n")
        print(f"System Decode Throughput: {average_decode_system_throughput:.2f} tokens/sec")
        print(f"Decode Throughput per GPU: {average_decode_system_throughput / dist.get_world_size():.2f} tokens/sec/GPU")
        print(f"Average Decode Latency: {average_decode_latency:.2f} ms/token/user")
        print(f"Per-user Decode Throughput: {average_decode_throughput_per_user:.2f} tokens/sec/user")
        print(f"{'='*60}\n")

        # Save benchmark results
        metrics = {
            'total_decode_time_ms': total_time_ms,
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
        # "attention_config": AttentionConfig(backend="FLASH_ATTN"), 
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
    
    inference_benchmark(llm, prompts, args)


def run_data_collection(args, llm):
    """
    Run systematic data collection across multiple batch sizes and sequence lengths.
    """
    batch_sizes = [16, 32, 64]
    seq_lengths = [8192, 16384, 24576]
    
    total_runs = len(batch_sizes) * len(seq_lengths)
    current_run = 0
    successful_runs = 0
    skipped_runs = 0
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("STARTING SYSTEMATIC DATA COLLECTION")
        print("="*80)
        print(f"Total configurations to test: {total_runs}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Sequence lengths: {seq_lengths}")
        print("="*80 + "\n")
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            current_run += 1
            
            if dist.get_rank() == 0:
                print("\n" + "#"*80)
                print(f"RUN {current_run}/{total_runs}: Batch Size = {batch_size}, Seq Length = {seq_length}")
                print("#"*80 + "\n")
            
            # Update args for capacity check
            args.batch_size = batch_size
            args.seq_length = seq_length
            
            # Check if configuration fits in KV cache
            fits, _, _, _ = check_kv_cache_capacity(llm, args)
            
            if not fits:
                skipped_runs += 1
                if dist.get_rank() == 0:
                    print(f"⚠️  SKIPPING benchmark - configuration exceeds KV cache capacity")
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


# warm up the model before benchmarking to ensure accurate measurements
def warmup_model(llm, args):
    if dist.get_rank() == 0:
        print("\nWarming up the model with a few generations...")
    
    warmup_prompts = generate_benchmark_prompts(
        batch_size=args.batch_size,
        seq_length=128,
        tokenizer=None,
        model_name=args.model,
        vocab_style="natural",
        seed=42
    )
    
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=16)
    
    llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
    
    torch.cuda.synchronize()
    dist.barrier()
    
    if dist.get_rank() == 0:
        print("Model warmup complete.\n")

def main():
    args = parse_args()

    llm = setup_vllm_model(args)

    # Check KV cache capacity before warmup
    fits, _, _, _ = check_kv_cache_capacity(llm, args)
    if not fits and not args.collect_data:
        if dist.get_rank() == 0:
            print("⚠️  ABORTING: Benchmark configuration exceeds KV cache capacity")
            print("   Results would reflect queued behavior, not concurrent throughput")
            print("   Please reduce batch_size or seq_length and try again")
        dist.destroy_process_group()
        return

    # Warm up the model before benchmarking
    warmup_model(llm, args)
    
    if args.collect_data:
        # Run systematic data collection across multiple configurations
        run_data_collection(args, llm)
    else:
        # Run single benchmark with specified batch size and sequence length
        run_inference_benchmark(args, llm, args.batch_size, args.seq_length)
    
    # destroy the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()