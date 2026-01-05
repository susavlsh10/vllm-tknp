"""
Inference benchmarking script for vLLM with Token Parallelism (TKNP) support.

Example usage:
Token parallelism: 
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --enable-token-parallel --token-parallel-size 2 --batch-size 16 --seq-length 4096

Tensor parallelism:
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 2 --batch-size 16 --seq-length 4096

Pipeline parallelism:
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py --tensor-parallel-size 1 --pipeline-parallel-size 2 --batch-size 16 --seq-length 4096

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
    parser.add_argument("--enable-token-parallel", action="store_true",
                        help="Enable token parallelism")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model length (default: 32768)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    # batch size and seq length for prompts
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for prompts (default: 8)")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Sequence length for prompts (default: 128)")
    parser.add_argument("--print-outputs", action="store_true",
                        help="Print generated outputs")
    parser.add_argument("--decode-tokens", type=int, default=128,
                        help="Number of tokens to decode during benchmarking (default: 128)")
    parser.add_argument("--output-dir", type=str, 
                        default="examples/offline_inference/TKNP/tknp_data",
                        help="Directory to save benchmark results (default: examples/offline_inference/TKNP/tknp_data)")
    parser.add_argument("--collect-data", action="store_true",
                        help="Enable systematic data collection across multiple batch sizes and sequence lengths")

    return parser.parse_args()

def get_gpu_name():
    """Get the GPU name for the current device."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        # Extract concise GPU name (e.g., "NVIDIA RTX A5000" -> "A5000")
        # Remove common prefixes and clean up
        gpu_name = gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace("RTX", "").strip()
        # Take last significant part (usually the model number)
        parts = gpu_name.split()
        gpu_name = parts[-1] if parts else "unknown"
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
    tknp_size = args.token_parallel_size if args.enable_token_parallel else 0
    
    filename = f"{model_name}_{gpu_name}_TP_{tp_size}_PP_{pp_size}_TKNP_{tknp_size}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    # Prepare data row with concise column names (rounded to 4 decimal places)
    data_row = {
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'prefill_time': round(metrics['prefill_warmup_time_ms'], 4),
        'generation_time': round(metrics['total_generation_time_ms'], 4),
        'decode_time': round(metrics['steady_state_decode_time_ms'], 4),
        'decode_tokens': metrics['steady_state_decode_tokens'],
        'sys_decode_tps': round(metrics['system_decode_throughput'], 4),
        'decode_tps_per_gpu': round(metrics['decode_throughput_per_gpu'], 4),
        'avg_decode_latency': round(metrics['average_decode_latency'], 4),
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

def inference_benchmark(llm, prompts, args):
    """
    Benchmark prefill and decode phases separately.
    
    Strategy:
    1. Run a generation with max_tokens=1 to measure time until all prompts are prefilled
       and first decode token is generated
    2. Run a separate generation with enough tokens to ensure all prompts complete prefill,
       then measure steady-state decode throughput
    """
    
    # Calculate how many tokens needed to complete all prefills with chunking
    # With batch_size queries of seq_length tokens and max_num_batched_tokens limit,
    # we need at least enough decode tokens for all queries to finish their chunked prefill
    total_prompt_tokens = args.batch_size * args.seq_length
    max_batched = 32768  # Should match max_num_batched_tokens in LLM config
    
    # Number of steps to complete chunked prefill (conservative estimate)
    # In worst case, we need ceil(total_prompt_tokens / max_batched) steps
    prefill_steps = math.ceil(total_prompt_tokens / max_batched)

    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Benchmark Configuration")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Max batched tokens: {max_batched}")
        print(f"Estimated prefill steps: {prefill_steps}")
        print(f"TKNP: {args.token_parallel_size}, TP: {args.tensor_parallel_size}, PP: {args.pipeline_parallel_size}")
        print(f"{'='*60}\n")
    
    # ===== Measurement 1: Prefill + Initial Decode =====
    # Generate enough tokens to ensure all prefills complete plus a few decode steps
    prefill_steps += 5 # Add a few extra tokens to ensure steady state
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=prefill_steps)
    
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    prefill_start.record()
    
    outputs = llm.generate(prompts, sampling_params)
    
    prefill_end.record()
    torch.cuda.synchronize()
    
    prefill_warmup_time_ms = prefill_start.elapsed_time(prefill_end)

    # ===== Measurement 2: Steady-State Decode =====
    # Now measure pure decode performance with a longer generation
    # All prompts are already processed, so this is pure decode
    decode_tokens = args.decode_tokens
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=decode_tokens)
    
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    decode_start.record()
    
    outputs = llm.generate(prompts, sampling_params)
    
    decode_end.record()
    torch.cuda.synchronize()
    
    total_time_ms = decode_start.elapsed_time(decode_end)
    
    # Since we're generating from scratch again, estimate prefill time based on
    # the first measurement, then subtract to get pure decode time
    # More accurate: measure the last N tokens where N is large enough to be in steady state
    
    tokens_per_query = decode_tokens
    total_decode_tokens = args.batch_size * tokens_per_query
    
    # Estimate prefill completion time (time until all queries enter decode phase)
    # This is approximately the time for first `prefill_steps` tokens
    # estimated_prefill_time_ms = (total_time_ms / decode_tokens) * prefill_steps
    estimated_prefill_time_ms = prefill_warmup_time_ms
    decode_only_time_ms = total_time_ms - estimated_prefill_time_ms
    
    # Calculate metrics
    average_decode_latency = decode_only_time_ms / (decode_tokens - prefill_steps)
    average_decode_system_throughput = (args.batch_size * (decode_tokens - prefill_steps)) / (decode_only_time_ms / 1000)
    average_decode_throughput_per_user = 1 / (average_decode_latency / 1000)
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Timing Results")
        print(f"{'='*60}")
        print(f"Prefill + warmup time: {prefill_warmup_time_ms:.2f} ms")
        print(f"Total generation time ({decode_tokens} tokens): {total_time_ms:.2f} ms")
        print(f"Steady-state decode time: {decode_only_time_ms:.2f} ms")
        print(f"Steady-state decode tokens: {decode_tokens - prefill_steps}")
        print(f"\nDecoding Metrics\n")
        print(f"System Decode Throughput: {average_decode_system_throughput:.2f} tokens/sec")
        print(f"Decode Throughput per GPU: {average_decode_system_throughput / dist.get_world_size():.2f} tokens/sec/GPU")
        print(f"Average Decode Latency: {average_decode_latency:.2f} ms/token/user")
        print(f"Per-user Decode Throughput: {average_decode_throughput_per_user:.2f} tokens/sec/user")
        
        print(f"{'='*60}\n")

        # Save benchmark results
        metrics = {
            'prefill_warmup_time_ms': prefill_warmup_time_ms,
            'total_generation_time_ms': total_time_ms,
            'steady_state_decode_time_ms': decode_only_time_ms,
            'steady_state_decode_tokens': decode_tokens - prefill_steps,
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
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enforce_eager": True,
        "enable_prefix_caching": False,  # Disable prefix caching for benchmarking
        "attention_config": AttentionConfig(backend="FLASH_ATTN"),  # Add this line
        "gpu_memory_utilization": 0.86,  # Max GPU memory utilization
        "max_num_batched_tokens": 32768,  # max number of tokens in a single forward pass
    }
    
    # Only add token parallel configs if token parallelism is enabled
    if args.enable_token_parallel:
        if args.token_parallel_size <= 1:
            raise ValueError("Token parallelism requires token_parallel_size > 1")
        llm_kwargs["enable_token_parallel"] = True
        llm_kwargs["token_parallel_size"] = args.token_parallel_size
    
    llm = LLM(**llm_kwargs)
    if dist.get_rank() == 0:
        print(f"LLM initialized with tensor_parallel_size={args.tensor_parallel_size}, pipeline_parallel_size={args.pipeline_parallel_size}, data_parallel_size={args.data_parallel_size}, token_parallel_size={args.token_parallel_size}")
    return llm  

def run_inference_benchmark(args, llm, batch_size, seq_length):
    prompts = None
    if dist.get_rank() == 0:
        
        # Generate benchmark prompts
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
    
    Args:
        args: Command line arguments
        llm: Initialized LLM instance
    """
    # Define batch sizes and sequence lengths to benchmark
    batch_sizes = [16, 32, 64, 128]
    seq_lengths = [4096, 8192, 16384]
    
    total_runs = len(batch_sizes) * len(seq_lengths)
    current_run = 0
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("STARTING SYSTEMATIC DATA COLLECTION")
        print("="*80)
        print(f"Total configurations to test: {total_runs}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Sequence lengths: {seq_lengths}")
        print(f"Parallelism config: TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}, TKNP={args.token_parallel_size if args.enable_token_parallel else 0}")
        print("="*80 + "\n")
    
    # Iterate through all combinations
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            current_run += 1
            
            if dist.get_rank() == 0:
                print("\n" + "#"*80)
                print(f"RUN {current_run}/{total_runs}: Batch Size = {batch_size}, Seq Length = {seq_length}")
                print("#"*80 + "\n")
            
            # Update args for this run
            args.batch_size = batch_size
            args.seq_length = seq_length
            
            try:
                # Run the benchmark for this configuration
                run_inference_benchmark(args, llm, batch_size, seq_length)
                
                if dist.get_rank() == 0:
                    print(f"\n✓ Completed run {current_run}/{total_runs}")
                    
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"\n✗ Error in run {current_run}/{total_runs}: {str(e)}")
                    print("Continuing with next configuration...\n")
                continue
            
            # Add a small delay between runs to allow system to stabilize
            torch.cuda.synchronize()
            dist.barrier()
    
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("DATA COLLECTION COMPLETE")
        print("="*80)
        print(f"Total runs completed: {total_runs}")
        print(f"Results saved to: {args.output_dir}")
        print("="*80 + "\n")

def main():
    args = parse_args()

    llm = setup_vllm_model(args)
    
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