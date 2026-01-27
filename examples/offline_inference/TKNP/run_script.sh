#!/bin/bash

# Bash script to run vLLM inference benchmarks with different parallelism strategies

set -e  # Exit on error

echo "=========================================="
echo "Starting vLLM Inference Benchmarks"
echo "=========================================="
echo ""


# Pipeline Parallelism
echo "Running Pipeline Parallelism benchmark..."
echo "------------------------------------------"
# torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
#     --tensor-parallel-size 1 \
#     --pipeline-parallel-size 2 \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --collect-data

echo ""
echo "✓ Pipeline Parallelism benchmark completed"
echo ""

# Token Parallelism
echo "Running Token Parallelism benchmark..."
echo "------------------------------------------"
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 1 \
    --token-parallel-size 2 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --collect-data

echo ""
echo "✓ Token Parallelism benchmark completed"
echo ""

# Tensor Parallelism
echo "Running Tensor Parallelism benchmark..."
echo "------------------------------------------"
torchrun --nproc-per-node=2 examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
    --tensor-parallel-size 2 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --collect-data

echo ""
echo "✓ Tensor Parallelism benchmark completed"
echo ""


echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="