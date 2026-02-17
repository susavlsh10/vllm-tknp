#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -p batch
#SBATCH -J hw_nresearch_snoise-snoise:tknp-benchmark
#SBATCH -t 01:00:00
#SBATCH -o logs/tknp_benchmark_%j.out
#SBATCH -e logs/tknp_benchmark_%j.err

# ============================================================================
# CONFIGURATION PARAMETERS - EDIT THESE AS NEEDED
# ============================================================================

NODES=${SLURM_JOB_NUM_NODES:-1}
echo "Allocated nodes: $NODES"

# Container configuration
IMAGE=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp.sqsh
CONTAINER_MOUNTS=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise/sshrestha/:/lustre/fsw/hw_nresearch_snoise/sshrestha/
CONTAINER_WORKDIR=/home/sshrestha/workspace/2026/vllm-tknp

# Model configuration
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME="Qwen/Qwen3-32B"

# Benchmark parameters
BATCH_SIZE=64
# SEQ_LENGTH=32768
SEQ_LENGTH=65536
DECODE_TOKENS=1000

# Parallelism configuration
TP_SIZE=8
TKNP_SIZE=$NODES
PP_SIZE=$NODES

# GPUs per node
GPUS_PER_NODE=8

# Create logs directory
mkdir -p $CONTAINER_WORKDIR/logs

# Get master node address (before entering container)
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

echo "======================================================================"
echo "Starting TKNP Benchmarks at $(date)"
echo "======================================================================"
echo "Model: $MODEL_NAME"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Batch Size: $BATCH_SIZE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Decode Tokens: $DECODE_TOKENS"
echo "TP Size: $TP_SIZE"
echo "TKNP Size: $TKNP_SIZE"
echo "PP Size: $PP_SIZE"
echo "======================================================================"
echo ""

# Configuration 1: Pipeline Parallel (TP + PP, TKNP=1)
echo "----------------------------------------------------------------------"
echo "Configuration 1: Pipeline Parallel"
echo "TP=$TP_SIZE, PP=$PP_SIZE, TKNP=1"
echo "----------------------------------------------------------------------"
srun --mpi=pmix \
     --nodes=$NODES \
     --ntasks-per-node=1 \
     --container-image=$IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     --container-workdir=$CONTAINER_WORKDIR \
     --no-container-mount-home \
     --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
     bash examples/offline_inference/TKNP/jobs/run_tknp_worker.sh \
          "$MODEL_NAME" \
          $TP_SIZE \
          $PP_SIZE \
          1 \
          $BATCH_SIZE \
          $SEQ_LENGTH \
          $DECODE_TOKENS \
          $NODES \
          $GPUS_PER_NODE

echo ""
echo "✓ Configuration 1 completed"
echo ""
sleep 5

# Configuration 2: Token Parallel (TP + TKNP, PP=1)
echo "----------------------------------------------------------------------"
echo "Configuration 2: Token Parallel"
echo "TP=$TP_SIZE, TKNP=$TKNP_SIZE, PP=1"
echo "----------------------------------------------------------------------"
srun --mpi=pmix \
     --nodes=$NODES \
     --ntasks-per-node=1 \
     --container-image=$IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     --container-workdir=$CONTAINER_WORKDIR \
     --no-container-mount-home \
     --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT \
     bash examples/offline_inference/TKNP/jobs/run_tknp_worker.sh \
          "$MODEL_NAME" \
          $TP_SIZE \
          1 \
          $TKNP_SIZE \
          $BATCH_SIZE \
          $SEQ_LENGTH \
          $DECODE_TOKENS \
          $NODES \
          $GPUS_PER_NODE

echo ""
echo "✓ Configuration 2 completed"
echo ""

# ============================================================================
# COMPLETION
# ============================================================================

echo "======================================================================"
echo "All benchmarks completed at $(date)"
echo "======================================================================"