#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p batch
#SBATCH -J hw_nresearch_snoise-snoise:tknp-benchmark
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp.sqsh
#SBATCH --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise/sshrestha/:/lustre/fsw/hw_nresearch_snoise/sshrestha/
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
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Benchmark parameters
BATCH_SIZE=32
SEQ_LENGTH=32768
DECODE_TOKENS=1000

# Parallelism configurations to test
TP_SIZE=2
TKNP_SIZE=4
PP_SIZE=4

# GPUs per node
GPUS_PER_NODE=8

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

echo "======================================================================"
echo "Starting TKNP Benchmarks at $(date)"
echo "======================================================================"
echo "Model: $MODEL_NAME"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((NODES * GPUS_PER_NODE))"
echo "Batch Size: $BATCH_SIZE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Decode Tokens: $DECODE_TOKENS"
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
     --gpus-per-node=$GPUS_PER_NODE \
     --container-image=$IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     --container-workdir=$CONTAINER_WORKDIR \
     --container-writable \
     --no-container-mount-home \
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
sleep 10

# Configuration 2: Token Parallel (TP + TKNP, PP=1)
echo "----------------------------------------------------------------------"
echo "Configuration 2: Token Parallel"
echo "TP=$TP_SIZE, TKNP=$TKNP_SIZE, PP=1"
echo "----------------------------------------------------------------------"
srun --mpi=pmix \
     --nodes=$NODES \
     --ntasks-per-node=1 \
     --gpus-per-node=$GPUS_PER_NODE \
     --container-image=$IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     --container-workdir=$CONTAINER_WORKDIR \
     --container-writable \
     --no-container-mount-home \
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
echo "Results saved to: examples/offline_inference/TKNP/tknp_data"
echo "======================================================================"