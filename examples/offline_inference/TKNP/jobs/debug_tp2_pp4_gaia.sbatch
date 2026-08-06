#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p GAIA
#SBATCH -J hw_nresearch_snoise-snoise:tknp-pp4-debug
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o sweep_results/slurm_logs/tknp_pp4_debug_%j.out
#SBATCH -e sweep_results/slurm_logs/tknp_pp4_debug_%j.err

set -uo pipefail

CHECKOUT=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp
IMAGE=/mnt/lustre/gaia/sshrestha/containers/vllm-tknp-eos.sqsh
CONTAINER_MOUNTS=/labhome/sshrestha:/labhome/sshrestha,/mnt/lustre/gaia/sshrestha:/mnt/lustre/gaia/sshrestha

if [[ "${TKNP_IN_CONTAINER:-0}" != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-node=8 \
        --container-image="$IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir="$CHECKOUT" \
        env TKNP_IN_CONTAINER=1 \
        bash examples/offline_inference/TKNP/jobs/debug_tp2_pp4_gaia.sbatch
fi

RESULT_ROOT="${CHECKOUT}/sweep_results/gaia_pp4_debug_job_${SLURM_JOB_ID}"
BENCHMARK=examples/offline_inference/TKNP/tknp_inference_benchmarks.py
mkdir -p "$RESULT_ROOT"

source /labhome/sshrestha/init_container.sh || exit 11
export PYTHONPATH="$CHECKOUT"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_WORKSPACE_BASE=/mnt/lustre/gaia/sshrestha
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1
export CUDA_CACHE_PATH=/mnt/lustre/gaia/sshrestha/.cache/nv/ComputeCache
export TORCHINDUCTOR_CACHE_DIR=/mnt/lustre/gaia/sshrestha/.cache/torch/inductor-tknp
export VLLM_CACHE_ROOT=/mnt/lustre/gaia/sshrestha/.cache/vllm-tknp
export VLLM_CONFIG_ROOT=/mnt/lustre/gaia/sshrestha/.config/vllm-tknp
export PYTHONFAULTHANDLER=1
export NCCL_COMM_BLOCKING=1
export NCCL_CUMEM_ENABLE=0
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
cd "$CHECKOUT" || exit 12

hostname | tee "$RESULT_ROOT/node.txt"
nvidia-smi --query-gpu=name --format=csv,noheader > "$RESULT_ROOT/gpus.txt"

run_mode() {
    local mode=$1
    shift
    local mode_dir="$RESULT_ROOT/$mode"
    mkdir -p "$mode_dir/torchrun"

    timeout --signal=TERM --kill-after=30s 90s \
        env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        torchrun --nproc-per-node=8 --master-port=29621 \
        --log-dir "$mode_dir/torchrun" --redirects 3 --tee 3 \
        "$BENCHMARK" \
        --model Qwen/Qwen2.5-32B \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 4 \
        --token-parallel-size 1 \
        --batch-size 4 \
        --seq-length 128 \
        --decode-tokens 4 \
        --load-format dummy \
        --cudagraph-capture-sizes 4 \
        --output-dir "$mode_dir/results" \
        "$@" > "$mode_dir/combined.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$mode_dir/exit_code.txt"
    if (( rc == 0 )) && find "$mode_dir/results" -type f -name '*.csv' -print -quit | grep -q .; then
        touch "$mode_dir/PASS"
    else
        touch "$mode_dir/FAIL"
    fi
}

run_mode eager
run_mode cuda_graph --cuda-graph

echo "PP4 debug completed"
