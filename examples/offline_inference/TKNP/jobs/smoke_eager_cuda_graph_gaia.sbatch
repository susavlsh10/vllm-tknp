#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p GAIA
#SBATCH -J hw_nresearch_snoise-snoise:tknp-gaia-smoke
#SBATCH -t 00:45:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o smoke_results/slurm_logs/tknp_gaia_smoke_%j.out
#SBATCH -e smoke_results/slurm_logs/tknp_gaia_smoke_%j.err

set -uo pipefail

CHECKOUT=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp
IMAGE=/mnt/lustre/gaia/sshrestha/containers/vllm-tknp-eos.sqsh
CONTAINER_MOUNTS=/labhome/sshrestha:/labhome/sshrestha,/mnt/lustre/gaia/sshrestha:/mnt/lustre/gaia/sshrestha

# GAIA exposes Pyxis options on srun but not sbatch. Re-enter this script once
# inside the frozen container, preserving the outer allocation and log files.
if [[ "${TKNP_IN_CONTAINER:-0}" != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-node=8 \
        --container-image="$IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir="$CHECKOUT" \
        env TKNP_IN_CONTAINER=1 \
        bash examples/offline_inference/TKNP/jobs/smoke_eager_cuda_graph_gaia.sbatch
fi

RESULT_ROOT="${CHECKOUT}/smoke_results/gaia_job_${SLURM_JOB_ID}"
EAGER_DIR="${RESULT_ROOT}/eager"
GRAPH_DIR="${RESULT_ROOT}/cuda_graph"
BENCHMARK=examples/offline_inference/TKNP/tknp_inference_benchmarks.py

mkdir -p "$EAGER_DIR" "$GRAPH_DIR" || exit 10

# The legacy EOS initializer appends to PYTHONPATH and therefore expects the
# variable to exist even when this script is running with nounset enabled.
export PYTHONPATH="${PYTHONPATH:-}"
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
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" || exit 12
cd "$CHECKOUT" || exit 12

hostname | tee "$RESULT_ROOT/node.txt"
git rev-parse HEAD | tee "$RESULT_ROOT/git_head.txt"
git status --short > "$RESULT_ROOT/git_status.txt"

if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
    > "$RESULT_ROOT/gpus.txt"; then
    echo "Environment check failed: nvidia-smi could not enumerate GPUs." \
        | tee "$RESULT_ROOT/environment_error.txt"
    exit 13
fi
cat "$RESULT_ROOT/gpus.txt"

if [[ ! -s "$RESULT_ROOT/gpus.txt" ]] || grep -vq 'H100' "$RESULT_ROOT/gpus.txt"; then
    echo "Environment check failed: every visible GPU must be an H100." \
        | tee "$RESULT_ROOT/environment_error.txt"
    exit 14
fi

python3 - <<'PY' | tee "$RESULT_ROOT/environment.txt"
import flashinfer
import torch
import triton
import vllm
import vllm._C

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"triton={triton.__version__}")
print(f"flashinfer={flashinfer.__version__}")
print(f"vllm={vllm.__version__}")
print(f"vllm_path={vllm.__file__}")
print(f"vllm_C_path={vllm._C.__file__}")
PY
environment_rc=${PIPESTATUS[0]}

if (( environment_rc != 0 )); then
    echo "Environment check failed: Python or vllm._C import failed." \
        | tee "$RESULT_ROOT/environment_error.txt"
    exit 15
fi

if ! grep -q '^vllm_path=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp/' \
    "$RESULT_ROOT/environment.txt"; then
    echo "Environment check failed: vLLM did not import from the GAIA TKNP checkout." \
        | tee "$RESULT_ROOT/environment_error.txt"
    exit 16
fi

COMMON_ARGS=(
    "$BENCHMARK"
    --model Qwen/Qwen2.5-1.5B-Instruct
    --tensor-parallel-size 1
    --token-parallel-size 2
    --batch-size 2
    --seq-length 128
    --decode-tokens 4
    --load-format dummy
)

echo "Starting eager smoke test at $(date --iso-8601=seconds)"
timeout --signal=TERM --kill-after=30s 20m \
    env CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --nproc-per-node=2 "${COMMON_ARGS[@]}" \
    --output-dir "$EAGER_DIR" 2>&1 | tee "$EAGER_DIR/run.log"
eager_rc=${PIPESTATUS[0]}
printf '%s\n' "$eager_rc" > "$EAGER_DIR/exit_code"

if (( eager_rc != 0 )); then
    echo "Eager smoke test failed with exit code $eager_rc; CUDA Graph was skipped."
    touch "$GRAPH_DIR/SKIPPED_EAGER_FAILED"
    exit "$eager_rc"
fi
touch "$EAGER_DIR/SUCCESS"

echo "Starting CUDA Graph smoke test at $(date --iso-8601=seconds)"
timeout --signal=TERM --kill-after=30s 20m \
    env CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --nproc-per-node=2 "${COMMON_ARGS[@]}" \
    --cuda-graph \
    --cudagraph-capture-sizes 2 \
    --output-dir "$GRAPH_DIR" 2>&1 | tee "$GRAPH_DIR/run.log"
graph_rc=${PIPESTATUS[0]}
printf '%s\n' "$graph_rc" > "$GRAPH_DIR/exit_code"

if (( graph_rc != 0 )); then
    echo "CUDA Graph smoke test failed with exit code $graph_rc."
    exit "$graph_rc"
fi
touch "$GRAPH_DIR/SUCCESS"

echo "Both GAIA TKNP smoke tests completed successfully at $(date --iso-8601=seconds)."
