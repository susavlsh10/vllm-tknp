#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p batch
#SBATCH -J hw_nresearch_snoise-snoise:tknp-smoke
#SBATCH -t 00:45:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp.sqsh
#SBATCH --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise/sshrestha/:/lustre/fsw/hw_nresearch_snoise/sshrestha/
#SBATCH -o smoke_results/slurm_logs/tknp_smoke_%j.out
#SBATCH -e smoke_results/slurm_logs/tknp_smoke_%j.err

set -uo pipefail

CHECKOUT=/home/sshrestha/workspace/2026/vllm-tknp
RESULT_ROOT="${CHECKOUT}/smoke_results/job_${SLURM_JOB_ID}"
EAGER_DIR="${RESULT_ROOT}/eager"
GRAPH_DIR="${RESULT_ROOT}/cuda_graph"
BENCHMARK=examples/offline_inference/TKNP/tknp_inference_benchmarks.py

mkdir -p "$EAGER_DIR" "$GRAPH_DIR" || exit 10

source /home/sshrestha/init_container.sh || exit 11
export PYTHONPATH="$CHECKOUT"
cd "$CHECKOUT" || exit 12

hostname | tee "$RESULT_ROOT/node.txt"
if ! nvidia-smi --query-gpu=name --format=csv,noheader > "$RESULT_ROOT/gpus.txt"; then
    echo "Environment check failed: nvidia-smi could not enumerate GPUs." | tee "$RESULT_ROOT/environment_error.txt"
    exit 13
fi
cat "$RESULT_ROOT/gpus.txt"

if [[ ! -s "$RESULT_ROOT/gpus.txt" ]] || grep -vq 'H100' "$RESULT_ROOT/gpus.txt"; then
    echo "Environment check failed: every visible GPU must be an H100." | tee "$RESULT_ROOT/environment_error.txt"
    exit 2
fi

python3 -c 'import vllm; print(vllm.__file__)' | tee "$RESULT_ROOT/vllm_path.txt"
if ! grep -q '^/home/sshrestha/workspace/2026/vllm-tknp/' "$RESULT_ROOT/vllm_path.txt"; then
    echo "Environment check failed: vLLM did not import from the TKNP checkout." | tee "$RESULT_ROOT/environment_error.txt"
    exit 3
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
    --output-dir "$GRAPH_DIR" 2>&1 | tee "$GRAPH_DIR/run.log"
graph_rc=${PIPESTATUS[0]}
printf '%s\n' "$graph_rc" > "$GRAPH_DIR/exit_code"

if (( graph_rc != 0 )); then
    echo "CUDA Graph smoke test failed with exit code $graph_rc."
    exit "$graph_rc"
fi
touch "$GRAPH_DIR/SUCCESS"

echo "Both TKNP smoke tests completed successfully at $(date --iso-8601=seconds)."
