#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 <on|off> <absolute-result-dir>}"
RESULT_DIR="${2:?usage: $0 <on|off> <absolute-result-dir>}"

if [[ "${MODE}" != "on" && "${MODE}" != "off" ]]; then
  echo "mode must be 'on' or 'off'" >&2
  exit 2
fi

WORKTREE=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
VENV=/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv

source "${VENV}/bin/activate"
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64

cd "${WORKTREE}"
mkdir -p "${RESULT_DIR}"

dbo_args=()
if [[ "${MODE}" == "on" ]]; then
  dbo_args=(--enable-dbo)
fi

# This stale fork specializes compiled TKNP communication shapes.  Use one
# process and one exact CUDA graph per point rather than sharing a multi-shape
# piecewise cache across asymmetric rank-local shapes.
for repeat in 1 2 3; do
  for batch_size in 16 32 64 96 128; do
    log_file="${RESULT_DIR}/${MODE}_b${batch_size}_r${repeat}.log"
    python -m torch.distributed.run --nproc-per-node=4 \
      examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tensor-parallel-size 1 \
      --token-parallel-size 4 \
      --batch-size "${batch_size}" \
      --seq-length 32768 \
      --decode-tokens 256 \
      --load-format dummy \
      --cuda-graph \
      "${dbo_args[@]}" \
      --cudagraph-capture-sizes "${batch_size}" \
      --gpu-memory-utilization 0.9 \
      --output-dir "${RESULT_DIR}" \
      2>&1 | tee "${log_file}"
  done
done

echo "TKNP_WAVEFRONT_SWEEP_COMPLETE mode=${MODE} result_dir=${RESULT_DIR}"
