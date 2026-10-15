#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${1:?usage: $0 <absolute-result-dir>}"
if [[ "${RESULT_ROOT}" != /* ]]; then
  echo "result directory must be absolute" >&2
  exit 2
fi

WORKTREE=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
VENV=/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv
export PYTHONPATH="${WORKTREE}"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64

cd "${WORKTREE}"
mkdir -p "${RESULT_ROOT}"

for repeat in 1 2 3; do
  # Rotate order to reduce systematic first-run and thermal bias.
  case "${repeat}" in
    1) variants=(old new off) ;;
    2) variants=(new off old) ;;
    3) variants=(off old new) ;;
  esac
  for variant in "${variants[@]}"; do
    run_dir="${RESULT_ROOT}/${variant}_r${repeat}"
    mkdir -p "${run_dir}"
    dbo_args=()
    case "${variant}" in
      old)
        export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
        dbo_args=(--enable-dbo)
        ;;
      new)
        export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1
        dbo_args=(--enable-dbo)
        ;;
      off)
        export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
        ;;
    esac

    "${VENV}/bin/python3" -m torch.distributed.run --nproc-per-node=4 \
      examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tensor-parallel-size 1 --token-parallel-size 4 \
      --batch-size 128 --seq-length 32768 --decode-tokens 256 \
      --load-format dummy --cuda-graph "${dbo_args[@]}" \
      --cudagraph-capture-sizes 128 --gpu-memory-utilization 0.9 \
      --output-dir "${run_dir}" > "${run_dir}/run.log" 2>&1

    replay_count=$(grep -c 'TKNP_CUDAGRAPH_REPLAY padded_size=128' \
      "${run_dir}/run.log" || true)
    if (( replay_count < 4 )); then
      echo "Missing four-rank graph replay: ${variant} repeat ${repeat}" >&2
      exit 1
    fi
    if [[ "${variant}" != off ]]; then
      ubatch_replay_count=$(grep -c 'TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=128' \
        "${run_dir}/run.log" || true)
      if (( ubatch_replay_count < 4 )); then
        echo "Missing four-rank microbatch graph replay: ${variant} repeat ${repeat}" >&2
        exit 1
      fi
    fi
    grep -q 'Results saved to:' "${run_dir}/run.log"
    grep 'System Decode Throughput:' "${run_dir}/run.log"
    echo "TKNP_ROOT_STREAM_POINT_DONE variant=${variant} repeat=${repeat}"
  done
done

echo "TKNP_ROOT_STREAM_COMPARISON_COMPLETE result_root=${RESULT_ROOT}"
