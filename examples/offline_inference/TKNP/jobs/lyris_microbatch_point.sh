#!/usr/bin/env bash
set -euo pipefail

result_root=${1:?absolute result root}
variant=${2:?off, legacy, independent, split, balanced}
batch=${3:-128}
context=${4:-32768}
decode=${5:-128}
profile=${6:-0}
tp=${TKNP_TP_SIZE:-1}
tknp=${TKNP_WORLD_SIZE:-4}
(( tp * tknp == 4 )) || { echo 'Single-node topology must use four GPUs'; exit 2; }
worktree=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_venv=/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv
[[ "$result_root" == /* ]] || exit 2
export PYTHONPATH="$worktree"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export VLLM_TKNP_DBO_INDEPENDENT_COMM=0
export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
export VLLM_TKNP_DBO_LAYOUT=asymmetric
export VLLM_TKNP_DBO_LOOKAHEAD=0
export VLLM_TKNP_DBO_ZERO_ROOT_C1=0
export VLLM_TKNP_DBO_SYMM_MEM=0
dbo_args=()
case "$variant" in
  off) ;;
  legacy) dbo_args=(--enable-dbo) ;;
  independent) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 ;;
  split) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 ;;
  balanced) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_LAYOUT=balanced ;;
  lookahead) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_LOOKAHEAD=1 ;;
  lookahead_balanced) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_LOOKAHEAD=1 VLLM_TKNP_DBO_LAYOUT=balanced ;;
  wavefront) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_LOOKAHEAD=1 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 ;;
  zero_legacy) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 ;;
  zero_independent) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 ;;
  pipelined) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 VLLM_TKNP_DBO_LOOKAHEAD=1 ;;
  serial_root) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_INDEPENDENT_COMM=1 VLLM_TKNP_DBO_INDEPENDENT_TP=0 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 VLLM_TKNP_DBO_LOOKAHEAD=1 ;;
  symmetric) dbo_args=(--enable-dbo); export VLLM_TKNP_DBO_SYMM_MEM=1 VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1 VLLM_TKNP_DBO_ZERO_ROOT_C1=1 ;;
  *) exit 2 ;;
esac
profile_args=()
if [[ "${TKNP_GRAPH_BENCH_ITERS:-0}" != 0 ]]; then
  export VLLM_TKNP_KEEP_GRAPH_INPUTS_FOR_DIAGNOSTIC=1
  profile_args+=(--graph-replay-bench-iters "$TKNP_GRAPH_BENCH_ITERS")
fi
if [[ "${TKNP_DUMP_TOKEN_IDS:-0}" == 1 ]]; then
  profile_args+=(--dump-token-ids)
fi
if [[ "${TKNP_DUMP_FIRST_LOGITS:-0}" == 1 ]]; then
  profile_args+=(--dump-first-step-logits)
fi
if [[ "$profile" == 1 ]]; then
  profile_args+=(--torch-profiler-dir "$result_root/traces" --torch-profiler-delay-iters 4 --torch-profiler-max-iters 8)
  if [[ "${TKNP_PROFILE_AFTER:-0}" == 1 ]]; then
    profile_args+=(--profile-after-benchmark)
  fi
fi
if [[ "${TKNP_DISABLE_CUSTOM_AR:-0}" == 1 ]]; then
  profile_args+=(--disable-custom-all-reduce)
fi
mkdir -p "$result_root"
cd "$worktree"
"$tknp_venv/bin/python3" -c 'import torch, vllm; print(vllm.__file__, torch.__version__, torch.cuda.nccl.version())' > "$result_root/environment.log" 2>&1
"$tknp_venv/bin/python3" -c 'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(("VLLM_TKNP", "NCCL_", "TKNP_"))}, sort_keys=True))' >> "$result_root/environment.log"
sha1sum vllm/v1/worker/gpu_model_runner.py vllm/model_executor/layers/token_parallel_linear.py vllm/v1/worker/gpu_ubatch_wrapper.py vllm/forward_context.py vllm/v1/attention/backends/flashinfer.py vllm/config/parallel.py vllm/distributed/device_communicators/cuda_communicator.py >> "$result_root/environment.log"
timeout 900 "$tknp_venv/bin/python3" -m torch.distributed.run --nproc-per-node=4 \
  "${TKNP_BENCHMARK_SCRIPT:-examples/offline_inference/TKNP/tknp_inference_benchmarks.py}" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size "$tp" --token-parallel-size "$tknp" \
  --batch-size "$batch" --seq-length "$context" --decode-tokens "$decode" \
  --kv-cache-dtype "${TKNP_KV_CACHE_DTYPE:-auto}" \
  --load-format dummy --cuda-graph --cuda-graph-mode "${TKNP_GRAPH_MODE:-FULL_AND_PIECEWISE}" \
  --steady-state-metrics "${dbo_args[@]}" "${profile_args[@]}" \
  --cudagraph-capture-sizes "$batch" --gpu-memory-utilization 0.9 \
  --output-dir "$result_root" > "$result_root/run.log" 2>&1
replay_count=$(grep -c "TKNP_CUDAGRAPH_REPLAY padded_size=$batch" "$result_root/run.log" || true)
(( replay_count >= 4 )) || { echo "MISSING_GRAPH_REPLAY $replay_count"; exit 1; }
if [[ "$variant" != off ]]; then
  ubatch_count=$(grep -c "TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=$batch" "$result_root/run.log" || true)
  (( ubatch_count >= 4 )) || { echo "MISSING_MICROBATCH_REPLAY $ubatch_count"; exit 1; }
fi
grep 'System Decode Throughput:' "$result_root/run.log"
grep 'STEADY_STATE_DECODE' "$result_root/run.log"
echo "POINT_PASS variant=$variant batch=$batch context=$context result=$result_root"
