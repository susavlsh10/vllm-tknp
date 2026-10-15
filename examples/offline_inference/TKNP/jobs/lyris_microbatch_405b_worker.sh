#!/usr/bin/env bash
set -euo pipefail

variant=${1:?off or zero_independent}
point_root=${2:?absolute result directory}
worktree=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_venv=/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv
node_rank=${SLURM_PROCID:?}
nodes=${SLURM_JOB_NUM_NODES:?}
batch=${TKNP_405B_BATCH:-256}
context=${TKNP_405B_CONTEXT:-8192}
decode=${TKNP_405B_DECODE:-512}
tknp=${TKNP_405B_TKNP:-2}
(( nodes == 2 * tknp )) || { echo 'TP8 TKNP topology requires two four-GPU nodes per TKNP rank'; exit 2; }
export PYTHONPATH="$worktree"
# Preserve compile/JIT caches between srun containers and paired repetitions.
cache_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_microbatch_cache_20261002
export VLLM_CACHE_ROOT="$cache_root/405b_microbatch"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/405b_inductor_node${node_rank}"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export VLLM_TKNP_DBO_LAYOUT=asymmetric
export VLLM_TKNP_DBO_LOOKAHEAD=0
export VLLM_TKNP_DBO_SYMM_MEM=0
export VLLM_TKNP_DBO_INDEPENDENT_COMM=0
export VLLM_TKNP_DBO_INDEPENDENT_TP=0
export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
export VLLM_TKNP_DBO_ZERO_ROOT_C1=0
export VLLM_TKNP_DBO_C0_FRACTION=0.5
export VLLM_TKNP_DBO_C1_PRIORITY=0
unset VLLM_TKNP_DBO_ROOT_REQUESTS
dbo_args=()
case "$variant" in
  off) ;;
  zero_independent|pipelined|serial_root)
    dbo_args=(--enable-dbo)
    export VLLM_TKNP_DBO_INDEPENDENT_COMM=1
    export VLLM_TKNP_DBO_INDEPENDENT_TP=1
    export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1
    export VLLM_TKNP_DBO_ZERO_ROOT_C1=1
    export VLLM_TKNP_DBO_ROOT_REQUESTS=${TKNP_405B_ROOT_REQUESTS:-0}
    export VLLM_TKNP_DBO_C0_FRACTION=${TKNP_405B_C0_FRACTION:-0.5}
    export VLLM_TKNP_DBO_C1_PRIORITY=${TKNP_405B_C1_PRIORITY:-0}
    if [[ "$variant" == pipelined || "$variant" == serial_root ]]; then
      export VLLM_TKNP_DBO_LOOKAHEAD=1
    fi
    if [[ "$variant" == serial_root ]]; then
      export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
      export VLLM_TKNP_DBO_INDEPENDENT_TP=0
    fi
    ;;
  *) exit 2 ;;
esac
cd "$worktree"
mkdir -p "$point_root"
"$tknp_venv/bin/python3" -c 'import torch,vllm,os,json; print(vllm.__file__,torch.__version__,torch.cuda.nccl.version()); print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(("VLLM_TKNP", "NCCL_", "TKNP_"))},sort_keys=True))' > "$point_root/environment_node${node_rank}.log" 2>&1
sha1sum vllm/v1/worker/gpu_model_runner.py vllm/model_executor/layers/token_parallel_linear.py vllm/v1/worker/gpu_ubatch_wrapper.py vllm/forward_context.py vllm/v1/attention/backends/flashinfer.py vllm/config/parallel.py vllm/distributed/device_communicators/cuda_communicator.py >> "$point_root/environment_node${node_rank}.log"
profile_args=()
if [[ "${TKNP_405B_PROFILE:-0}" == 1 || "${TKNP_405B_PROFILE_AFTER:-0}" == 1 ]]; then
  profile_args=(--torch-profiler-dir "$point_root/traces" --torch-profiler-delay-iters 4 --torch-profiler-max-iters 8)
fi
if [[ "${TKNP_405B_PROFILE_AFTER:-0}" == 1 ]]; then
  profile_args+=(--profile-after-benchmark)
fi
if [[ "${TKNP_DUMP_FIRST_LOGITS:-0}" == 1 ]]; then
  profile_args+=(--dump-first-step-logits)
fi
log="$point_root/node${node_rank}.log"
timeout --signal=TERM --kill-after=30s 1800 "$tknp_venv/bin/python3" -m torch.distributed.run \
  --nnodes="$nodes" --nproc-per-node=4 --node-rank="$node_rank" \
  --rdzv-id="${SLURM_JOB_ID}_${TKNP_POINT_ID}" --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  examples/offline_inference/TKNP/tknp_inference_benchmarks.py \
  --model /home/sshrestha/workspace/2026/vllm-tknp-lyris-clean/examples/offline_inference/TKNP/model_configs/llama31_405b \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 8 --token-parallel-size "$tknp" \
  --batch-size "$batch" --seq-length "$context" --decode-tokens "$decode" \
  --kv-cache-dtype "${TKNP_405B_KV_DTYPE:-auto}" \
  --load-format dummy --cuda-graph --cuda-graph-mode FULL_DECODE_ONLY --steady-state-metrics --dump-token-ids \
  "${dbo_args[@]}" "${profile_args[@]}" \
  --cudagraph-capture-sizes "$batch" --gpu-memory-utilization 0.9 \
  --output-dir "$point_root" > "$log" 2>&1
replays=$(grep -c "TKNP_CUDAGRAPH_REPLAY padded_size=$batch" "$log" || true)
(( replays >= 4 )) || { echo "MISSING_GRAPH_REPLAY node=$node_rank count=$replays"; exit 1; }
if [[ "$variant" != off ]]; then
  replays=$(grep -c "TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=$batch" "$log" || true)
  (( replays >= 4 )) || { echo "MISSING_UBATCH_REPLAY node=$node_rank count=$replays"; exit 1; }
fi
touch "$point_root/NODE_${node_rank}_PASS"
grep 'STEADY_STATE_DECODE' "$log"
grep 'System Decode Throughput:' "$log" || true
echo "NODE_PASS variant=$variant node=$node_rank point=$point_root"

if [[ "${TKNP_405B_DENSITY_COMPANION:-0}" == 1 ]]; then
  # A second, explicitly matched precision/workload point amortizes allocation
  # and model startup costs. Recursive workers disable the companion so this
  # is exactly one extra point, not an open-ended sweep. Use a distinct store
  # and run ID while the preceding torchrun's sockets are being released.
  TKNP_405B_DENSITY_COMPANION=0 TKNP_405B_BATCH=1920 TKNP_405B_CONTEXT=8192 \
  TKNP_405B_DECODE=128 TKNP_405B_KV_DTYPE=fp8 TKNP_405B_ROOT_REQUESTS=64 \
  TKNP_405B_C0_FRACTION=0.5 TKNP_405B_PROFILE=0 TKNP_405B_PROFILE_AFTER=0 \
  TKNP_DUMP_FIRST_LOGITS=0 TKNP_POINT_ID="${TKNP_POINT_ID}_density" \
  MASTER_PORT=$((MASTER_PORT + 100)) \
  bash "$0" "$variant" "${point_root}_density"
fi
