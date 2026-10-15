#!/usr/bin/env bash
# One torchrun agent per allocated node. Never allocates or submits a job.
set -euo pipefail

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
if (( $# != 5 )); then
  echo 'Usage: bash eos_microbatch_worker.sh [--dry-run] ABS_RESULT_DIR off|serial_root|zero_independent|pipelined BATCH CONTEXT DECODE' >&2
  exit 2
fi
result_root=$1 variant=$2 batch=$3 context=$4 decode=$5
worktree=${TKNP_WORKTREE:?Set TKNP_WORKTREE to this checkout on EOS}
tknp_python=${TKNP_PYTHON:?Set TKNP_PYTHON to the compatible x86 CUDA environment}
model=${TKNP_MODEL:?Set TKNP_MODEL to a cached model/config path or repository ID}
tp=${TKNP_TP_SIZE:-8}
tknp=${TKNP_WORLD_SIZE:-2}
nodes=${TKNP_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
node_rank=${TKNP_NODE_RANK:-${SLURM_PROCID:-0}}
local_gpus=${TKNP_LOCAL_GPUS:?Set TKNP_LOCAL_GPUS to visible GPUs per participating node}
for value in "$batch" "$context" "$decode" "$tp" "$tknp" "$nodes" "$local_gpus"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'Positive integer topology/workload required' >&2; exit 2; }
done
[[ $node_rank =~ ^(0|[1-9][0-9]*)$ ]] && (( node_rank < nodes )) || exit 2
[[ $result_root == /* && $worktree == /* && $tknp_python == /* ]] || { echo 'Absolute result, checkout and Python paths required' >&2; exit 2; }
(( tp * tknp == nodes * local_gpus && tknp >= 2 )) || { echo 'TP x TKNP must equal all participating GPUs, with TKNP >= 2' >&2; exit 2; }
(( batch % tknp == 0 && batch >= 4 && decode >= 16 )) || { echo 'Batch must divide into TKNP lanes; at least four requests and sixteen decode steps required' >&2; exit 2; }
if (( tp != 8 )) && [[ ${TKNP_DEVELOPMENT_SMOKE:-0} != 1 ]]; then
  echo 'Paper runs require TP8; set TKNP_DEVELOPMENT_SMOKE=1 only for a small-model functional test' >&2
  exit 2
fi
if (( nodes > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR to the first participating compute node}"
  : "${MASTER_PORT:?Set one shared, unused rendezvous port}"
fi

# Keep off/on transport settings explicit and do not inherit old recipe flags.
export PYTHONPATH="$worktree"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export VLLM_TKNP_DBO_LAYOUT=asymmetric
export VLLM_TKNP_DBO_SYMM_MEM=0
export VLLM_TKNP_DBO_LOOKAHEAD=0
export VLLM_TKNP_DBO_INDEPENDENT_COMM=0
export VLLM_TKNP_DBO_INDEPENDENT_TP=0
export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=0
export VLLM_TKNP_DBO_ZERO_ROOT_C1=0
export VLLM_TKNP_DBO_ROOT_REQUESTS=${TKNP_ROOT_REQUESTS:-0}
export VLLM_TKNP_DBO_C0_FRACTION=${TKNP_C0_FRACTION:-0.5}
export VLLM_TKNP_DBO_C1_PRIORITY=${TKNP_C1_PRIORITY:-0}
export VLLM_TKNP_DBO_INDEPENDENT_TP_SYMM=${TKNP_PRIVATE_TP_SYMM:-1}
case "$variant" in
  off) ;;
  serial_root|zero_independent|pipelined)
    export VLLM_TKNP_DBO_INDEPENDENT_COMM=1
    export VLLM_TKNP_DBO_ZERO_ROOT_C1=1
    if [[ $variant != serial_root ]]; then
      export VLLM_TKNP_DBO_INDEPENDENT_TP=1
      export VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE=1
    fi
    if [[ $variant != zero_independent ]]; then export VLLM_TKNP_DBO_LOOKAHEAD=1; fi
    ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac

command=("$tknp_python" -m torch.distributed.run --nnodes="$nodes"
         --nproc-per-node="$local_gpus" --node-rank="$node_rank")
if (( nodes == 1 )); then
  command+=(--standalone)
else
  command+=(--rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
            --rdzv-id="${TKNP_POINT_ID:-${SLURM_JOB_ID:-manual}_${variant}_${batch}_${context}}")
fi
command+=("${TKNP_BENCHMARK_SCRIPT:-examples/offline_inference/TKNP/tknp_inference_benchmarks.py}"
          --model "$model" --tensor-parallel-size "$tp" --token-parallel-size "$tknp"
          --batch-size "$batch" --seq-length "$context" --decode-tokens "$decode"
          --kv-cache-dtype auto --load-format "${TKNP_LOAD_FORMAT:-dummy}"
          --cuda-graph --cuda-graph-mode FULL_DECODE_ONLY
          --cudagraph-capture-sizes "$batch" --gpu-memory-utilization "${TKNP_MEMORY_UTILIZATION:-0.9}"
          --steady-state-metrics --dump-token-ids --output-dir "$result_root")
if [[ $variant != off ]]; then command+=(--enable-dbo); fi
if [[ -n ${TKNP_TOKENIZER:-} ]]; then command+=(--tokenizer "$TKNP_TOKENIZER"); fi
if [[ ${TKNP_PROFILE:-0} == 1 ]]; then
  command+=(--torch-profiler-dir "$result_root/traces"
            --torch-profiler-delay-iters 4 --torch-profiler-max-iters 8)
fi
if (( dry_run )); then
  printf 'WORKTREE=%q\nVARIANT=%q\nNODE_RANK=%q\n' "$worktree" "$variant" "$node_rank"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "$worktree"
mkdir -p "$result_root"
log="$result_root/node${node_rank}.log"
[[ ! -e $log && ! -e $result_root/steady_rank$((node_rank * local_gpus)).json ]] || {
  echo 'Result already exists; use a fresh directory for each point' >&2; exit 2;
}
"$tknp_python" - <<'PY' > "$result_root/environment_node${node_rank}.json"
import importlib.metadata as metadata
import json, os, platform, subprocess
import torch, vllm
packages = {}
for name in ('torch', 'flashinfer-python', 'transformers', 'nixl'):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps(dict(architecture=platform.machine(), packages=packages,
    vllm_file=vllm.__file__, torch_cuda=torch.version.cuda,
    nccl=torch.cuda.nccl.version(),
    gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True),
    recipe={k: v for k, v in os.environ.items() if k.startswith('VLLM_TKNP_DBO_')}), indent=2))
PY
timeout --signal=TERM --kill-after=30s "${TKNP_POINT_TIMEOUT:-1800}" "${command[@]}" > "$log" 2>&1
for ((rank=node_rank * local_gpus; rank < (node_rank + 1) * local_gpus; rank++)); do
  test -s "$result_root/steady_rank${rank}.json"
done
replays=$(grep -c "TKNP_CUDAGRAPH_REPLAY padded_size=$batch" "$log" || true)
(( replays >= local_gpus )) || { echo "MISSING_GRAPH_REPLAY node=$node_rank count=$replays" >&2; exit 1; }
if [[ $variant != off ]]; then
  replays=$(grep -c "TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=$batch" "$log" || true)
  (( replays >= local_gpus )) || { echo "MISSING_MICROBATCH_REPLAY node=$node_rank count=$replays" >&2; exit 1; }
fi
grep 'STEADY_STATE_DECODE' "$log"
printf 'NODE_PASS variant=%s node=%s result=%s\n' "$variant" "$node_rank" "$result_root"
