#!/usr/bin/env bash
set -euo pipefail
tknp_nodes=$1
tknp_local_gpus=$2
tknp_master=$3
tknp_port=$4
tknp_style=$5
shift 5
export VLLM_NIXL_SIDE_CHANNEL_HOST="$(hostname -s)"
tknp_python=${TKNP_PYTHON:-/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv/bin/python3}
# This launcher is dedicated to the matched collection suite. Ordinary
# baselines use native graph buckets, even when resuming an older controller.
tknp_args=("$@")
tknp_token_parallel=0
for ((tknp_i=0; tknp_i<${#tknp_args[@]}-1; tknp_i++)); do
  if [[ "${tknp_args[$tknp_i]}" == --tknp ]] && (( tknp_args[tknp_i+1] > 1 )); then
    tknp_token_parallel=1
  fi
done
if (( ! tknp_token_parallel )); then
  set -- "$@" --native-graph-shapes
  if (( tknp_nodes > 1 )); then
    set -- "$@" --nnodes "$tknp_nodes" --node-rank "${SLURM_PROCID:-0}" \
      --master-addr "$tknp_master" --master-port "$tknp_port"
  fi
  exec "$tknp_python" "$@"
fi
if [[ "$tknp_style" == external ]]; then
  exec "$tknp_python" -m torch.distributed.run --nnodes="$tknp_nodes" \
    --nproc-per-node="$tknp_local_gpus" --node-rank="${SLURM_PROCID:-0}" \
    --rdzv-backend=c10d --rdzv-endpoint="$tknp_master:$tknp_port" "$@"
fi
exec "$tknp_python" "$@"
