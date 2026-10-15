#!/usr/bin/env bash
# Temporary validation: controller was explicitly paused; always resume it.
set -euo pipefail
tknp_job=$1
tknp_controller=$2
tknp_root=$3
[[ "$tknp_job" =~ ^[0-9]+$ && "$tknp_controller" =~ ^[0-9]+$ ]] || exit 2
trap 'srun --jobid="$tknp_job" --overlap -N1 --ntasks=1 --nodelist=lyris0121 kill -CONT "$tknp_controller"' EXIT
tknp_summary="$tknp_root/multi/e2e/e2e-c32768-b51-o1025-tp/client/summary.json"
for ((tknp_wait=0; tknp_wait<120; tknp_wait++)); do
  [[ ! -f "$tknp_summary" ]] || break
  sleep 5
done
[[ -f "$tknp_summary" ]] || { echo 'Client still active; skip diagnostic and resume'; exit 1; }
timeout 600 python3 /home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP/e1_pp_validation.py \
  --job "$tknp_job" --nodes lyris0121 lyris0126 --output "$tknp_root/pp_validation"
