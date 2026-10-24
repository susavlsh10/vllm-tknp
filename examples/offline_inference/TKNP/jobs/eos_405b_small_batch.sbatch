#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-smallbatch
set -Euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
tknp_source=$tknp_parent/source_standard_pp_v26
tknp_output=$tknp_root/smallbatch405b_20261019_job$SLURM_JOB_ID
mkdir -p "$tknp_output"
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo SMALL_BATCH_EXIT:$?; echo SLURM_releases_four_nodes' EXIT
tknp_failed=0
for tknp_method in dp-tp8 tknp-tp8 dcp tp; do
  echo METHOD_START:"$tknp_method"
  if python3 "$tknp_source/examples/offline_inference/TKNP/eos_405b_efficiency.py" \
    --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --model "$tknp_parent/model405b" \
    --output "$tknp_output/$tknp_method" --stage matrix --methods "$tknp_method" \
    --context 32768 --tokens 128 --batches 4 8 16 --repeats 1 \
    --attention-backend FLASH_ATTN > "$tknp_output/$tknp_method.log" 2>&1; then
    echo METHOD_PASS:"$tknp_method"
  else
    echo METHOD_FAILED:"$tknp_method"
    tknp_failed=1
  fi
done
echo METHOD_START:pp-scheduling-only
if python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_pp_scheduling_only.py" \
  --source "$tknp_source" --parent "$tknp_parent" --job "$SLURM_JOB_ID" \
  --nodes "${tknp_nodes[@]}" --output "$tknp_output/pp-scheduling-only" \
  --batches 4 8 16 --repeats 1 > "$tknp_output/pp-scheduling-only.log" 2>&1; then
  echo METHOD_PASS:pp-scheduling-only
else
  echo METHOD_FAILED:pp-scheduling-only
  tknp_failed=1
fi
echo SMALL_BATCH_COMPLETE:failures="$tknp_failed"
exit "$tknp_failed"
