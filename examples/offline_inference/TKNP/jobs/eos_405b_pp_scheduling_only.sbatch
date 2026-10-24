#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-pp-sched
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo PP_SCHED_ONLY_EXIT:$?; echo SLURM_releases_four_nodes' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_pp_scheduling_only.py" \
  --source "$tknp_parent/source_standard_pp_v26" --parent "$tknp_parent" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" \
  --output "$tknp_root/pp405b_scheduling_only_20261019_job$SLURM_JOB_ID"
