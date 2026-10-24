#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:hetero-dp-tp
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo DP_TP_RETRY_EXIT:$?; echo SLURM_releases_four_nodes' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_heterogeneous.py" \
  --source "$tknp_parent/source_hetero_actual_admission_v30" --parent "$tknp_parent" \
  --output "$tknp_root/hetero405b_dp_tp_actual_20261021_job$SLURM_JOB_ID" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" \
  --methods dp-tp8 tp --actual-length-admission \
  --reference-summary "$tknp_root/hetero405b_4node_20261021_job6312408/summary.json"
