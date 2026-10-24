#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:hetero-scheduler
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_source=$tknp_root/efficiency405b_20261018_job6278892/source_hetero_scheduler_v29
tknp_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo SCHEDULER_TEST_EXIT:$?; echo SLURM_releases_one_node' EXIT
for tknp_variant in fifo window root-aware; do
 python3 "$tknp_repo/examples/offline_inference/TKNP/eos_heterogeneous_scheduler.py" \
  --source "$tknp_source" --job "$SLURM_JOB_ID" --node "$tknp_node" --variant "$tknp_variant" \
  --output "$tknp_root/hetero_scheduler_20261021_job${SLURM_JOB_ID}_$tknp_variant"
done
