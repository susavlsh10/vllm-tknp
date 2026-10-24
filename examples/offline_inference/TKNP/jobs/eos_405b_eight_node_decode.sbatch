#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-8node
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo EIGHT_NODE_EXIT:$?; echo SLURM_releases_eight_nodes' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_eight_node_decode.py" \
  --source "$tknp_parent/source_standard_pp_v26" --parent "$tknp_parent" \
  --output "$tknp_root/decode405b_8node_20261020_job$SLURM_JOB_ID" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}"
