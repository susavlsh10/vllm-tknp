#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:pp-ablation
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_source=$tknp_root/efficiency405b_20261018_job6278892/source_standard_pp_v26
tknp_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo PP_ABLATION_EXIT:$?; echo SLURM_releases_owned_node' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_pp_ablation.py" \
  --source "$tknp_source" --job "$SLURM_JOB_ID" --node "$tknp_node" \
  --output "$tknp_root/pp_ablation_20261019_job$SLURM_JOB_ID"
