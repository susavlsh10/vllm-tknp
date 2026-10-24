#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:hetero-validation
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_source=$tknp_root/efficiency405b_20261018_job6278892/source_hetero_validation_v28
tknp_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo HETERO_VALIDATION_EXIT:$?; echo SLURM_releases_one_node' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/test_heterogeneous_decode.py"
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_heterogeneous_validation.py" \
  --source "$tknp_source" --job "$SLURM_JOB_ID" --node "$tknp_node" \
  --output "$tknp_root/heterogeneous_8b_20261021_job$SLURM_JOB_ID"
