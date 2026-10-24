#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:70b96k-capacity
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_source=$tknp_root/efficiency405b_20261018_job6278892/source_hetero_actual_admission_v30
tknp_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo CAPACITY_70B_EXIT:$?; echo SLURM_releases_one_node' EXIT
# Materialize the shared named container once before the two DP replicas
# launch concurrently; otherwise Pyxis extraction races on the same path.
srun --overlap --nodes=1 --ntasks=1 --nodelist="$tknp_node" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-name="tknp_efficiency$SLURM_JOB_ID" \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre:/lustre \
  /bin/true
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_70b_capacity_single.py" \
  --source "$tknp_source" --job "$SLURM_JOB_ID" --node "$tknp_node" \
  --output "$tknp_root/capacity70b_single_96k_20261022_job$SLURM_JOB_ID"
