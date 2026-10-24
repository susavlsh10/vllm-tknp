#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-pp8-retry
set -Eeuo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo PP8_RETRY_EXIT:$?; echo SLURM_releases_eight_nodes' EXIT
srun --overlap --mpi=pmix --cpu-bind=none -N8 -n8 --ntasks-per-node=1 \
  --container-name="fp8_decode_$SLURM_JOB_ID" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  /home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3 \
  "$tknp_repo/examples/offline_inference/TKNP/test_mq_atomic_bind.py"
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_eight_node_decode.py" \
  --source "$tknp_parent/source_pp_mqbind_v27" --parent "$tknp_parent" \
  --output "$tknp_root/decode405b_8node_pp_retry_20261020_job$SLURM_JOB_ID" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --methods tp8-pp8
