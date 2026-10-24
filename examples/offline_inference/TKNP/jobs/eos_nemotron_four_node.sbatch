#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:nemotron96k
set -Eeuo pipefail
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_source=$tknp_root/nemotron_four_node_20261023/source_v35
tknp_output=$tknp_root/nemotron_4node_96k_o1024_20261023_job$SLURM_JOB_ID
tknp_container=nemotron_four_$SLURM_JOB_ID
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo NEMOTRON_JOB_EXIT:$?; echo Slurm_releases_four_nodes' EXIT
# Materialize exactly one named container per node before any DP replicas.
srun --overlap -N4 -n4 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-name="$tknp_container" \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  nvidia-smi --query-gpu=name --format=csv,noheader
python3 "$tknp_source/examples/offline_inference/TKNP/nemotron_four_node_sweep.py" \
  --source "$tknp_source" --output "$tknp_output" --job "$SLURM_JOB_ID" \
  --nodes "${tknp_nodes[@]}" --container-name "$tknp_container" \
  --tokenizer "$tknp_root/nemotron_modelopt_20261023/tokenizer"
