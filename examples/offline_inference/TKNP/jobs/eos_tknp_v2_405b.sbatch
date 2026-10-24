#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:tknp-v2-405b
set -Eeuo pipefail
tknp_source=${1:?Frozen validated source required}
tknp_output=${2:?Unique result directory required}
tknp_readiness=${3:?Single-node readiness gate required}
tknp_parent=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/efficiency405b_20261018_job6278892
tknp_baseline=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/decode405b_4node_c32768_o1024_20261020_job6299009/summary.json
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo TKNP_V2_EXIT:$?; echo SLURM_releases_four_nodes' EXIT
# Materialize one named container per node before any concurrent worker launch.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --mpi=pmix \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-name="tknpv2_$SLURM_JOB_ID" \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  /bin/true
python3 -u "$tknp_source/examples/offline_inference/TKNP/tknp_v2_405b_sweep.py" \
  --source "$tknp_source" --parent "$tknp_parent" --output "$tknp_output" \
  --baseline "$tknp_baseline" --readiness "$tknp_readiness" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}"
