#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:nemotron4-v2
set -Eeuo pipefail
nemotron_source=${1:?Frozen source required}
nemotron_parent=${2:?Model/tokenizer metadata parent required}
nemotron_gate=${3:?Passed single-node gate required}
nemotron_output=${4:?Unique result directory required}
nemotron_method=${5:?Method required}
mapfile -t nemotron_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nemotron_nodes[@]} == 4 ]]
trap 'echo NEMOTRON4_EXIT:$?; echo SLURM_releases_four_owned_nodes' EXIT
# DP replicas must not race named-container extraction.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --mpi=pmix \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-name="nemotron4_$SLURM_JOB_ID" \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  /bin/true
python3 -u "$nemotron_source/examples/offline_inference/TKNP/nemotron4_sweep.py" \
  --source "$nemotron_source" --parent "$nemotron_parent" --readiness "$nemotron_gate" \
  --output "$nemotron_output" --method "$nemotron_method" \
  --job "$SLURM_JOB_ID" --nodes "${nemotron_nodes[@]}"
