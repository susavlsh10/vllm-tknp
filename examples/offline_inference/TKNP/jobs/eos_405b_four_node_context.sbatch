#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-context
set -Eeuo pipefail
tknp_context=${1:?Context length required}
case "$tknp_context" in 32768|65536|98304) ;; *) exit 2 ;; esac
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results
tknp_parent=$tknp_root/efficiency405b_20261018_job6278892
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
trap 'echo CONTEXT_SWEEP_EXIT:$?; echo SLURM_releases_four_nodes' EXIT
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_four_node_context.py" \
  --source "$tknp_parent/source_pp_mqbind_v27" --parent "$tknp_parent" \
  --output "$tknp_root/decode405b_4node_c${tknp_context}_o1024_20261020_job$SLURM_JOB_ID" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --context "$tknp_context"
