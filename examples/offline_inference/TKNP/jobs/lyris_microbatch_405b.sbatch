#!/usr/bin/env bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=4
#SBATCH -p gb200
#SBATCH -J hw_nresearch_snoise-tknp:microbatch405b
#SBATCH -t 02:00:00
#SBATCH -o /home/sshrestha/workspace/2026/vllm-tknp-ubatch/ubatch_results/redesign_20261002/405b_%j.out
#SBATCH -e /home/sshrestha/workspace/2026/vllm-tknp-ubatch/ubatch_results/redesign_20261002/405b_%j.err
set -euo pipefail
worktree=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-arm64-v0.27.0.sqsh
mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export RESULT_ROOT=${RESULT_ROOT:-${TKNP_405B_RESULT_BASE:-$worktree/ubatch_results/redesign_20261002}/405b_job_$SLURM_JOB_ID}
mkdir -p "$RESULT_ROOT"
printf 'job=%s\nnodes=%s\nmodel=405b_architecture_dummy_bf16\ntp=8\ntknp=%s\nbatch=%s\ncontext=%s\ndecode=%s\n' "$SLURM_JOB_ID" "$SLURM_NODELIST" "${TKNP_405B_TKNP:-2}" "${TKNP_405B_BATCH:-256}" "${TKNP_405B_CONTEXT:-8192}" "${TKNP_405B_DECODE:-512}" > "$RESULT_ROOT/job_manifest.txt"
point_index=0
for repeat in $(seq 1 "${TKNP_405B_REPEATS:-3}"); do
  # Reverse the second pair to reduce order/drift bias.
  read -r -a variants <<< "${TKNP_405B_VARIANTS:-off zero_independent}"
  for (( index=0; index < ${#variants[@]}; index++ )); do
    chosen=$index
    if (( repeat % 2 == 0 )); then chosen=$((${#variants[@]} - 1 - index)); fi
    variant=${variants[$chosen]}
    point_index=$((point_index + 1))
    export TKNP_POINT_ID="${repeat}_${variant}"
    export MASTER_PORT=$((23000 + SLURM_JOB_ID % 18000 + point_index))
    point_root="$RESULT_ROOT/repeat${repeat}_${variant}"
    srun --mpi=pmix --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
      --container-image="$image" --container-mounts="$mounts" --container-workdir="$worktree" \
      --export=ALL \
      bash examples/offline_inference/TKNP/jobs/lyris_microbatch_405b_worker.sh "$variant" "$point_root"
    for (( node=0; node < SLURM_JOB_NUM_NODES; node++ )); do test -f "$point_root/NODE_${node}_PASS"; done
  done
done
touch "$RESULT_ROOT/JOB_SUCCESS"
echo "JOB_SUCCESS result=$RESULT_ROOT"
