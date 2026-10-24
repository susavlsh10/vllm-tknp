#!/usr/bin/env bash
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-fp8-4n
#SBATCH --array=0-5

set -Eeuo pipefail
umask 002
: "${TKNP_SWEEP_SUITE_ROOT:?Submit with eos_fp8_four_node_suite.py submit}"
: "${SLURM_ARRAY_TASK_ID:?An independent array element is required}"
methods=(dp-tp8 tknp-tp8 tknp-tp8-dbo tp8-pp4 tp dcp)
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < ${#methods[@]} ))
tknp_method=${methods[$SLURM_ARRAY_TASK_ID]}
tknp_repo=$TKNP_SWEEP_SUITE_ROOT/source
tknp_output=$TKNP_SWEEP_SUITE_ROOT/task${SLURM_ARRAY_TASK_ID}-${tknp_method}-job${SLURM_JOB_ID}
mkdir -p "$tknp_output"
tknp_child=

tknp_finish() {
  tknp_rc=$?
  trap - EXIT TERM INT
  if [[ -n $tknp_child ]] && kill -0 "$tknp_child" 2>/dev/null; then
    kill -INT "$tknp_child" 2>/dev/null || true
    for ((tknp_wait=0; tknp_wait<20; tknp_wait++)); do
      kill -0 "$tknp_child" 2>/dev/null || break
      sleep 1
    done
    kill -TERM "$tknp_child" 2>/dev/null || true
  fi
  python3 "$tknp_repo/examples/offline_inference/TKNP/eos_fp8_four_node_suite.py" finish \
    --result "$tknp_output" --return-code "$tknp_rc" || tknp_rc=1
  # Do not leave an interactive salloc shell. Exiting this batch element
  # releases its exclusive four nodes, including any surviving SLURM steps.
  exit "$tknp_rc"
}
trap tknp_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

python3 "$tknp_repo/examples/offline_inference/TKNP/eos_fp8_four_node_suite.py" verify --repo "$tknp_repo"
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#tknp_nodes[@]} == 4 ))
printf 'FOUR_NODE_EXPERIMENT method=%s job=%s nodes=%s\n' "$tknp_method" "$SLURM_JOB_ID" "${tknp_nodes[*]}"

srun --overlap --mpi=pmix --cpu-bind=none --nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-name="fp8_decode_$SLURM_JOB_ID" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$tknp_repo" env TKNP_WORKTREE="$tknp_repo" \
  bash -c 'source examples/offline_inference/TKNP/jobs/eos_legacy_setup.sh
    "$TKNP_PYTHON" -m pytest --noconftest -q \
      tests/v1/worker/test_fp8_405b_smoke.py \
      tests/v1/worker/test_fp8_decode_sweep.py \
      tests/v1/worker/test_tknp_graph_sweep.py \
      tests/v1/worker/test_eos_readiness.py \
      tests/v1/worker/test_eos_multinode_readiness.py' \
  > "$tknp_output/bootstrap.log" 2>&1

python3 "$tknp_repo/examples/offline_inference/TKNP/fp8_decode_sweep.py" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" \
  --model "$TKNP_SWEEP_SUITE_ROOT/model405b" --output "$tknp_output/sweep" \
  --methods "$tknp_method" --context 32768 --tokens 1024 --smoke-tokens 32 \
  --batches 4 8 16 32 64 128 > "$tknp_output/controller.log" 2>&1 &
tknp_child=$!
wait "$tknp_child"
tknp_child=
printf 'FOUR_NODE_EXPERIMENT_COMPLETE method=%s\n' "$tknp_method"
