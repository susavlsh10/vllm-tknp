#!/usr/bin/env bash
# Run from the salloc shell. Owns no allocation other than the supplied job.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Run inside the four-node salloc shell}"
: "${TKNP_EFFICIENCY_ROOT:?Set the frozen experiment directory}"
tknp_repo=${TKNP_EFFICIENCY_SOURCE:-$TKNP_EFFICIENCY_ROOT/source}
tknp_backend=${TKNP_EFFICIENCY_BACKEND:-FLASH_ATTN}
tknp_output=${TKNP_EFFICIENCY_OUTPUT:-$TKNP_EFFICIENCY_ROOT/native_fa3}
read -ra tknp_batches <<< "${TKNP_EFFICIENCY_BATCHES:-32 64 96 128}"
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#tknp_nodes[@]} == 4 ))
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_fp8_four_node_suite.py" verify --repo "$tknp_repo"
srun --overlap --mpi=pmix --cpu-bind=none -N4 -n4 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-name="fp8_decode_$SLURM_JOB_ID" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$tknp_repo" env TKNP_WORKTREE="$tknp_repo" \
  bash -c 'source examples/offline_inference/TKNP/jobs/eos_legacy_setup.sh
    "$TKNP_PYTHON" -m pytest --noconftest -q \
      tests/v1/worker/test_eos_405b_efficiency.py \
      tests/v1/worker/test_fp8_decode_sweep.py \
      tests/v1/worker/test_eos_efficiency.py' \
  > "$TKNP_EFFICIENCY_ROOT/bootstrap.log" 2>&1
if [[ "$tknp_backend" == FLASH_ATTN ]]; then
  tknp_diagnostic=(examples/offline_inference/TKNP/check_h100_fa3_decode.py --batches 4 32 --aot-splits 16)
elif [[ "$tknp_backend" == FLASHINFER ]]; then
  tknp_diagnostic=(examples/offline_inference/TKNP/check_h100_decode_kernels.py --q-heads 16 --kv-heads 1 --variants tc)
else
  printf 'Unsupported attention backend: %s\n' "$tknp_backend" >&2
  exit 2
fi
srun --overlap --mpi=pmix --cpu-bind=none -N1 -n1 --nodelist="${tknp_nodes[0]}" \
  --container-name="fp8_decode_$SLURM_JOB_ID" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$tknp_repo" env PYTHONPATH="$tknp_repo" \
    HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache HF_HUB_OFFLINE=1 \
  /home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3 \
    "${tknp_diagnostic[@]}" \
    --output "$TKNP_EFFICIENCY_ROOT/gqa16-numerics.json" \
  > "$TKNP_EFFICIENCY_ROOT/gqa16-numerics.log" 2>&1
# Short discovery first; no long sweep or asserted target speedup.
python3 "$tknp_repo/examples/offline_inference/TKNP/eos_405b_efficiency.py" \
  --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" \
  --model "$TKNP_EFFICIENCY_ROOT/model405b" --output "$tknp_output" \
  --attention-backend "$tknp_backend" --query-groups 0 \
  --stage matrix --methods tknp-tp8 tknp-tp8-dbo dcp dp-tp8 tp8-pp4 tp \
  --batches "${tknp_batches[@]}" --tokens "${TKNP_EFFICIENCY_TOKENS:-128}" \
  --context "${TKNP_EFFICIENCY_CONTEXT:-32768}" \
  > "$tknp_output.log" 2>&1
printf '405B_BASELINE_COMPLETE\n'
