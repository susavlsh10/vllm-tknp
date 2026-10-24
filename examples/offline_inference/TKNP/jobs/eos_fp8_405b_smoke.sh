#!/usr/bin/env bash
# Run only inside an existing one-node EOS allocation/container. No submission.
set -euo pipefail
[[ $# == 1 && $1 == /* ]] || { echo 'Usage: eos_fp8_405b_smoke.sh ABS_NEW_RESULT_DIR' >&2; exit 2; }
result=$1
[[ $(hostname -s) =~ ^eos[0-9]+$ && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || {
  echo 'Requires one allocated EOS compute node' >&2; exit 2;
}
[[ ! -e $result ]] || { echo 'Use a new result directory' >&2; exit 2; }
mkdir -p "$result"
source "$(dirname "$0")/eos_legacy_setup.sh" > "$result/environment.log" 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
helper=examples/offline_inference/TKNP/fp8_405b_smoke.py
"$TKNP_PYTHON" -m pytest --noconftest -q tests/v1/worker/test_fp8_405b_smoke.py > "$result/unit.log" 2>&1
prepare_args=()
if [[ -n ${TKNP_FP8_PUBLIC_METADATA:-} ]]; then
  prepare_args+=(--metadata-json "$TKNP_FP8_PUBLIC_METADATA")
fi
"$TKNP_PYTHON" "$helper" prepare --model-dir "$result/model405b" "${prepare_args[@]}" > "$result/prepare.log" 2>&1
sha256sum "$helper" examples/offline_inference/TKNP/jobs/eos_fp8_405b_smoke.sh > "$result/helper_source.sha256"
timeout --signal=TERM --kill-after=30s 2700 "$TKNP_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=8 "$helper" run --model "$result/model405b" \
  --output "$result/probe" --contexts 512 32768 65536 98304 --batches 1 2 4 \
  --tokens 33 --gpu-memory-utilization 0.9 > "$result/probe.log" 2>&1
test -s "$result/probe/summary.json"
for ((rank=0; rank<8; rank++)); do test -s "$result/probe/resources_final_rank${rank}.json"; done
echo EOS_FP8_405B_SMOKE_PASS
