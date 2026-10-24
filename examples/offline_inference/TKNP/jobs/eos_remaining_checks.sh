#!/usr/bin/env bash
# Finish the bounded readiness campaign after decode70b_85pct passes.
set -euo pipefail
(( $# == 6 )) || { echo 'Usage: JOB RESULT PREFILL_NODE DECODE0_NODE DECODE1_NODE MODEL70' >&2; exit 2; }
job=$1 result=$2 pnode=$3 dnode0=$4 dnode1=$5 model70=$6
cd "$(dirname "$0")/../../../.."
here=examples/offline_inference/TKNP
common=(--job "$job" --nodes "$pnode" "$dnode0" "$dnode1")
python3 -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); s=json.loads((r/"decode70b_85pct/readiness.json").read_text()); good={v["key"] for v in s["attempts"] if v["status"]=="passed"}; assert len(good)==15; c=json.loads((r/"decode70b_85pct/matrix-c32768.json").read_text()); c["runtime"]["container_name"]="tknp_ready_"+str(c["slurm_job_id"]); c["runtime"]["env"]["UCX_TLS"]="^gdr_copy"; (r/"runtime85.json").write_text(json.dumps(c["runtime"],indent=2)+"\n"); (r/"pair_config85.json").write_text(json.dumps(c,indent=2)+"\n")' "$result"
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/thread_device_fix90" \
  --family decode --contexts 32768 --tokens 33 --count 16 --lanes 8 --methods tknp-dbo --gpu-memory-utilization 0.9
if [[ ! -s "$result/pair70b_85pct/pair.json" ]]; then
  python3 "$here/eos_multinode_pair.py" --config "$result/pair_config85.json" --output "$result/pair70b_85pct"
fi
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/mixed70b_85pct" \
  --family decode --contexts 98304 --mixed-lengths 32768 65536 98304 --rate 20 \
  --tokens 33 --count 16 --lanes 8 --methods tknp tknp-dbo --gpu-memory-utilization 0.85 --lookahead 1
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/e2e70b_supported_85pct" \
  --family e2e --contexts 32768 --tokens 33 --count 8 --lanes 8 --methods tknp tknp-dbo dp pp --staged-decode --staged-skip-pp --gpu-memory-utilization 0.85 --lookahead 1
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/resident_synthetic70b_supported_85pct" \
  --family decode --contexts 32768 --tokens 33 --count 8 --lanes 8 --methods tknp tknp-dbo dp pp --staged-decode --staged-skip-pp --gpu-memory-utilization 0.85 --lookahead 1
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/e2e70b96k_supported_85pct" \
  --family e2e --contexts 98304 --tokens 33 --count 8 --lanes 8 --methods tknp-dbo pp --staged-decode --staged-skip-pp --gpu-memory-utilization 0.85 --lookahead 1
python3 "$here/e1_70b_suite.py" "${common[@]}" --model "$model70" --runtime-json "$result/runtime85.json" \
  --gpus-per-node 8 --prefill-tp 8 --prefill-gpus 0 1 2 3 4 5 6 7 \
  --decoder-gpus 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --output "$result/capacity70b_85pct" --family decode --contexts 32768 --tokens 33 \
  --methods tknp-tp8 dp-tp8 tp8-pp2 dcp --regimes overload --max-hours 1 --gpu-memory-utilization 0.85
python3 "$here/eos_crossover_readiness.py" --suite "$result/capacity70b_85pct" --output "$result/crossover70b_85pct"
echo EOS_REMAINING_CHECKS_FINISHED
