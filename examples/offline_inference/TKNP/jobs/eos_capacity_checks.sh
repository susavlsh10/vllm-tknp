#!/usr/bin/env bash
# Decode-only continuation: exactly two allocated 8-H100 nodes, no prefill node.
set -euo pipefail
(( $# == 5 )) || { echo 'Usage: JOB ABS_RESULT NODE0 NODE1 ABS_MODEL70' >&2; exit 2; }
job=$1 result=$2 node0=$3 node1=$4 model70=$5
cd "$(dirname "$0")/../../../.."
here=examples/offline_inference/TKNP
python3 -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); c=json.loads((r/"runtime85.json").read_text()); c["container_name"]="tknp_ready_"+sys.argv[2]; c["env"]["UCX_TLS"]="^gdr_copy"; (r/"runtime_capacity.json").write_text(json.dumps(c,indent=2)+"\n")' "$result" "$job"
python3 "$here/e1_70b_suite.py" --job "$job" --nodes "$node0" "$node1" \
  --model "$model70" --runtime-json "$result/runtime_capacity.json" \
  --gpus-per-node 8 --prefill-tp 8 --decoder-gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --output "$result/capacity70b_85pct" --family decode --contexts 32768 --tokens 33 \
  --methods tknp-tp8 dp-tp8 tp8-pp2 dcp --regimes overload --max-hours 0.8 \
  --request-timeout 900 --gpu-memory-utilization 0.85
python3 "$here/eos_crossover_readiness.py" --suite "$result/capacity70b_85pct" --output "$result/crossover70b_85pct"
echo EOS_CAPACITY_CHECKS_FINISHED
