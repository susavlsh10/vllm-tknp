#!/usr/bin/env bash
# Validate the bounded native P/D admission path, then rerun the same burst.
set -euo pipefail
tknp_job=${1:?existing job}
tknp_base=${2:?pilot base}
tknp_launcher=/home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP/jobs/lyris_e2e_interactive.sh
tknp_deadline=$((SECONDS + 2700))
until python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(not (len(p["attempts"]) == 6 and all(a["status"] == "completed" for a in p["attempts"]) and len(p.get("paired_tokens",[])) == 3))' "$tknp_base/repeat_capacity64/pilot.json" 2>/dev/null; do
    [[ $SECONDS -lt $tknp_deadline ]] || { echo "Selected repeats not complete; stop" >&2; exit 1; }
    sleep 15
done
bash "$tknp_launcher" "$tknp_job" readiness --section unit --output "$tknp_base/unit_final"
bash "$tknp_launcher" "$tknp_job" pilot --section queue --output "$tknp_base/queue32_lifetime" \
    --reference "$tknp_base/resident_capacity_tail" --contexts 32768 --tokens 4097 \
    --lifetime-admission --dp-reference "$tknp_base/queue32/c32768-b128-rep0-dp2"
echo "PILOT_CLEAN_QUEUE_COMPLETE"
