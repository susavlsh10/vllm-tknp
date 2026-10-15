#!/usr/bin/env bash
# Confirm the positive matched-layout point and largest resident control.
# No new allocation: wait for the sequential real P/D queue tests to drain.
set -euo pipefail
tknp_job=${1:?existing job}
tknp_base=${2:?pilot base}
tknp_launcher=/home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP/jobs/lyris_e2e_interactive.sh
tknp_deadline=$((SECONDS + 3600))
until python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(not (len(p["attempts"]) == 2 and all(a["status"] == "completed" for a in p["attempts"])))' "$tknp_base/queue32/pilot.json" 2>/dev/null; do
    [[ $SECONDS -lt $tknp_deadline ]] || { echo "Queue tests not complete; stop" >&2; exit 1; }
    sleep 15
done
sleep 3
bash "$tknp_launcher" "$tknp_job" pilot --section resident --output "$tknp_base/repeat_quarter32" \
    --reference "$tknp_base/resident" --contexts 32768 --batches 32 --tokens 129 --layout quarter-root --repetitions 3
bash "$tknp_launcher" "$tknp_job" pilot --section resident --output "$tknp_base/repeat_capacity64" \
    --reference "$tknp_base/resident_capacity_tail" --contexts 32768 --batches 64 --tokens 129 --repetitions 3
echo "PILOT_REPEATS_COMPLETE"
