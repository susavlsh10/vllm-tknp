#!/usr/bin/env bash
# Run sequentially within one ALREADY allocated node; no sbatch/allocation.
set -euo pipefail
tknp_job=${1:?existing single-node job}
tknp_base=${2:?new pilot base directory}
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_launcher="$tknp_repo/examples/offline_inference/TKNP/jobs/lyris_e2e_interactive.sh"
# Wait for the first driver's final 96K capacity stop, never overlap GPU trials.
tknp_deadline=$((SECONDS + 900))
until python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(not (all(a["status"] == "completed" for a in p["attempts"]) and any(r["context"] == 98304 for r in p["capacity_stops"])))' "$tknp_base/resident/pilot.json"; do
    [[ $SECONDS -lt $tknp_deadline ]] || { echo "Initial sweep not complete; stop" >&2; exit 1; }
    sleep 10
done
bash "$tknp_launcher" "$tknp_job" pilot --section resident --output "$tknp_base/resident_capacity_tail" \
    --reference "$tknp_base/resident" --contexts 32768 --batches 64 128 --tokens 129
bash "$tknp_launcher" "$tknp_job" pilot --section resident --output "$tknp_base/resident_quarter_root" \
    --reference "$tknp_base/resident" --contexts 32768 98304 --batches 16 32 --tokens 129 --layout quarter-root
bash "$tknp_launcher" "$tknp_job" pilot --section proxy --output "$tknp_base/proxy32" \
    --reference "$tknp_base/resident_capacity_tail" --contexts 32768 --tokens 129
bash "$tknp_launcher" "$tknp_job" pilot --section proxy --output "$tknp_base/proxy96" \
    --reference "$tknp_base/resident" --contexts 98304 --tokens 129
# Long forced outputs make both decoder queues fill despite real-P pacing.
# Identical prompt/arrival/output traces, two decode GPUs, utilization .8.
bash "$tknp_launcher" "$tknp_job" pilot --section queue --output "$tknp_base/queue32" \
    --reference "$tknp_base/resident_capacity_tail" --contexts 32768 --tokens 4097
echo "PILOT_FOLLOWUPS_COMPLETE"
