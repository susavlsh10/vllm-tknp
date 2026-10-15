#!/usr/bin/env bash
# Login-host controller for one EXISTING node. Never allocates or submits.
set -euo pipefail
tknp_job=${1:?Existing allocation}
tknp_node=${2:?Allocated node}
tknp_root=${3:?Unique output root}
[[ "$tknp_job" =~ ^[0-9]+$ ]] || exit 2
[[ "$(squeue -h -j "$tknp_job" -o '%T %D %a')" == 'RUNNING 1 coreai_comparch_inferencex' ]] || exit 2
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
tknp_common=(--job "$tknp_job" --nodes "$tknp_node" --decoder-gpus 1 2)
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_main" \
  --tokens 1025 --max-hours 1.2 --regimes crossover \
  --methods tknp dp-native dp-lifetime tp pp
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/single_decode" \
  --tokens 257 --max-hours 0.2 --contexts 32768 --regimes crossover --methods tknp dp-native dp-lifetime tp pp
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
# Release only this explicitly supplied experiment allocation after collection.
scancel "$tknp_job"
