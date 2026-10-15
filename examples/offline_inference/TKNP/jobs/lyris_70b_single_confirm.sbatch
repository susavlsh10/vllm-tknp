#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-confirm
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Result root}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 1 2 --prefill-gpus 0 3)
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/confirm-allocation.txt"
# Resume first: completed cells are reused, never silently replaced.
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_fast_prefill" --tokens 4097 \
  --max-hours 0.55 --contexts 65536 98304 --regimes crossover \
  --methods tknp tp pp-pipelined dp-native
# Confirm the capacity win against conservative DP, reversing pair two.
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_fast_prefill" --tokens 4097 \
  --max-hours 0.45 --contexts 32768 --batches 18 --methods tknp dp-lifetime --repeats 3
# A fit-in-both control is essential to show the actual crossover.
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_fast_prefill" --tokens 4097 \
  --max-hours 0.15 --contexts 32768 --batches 6 --methods tknp dp-lifetime
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
