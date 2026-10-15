#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:45:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-coverage
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Result root}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 1 2 5 6)
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/coverage-allocation.txt"
# Resume only missing primary cells, including the startup-port-failed hybrid.
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/fast_prefill" --tokens 4097 \
  --max-hours 1.15 --prefill-gpus 0 3 4 7 --regimes crossover \
  --methods tknp-tp2 tknp dp-tp2 tp pp-pipelined dp-native
# Repair the single failed earlier HTTP control without pooling experiment families.
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/e2e" --tokens 1025 \
  --max-hours 0.1 --contexts 32768 --batches 51 --methods dp-native
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
