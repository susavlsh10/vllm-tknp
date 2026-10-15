#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-matrix
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Set a unique Lustre result root}
mkdir -p "$tknp_root"
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/allocation.txt"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 1 2 5 6)
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/e2e" --tokens 1025 \
  --max-hours 2.2 --regimes crossover \
  --methods tknp tknp-tp2 dp-native dp-lifetime dp-tp2 tp pp
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/decode" --tokens 257 \
  --max-hours 0.7 --regimes crossover \
  --methods tknp tknp-tp2 dp-native dp-lifetime dp-tp2 tp pp dcp
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/e2e" --tokens 1025 \
  --max-hours 0.4 --contexts 32768 --regimes below overload \
  --methods tknp tknp-tp2 dp-native dp-lifetime dp-tp2 tp pp
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
