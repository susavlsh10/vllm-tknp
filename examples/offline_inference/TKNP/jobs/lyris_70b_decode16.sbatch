#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-dcp16
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Set a unique Lustre result root}
mkdir -p "$tknp_root"
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/allocation.txt"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
tknp_methods=(dcp tp tknp tknp-tp2 tknp-tp4 tknp-tp8 dp-native dp-lifetime dp-tp2 dp-tp4 dp-tp8 pp tp4-pp4 tp8-pp2)
tknp_controls=(dcp tp tknp-tp8 dp-tp8 tp8-pp2 dp-native)
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/decode" --tokens 257 \
  --max-hours 3.0 --regimes crossover --methods "${tknp_methods[@]}"
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/decode" --tokens 257 \
  --max-hours 0.3 --contexts 32768 --regimes below --methods "${tknp_controls[@]}"
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/decode" --tokens 257 \
  --max-hours 0.35 --contexts 98304 --regimes overload --methods "${tknp_controls[@]}"
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
