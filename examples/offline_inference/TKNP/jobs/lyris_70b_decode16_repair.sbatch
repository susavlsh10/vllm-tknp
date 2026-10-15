#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-repair
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Existing four-node result root required}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --family decode --output "$tknp_root/decode" --tokens 257 --request-timeout 300)
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
# First gate the two diagnosed source repairs. Completed cases are reused;
# failures retain their original output and get a new retry-named directory.
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 32768 --batches 260 --methods dcp pp --max-hours 0.4
# Resumes only missing cells. Native replica controls are retained.
python3 "$tknp_entry" "${tknp_common[@]}" --regimes crossover --max-hours 1.9 \
  --methods dcp tp tknp-tp2 tknp-tp4 tknp-tp8 dp-native dp-lifetime dp-tp2 dp-tp4 dp-tp8 pp tp4-pp4 tp8-pp2
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 32768 --batches 64 --methods dcp --max-hours 0.15
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 98304 --batches 108 --methods dcp --max-hours 0.15
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
