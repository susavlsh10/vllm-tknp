#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 2
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:15:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-multi-resume
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Result root}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 1 2 5 6)
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/resume-allocation.txt"
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/e2e" --tokens 1025 --max-hours 0.2 \
  --contexts 32768 --batches 51 --methods tknp tknp-tp2 dp-native
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/fast_prefill" --tokens 4097 --max-hours 0.65 \
  --prefill-gpus 0 3 4 7 --contexts 32768 --regimes crossover \
  --methods tknp-tp2 dp-native dp-tp2 tp pp-pipelined tknp dp-lifetime
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/fast_prefill" --tokens 4097 --max-hours 0.95 \
  --prefill-gpus 0 3 4 7 --contexts 65536 98304 --regimes crossover \
  --methods tknp-tp2 dp-native dp-tp2 tp pp-pipelined
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
