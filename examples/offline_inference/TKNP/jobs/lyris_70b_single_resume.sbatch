#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-resume
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Result root}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 1 2)
printf '%s\n' "job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT nodes=${tknp_nodes[*]}" > "$tknp_root/single-resume-allocation.txt"
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_main" --tokens 1025 --max-hours 0.1 \
  --contexts 32768 --batches 19 --methods tknp
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_fast_prefill" --tokens 4097 --max-hours 0.5 \
  --prefill-gpus 0 3 --contexts 32768 --regimes crossover --methods tknp dp-native tp pp-pipelined dp-lifetime
python3 "$tknp_entry" "${tknp_common[@]}" --output "$tknp_root/single_fast_prefill" --tokens 4097 --max-hours 0.75 \
  --prefill-gpus 0 3 --contexts 65536 98304 --regimes crossover --methods tknp dp-native tp pp-pipelined
python3 "$tknp_entry" "${tknp_common[@]}" --family decode --output "$tknp_root/single_decode" --tokens 257 \
  --max-hours 0.15 --contexts 32768 --regimes crossover --methods tknp dp-native dp-lifetime tp pp
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
