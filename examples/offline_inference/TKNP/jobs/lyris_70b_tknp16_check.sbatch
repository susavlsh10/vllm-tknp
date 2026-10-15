#!/usr/bin/env bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p gb200
#SBATCH -N 4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -J coreai_comparch_inferencex-tknp.70b-ncclcheck
set -euo pipefail
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_root=${RESULT_ROOT:?Existing four-node result root required}
mapfile -t tknp_nodes < <(scontrol show hostname "$SLURM_NODELIST")
export NCCL_DEBUG=INFO
tknp_common=(--job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --decoder-gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --family decode --output "$tknp_root/decode" --tokens 257 --request-timeout 300 --methods tknp)
tknp_entry="$tknp_repo/examples/offline_inference/TKNP/e1_70b_suite.py"
# One bounded diagnostic on a fresh allocation. Do not repeatedly run a
# broken topology or relabel an eager fallback as a CUDA-graph result.
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 32768 --probe-only --max-hours 0.12
python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); assert any(a["status"]=="completed" and a["name"].startswith("probe-decode-c32768-tknp") and "-tp" not in a["name"] for a in s["attempts"]), "TP1 x TKNP16 CUDA-graph diagnostic failed; inspect retained NCCL INFO log"' "$tknp_root/decode/suite.json"
# Disable verbose NCCL logging in measured runs, retaining its diagnostic.
unset NCCL_DEBUG
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 32768 --batches 260 --max-hours 0.18
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 65536 --batches 130 --max-hours 0.18
python3 "$tknp_entry" "${tknp_common[@]}" --contexts 98304 --batches 86 --max-hours 0.18
python3 "$tknp_repo/examples/offline_inference/TKNP/e1_70b_report.py" "$tknp_root" --output "$tknp_root/analysis.json"
