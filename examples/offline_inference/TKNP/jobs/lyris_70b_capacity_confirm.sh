#!/usr/bin/env bash
# Two confirmation pairs in ONE existing allocation; never submits a job.
set -euo pipefail
tknp_job=${1:?existing single-node allocation}
tknp_base=${2:?completed 70B capacity result parent}
tknp_launcher=/home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP/jobs/lyris_e2e_interactive.sh
tknp_model=/lustre/fsw/hw_nresearch_snoise/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
[[ -f "$tknp_base/run95/c32768-b18-o4097-dp2/trial.json" ]] || { echo 'Initial pair not complete' >&2; exit 1; }
for tknp_pair in 2 3; do
    tknp_order=()
    [[ "$tknp_pair" != 3 ]] || tknp_order=(--reverse-order)
    bash "$tknp_launcher" "$tknp_job" capacity --load-format dummy \
        --gpu-memory-utilization 0.95 --model "$tknp_model" \
        --context 32768 --tokens 4097 --batches 18 --reference "$tknp_base/run95" \
        --output "$tknp_base/pair$tknp_pair" "${tknp_order[@]}"
    PYTHONPATH=/home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP \
        python3 /home/sshrestha/workspace/2026/vllm-tknp-ubatch/examples/offline_inference/TKNP/e1_capacity_report.py \
        "$tknp_base/pair$tknp_pair" --output "$tknp_base/analysis_pair$tknp_pair.json"
done
echo 'CAPACITY_CONFIRMATION_COMPLETE'
