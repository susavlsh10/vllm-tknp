#!/usr/bin/env bash

set -uo pipefail

CHECKOUT=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp
RESULT_ROOT=${RESULT_ROOT:-${CHECKOUT}/sweep_results/gaia_graph_vs_eager_job_${SLURM_JOB_ID}}
BENCHMARK=examples/offline_inference/TKNP/tknp_inference_benchmarks.py
MODEL=Qwen/Qwen2.5-32B
SEQ_LENGTH=8192
DECODE_TOKENS=64
REPETITIONS=3
GPU_MEMORY_UTILIZATION=0.85
MASTER_PORT_BASE=31900

mkdir -p "${RESULT_ROOT}"
cd "${CHECKOUT}" || exit 10

cat > "${RESULT_ROOT}/experiment.txt" <<EOF
model=${MODEL}
load_format=dummy
sequence_length=${SEQ_LENGTH}
decode_tokens=${DECODE_TOKENS}
batch_sizes=32,64,128
repetitions=${REPETITIONS}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
topologies=tp2_pp2,tp2_tknp4
execution_modes=eager,cuda_graph
primary_comparison=cuda_graph_over_eager_within_each_topology
EOF

hostname > "${RESULT_ROOT}/node.txt"
git rev-parse HEAD > "${RESULT_ROOT}/git_head.txt"
git status --short > "${RESULT_ROOT}/git_status.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
    > "${RESULT_ROOT}/gpus.txt"

if [[ ! -s "${RESULT_ROOT}/gpus.txt" ]] || grep -vq H100 "${RESULT_ROOT}/gpus.txt"; then
    echo "Every visible GPU must be an H100." | tee "${RESULT_ROOT}/environment_error.txt"
    exit 11
fi

python3 - <<'PY' > "${RESULT_ROOT}/environment.txt"
import flashinfer
import torch
import vllm
import vllm._C

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"flashinfer={flashinfer.__version__}")
print(f"vllm={vllm.__version__}")
print(f"vllm_path={vllm.__file__}")
print(f"vllm_C_path={vllm._C.__file__}")
PY

if ! grep -q '^vllm_path=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp/' \
    "${RESULT_ROOT}/environment.txt"; then
    echo "vLLM did not import from the frozen checkout." | tee \
        "${RESULT_ROOT}/environment_error.txt"
    exit 12
fi

declare -a TOPOLOGIES=(
    "tp2_pp2 2 2 1 4 0,1,2,3"
    "tp2_tknp4 2 1 4 8 0,1,2,3,4,5,6,7"
)
declare -a BATCH_SIZES=(32 64 128)

pass_count=0
capacity_count=0
fail_count=0
run_index=0

run_one() {
    local topology=$1
    local tp=$2
    local pp=$3
    local tknp=$4
    local world_size=$5
    local visible_devices=$6
    local batch_size=$7
    local repetition=$8
    local mode=$9

    run_index=$((run_index + 1))
    local point_dir="${RESULT_ROOT}/${topology}/${mode}/bs${batch_size}/rep_${repetition}"
    local log="${point_dir}.log"
    local marker_base="${RESULT_ROOT}/${topology}/${mode}/bs${batch_size}/rep_${repetition}"
    local master_port=$((MASTER_PORT_BASE + run_index))
    mkdir -p "${point_dir}" "$(dirname "${log}")"
    rm -f "${marker_base}.PASS" "${marker_base}.FAIL" "${marker_base}.CAPACITY"

    local -a graph_args=()
    if [[ "${mode}" == cuda_graph ]]; then
        graph_args=(--cuda-graph --cudagraph-capture-sizes "${batch_size}")
    fi

    echo "START topology=${topology} mode=${mode} batch=${batch_size} repetition=${repetition} time=$(date --iso-8601=seconds)"
    timeout --signal=TERM --kill-after=30s 20m \
        env CUDA_VISIBLE_DEVICES="${visible_devices}" \
        torchrun --nproc-per-node="${world_size}" --master-port="${master_port}" \
        "${BENCHMARK}" \
        --model "${MODEL}" \
        --tensor-parallel-size "${tp}" \
        --pipeline-parallel-size "${pp}" \
        --token-parallel-size "${tknp}" \
        --batch-size "${batch_size}" \
        --seq-length "${SEQ_LENGTH}" \
        --decode-tokens "${DECODE_TOKENS}" \
        --load-format dummy \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        "${graph_args[@]}" \
        --output-dir "${point_dir}" > "${log}" 2>&1
    local run_rc=$?

    local csv_count
    csv_count=$(find "${point_dir}" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')
    if (( run_rc == 0 && csv_count == 0 )) && \
        grep -Eq 'ABORTING: Benchmark configuration exceeds|EXCEEDS KV cache capacity' "${log}"; then
        touch "${marker_base}.CAPACITY"
        capacity_count=$((capacity_count + 1))
        echo "CAPACITY topology=${topology} mode=${mode} batch=${batch_size} repetition=${repetition}"
        return 0
    fi

    local audit_ok=1
    if (( run_rc != 0 || csv_count != 1 )); then
        audit_ok=0
    fi
    if [[ "${mode}" == cuda_graph ]]; then
        grep -q 'Graph capturing finished' "${log}" || audit_ok=0
        if [[ "${topology}" == tp2_tknp4 ]]; then
            grep -q "TKNP_CUDAGRAPH_REPLAY padded_size=${batch_size}" "${log}" || audit_ok=0
        fi
        if grep -Eqi 'falling back|fallback to eager|dispatcher selected NONE|no exact captured split|split mismatch' "${log}"; then
            audit_ok=0
        fi
    else
        if grep -q 'Graph capturing finished' "${log}" || \
            grep -q 'TKNP_CUDAGRAPH_REPLAY' "${log}"; then
            audit_ok=0
        fi
    fi

    if (( audit_ok == 1 )); then
        touch "${marker_base}.PASS"
        pass_count=$((pass_count + 1))
        echo "PASS topology=${topology} mode=${mode} batch=${batch_size} repetition=${repetition}"
    else
        printf '%s\n' "${run_rc}" > "${marker_base}.exit_code"
        printf '%s\n' "${csv_count}" > "${marker_base}.csv_count"
        touch "${marker_base}.FAIL"
        fail_count=$((fail_count + 1))
        echo "FAIL topology=${topology} mode=${mode} batch=${batch_size} repetition=${repetition} rc=${run_rc} csv_count=${csv_count}"
    fi
}

for topology_spec in "${TOPOLOGIES[@]}"; do
    read -r topology tp pp tknp world_size visible_devices <<< "${topology_spec}"
    for batch_size in "${BATCH_SIZES[@]}"; do
        for repetition in $(seq 1 "${REPETITIONS}"); do
            if (( repetition % 2 == 1 )); then
                modes=(eager cuda_graph)
            else
                modes=(cuda_graph eager)
            fi
            for mode in "${modes[@]}"; do
                run_one "${topology}" "${tp}" "${pp}" "${tknp}" \
                    "${world_size}" "${visible_devices}" "${batch_size}" \
                    "${repetition}" "${mode}"
            done
        done
    done
done

cat > "${RESULT_ROOT}/counts.txt" <<EOF
pass=${pass_count}
capacity=${capacity_count}
fail=${fail_count}
EOF

if (( fail_count > 0 )); then
    touch "${RESULT_ROOT}/SWEEP_FAILED"
    echo "SWEEP_FAILED pass=${pass_count} capacity=${capacity_count} fail=${fail_count}"
    exit 20
fi

touch "${RESULT_ROOT}/SWEEP_SUCCESS"
echo "SWEEP_SUCCESS pass=${pass_count} capacity=${capacity_count} fail=${fail_count}"
