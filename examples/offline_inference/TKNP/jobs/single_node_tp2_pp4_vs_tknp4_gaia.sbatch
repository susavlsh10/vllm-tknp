#!/bin/bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p GAIA
#SBATCH -J hw_nresearch_snoise-snoise:tknp-single-node-sweep
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o sweep_results/slurm_logs/tknp_single_node_sweep_%j.out
#SBATCH -e sweep_results/slurm_logs/tknp_single_node_sweep_%j.err

set -uo pipefail

CHECKOUT=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp
IMAGE=/mnt/lustre/gaia/sshrestha/containers/vllm-tknp-eos.sqsh
CONTAINER_MOUNTS=/labhome/sshrestha:/labhome/sshrestha,/mnt/lustre/gaia/sshrestha:/mnt/lustre/gaia/sshrestha

# GAIA exposes Pyxis options on srun but not sbatch. Re-enter once inside the
# frozen EOS container while preserving the outer allocation and SLURM logs.
if [[ "${TKNP_IN_CONTAINER:-0}" != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 --gpus-per-node=8 \
        --container-image="$IMAGE" \
        --container-mounts="$CONTAINER_MOUNTS" \
        --container-workdir="$CHECKOUT" \
        env TKNP_IN_CONTAINER=1 \
        bash examples/offline_inference/TKNP/jobs/single_node_tp2_pp4_vs_tknp4_gaia.sbatch
fi

RESULT_ROOT="${CHECKOUT}/sweep_results/gaia_single_node_job_${SLURM_JOB_ID}"
BENCHMARK=examples/offline_inference/TKNP/tknp_inference_benchmarks.py
SUMMARIZER=examples/offline_inference/TKNP/summarize_single_node_sweep.py
MODEL=Qwen/Qwen2.5-32B
DECODE_TOKENS=64
REPETITIONS=3
MASTER_PORT=29619

mkdir -p "$RESULT_ROOT" || exit 10

export PYTHONPATH="${PYTHONPATH:-}"
source /labhome/sshrestha/init_container.sh || exit 11
export PYTHONPATH="$CHECKOUT"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_WORKSPACE_BASE=/mnt/lustre/gaia/sshrestha
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1
export CUDA_CACHE_PATH=/mnt/lustre/gaia/sshrestha/.cache/nv/ComputeCache
export TORCHINDUCTOR_CACHE_DIR=/mnt/lustre/gaia/sshrestha/.cache/torch/inductor-tknp
export VLLM_CACHE_ROOT=/mnt/lustre/gaia/sshrestha/.cache/vllm-tknp
export VLLM_CONFIG_ROOT=/mnt/lustre/gaia/sshrestha/.config/vllm-tknp
export NCCL_COMM_BLOCKING=1
export NCCL_CUMEM_ENABLE=0
mkdir -p "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" || exit 12
cd "$CHECKOUT" || exit 12

hostname | tee "$RESULT_ROOT/node.txt"
git rev-parse HEAD | tee "$RESULT_ROOT/git_head.txt"
git status --short > "$RESULT_ROOT/git_status.txt"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
    > "$RESULT_ROOT/gpus.txt" || exit 13
if [[ ! -s "$RESULT_ROOT/gpus.txt" ]] || grep -vq H100 "$RESULT_ROOT/gpus.txt"; then
    echo "Every visible GPU must be an H100." | tee "$RESULT_ROOT/environment_error.txt"
    exit 14
fi

python3 - <<'PY' | tee "$RESULT_ROOT/environment.txt"
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
environment_rc=${PIPESTATUS[0]}
if (( environment_rc != 0 )); then
    exit 15
fi
if ! grep -q '^vllm_path=/mnt/lustre/gaia/sshrestha/workspace/vllm-tknp/' \
    "$RESULT_ROOT/environment.txt"; then
    exit 16
fi

cat > "$RESULT_ROOT/experiment.txt" <<EOF
model=$MODEL
decode_tokens=$DECODE_TOKENS
repetitions=$REPETITIONS
batch_sweep=4,8,16,32,64
batch_sweep_sequence_length=2048
sequence_sweep=128,512,2048,8192,16384
sequence_sweep_batch_size=16
topologies=tp2_pp4,tp2_tknp4
cuda_graph=true
load_format=dummy
EOF

declare -a POINTS=(
    "batch 4 2048"
    "batch 8 2048"
    "batch 16 2048"
    "batch 32 2048"
    "batch 64 2048"
    "sequence 16 128"
    "sequence 16 512"
    "sequence 16 8192"
    "sequence 16 16384"
)

run_topology() {
    local topology=$1
    local tp=$2
    local pp=$3
    local tknp=$4
    local failed=0

    for point in "${POINTS[@]}"; do
        read -r sweep_kind batch_size seq_length <<< "$point"
        point_dir="${RESULT_ROOT}/${topology}/${sweep_kind}_bs${batch_size}_seq${seq_length}"
        mkdir -p "$point_dir"

        for repetition in $(seq 1 "$REPETITIONS"); do
            marker="${point_dir}/rep_${repetition}.PASS"
            if [[ -e "$marker" ]]; then
                continue
            fi

            rm -f "${point_dir}/rep_${repetition}.FAIL"
            rep_dir="${point_dir}/rep_${repetition}"
            mkdir -p "$rep_dir"
            log="${point_dir}/rep_${repetition}.log"

            echo "START topology=$topology sweep=$sweep_kind batch=$batch_size seq=$seq_length repetition=$repetition time=$(date --iso-8601=seconds)"
            timeout --signal=TERM --kill-after=30s 20m \
                env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                torchrun --nproc-per-node=8 --master-port="$MASTER_PORT" \
                "$BENCHMARK" \
                --model "$MODEL" \
                --tensor-parallel-size "$tp" \
                --pipeline-parallel-size "$pp" \
                --token-parallel-size "$tknp" \
                --batch-size "$batch_size" \
                --seq-length "$seq_length" \
                --decode-tokens "$DECODE_TOKENS" \
                --load-format dummy \
                --cuda-graph \
                --cudagraph-capture-sizes "$batch_size" \
                --output-dir "$rep_dir" > "$log" 2>&1
            run_rc=$?

            csv_count=$(find "$rep_dir" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')
            if (( run_rc == 0 && csv_count == 1 )); then
                touch "$marker"
                echo "PASS topology=$topology sweep=$sweep_kind batch=$batch_size seq=$seq_length repetition=$repetition"
            else
                printf '%s\n' "$run_rc" > "${point_dir}/rep_${repetition}.exit_code"
                printf '%s\n' "$csv_count" > "${point_dir}/rep_${repetition}.csv_count"
                touch "${point_dir}/rep_${repetition}.FAIL"
                echo "FAIL topology=$topology sweep=$sweep_kind batch=$batch_size seq=$seq_length repetition=$repetition rc=$run_rc csv_count=$csv_count"
                failed=1
                break
            fi
        done
    done

    return "$failed"
}

overall_failed=0
run_topology tp2_pp4 2 4 1 || overall_failed=1
run_topology tp2_tknp4 2 1 4 || overall_failed=1

python3 "$SUMMARIZER" "$RESULT_ROOT" || overall_failed=1

if (( overall_failed != 0 )); then
    touch "$RESULT_ROOT/SWEEP_FAILED"
    exit 20
fi

touch "$RESULT_ROOT/SWEEP_SUCCESS"
echo "Sweep completed successfully at $(date --iso-8601=seconds)"
