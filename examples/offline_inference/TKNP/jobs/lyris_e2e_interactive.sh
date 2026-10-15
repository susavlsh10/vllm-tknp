#!/usr/bin/env bash
# Reuse one existing allocation. This script NEVER allocates/submits a job.
# Usage on the Lyris login host:
# bash examples/offline_inference/TKNP/jobs/lyris_e2e_interactive.sh JOB_ID smoke|matrix|workload [arguments...]
set -euo pipefail
tknp_job_id=${1:?Supply an existing single-node SLURM job ID}
tknp_command=${2:?Choose smoke, matrix, or workload}
shift 2
[[ "$tknp_job_id" =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
tknp_allocation=$(squeue -h -j "$tknp_job_id" -o '%T %D')
[[ "$tknp_allocation" == 'RUNNING 1' ]] || { echo 'Require exactly one running node' >&2; exit 2; }
case "$tknp_command" in
  smoke|matrix|workload|readiness|pilot|capacity) ;;
  *) echo 'Choose smoke, matrix, workload, readiness, pilot, or capacity' >&2; exit 2 ;;
esac
tknp_repo=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
tknp_revision=$(git -C "$tknp_repo" rev-parse HEAD)
tknp_entry="e2e_${tknp_command}.py"
[[ "$tknp_command" != readiness ]] || tknp_entry=e0_readiness.py
[[ "$tknp_command" != pilot ]] || tknp_entry=e1_pilot.py
[[ "$tknp_command" != capacity ]] || tknp_entry=e1_capacity_crossover.py
exec srun --jobid="$tknp_job_id" --overlap -N1 --ntasks=1 \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-arm64-v0.27.0.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$tknp_repo" \
  env PYTHONPATH="$tknp_repo" \
  HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache HF_HUB_OFFLINE=1 \
  VLLM_RESEARCH_GIT_HEAD="$tknp_revision" NCCL_LAUNCH_ORDER_IMPLICIT=1 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  /home/sshrestha/workspace/2026/vllm-tknp-lyris-venv/bin/python3 \
  "examples/offline_inference/TKNP/$tknp_entry" "$@"
