#!/usr/bin/env bash
#SBATCH -A hw_nresearch_snoise
#SBATCH -p gb200
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --segment=2
#SBATCH -t 00:15:00
#SBATCH -J hw_nresearch_snoise-tknp:symmscratch
#SBATCH -o /home/sshrestha/workspace/2026/vllm-tknp-ubatch/ubatch_results/redesign_20261002/symm_cohort_%j.out
#SBATCH -e /home/sshrestha/workspace/2026/vllm-tknp-ubatch/ubatch_results/redesign_20261002/symm_cohort_%j.err
set -euo pipefail
worktree=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
export MASTER_ADDR MASTER_PORT PYTHONPATH LD_LIBRARY_PATH
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((23000 + SLURM_JOB_ID % 18000))
PYTHONPATH=$worktree
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
srun --mpi=pmix --nodes=2 --ntasks-per-node=1 \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-arm64-v0.27.0.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$worktree" --export=ALL \
  bash -c 'timeout 180 /home/sshrestha/workspace/2026/vllm-tknp-lyris-venv/bin/python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" --rdzv-backend=c10d --rdzv-id="symm_$SLURM_JOB_ID" --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" examples/offline_inference/TKNP/test_symmetric_cohort_allreduce.py --large'
echo "SYMM_COHORT_JOB_SUCCESS"
