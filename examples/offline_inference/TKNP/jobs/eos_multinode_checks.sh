#!/usr/bin/env bash
# Login-host controller. Uses only an existing allocation; no job submission.
set -euo pipefail
if (( $# != 6 )); then
  echo 'Usage: eos_multinode_checks.sh JOB ABS_RESULT_ROOT PREFILL_NODE DECODE0_NODE DECODE1_NODE ABS_70B_DUMMY_CONFIG' >&2
  exit 2
fi
job=$1 result=$2 pnode=$3 dnode0=$4 dnode1=$5 model70=$6
cd "$(dirname "$0")/../../../.."
here=examples/offline_inference/TKNP
common=(--job "$job" --nodes "$pnode" "$dnode0" "$dnode1")
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --output "$result/decode8b" --family decode --kv-gib 2
if [[ ! -s $result/changing_payload.log ]] || [[ $(grep -c INDEPENDENT_COHORT_NCCL_PASS "$result/changing_payload.log") != 4 ]]; then
srun --jobid="$job" --overlap -N2 --ntasks=2 --ntasks-per-node=1 --nodelist="$pnode,$dnode0" --kill-on-bad-exit=1 \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir=/home/sshrestha/workspace/2026/vllm-tknp-ubatch \
  env PYTHONPATH=/home/sshrestha/workspace/2026/vllm-tknp-ubatch CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_LAUNCH_ORDER_IMPLICIT=1 \
  bash -c 'exec timeout 300 /home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3 -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=2 --node-rank="$SLURM_PROCID" --rdzv-backend=c10d --rdzv-endpoint="$1:29651" \
    examples/offline_inference/TKNP/test_independent_cohort_comm.py' bash "$pnode" > "$result/changing_payload.log" 2>&1
test "$(grep -c INDEPENDENT_COHORT_NCCL_PASS "$result/changing_payload.log")" -eq 4
fi
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --output "$result/e2e8b" --family e2e --kv-gib 2 --methods tknp tknp-dbo dp pp
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/decode70b" \
  --family decode --contexts 32768 65536 98304 --tokens 33 --count 16 --lanes 8
python3 "$here/eos_multinode_pair.py" --config "$result/decode70b/matrix-c32768.json" --output "$result/pair70b"
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/mixed70b" \
  --family decode --contexts 98304 --mixed-lengths 32768 65536 98304 --rate 20 \
  --tokens 33 --count 16 --lanes 8 --methods tknp tknp-dbo
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/e2e70b" \
  --family e2e --contexts 32768 --tokens 33 --count 8 --lanes 8 --methods tknp dp pp --staged-decode --staged-skip-pp
python3 "$here/eos_multinode_readiness.py" "${common[@]}" --model "$model70" --output "$result/resident_synthetic70b" \
  --family decode --contexts 32768 --tokens 33 --count 8 --lanes 8 --methods tknp dp pp --staged-decode --staged-skip-pp
python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"decode70b/matrix-c32768.json").read_text()); (p/"runtime.json").write_text(json.dumps(c["runtime"],indent=2)+"\n")' "$result"
python3 "$here/e1_70b_suite.py" "${common[@]}" --model "$model70" --runtime-json "$result/runtime.json" \
  --gpus-per-node 8 --prefill-tp 8 --prefill-gpus 0 1 2 3 4 5 6 7 \
  --decoder-gpus 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --output "$result/capacity70b" --family decode --contexts 32768 --tokens 33 \
  --methods tknp-tp8 dp-tp8 tp8-pp2 dcp --regimes overload --max-hours 1 --gpu-memory-utilization 0.9
python3 "$here/eos_crossover_readiness.py" --suite "$result/capacity70b" --output "$result/crossover70b"
echo EOS_MULTINODE_CHECK_SEQUENCE_FINISHED
