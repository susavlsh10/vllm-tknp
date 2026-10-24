#!/usr/bin/env bash
# Sequential safety net for full-length confirmation; never an array.
# Submit with --dependency=afterany:<development-job> to keep <=4 nodes live.
#SBATCH --account=hw_nresearch_snoise
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=hw_nresearch_snoise-snoise:405b-confirm
set -Eeuo pipefail
tknp_parent=${1:?Preserved experiment root required}
tknp_source=${2:?Immutable runtime source required}
tknp_output=$tknp_parent/confirmation_followup_job$SLURM_JOB_ID
mkdir -p "$tknp_output"
mapfile -t tknp_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#tknp_nodes[@]} == 4 ))
python3 "$tknp_source/examples/offline_inference/TKNP/eos_fp8_four_node_suite.py" verify --repo "$tknp_source"
srun --jobid="$SLURM_JOB_ID" --overlap --mpi=pmix --cpu-bind=none -N4 -n4 \
  --ntasks-per-node=1 --container-name="fp8_decode_$SLURM_JOB_ID" \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir="$tknp_source" \
  /home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3 -c \
  'import json,socket,torch;names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())];assert len(names)==8 and all("H100" in n for n in names);print(json.dumps(dict(node=socket.gethostname(),gpu=names,torch=torch.__version__,cuda=torch.version.cuda)))' \
  > "$tknp_output/hardware.log" 2>&1
tknp_child=
tknp_cleanup() {
  tknp_rc=$?
  trap - EXIT TERM INT
  if [[ -n $tknp_child ]] && kill -0 "$tknp_child" 2>/dev/null; then
    kill -INT "$tknp_child" 2>/dev/null || true
    for ((tknp_wait=0;tknp_wait<20;tknp_wait++)); do
      kill -0 "$tknp_child" 2>/dev/null || break
      sleep 1
    done
    kill -TERM "$tknp_child" 2>/dev/null || true
  fi
  printf 'CONFIRM_FOLLOWUP_EXIT rc=%s; SLURM releases all four nodes\n' "$tknp_rc"
  exit "$tknp_rc"
}
trap tknp_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
tknp_candidate=$tknp_parent/projection_micro_native_tp
if ! python3 - "$tknp_candidate/summary.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text()) if p.exists() else {}
rows=s.get("trials",[])
valid={(r.get("batch"),r.get("repeat")) for r in rows if r.get("performance_valid")
       and r.get("full_graph_all_ranks") and not r.get("measured_preemptions")
       and r.get("output_length")==128 and not r.get("client",{}).get("failed",1)}
sys.exit(0 if valid=={(96,0),(128,0),(96,1),(128,1)} and not s.get("failures") else 1)
PY
then
  tknp_candidate=$tknp_output/native_tp_candidate
  if srun --jobid="$SLURM_JOB_ID" --overlap --mpi=pmix --cpu-bind=none -N1 -n1 \
      --nodelist="${tknp_nodes[0]}" --container-name="fp8_decode_$SLURM_JOB_ID" \
      --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh \
      --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
      --container-workdir="$tknp_source" env PYTHONPATH="$tknp_source" OMP_NUM_THREADS=1 \
      /home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3 -m torch.distributed.run \
      --standalone --nproc-per-node=8 "$tknp_parent/test_native_tp_selection.py" \
      --hidden-size 16384 --native-tp-selection > "$tknp_output/collective_correctness.log" 2>&1; then
    if python3 -c 'import re,sys;from pathlib import Path;s=Path(sys.argv[1]).read_text();assert sorted(map(int,re.findall(r"PRIVATE_COHORT_CUSTOM_PASS rank=(\d+) world=8",s)))==list(range(8))' "$tknp_output/collective_correctness.log"; then
      python3 "$tknp_source/examples/offline_inference/TKNP/eos_405b_efficiency.py" \
        --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" --model "$tknp_parent/model405b" \
        --output "$tknp_candidate" --tokens 128 --batches 96 128 --stage matrix \
        --methods tknp-tp8-dbo --attention-backend FLASH_ATTN --root-requests 0 \
        --max-tknp-resident 100 --dbo-threshold 96 --tail-downsize --extra-capture-batches 32 \
        --private-tp-custom 1 --private-tp-custom-bytes 262144 --repeats 2 \
        --capacity-source "$tknp_parent/fa3_matrix_retry" > "$tknp_output/native_tp_candidate.log" 2>&1 &
      tknp_child=$!
      if ! wait "$tknp_child"; then printf 'EXPERIMENTAL_CANDIDATE_FAILED; retain standard TKNP\n'; fi
      tknp_child=
    fi
  fi
fi
tknp_needed=$(python3 - "$tknp_parent" "$tknp_candidate" <<'PY'
import json,sys
import statistics
from pathlib import Path
root=Path(sys.argv[1])
c=Path(sys.argv[2])/"summary.json"
candidate=json.loads(c.read_text()) if c.exists() else {}
points=candidate.get("trials",[])
reference={b:max(r["output_tps"] for parent in ("fa3_matrix_retry","native_tail_control_retry")
                   for r in json.loads((root/parent/"summary.json").read_text())["trials"]
                   if r["method"]=="tknp-tp8" and r["batch"]==b) for b in (96,128)}
eligible=not candidate.get("failures")
overflow_eligible=False
for b in (96,128):
    rows=[r for r in points if r.get("batch")==b and r.get("performance_valid")
          and r.get("full_graph_all_ranks") and not r.get("measured_preemptions")
          and not r.get("client",{}).get("failed",1)]
    ratio=statistics.median(r["output_tps"] for r in rows)/reference[b] if len(rows)>=2 else 0
    eligible &= ratio>=1.03
    if b==128:overflow_eligible=not candidate.get("failures") and ratio>=1.03
    print("NATIVE_TP_CANDIDATE",b,"median_vs_best_native",ratio,file=sys.stderr)
if eligible or overflow_eligible:
    print("tknp-tp8-dbo")
    print("EXPLORATORY_FULL_LENGTH_MICROBATCH universal_candidate=",eligible,
          "overflow_candidate=",overflow_eligible,
          "not an automatic main-configuration promotion",file=sys.stderr)
for method,name in (("tknp-tp8","confirmation_1024_tknp"),
                    ("dp-tp8","confirmation_1024_baselines"),
                    ("dcp","confirmation_1024_baselines"),
                    ("tp8-pp4","confirmation_1024_baselines")):
    path=root/name/"summary.json"
    state=json.loads(path.read_text()) if path.exists() else {}
    valid=[r for r in state.get("trials",[]) if r.get("method")==method
           and r.get("batch")==128 and r.get("output_length")==1024
           and r.get("output_tokens")==128*1024 and r.get("performance_valid")
           and r.get("full_graph_all_ranks") and r.get("measured_preemptions")==0
           and not r.get("client",{}).get("failed",1)]
    # Repeat the native reference on this allocation even when development
    # already measured it, so the long-output comparisons use matched nodes.
    if method=="tknp-tp8" or not valid:print(method)
PY
)
tknp_methods=()
if [[ -n $tknp_needed ]]; then
  mapfile -t tknp_methods <<< "$tknp_needed"
fi
tknp_failed=0
# A targeted retry must not repeat already completed methods.
if [[ -n ${TKNP_CONFIRM_METHODS:-} ]]; then
  read -r -a tknp_methods <<< "$TKNP_CONFIRM_METHODS"
  for tknp_method in "${tknp_methods[@]}"; do
    case "$tknp_method" in
      tknp-tp8|tknp-tp8-dbo|dp-tp8|dcp|tp8-pp4) ;;
      *) printf 'Invalid confirmation method: %s\n' "$tknp_method" >&2; exit 2 ;;
    esac
  done
fi
for tknp_method in "${tknp_methods[@]}"; do
  tknp_flags=()
  if [[ "$tknp_method" == tknp-tp8 ]]; then
    tknp_flags=(--tail-downsize --extra-capture-batches 32)
  elif [[ "$tknp_method" == tknp-tp8-dbo ]]; then
    tknp_flags=(--tail-downsize --extra-capture-batches 32 --root-requests 0 \
      --max-tknp-resident 100 --dbo-threshold 96 --private-tp-custom 1 --private-tp-custom-bytes 262144)
  fi
  printf 'CONFIRM_START method=%s job=%s nodes=%s\n' "$tknp_method" "$SLURM_JOB_ID" "${tknp_nodes[*]}"
  python3 "$tknp_source/examples/offline_inference/TKNP/eos_405b_efficiency.py" \
    --job "$SLURM_JOB_ID" --nodes "${tknp_nodes[@]}" \
    --model "$tknp_parent/model405b" --output "$tknp_output/$tknp_method" \
    --tokens 1024 --context 32768 --batches 128 --stage matrix \
    --methods "$tknp_method" --attention-backend FLASH_ATTN --pp-baseline standard \
    "${tknp_flags[@]}" > "$tknp_output/$tknp_method.log" 2>&1 &
  tknp_child=$!
  if wait "$tknp_child"; then
    printf 'CONFIRM_PASS method=%s\n' "$tknp_method"
  else
    printf 'CONFIRM_FAILED method=%s\n' "$tknp_method"
    tknp_failed=1
  fi
  tknp_child=
done
printf 'CONFIRM_FOLLOWUP_COMPLETE failures=%s\n' "$tknp_failed"
exit "$tknp_failed"
