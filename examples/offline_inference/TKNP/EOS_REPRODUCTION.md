# Token Parallel reproduction on EOS H100

Updated October 15, 2026. Run the committed `tknp-microbatch-prototype` branch,
not EOS's older `vllm-tknp` checkout. This guide covers environment preparation,
fixed two-cohort TKNP with CUDA graphs, matched measurements, all-rank traces,
and a small real prefill/decode check. The new launcher has CPU contract tests;
the current branch has not yet been GPU-validated on EOS. LYRIS GB200 results
are not H100 results.

The paper policy is [PAPER_EXPERIMENT_PLAN.md](PAPER_EXPERIMENT_PLAN.md): TP8
first, BF16 weights/KV, 32K/64K/96K inputs. Development smokes below TP8 are
functional checks only. Do not submit a full H100 sweep before those gates pass.

## Source and runtime requirements

Use a separate EOS checkout and environment:

| Item | EOS path or requirement |
|---|---|
| Checkout | `/home/sshrestha/workspace/2026/vllm-tknp-ubatch` |
| Branch | `tknp-microbatch-prototype`; pin the pushed commit, not just the branch name |
| Environment | `/home/sshrestha/workspace/2026/vllm-tknp-eos-venv` |
| New image | `/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-x86_64-v0.27.0.sqsh` after preparation below |
| Results | A new directory under `/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/` |
| SSH config on the local Mac | `/Users/susavshrestha/Workspace/shared/ssh-config` |

The current LYRIS image is ARM64 and uses torch `2.13.0+cu130`; the research
overlay identifies itself as `0.27.1+tknp`. Its binary/runtime pairing is not
described by the fork's stale `requirements/cuda.txt` (torch 2.9.1,
FlashInfer 0.5.3). Do not install those requirements over the working image.
EOS needs x86-64 CUDA binaries and an SM90-capable kernel/runtime stack. Never
copy LYRIS `.so` files, its venv, caches, or ARM64 squashfs to EOS. Do not use
the old `/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp.sqsh` for this
branch without a separate compatibility port. Do not source
`/home/sshrestha/init_container.sh`; it points at another project.

Start with 8B, then 70B. **405B BF16 does not fit a TP8 root group on 80-GB
H100s**: weights alone require roughly 94 GiB per root GPU, before KV/runtime.
The LYRIS root shards measured about 95.6 GiB. More attention-only ranks do
not reduce that root weight footprint. Skip 405B on that hardware under the
TP8/BF16 policy; do not silently quantize, offload or change TP to make it fit.

## Prepare the checkout

On the Mac, connect to EOS using the existing SSH configuration. On the EOS
login node, clone into the new path only if it does not exist:

```bash
git clone --single-branch --branch tknp-microbatch-prototype \
  https://github.com/susavlsh10/vllm-tknp.git \
  /home/sshrestha/workspace/2026/vllm-tknp-ubatch
cd /home/sshrestha/workspace/2026/vllm-tknp-ubatch
git status --short
git rev-parse HEAD
```

Record that exact HEAD with the experiment. For an existing checkout, inspect
its branch/status before `git fetch origin`; use `git pull --ff-only` only
when the worktree is clean and already on the intended branch. Never reset or
rsync-delete the older EOS working directory. The old EOS pull/push scripts
target a different checkout and use `--delete`; do not use them for this port.

## Prepare a compatible x86 image

Reuse a compatible *verified* x86 image if one is already available. Otherwise,
inside a separately approved one-node interactive allocation, have Pyxis
import the matching upstream base on an EOS compute node. Do not import/build
on the login node or assume that the filename denotes compatibility:

```bash
eos_image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-x86_64-v0.27.0.sqsh
test ! -e "$eos_image"
srun -A hw_nresearch_snoise -p batch -N1 --ntasks=1 -t 01:00:00 \
  --container-image=docker://vllm/vllm-openai:v0.27.0 \
  --container-save="$eos_image" \
  --container-mounts=/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  bash -lc 'uname -m; nvidia-smi; python3 -c "import torch,vllm; print(torch.__version__,torch.version.cuda,vllm.__version__); print(torch.ones(1,device=\"cuda\"))"'
sha256sum "$eos_image"
```

The tag is a bring-up candidate, not an EOS validation claim. Verify that the
import provides x86-64, working H100 CUDA allocation and the intended torch
stack. If the tag is unavailable, CUDA allocation fails, or required APIs
below are missing, stop and resolve the image/driver pairing rather than
falling back to the legacy image or installing arbitrary latest packages.
Pin the successful image checksum and package versions for every comparison.

The upstream [GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
documents binary/torch compatibility and source builds. This fork changes
Python rather than CUDA/C++ source, so a matching installed wheel can supply
the native libraries; successful import and graph tests are still required.
If a source build is needed, use the local installation instructions and the
existing-torch workflow in `docs/getting_started/installation/gpu.cuda.inc.md`
in a separate build checkout. Do not let dependency resolution replace the
pinned torch stack or rebuild the benchmark checkout in place.

## Allocate one development node

Inspect existing sessions/jobs first and reuse an appropriate allocation:

```bash
tmux list-sessions
squeue -u sshrestha
```

For a new approved session, use a unique name; do not kill other sessions:

```bash
tmux new-session -s eos-tknp-ubatch
srun -A hw_nresearch_snoise -p batch -N1 --ntasks-per-node=1 \
  -J hw_nresearch_snoise-tknp:ubatch-dev -t 02:00:00 --exclusive \
  --container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-x86_64-v0.27.0.sqsh \
  --container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise \
  --container-workdir=/home/sshrestha/workspace/2026/vllm-tknp-ubatch \
  --pty /bin/bash
```

Mount the parent Lustre directory: the Hugging Face cache is a sibling of
`sshrestha/`. The account/partition shown are the documented EOS defaults, not
an assumption that LYRIS's alternate account is authorized on EOS.

## Install the Python overlay and verify imports

Run once inside the new x86 container, before setting `PYTHONPATH`. The
following copies only matching x86 native/generated assets, not upstream
Python over the research implementation:

```bash
set -euo pipefail
export TKNP_WORKTREE=/home/sshrestha/workspace/2026/vllm-tknp-ubatch
export TKNP_PYTHON=/home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3
tknp_installed_vllm=$(python3 -I -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')
test "$tknp_installed_vllm" != "$TKNP_WORKTREE/vllm"
python3 -m venv --system-site-packages /home/sshrestha/workspace/2026/vllm-tknp-eos-venv
find "$tknp_installed_vllm" -maxdepth 1 -type f -name '*.so' \
  -exec cp --target-directory="$TKNP_WORKTREE/vllm" {} +
cp "$tknp_installed_vllm/_version.py" "$TKNP_WORKTREE/vllm/_version.py"
for tknp_asset in vllm_flash_attn third_party/triton_kernels; do
  if test -d "$tknp_installed_vllm/$tknp_asset"; then
    mkdir -p "$TKNP_WORKTREE/vllm/$tknp_asset"
    cp -a "$tknp_installed_vllm/$tknp_asset/." "$TKNP_WORKTREE/vllm/$tknp_asset/"
  fi
done
export PYTHONPATH="$TKNP_WORKTREE"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_CACHE_ROOT=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/inductor
cd "$TKNP_WORKTREE"
"$TKNP_PYTHON" -c 'import pathlib,platform,torch,vllm; import vllm._C; from flashinfer.decode import fast_decode_plan; from vllm.v1.attention.backends.flashinfer import FlashInferBackend; print(platform.machine(),vllm.__file__,torch.__version__,torch.version.cuda,torch.cuda.nccl.version()); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path(".").resolve()); assert platform.machine()=="x86_64"; print(torch.ones(1,device="cuda"))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m
```

Confirm H100s, actual GPU count/memory and topology. The matching image must
provide the fork's FlashInfer APIs and `nixl` for P/D. If imports fail, preserve
the first traceback and repair the environment before collecting results.
Do not print tokens, private initialization scripts or the whole environment.
Model IDs require cached config/tokenizer files even with dummy weights;
keep offline mode and supply explicit cached paths if EOS's cache differs.
Do not download a gated checkpoint or silently switch model architecture.

On subsequent containers reuse the venv and export the same scoped settings.
Use a separate environment for any alternative runtime. Do not reuse LYRIS's
NIC names, NVLink assumptions, allocator caches or compiled kernel caches.

## Run short functional gates

Inside the allocated compute container, run targeted tests. The image must
provide pytest; install it into the new venv only if missing. The global
upstream fixtures are intentionally bypassed:

```bash
"$TKNP_PYTHON" -m pytest --noconftest -q \
  tests/v1/worker/test_tknp_cohort_plan.py \
  tests/v1/worker/test_tknp_customop_contract.py \
  tests/v1/worker/test_tknp_placement.py \
  tests/v1/worker/test_tknp_pilot.py \
  tests/v1/worker/test_tknp_capacity_crossover.py \
  tests/v1/worker/test_pp_nixl_layout.py \
  tests/v1/worker/test_tknp_70b_suite.py
"$TKNP_PYTHON" examples/offline_inference/TKNP/test_flashinfer_fast_plan.py
CUDA_VISIBLE_DEVICES=0,1,2,3 "$TKNP_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  examples/offline_inference/TKNP/test_independent_cohort_comm.py
```

Require the FlashInfer pass marker and four
`INDEPENDENT_COHORT_NCCL_PASS` markers. For TP split-compute recipes, also run
`test_symmetric_cohort_allreduce.py` on a supported local group if the selected
TP reduction backend uses symmetric memory. Its test intentionally fails if
that backend is unavailable; record the limitation instead of claiming a pass.

Run an 8B **development-only** four-GPU pair first:

```bash
export TKNP_MODEL=meta-llama/Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TKNP_TP_SIZE=2 TKNP_WORLD_SIZE=2 TKNP_LOCAL_GPUS=4 TKNP_NNODES=1
export TKNP_DEVELOPMENT_SMOKE=1
export TKNP_ROOT_REQUESTS=4 TKNP_C0_FRACTION=0.5 TKNP_C1_PRIORITY=0
export TKNP_LOAD_FORMAT=dummy TKNP_MEMORY_UTILIZATION=0.9 TKNP_PROFILE=0
tknp_result_base=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/dev_NEW_TAG
bash examples/offline_inference/TKNP/jobs/eos_microbatch_worker.sh \
  "$tknp_result_base/off" off 32 32768 128
bash examples/offline_inference/TKNP/jobs/eos_microbatch_worker.sh \
  "$tknp_result_base/on" pipelined 32 32768 128
"$TKNP_PYTHON" examples/offline_inference/TKNP/summarize_microbatch_pairs.py \
  "$tknp_result_base/off" "$tknp_result_base/on"
```

Use a fresh tag for every attempt. The launcher rejects existing output logs,
invalid topology and low TP unless the development exception is explicit.
The off baseline uses standard ownership; its quotas may differ from the
two-cohort plan. For causal matched-ownership comparison, use positive root
ownership and set `TKNP_BENCHMARK_SCRIPT` to
`examples/offline_inference/TKNP/benchmark_weighted_nomicro.py` **only for the
off run**, then unset it for on. A zero-root unsplit ablation is unsupported.
This synthetic-KV token agreement is a plumbing gate, not trained quality.

## Reproduce TP8 with two TKNP groups

After the single-node gate, the target uses 16 decode GPUs: two eight-GPU EOS
nodes **only if inventory verified eight GPUs/node**; otherwise four four-GPU
nodes. Obtain a new allocation explicitly for that size. For the eight-GPU
case, inside a login-node tmux session:

```bash
salloc -A hw_nresearch_snoise -p batch -N2 --ntasks-per-node=1 \
  --exclusive -t 02:00:00 -J hw_nresearch_snoise-tknp:tp8-tknp2
```

In the allocated shell, export paths/environment from above, then:

```bash
export TKNP_NNODES=2 TKNP_LOCAL_GPUS=8 TKNP_TP_SIZE=8 TKNP_WORLD_SIZE=2
export TKNP_DEVELOPMENT_SMOKE=0
export TKNP_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
export TKNP_ROOT_REQUESTS=8 TKNP_C0_FRACTION=0.5 TKNP_C1_PRIORITY=0
export TKNP_PRIVATE_TP_SYMM=1 TKNP_PROFILE=0 TKNP_LOAD_FORMAT=dummy
export TKNP_MEMORY_UTILIZATION=0.9
unset CUDA_VISIBLE_DEVICES TKNP_NODE_RANK TKNP_BENCHMARK_SCRIPT
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
eos_image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-x86_64-v0.27.0.sqsh
eos_mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise
tknp_result_base=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/70b_tp8_tknp2_NEW_TAG
for repeat in 1 2 3; do
  variants=(off pipelined)
  if (( repeat == 2 )); then variants=(pipelined off); fi
  for variant in "${variants[@]}"; do
    export TKNP_POINT_ID="${SLURM_JOB_ID}_${repeat}_${variant}"
    if [[ $variant == off ]]; then export MASTER_PORT=$((24000 + repeat * 10));
    else export MASTER_PORT=$((24001 + repeat * 10)); fi
    srun --nodes="$TKNP_NNODES" --ntasks="$TKNP_NNODES" --ntasks-per-node=1 \
      --kill-on-bad-exit=1 --export=ALL \
      --container-image="$eos_image" --container-mounts="$eos_mounts" \
      --container-workdir="$TKNP_WORKTREE" \
      bash examples/offline_inference/TKNP/jobs/eos_microbatch_worker.sh \
      "$tknp_result_base/repeat${repeat}_${variant}" "$variant" 32 32768 128
  done
done
```

The quota/batch above is a conservative bring-up seed, not a frozen optimal
70B recipe. Probe actual per-owner capacity before increasing batch. Keep one
exact capture shape/ownership plan per engine. Change context to 65,536 and
98,304 only after the 32K pair passes. Do not reuse GB200-fit batch sizes on
H100s. Require successful completion on every node, not only rank zero.

The recipes map to these capture-time settings; replay is one captured graph,
not a Python scheduling loop:

| Recipe | Private TKNP contexts | Split root compute and private TP | Zero root C1 attention | Lookahead |
|---|---|---|---|---|
| off | No | No | Standard planner | No |
| serial_root | Yes | No | Yes | Yes |
| zero_independent | Yes | Yes | Yes | No |
| pipelined | Yes | Yes | Yes | Yes |

Use the one selected recipe for a bounded pilot, not a full variant search.
If private symmetric TP reduction is supported, the split recipes retain
cohort-private scratch. Otherwise private NCCL may be used: record that backend
change and its cost. Never enable experimental one-sided mailboxes to conceal
a transport stall. Equal root/peer compute or reduced attention gaps alone do
not establish a throughput gain.

## Verify results and capture traces

After all workers complete, run the pair summarizer inside the compatible
container with the shared result directory mounted:

```bash
"$TKNP_PYTHON" examples/offline_inference/TKNP/summarize_microbatch_pairs.py \
  "$tknp_result_base/repeat1_off" "$tknp_result_base/repeat1_pipelined" \
  "$tknp_result_base/repeat2_off" "$tknp_result_base/repeat2_pipelined" \
  "$tknp_result_base/repeat3_off" "$tknp_result_base/repeat3_pipelined"
```

Require 16 `steady_rank*.json` records per point, all-rank
`TKNP_CUDAGRAPH_REPLAY padded_size=32` and, for on,
`TKNP_UBATCH_CUDAGRAPH_REPLAY total_tokens=32`, no fallback/preemption or
missing steps, and request-ordered token equality. Rank totals and measured
full-batch intervals must agree. Save actual ownership, private communicator
markers and all stderr. The summarizer uses the slowest rank's unprofiled
`engine.step` mean, trims first/last four steps, and rejects profiled/mismatched
configurations. It is **decode-only steady-state throughput**, not client
burst throughput including queueing and not end-to-end serving throughput.

For all-rank PyTorch traces, rerun one passing shape into a fresh directory
with `TKNP_PROFILE=1` on all nodes and the same selected on recipe. The
launcher enables an eight-iteration trace after four warmup iterations. Check
that every assigned worker produced a compressed trace under `traces/` and
that profiling completed. Exclude the profiled run from performance ratios.
CPU NVTX ranges around replay do not create per-layer NVTX ranges inside a
captured graph; analyze kernels, CUDA streams and graph dependencies.

## Small real prefill decode check

Only after the environment/model gates, use the real 8B checkpoint for a
three-GPU single-node NIXL smoke. This is an artificially capped infrastructure
check, not a paper capacity result. The supplied graph smoke matrix uses
small pools appropriate for an H100, unlike `e0_readiness.py`'s GB200-sized
96-GiB defaults:

```bash
tknp_pd_root=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/pd_NEW_TAG
mkdir -p "$tknp_pd_root"
"$TKNP_PYTHON" examples/offline_inference/TKNP/e2e_workload.py generate \
  --natural --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --count 8 --lengths 128 --output-lengths 33 \
  --output "$tknp_pd_root/workload.jsonl"
"$TKNP_PYTHON" examples/offline_inference/TKNP/e2e_matrix.py \
  --matrix examples/offline_inference/TKNP/single_node_e2e_graph_smoke.json \
  --workload "$tknp_pd_root/workload.jsonl" --output "$tknp_pd_root/trials"
```

Require real KV handoff, complete/drained requests, BF16 worker manifests and
graph replay for off/on. Keep full-logit numerical diagnostics distinct from
exact greedy matches. For a 32K handoff, create a new explicit H100-sized
matrix and validate lifetime capacities before running; do not enlarge the
tiny capped smoke in place.

The multi-node `e1_70b_suite.py`/`e2e_slurm.py` collection stack is still
LYRIS-specific: four GPUs/node, LYRIS Python/image paths and network defaults.
Do **not** invoke it unchanged on EOS or mistake the portable microbatch
worker for a port of the whole collection controller. Multi-node EOS real P/D
and its TP8 shard/PP completion paths require a subsequent explicit adapter
and smoke. Real NIXL-to-DCP mapping remains unsupported on either cluster.

## Retrieve artifacts and release only your allocation

From the Mac, substitute the actual unique result tag:

```bash
rsync -avz -e 'ssh -F /Users/susavshrestha/Workspace/shared/ssh-config' \
  login-eos:/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/70b_tp8_tknp2_NEW_TAG/ \
  /Users/susavshrestha/Workspace/project-workspace/eos/tknp_eos_results/70b_tp8_tknp2_NEW_TAG/
```

Save source commit, image checksum, package/runtime inventory, SLURM job and
account, node list/topology, commands/recipe, per-rank metrics, token IDs,
traces and failure logs. Large outputs stay outside Git. Exit only this job's
interactive shell/allocation when finished; leave unrelated jobs and tmux
sessions untouched. No EOS performance improvement is claimed until these
paired tests actually run and pass.
