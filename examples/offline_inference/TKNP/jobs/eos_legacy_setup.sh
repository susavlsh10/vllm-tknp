#!/usr/bin/env bash
# Source in an allocated EOS container. Does not replace the old checkout.
set -euo pipefail
export TKNP_WORKTREE=${TKNP_WORKTREE:-/home/sshrestha/workspace/2026/vllm-tknp-ubatch}
tknp_native_root=${TKNP_NATIVE_ROOT:-/home/sshrestha/workspace/2026/vllm-tknp/vllm}
tknp_venv=${TKNP_EOS_VENV:-/home/sshrestha/workspace/2026/vllm-tknp-eos-venv}
export TKNP_PYTHON="$tknp_venv/bin/python3"
[[ $(uname -m) == x86_64 ]] || { echo 'EOS setup requires x86_64'; return 2; }
test -f "$tknp_native_root/_C.abi3.so"
if [[ ! -x $TKNP_PYTHON ]]; then python3 -m venv --system-site-packages "$tknp_venv"; fi
find "$tknp_native_root" -maxdepth 1 -type f -name '*.so' \
  -exec cp -n --target-directory="$TKNP_WORKTREE/vllm" {} +
cp -n "$tknp_native_root/_version.py" "$TKNP_WORKTREE/vllm/_version.py"
for tknp_asset in vllm_flash_attn third_party/triton_kernels; do
  if [[ -d $tknp_native_root/$tknp_asset ]]; then
    mkdir -p "$TKNP_WORKTREE/vllm/$tknp_asset"
    cp -an "$tknp_native_root/$tknp_asset/." "$TKNP_WORKTREE/vllm/$tknp_asset/"
  fi
done
export PYTHONPATH="$TKNP_WORKTREE"
export HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_CACHE_ROOT=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/vllm_torch291
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/inductor_torch291
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export OMP_NUM_THREADS=1
unset FLASHINFER_DISABLE_VERSION_CHECK
cd "$TKNP_WORKTREE"
"$TKNP_PYTHON" - <<'PY'
import importlib.metadata as metadata
import json, pathlib, subprocess
import torch, vllm, flashinfer, nixl
from flashinfer.decode import fast_decode_plan
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
assert pathlib.Path(vllm.__file__).resolve().parent == pathlib.Path.cwd() / 'vllm'
print(json.dumps(dict(vllm_path=vllm.__file__, packages={
    name: metadata.version(name) for name in
    ('torch', 'flashinfer-python', 'flashinfer-cubin', 'transformers', 'nixl')},
    cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
    gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True),
    devices=torch.cuda.device_count()), indent=2))
print('CUDA_ALLOCATION', torch.ones(1, device='cuda'))
print('EOS_TKNP_ENV_READY', flush=True)
PY
