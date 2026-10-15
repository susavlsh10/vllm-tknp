"""Launch roles in an EXISTING allocation; never allocates nodes or changes accounts."""
import os
from pathlib import Path

REPO = "/home/sshrestha/workspace/2026/vllm-tknp-ubatch"
PYTHON = "/home/sshrestha/workspace/2026/vllm-tknp-lyris-venv/bin/python3"
IMAGE = "/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-arm64-v0.27.0.sqsh"
MOUNTS = "/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise"


def container_command(config, nodes, command, env=None):
    return ["srun", f"--jobid={config['slurm_job_id']}", "--overlap", "--mpi=pmix",
            f"--nodes={len(nodes)}", f"--ntasks={len(nodes)}", "--ntasks-per-node=1",
            "--nodelist=" + ",".join(nodes), "--kill-on-bad-exit=1",
            "--container-image=" + IMAGE, "--container-mounts=" + MOUNTS,
            "--container-workdir=" + REPO, "env",
            "PYTHONPATH=" + REPO, "HF_HOME=/lustre/fsw/hw_nresearch_snoise/hf_cache",
            "HF_HUB_OFFLINE=1", "NCCL_LAUNCH_ORDER_IMPLICIT=1",
            "LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64",
            "VLLM_RESEARCH_GIT_HEAD=ba5fef1d88b4cbc71f7abcd9bc1a370c1e51d5cd",
            *[f"{key}={value}" for key, value in (env or {}).items()], *command]


def launch_role(config, server, index, command, env):
    devices = {}
    for gpu in server["gpus"]:
        devices.setdefault(gpu // 4, []).append(gpu % 4)
    mappings = list(devices.values())
    if any(ids != mappings[0] for ids in mappings):
        raise ValueError("A distributed role needs identical local GPU indices on its nodes")
    nodes = [config["slurm_nodes"][i] for i in devices]
    local_ids = ",".join(map(str, mappings[0]))
    env = dict(env, CUDA_VISIBLE_DEVICES=local_ids,
               VLLM_NIXL_SIDE_CHANNEL_HOST=nodes[0])
    # Each node needs its OWN reachable NIXL address, not the rendezvous host.
    env.pop("VLLM_NIXL_SIDE_CHANNEL_HOST")
    args = command[1:]
    if server.get("tknp", 1) > 1:
        args = args[4:]  # -m torch.distributed.run --standalone --nproc-per-node
    if not args or not args[0].endswith("e2e_server.py"):
        raise ValueError("Unexpected server command")
    args += ["--host", "0.0.0.0"]
    worker = ["bash", str(Path(REPO) / "examples/offline_inference/TKNP/jobs/lyris_e2e_role.sh"),
              str(len(nodes)), str(len(mappings[0])), nodes[0], str(config.get("rdzv_port_base", 23500) + index * 31),
              "external" if len(nodes) > 1 or server.get("tknp", 1) > 1 else "mp",
              *args]
    return container_command(config, nodes, worker, env), f"http://{nodes[0]}:{config.get('port_base', 8200) + index}"


def client_command(config, command):
    return container_command(config, [config["slurm_nodes"][0]], [PYTHON, *command[1:]])
