#!/usr/bin/env python3
"""One bounded unprofiled TP8xTKNP2 pair in an existing EOS allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from e2e_slurm import container_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--root-requests", type=int, default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = config["runtime"]
    nodes = config["slurm_nodes"][:2]
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in ("off", "pipelined"):
        point = args.output / variant
        if point.exists():
            raise ValueError("Use a fresh pair directory; never overwrite prior measurements")
        env = dict(TKNP_WORKTREE=runtime["repo"], TKNP_PYTHON=runtime["python"],
            TKNP_MODEL=config["model"], TKNP_TP_SIZE="8", TKNP_WORLD_SIZE="2",
            TKNP_LOCAL_GPUS="8", TKNP_NNODES="2", TKNP_LOAD_FORMAT="dummy",
            MASTER_ADDR=nodes[0], MASTER_PORT="29661", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            TKNP_ROOT_REQUESTS=str(args.root_requests), TKNP_C0_FRACTION="0.5",
            TKNP_MEMORY_UTILIZATION=str(config["gpu_memory_utilization"]), TKNP_PROFILE="0", TKNP_POINT_TIMEOUT="900")
        command = container_command(config, nodes, ["bash", runtime["repo"]+
            "/examples/offline_inference/TKNP/jobs/eos_microbatch_worker.sh", str(point), variant,
            str(args.batch), str(args.context), str(args.tokens)], env)
        (args.output / f"{variant}-launch.json").write_text(json.dumps(command, indent=2)+"\n")
        with (args.output / f"{variant}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1080)
        if len(list(point.glob("steady_rank*.json"))) != 16:
            raise RuntimeError("Missing rank timing evidence")
    script = Path(__file__).with_name("summarize_microbatch_pairs.py")
    with (args.output / "pair.json").open("x") as report:
        subprocess.run([sys.executable, str(script), str(args.output/"off"), str(args.output/"pipelined")],
                       stdout=report, check=True)
    print("EOS_MULTINODE_PAIR_PASS", flush=True)


if __name__ == "__main__":
    main()
