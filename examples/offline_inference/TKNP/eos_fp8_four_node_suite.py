#!/usr/bin/env python3
"""Freeze and submit six independent four-node EOS FP8 decode experiments.

Only `submit` creates a SLURM job. Each array element owns/relinquishes its own
exclusive four-node allocation. No checkpoint download or credential handling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from fp8_decode_sweep import FOUR_NODE_METHODS

HERE = Path(__file__).resolve().parent
BATCHES = [4, 8, 16, 32, 64, 128]
MODEL = "/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/fp8_405b_20261016_job6263959_retry1/model405b"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_snapshot(repo):
    manifest = json.loads((repo / ".research-source.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen source changed: " + name)
    return manifest


def freeze_source(repo, destination):
    destination.mkdir()
    names = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo).decode().split("\0")
    for name in sorted(set(filter(None, names))):
        source = repo / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Generated native runtime assets are ignored by Git, but are required by
    # this fork. Copy them before any parallel bootstrap can import the source.
    for source in (repo / "vllm").glob("*.so"):
        shutil.copy2(source, destination / "vllm" / source.name)
    version = repo / "vllm/_version.py"
    if version.exists():
        shutil.copy2(version, destination / "vllm/_version.py")
    for name in ("vllm/vllm_flash_attn", "vllm/third_party/triton_kernels"):
        if (repo / name).is_dir():
            shutil.copytree(repo / name, destination / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        original_repo=str(repo), frozen_at=time.time(), source_sha256={
            str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}})
    write_json(destination / ".research-source.json", manifest)
    return manifest


def finish(result, return_code):
    errors = []
    summary = result / "sweep/summary.json"
    state = json.loads(summary.read_text()) if summary.exists() else {}
    if return_code == 0:
        trials = state.get("trials", [])
        if len(trials) != len(BATCHES) or {r["batch"] for r in trials} != set(BATCHES):
            errors.append("Incomplete primary batch grid")
        if not state.get("finished_time") or state.get("failures"):
            errors.append("Missing completion marker or controller failure")
        for row in trials:
            if (row["output_tokens"] != row["batch"] * 1024 or not row["full_graph_all_ranks"]
                    or row["measured_preemptions"] or row["client"]["failed"]):
                errors.append("Primary validation failed")
    status = dict(passed=return_code == 0 and not errors, return_code=return_code,
        validation_errors=errors, completed_points=len(state.get("trials", [])),
        job_id=os.environ.get("SLURM_JOB_ID"), array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        finished_time=time.time(), cleanup="SLURM releases all owned nodes when the batch script exits")
    write_json(result / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["passed"] else 1


def collect_suite(output):
    tasks, trials = {}, []
    for index, method in enumerate(FOUR_NODE_METHODS):
        paths = sorted(output.glob(f"task{index}-*/status.json"))
        if not paths:
            tasks[method] = dict(passed=False, status="pending/running")
            continue
        status_path = paths[-1]
        status = json.loads(status_path.read_text())
        tasks[method] = dict(status, path=str(status_path.parent))
        if not status["passed"]:
            continue
        state = json.loads((status_path.parent / "sweep/summary.json").read_text())
        if any(row["method"] != method for row in state["trials"]):
            raise RuntimeError("Array result method does not match its assigned configuration")
        trials.extend(state["trials"])
    passed = all(task["passed"] for task in tasks.values()) and len(trials) == len(FOUR_NODE_METHODS) * len(BATCHES)
    result = dict(passed=passed, expected_points=len(FOUR_NODE_METHODS) * len(BATCHES),
        validated_points=len(trials), tasks=tasks, trials=trials,
        scope="Four-node per configuration, 405B mixed FP8/BF16 dummy weights, BF16 random KV, ISL32768/OSL1024 decode-only burst completion; queueing included; initialization, prefill, and smoke excluded",
        output_tokens=sum(row["output_tokens"] for row in trials))
    write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--output", type=Path, required=True)
    submit.add_argument("--model", type=Path, default=Path(MODEL))
    verify = commands.add_parser("verify")
    verify.add_argument("--repo", type=Path, required=True)
    done = commands.add_parser("finish")
    done.add_argument("--result", type=Path, required=True)
    done.add_argument("--return-code", type=int, required=True)
    report = commands.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        record = verify_snapshot(args.repo)
        print("FROZEN_SOURCE_VERIFIED", len(record["source_sha256"]), record["commit"])
        return
    if args.command == "finish":
        raise SystemExit(finish(args.result, args.return_code))
    if args.command == "report":
        for task in range(len(FOUR_NODE_METHODS)):
            paths = sorted(args.output.glob(f"task{task}-*/status.json"))
            print(FOUR_NODE_METHODS[task], json.loads(paths[-1].read_text()) if paths else "pending/running")
        return
    if args.command == "collect":
        result = collect_suite(args.output)
        print("VALIDATED_POINTS", result["validated_points"], "/", result["expected_points"], "PASSED", result["passed"])
        for method in FOUR_NODE_METHODS:
            rows = sorted((r for r in result["trials"] if r["method"] == method), key=lambda r: r["batch"])
            print(method, [(r["batch"], round(r["output_tokens_per_second"], 2)) for r in rows])
        return
    repo = HERE.parents[2]
    config = json.loads((args.model / "config.json").read_text())
    if config["quantization_config"]["quant_method"] != "fbgemm_fp8":
        raise ValueError("Use the validated 405B FBGEMM FP8 recipe")
    args.output = args.output.resolve()
    if "," in str(args.output):
        raise ValueError("Suite path cannot contain commas (SLURM export syntax)")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "slurm").mkdir()
    snapshot = args.output / "source"
    manifest = freeze_source(repo, snapshot)
    model = args.output / "model405b"
    model.mkdir()
    for source in args.model.glob("*.json"):
        shutil.copy2(source, model / source.name)
    (args.output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo))
    write_json(args.output / "suite.json", dict(methods=FOUR_NODE_METHODS, nodes_per_method=4,
        max_concurrent_nodes=24, batches=BATCHES, input_tokens=32768, output_tokens=1024,
        smoke_output_tokens=32, dummy_weights=True, kv_dtype="bfloat16", source_commit=manifest["commit"],
        source=str(snapshot), model=str(model), account="hw_nresearch_snoise"))
    script = snapshot / "examples/offline_inference/TKNP/jobs/eos_fp8_four_node_sweep.sbatch"
    command = ["sbatch", "--parsable", "--array=0-5",
        "--output=" + str(args.output / "slurm/%A_%a.out"),
        "--export=ALL,TKNP_SWEEP_SUITE_ROOT=" + str(args.output), str(script)]
    job = subprocess.check_output(command, text=True).strip()
    write_json(args.output / "submission.json", dict(job_id=job, command=command, submitted_at=time.time()))
    print("SUBMITTED_FOUR_NODE_ARRAY", job, "METHODS", ",".join(FOUR_NODE_METHODS), flush=True)


if __name__ == "__main__":
    main()
