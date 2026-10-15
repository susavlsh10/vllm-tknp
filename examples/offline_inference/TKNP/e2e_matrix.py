#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-node experiment infrastructure validator, never submits jobs.

Run the supplied declarative matrix sequentially in an existing allocation.
P/D and colocated trials are explicitly distinct; no budget-equivalent claim
is inferred from this smoke matrix. Independent DP replicas really execute.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def validate_matrix(config, gpu_count):
    names = set()
    for trial in config["trials"]:
        if trial.get("disabled_reason"):
            continue
        if trial["name"] in names:
            raise ValueError("Duplicate trial name")
        names.add(trial["name"])
        if (any(r.get("role") == "prefill" for r in trial["servers"])
                and any(r.get("connector") == "synthetic" for r in trial["servers"] if r.get("role") == "decode")):
            raise ValueError("Synthetic KV must not be labeled as real prefill handoff")
        if not trial["servers"] or not any(role.get("role") == "decode" for role in trial["servers"]):
            raise ValueError("A trial needs at least one decode endpoint")
        used = []
        for role in trial["servers"]:
            if role.get("role") not in {"prefill", "decode"}:
                raise ValueError("Each server must have a prefill/decode role")
            ids = role["gpus"]
            if not ids or any(not isinstance(i, int) or not 0 <= i < gpu_count for i in ids):
                raise ValueError("GPU outside this single-node allocation")
            used += ids
            expected = role.get("tp", 1) * role.get("pp", 1) * role.get("tknp", 1)
            if expected != len(ids):
                raise ValueError("TP x PP x TKNP must match assigned GPU count; DCP does not multiply it")
            if role.get("dcp", 1) > role.get("tp", 1) or role.get("tp", 1) % role.get("dcp", 1):
                raise ValueError("DCP must divide TP")
            if role.get("dcp", 1) > 1 and role.get("connector") == "nixl":
                raise ValueError("Real NIXL handoff into DCP interleaved KV is not implemented")
            if role.get("tknp", 1) > 1 and role.get("connector") not in {"nixl", "synthetic"}:
                raise ValueError("TKNP requires real NIXL or explicitly labeled synthetic KV")
            if any(not name.startswith(("VLLM_TKNP_", "NCCL_")) and name != "VLLM_KV_CACHE_LAYOUT"
                   for name in role.get("env", {})):
                raise ValueError("Role environment overrides are limited to TKNP/NCCL settings")
        if len(set(used)) != len(used):
            raise ValueError("Concurrent roles overlap GPUs")


def run_trial(trial, config, workload, output):
    here = Path(__file__).resolve().parent
    processes, logs, urls = [], [], {"prefill": [], "decode": []}
    launches = []
    output.mkdir()
    try:
        for index, server in enumerate(trial["servers"]):
            port = config.get("port_base", 8200) + index
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, server["gpus"])),
                       VLLM_ENABLE_V1_MULTIPROCESSING="0", OMP_NUM_THREADS="1",
                       VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1",
                       VLLM_NIXL_SIDE_CHANNEL_PORT=str(config.get("nixl_port_base", 19200) + index * 16),
                       VLLM_NIXL_ABORT_REQUEST_TIMEOUT=str(int(config.get("timeout", 600)) + 120),
                       VLLM_CACHE_ROOT=str(Path(config.get("cache_root", output / "cache")) / str(index)),
                       TRITON_CACHE_DIR=str(Path(config.get("triton_root", output / "triton")) / str(index)))
            env.update({key: str(value) for key, value in server.get("env", {}).items()})
            command = [sys.executable]
            if server.get("tknp", 1) > 1:
                command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(server['gpus'])}"]
            command += [str(here / "e2e_server.py"), "--port", str(port), "--result-dir", str(output / f"server{index}"),
                        "--model", config.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
                        "--load-format", config.get("load_format", "auto"),
                        "--max-model-len", str(config.get("max_model_len", 2048)),
                        "--max-num-seqs", str(server.get("max_num_seqs", config.get("max_num_seqs", 16))),
                        "--max-num-batched-tokens", str(server.get("max_num_batched_tokens", config.get("max_num_batched_tokens", 2048))),
                        "--kv-cache-dtype", config.get("kv_cache_dtype", "auto")]
            kv_bytes = server.get("kv_cache_memory_bytes", config.get("kv_cache_memory_bytes"))
            if kv_bytes is not None:
                command += ["--kv-cache-memory-bytes", str(kv_bytes)]
            command += ["--gpu-memory-utilization", str(server.get("gpu_memory_utilization", config.get("gpu_memory_utilization", 0.8)))]
            for option in ("tp", "pp", "tknp", "dcp", "connector", "placement"):
                if option in server:
                    command += ["--" + option, str(server[option])]
            for option in ("cuda_graph", "enable_dbo", "graph_slots", "fixed_wave", "disable_compile", "disable_combo_kernels", "record_logit_topk", "record_full_logits", "native_graph_shapes"):
                if server.get(option):
                    command += ["--" + option.replace("_", "-")]
            for option in ("graph_batch_size", "fixed_output_tokens", "profile_steps", "profile_after_step", "pp_decode_microbatch"):
                if option in server:
                    command += ["--" + option.replace("_", "-"), str(server[option])]
            if server.get("owner_counts"):
                command += ["--owner-counts"] + list(map(str, server["owner_counts"]))
            if config.get("require_bf16"):
                command += ["--require-bf16"]
            url = f"http://127.0.0.1:{port}"
            if config.get("slurm_job_id"):
                from e2e_slurm import launch_role
                # Do not forward the login host's complete environment as
                # explicit values. Only the scoped, recorded role settings.
                scoped = {key: value for key, value in env.items()
                          if key.startswith(("VLLM_", "TRITON_")) or key in {"OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}}
                scoped.update(server.get("env", {}))
                command, url = launch_role(config, server, index, command, scoped)
            launches.append({"command": command, "gpus": server["gpus"], "env_overrides": server.get("env", {})})
            (output / "launches.json").write_text(json.dumps(launches, indent=2) + "\n")
            log = (output / f"server{index}.log").open("x")
            logs.append(log)
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            urls[server["role"]].append(url)
        pending = set(urls["decode"] + urls["prefill"])
        deadline = time.monotonic() + 900
        while pending:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Server exited during initialization")
            for url in list(pending):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if json.load(response)["ready"]:
                            pending.remove(url)
                except (OSError, ValueError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server initialization timeout")
            time.sleep(1)
        command = [sys.executable, str(here / "e2e_workload.py"), "run", "--workload", str(workload),
                   "--output", str(output / "client"), "--decode-urls"] + urls["decode"]
        if urls["prefill"]:
            command += ["--prefill-urls"] + urls["prefill"]
        if trial.get("staged_decode"):
            command += ["--staged-decode"]
        if trial.get("prefill_barrier"):
            command += ["--prefill-barrier"]
        for key in ("warmup_requests", "routing", "timeout", "ttft_slo", "tpot_slo"):
            if key in config:
                command += ["--" + key.replace("_", "-"), str(config[key])]
        if config.get("slurm_job_id"):
            from e2e_slurm import client_command
            command = client_command(config, command)
        subprocess.run(command, check=True, timeout=config.get("timeout", 3600) + 120)
        # Transfer acknowledgements must drain, not just the client streams.
        deadline = time.monotonic() + 30
        while True:
            counts = []
            for url in urls["prefill"] + urls["decode"]:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    counts.append(json.load(response)["outstanding_requests"])
            if not any(counts):
                break
            if any(p.poll() is not None for p in processes) or time.monotonic() > deadline:
                raise RuntimeError("Request/KV state did not drain after completed workload")
            time.sleep(0.02)
    finally:
        for url in urls["prefill"] + urls["decode"]:
            try:
                with urllib.request.urlopen(urllib.request.Request(url + "/shutdown", data=b"{}", method="POST"), timeout=2):
                    pass
            except (OSError, ValueError):
                pass
        if config.get("slurm_job_id"):
            for index, role in enumerate(trial["servers"]):
                nodes = sorted({gpu // 4 for gpu in role["gpus"]})
                if len(nodes) < 2 or role.get("tknp", 1) > 1:
                    continue
                # First let the leader collect ALL worker execution counters.
                # Only then stop the remote native worker supervisors.
                evidence = output / f"server{index}/execution.json"
                deadline = time.monotonic() + 10
                while not evidence.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                for node in nodes[1:]:
                    url = f"http://{config['slurm_nodes'][node]}:{config.get('port_base', 8200) + 1000 + index}/shutdown"
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"), timeout=2):
                            pass
                    except (OSError, ValueError):
                        pass
            # Give the engines time to persist evidence and terminate their
            # own workers before signalling the SLURM step supervisors.
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        for process in processes:
            if process.poll() is None:
                # Let the supervisor drain its worker RPCs before teardown.
                # Killing the whole group first destroys MP workers before
                # the server can persist execution/cleanup evidence.
                process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
    rows = [json.loads(line) for line in (output / "client/requests.jsonl").read_text().splitlines()]
    telemetry = {}
    measured_ids = {row["server_request_id"] for row in rows}
    for journal in output.glob("server*/rank*.jsonl"):
        admitted = {}
        queue_seconds = {}
        ready_seconds = {}
        ready_times = {}
        compute_after_kv_seconds = {}
        measured_preemptions = 0
        steps = []
        shutdown = None
        for event in map(json.loads, journal.read_text().splitlines()):
            if event["event"] == "admit":
                admitted[event["request_id"]] = event["time"]
            elif event["event"] == "first_compute" and event["request_id"] in measured_ids:
                queue_seconds[event["request_id"]] = event["time"] - admitted[event["request_id"]]
                if event["request_id"] in ready_times:
                    compute_after_kv_seconds[event["request_id"]] = event["time"] - ready_times[event["request_id"]]
            elif event["event"] == "kv_ready":
                ready_times[event["request_id"]] = event["time"]
                if event["request_id"] in measured_ids:
                    ready_seconds[event["request_id"]] = event["time"] - admitted[event["request_id"]]
            elif event["event"] == "step":
                steps.append(event)
            elif event["event"] == "shutdown":
                shutdown = event
            elif event["event"] == "preempt" and event["request_id"] in measured_ids:
                measured_preemptions += 1
        telemetry[f"{journal.parent.name}/{journal.stem}"] = {
            "first_compute_queue_s": queue_seconds,
            "admit_to_kv_ready_s": ready_seconds,
            "kv_ready_to_first_compute_s": compute_after_kv_seconds,
            "kv_ready_timing_scope": "scheduler-visible completion; includes KV admission/handshake/transfer polling, NOT pure wire time",
            "max_waiting": max((step["waiting"] for step in steps), default=0),
            "max_waiting_capacity": max((step.get("waiting_capacity", 0) for step in steps), default=0),
            "max_preemptions_live": max((step["preemptions"] for step in steps), default=0),
            "preemptions_total": max((step.get("preemptions_total", 0) for step in steps), default=0),
            "preemptions_measured": measured_preemptions,
            "min_free_blocks": min((step["kv_free_blocks"] for step in steps), default=None),
            "shutdown": shutdown}
        if shutdown is None or shutdown["remaining_requests"] or not shutdown.get("all_kv_free", False):
            raise RuntimeError(f"Request/KV leak or incomplete shutdown: {journal}")
    result = {"trial": trial, "launches": launches, "telemetry": telemetry,
              "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
              "completed": sum(not row.get("error") for row in rows),
              "scope": config.get("scope", "single-node infrastructure smoke, not a throughput comparison")}
    from e2e_workload import percentile
    result["engine_stage_latency"] = {}
    for phase in ("prefill", "decode", "mixed", "transfer_only"):
        grouped = {}
        for journal in output.glob("server*/rank*.jsonl"):
            for step in map(json.loads, journal.read_text().splitlines()):
                if step["event"] != "step" or step.get("phase") != phase:
                    continue
                # The first server can be a producer or decoder. Group by
                # server and step, not by rank; count slowest-rank wall time.
                key = (journal.parent.name, step["step"])
                grouped[key] = max(grouped.get(key, 0), step["seconds"])
        result["engine_stage_latency"][phase] = {
            "steps": len(grouped), "slowest_rank_engine_seconds_sum": sum(grouped.values()),
            "p50_s": percentile(list(grouped.values()), 0.5),
            "p95_s": percentile(list(grouped.values()), 0.95),
            "scope": "batched engine-step wall time, NOT exclusive per-request GPU latency; stages may overlap"}
    result["worker_execution"] = {
        path.parent.name: json.loads(path.read_text()) for path in output.glob("server*/execution.json")}
    (output / "trial.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Fresh servers and warmup for each repetition")
    parser.add_argument("--alternate-order", action="store_true",
                        help="Reverse baseline order on odd repetitions to reduce order bias")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("Positive repetition count required")
    config = json.loads(args.matrix.read_text())
    import torch
    validate_matrix(config, torch.cuda.device_count())
    if args.only:
        if set(args.only) - {trial["name"] for trial in config["trials"]}:
            raise ValueError("Unknown selected trial")
        config["trials"] = [trial for trial in config["trials"] if trial["name"] in args.only]
    if args.validate_only:
        print("Matrix fits this single node")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "matrix.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "execution.json").write_text(json.dumps({
        "repetitions": args.repetitions, "alternate_order": args.alternate_order,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    results = []
    for repetition in range(args.repetitions):
        trials = config["trials"]
        if args.alternate_order and repetition % 2:
            trials = list(reversed(trials))
        for trial in trials:
            if trial.get("disabled_reason"):
                result = {"trial": trial["name"], "status": "not-run", "reason": trial["disabled_reason"]}
            else:
                name = (trial["name"] if args.repetitions == 1 else
                        f"rep{repetition:02d}-" + trial["name"])
                print(f"Validating {name}", flush=True)
                result = run_trial(trial, config, args.workload, args.output / name)
            result["repetition"] = repetition
            result["execution_index"] = len(results)
            results.append(result)
            # Keep completed evidence if a later trial fails; never silently
            # skip failed trials or pool repeats into one throughput number.
            (args.output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
