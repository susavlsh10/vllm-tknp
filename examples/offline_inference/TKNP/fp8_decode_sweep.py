#!/usr/bin/env python3
"""FP8 decode sweep with one loaded engine per method in an existing allocation.

Real KV page capacity and admission, synthetic random prefilled KV, no prefill
compute/transfer/fill in the request clock. No allocation or checkpoint download.
"""
import argparse
import hashlib
import json
import math
import signal
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from e1_70b_suite import build_trial, measured_caps, quotas, tknp_capacity_rebalance_needed
from e1_pilot_report import analyze
from e2e_matrix import run_trial, validate_matrix
from e2e_workload import make_workload, percentile

METHODS = ("dp-tp8", "tknp-tp8", "tknp-tp8-dbo", "tp8-pp2", "tp8-pp4", "tp", "dcp")
FOUR_NODE_METHODS = ("dp-tp8", "tknp-tp8", "tknp-tp8-dbo", "tp8-pp4", "tp", "dcp")
HERE = Path(__file__).resolve().parent


def configuration(job, nodes, model, context, tokens):
    repo = str(HERE.parents[2])
    snapshot = Path(repo) / ".research-source.json"
    commit = (json.loads(snapshot.read_text())["commit"] if snapshot.exists() else
              subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
    return dict(model=model, tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        load_format="dummy", kv_cache_dtype="auto", require_fbgemm_fp8=True,
        dummy_fbgemm_fp8=True, preinitialize_random_kv=True,
        max_model_len=context + tokens, max_num_seqs=48, max_num_batched_tokens=512,
        gpu_memory_utilization=0.9, warmup_requests=max(2, len(nodes)), warmup_output_tokens=16,
        timeout=1800, routing="least-work", slurm_job_id=job, slurm_nodes=nodes,
        gpus_per_node=8, port_base=8700, nixl_port_base=22000, rdzv_port_base=25400,
        cache_root=f"/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/fp8_sweep_roles/{job}",
        triton_root=f"/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/fp8_sweep_triton/{job}",
        runtime=dict(repo=repo,
            python="/home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3",
            image="/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh",
            source_commit=commit,
            container_name="fp8_decode_" + str(job),
            env=dict(VLLM_ATTENTION_BACKEND="FLASHINFER", OMP_NUM_THREADS="1",
                TORCHINDUCTOR_CACHE_DIR="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/inductor_torch291")),
        scope="405B FBGEMM mixed FP8/BF16 dummy weights, BF16 random KV, decode-only; no accuracy claims")


def dbo_layout(offered, caps):
    """Capacity-safe equal cohorts, with >=2 rows per attention owner/cohort.

    Small offered batches use a larger masked graph, never dummy requests/KV.
    Root C1 has no attention; the root C0 shape stays nonsingleton.
    """
    degree = len(caps)
    minimum = 4 * degree
    size = min(max(minimum, math.ceil(offered / degree) * degree), sum(caps) // degree * degree)
    if degree < 2 or size < minimum or caps[0] < 2 or min(caps[1:]) < 4:
        raise ValueError("Insufficient capacity for supported two-cohort TKNP layouts")
    root = min(caps[0], size // degree, size // 2 - 2 * (degree - 1))
    counts = [root] + quotas(caps[1:], size - root)
    if sum(counts) != size or min(counts[1:]) < 4:
        raise ValueError("Unsupported capacity distribution for two-cohort TKNP")
    return size, counts


def configure_pp_baseline(trial, mode="standard"):
    """Keep native PP concurrency; explicitly separate historical tuning.

    Standard PP uses the normal scheduler token budget and graph buckets, not
    our ceil(residents / PP) scheduler override or PP CPU fast paths.
    """
    if mode not in ("standard", "tuned"):
        raise ValueError("Unknown PP baseline: " + mode)
    for role in trial["servers"]:
        if role.get("pp", 1) <= 1:
            continue
        role["pp_baseline"] = mode
        role["env"].update(VLLM_PP_COMPACT_CACHED_TOKENS="1" if mode == "tuned" else "0",
                           VLLM_CACHE_PROMPT_TOKEN_IDS="1" if mode == "tuned" else "0")
        if mode == "standard":
            role.pop("pp_decode_microbatch", None)
            role["graph_batch_size"] = role["max_num_seqs"]
    return trial


def pp_segment_override(role, batch, capacity):
    if role.get("pp", 1) > 1 and role.get("pp_baseline") == "tuned":
        return {"pp_decode_microbatch": max(1, math.ceil(min(batch, capacity) / role["pp"]))}
    return {}


def trial_for(method, context, tokens, offered, caps, gpu_count=16, pp_baseline="standard"):
    trial = build_trial(method, context, tokens, offered, list(range(gpu_count)), caps, "decode")
    for role in trial["servers"]:
        role["disable_combo_kernels"] = False
        if method == "dcp":
            # DCP pages cover dcp times the ordinary logical token span.
            # The lifetime guard now accounts for this, so the same known-OSL
            # admission contract as DP/PP/TKNP avoids decode-only recomputation
            # without artificially shrinking DCP's physical KV capacity.
            lanes = min(offered, caps[0])
            role.update(max_num_seqs=lanes, graph_batch_size=lanes)
            role["env"]["VLLM_TKNP_LIFETIME_ADMISSION"] = "1"
        if method == "tknp-tp8-dbo":
            size, counts = dbo_layout(offered, caps)
            role.update(graph_batch_size=size, max_num_seqs=size, owner_counts=counts)
            # Match the validated multi-layout prototype, not unvalidated
            # cross-layer lookahead. Keep on/off results explicitly separate.
            role["env"].update(VLLM_TKNP_DBO_LOOKAHEAD="0",
                VLLM_TKNP_DBO_ROOT_REQUESTS=str(counts[0]))
    return configure_pp_baseline(trial, pp_baseline)


def audit(path, trial, count, tokens):
    evidence = analyze(path)
    for key in ("all_decode_workers_full_graph", "all_servers_zero_measured_preemptions", "all_servers_kv_drained"):
        if not evidence[key]:
            raise RuntimeError("Validation failed: " + key)
    for index, role in enumerate(trial["servers"]):
        name = f"server{index}"
        if role.get("pp_baseline") == "standard":
            manifest=json.loads((path/name/"manifest.json").read_text())
            settings=manifest["runtime_env"]
            if (manifest["arguments"].get("pp_decode_microbatch") is not None
                    or manifest["engine_batch_queue_size"] != role["pp"]
                    or manifest["scheduler_max_num_scheduled_tokens"] != manifest["arguments"]["max_num_batched_tokens"]
                    or any(settings.get(key)!="0" for key in
                           ("VLLM_PP_COMPACT_CACHED_TOKENS","VLLM_CACHE_PROMPT_TOKEN_IDS"))):
                raise RuntimeError("Standard PP accidentally retained tuning or lost native concurrency")
        expected = list(range(len(role["gpus"])))
        resources = evidence["resources"][name]
        executions = evidence["execution"].get(name, [])
        if sorted(r["rank"] for r in resources) != expected or sorted(r["rank"] for r in executions) != expected:
            raise RuntimeError("Missing assigned rank evidence")
        for resource in resources:
            if (resource["model_dtype"] != "torch.bfloat16" or resource["quantization"] != "fbgemm_fp8"
                    or resource["kv_storage_dtypes"] != ["torch.bfloat16"]):
                raise RuntimeError("Weight/KV precision mismatch")
            kv = resource["synthetic_kv"]
            if not kv["preinitialized_bytes"] or kv["std"] <= 0 or kv["per_request_fill_calls"]:
                raise RuntimeError("Random KV was not preinitialized outside requests")
            if any(backend != "cutlass" for backend in resource["fp8_backends"]):
                raise RuntimeError("Unexpected FP8 kernel backend")
        if not any(resource["fp8_backends"] for resource in resources):
            raise RuntimeError("No FP8 dense kernels in this engine")
        for record in executions:
            if record["counts"].get("graph:FULL_REPLAY", 0) <= 0 or record["per_request_kv_fill_calls"]:
                raise RuntimeError("Graph replay or no-timed-fill evidence failed")
    if evidence["client"]["completed"] != count or evidence["client"]["output_tokens"] != count * tokens:
        raise RuntimeError("Incomplete requests/output tokens")
    return evidence


def summarize(path, method, batch, context, tokens, trial):
    evidence = audit(path, trial, batch, tokens)
    client = evidence["client"]
    requests = list(map(json.loads, (path / "client/requests.jsonl").read_text().splitlines()))
    measured_ids = {r["server_request_id"] for r in requests}
    queue = []
    step_groups = {}
    peaks = {}
    first = {}
    for file in sorted(path.glob("server*/rank*.jsonl")):
        events = list(map(json.loads, file.read_text().splitlines()))
        admitted = {}
        for event in events:
            if event["event"] == "admit":
                admitted[event["request_id"]] = event["time"]
            elif event["event"] == "first_compute" and event["request_id"] in measured_ids:
                first.setdefault((file.parent.name, event["request_id"]), event["time"])
                if file.stem == "rank0":
                    queue.append(event["time"] - admitted[event["request_id"]])
            elif event["event"] == "step" and set(event.get("scheduled_request_ids", [])) & measured_ids:
                key = (file.parent.name, event["step"])
                previous = step_groups.get(key)
                if previous is None or event["seconds"] > previous["seconds"]:
                    step_groups[key] = event
                peaks[file.parent.name] = max(peaks.get(file.parent.name, 0), event["running"])
    # Exclude admission and completion edges independently for each scheduled
    # request set (each wave/microbatch), grouping replicated ranks by max time.
    segments = {}
    for key, event in sorted(step_groups.items()):
        signature = (key[0], tuple(sorted(event["scheduled_request_ids"])))
        segments.setdefault(signature, []).append(event["seconds"])
    steady = [seconds for values in segments.values() for seconds in values[4:-4]]
    # Quantify waves by first compute times; streaming HTTP arrival jitter
    # should not turn one resident wave into several apparent waves.
    starts = {}
    for (server, req), timestamp in first.items():
        starts.setdefault(server, []).append(timestamp)
    waves = {}
    for server, values in starts.items():
        groups = []
        for timestamp in sorted(values):
            if not groups or timestamp - groups[-1][-1] > 2.0:
                groups.append([])
            groups[-1].append(timestamp)
        waves[server] = [len(group) for group in groups]
    return dict(method=method, batch=batch, context=context, output_length=tokens,
        completed=client["completed"], output_tokens=client["output_tokens"],
        output_tokens_per_second=client["system_output_tps"],
        client=client, capacities=measured_caps(path, context, tokens, trial),
        max_running=peaks, admission_wave_sizes=waves,
        queue_p50_s=percentile(queue, 0.5), queue_p95_s=percentile(queue, 0.95),
        steady_engine_step_ms_p50=percentile(steady, 0.5) * 1000 if steady else None,
        steady_engine_step_ms_p95=percentile(steady, 0.95) * 1000 if steady else None,
        full_graph_all_ranks=True, random_kv_preinitialized=True, measured_preemptions=0,
        path=str(path), scope="cached-decode burst completion; queueing included; prefill/KV fill/init excluded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16, 32, 48])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--pp-baseline", choices=["standard", "tuned"], default="standard")
    parser.add_argument("--smoke-tokens", type=int, default=0,
                        help="Run every captured layout briefly before primary measurements, retaining the full output-length reservation")
    parser.add_argument("--fresh-engines", action="store_true",
                        help="Diagnostic: reload every method per point instead of reusing graph/layout caches")
    args = parser.parse_args()
    gpu_count = len(args.nodes) * 8
    if args.methods is None:
        args.methods = list(FOUR_NODE_METHODS if gpu_count == 32 else
                            ("dp-tp8", "tknp-tp8", "tp8-pp2", "tp", "dcp"))
    if len(args.nodes) < 2:
        parser.error("405B collection requires at least two nodes; use the 8B readiness driver for single-node development")
    if gpu_count != 16 and "tp8-pp2" in args.methods:
        parser.error("TP8xPP2 consumes 16 GPUs; select the desired full-budget methods explicitly for larger allocations")
    if gpu_count != 32 and "tp8-pp4" in args.methods:
        parser.error("TP8xPP4 requires exactly 32 GPUs")
    if (args.tokens < 1 or args.context < 1 or args.smoke_tokens < 0
            or not args.batches or any(b < 1 for b in args.batches)
            or len(set(args.batches)) != len(args.batches)):
        parser.error("Positive lengths and distinct positive batch sizes required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = configuration(args.job, args.nodes, args.model, args.context, args.tokens)
    plan = dict(config=config, methods=args.methods, batches=args.batches,
                context=args.context, tokens=args.tokens, native_engine_reuse=not args.fresh_engines,
                tknp_multi_layout_reuse=not args.fresh_engines, smoke_tokens=args.smoke_tokens,
                pp_baseline=args.pp_baseline)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    state_path = args.output / "summary.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(capacities={}, trials=[], failures=[])
    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    def execute(method, batch, output_tokens, caps, tag):
        trial = trial_for(method, args.context, args.tokens, batch, caps, gpu_count, args.pp_baseline)
        point_config = dict(config, trials=[trial])
        validate_matrix(point_config, gpu_count)
        destination = args.output / tag
        attempt = 0
        while destination.exists():
            attempt += 1
            destination = args.output / f"{tag}-retry{attempt}"
        workload = args.output / f"workload-b{batch}-o{output_tokens}.jsonl"
        if not workload.exists():
            workload.write_text("".join(json.dumps(row) + "\n" for row in make_workload(
                batch, [args.context], [output_tokens], 0, 20261016)))
        (args.output / f"{destination.name}-matrix.json").write_text(json.dumps(point_config, indent=2) + "\n")
        print("START", destination.name, "caps", caps, flush=True)
        run_trial(trial, point_config, workload, destination)
        row = summarize(destination, method, batch, args.context, output_tokens, trial)
        row["workload_sha256"] = hashlib.sha256(workload.read_bytes()).hexdigest()
        print("PASS", destination.name, "TPS", row["output_tokens_per_second"],
              "capacity", row["capacities"], flush=True)
        return row, trial
    def execute_group(method, batches, caps):
        trial = trial_for(method, args.context, args.tokens, max(batches), caps, gpu_count, args.pp_baseline)
        point_layouts = {}
        if method.startswith("tknp-tp8"):
            role = trial["servers"][0]
            layouts = {}
            for batch in batches:
                point = trial_for(method, args.context, args.tokens, batch, caps, gpu_count)["servers"][0]
                size = point["graph_batch_size"]
                layouts[size] = point["owner_counts"]
                point_layouts[batch] = size
            role.pop("owner_counts")
            role["graph_layouts"] = layouts
        destination = args.output / ("persistent-" + method)
        attempt = 0
        while destination.exists():
            attempt += 1
            destination = args.output / f"persistent-{method}-retry{attempt}"
        segments = []
        for batch in batches:
            workload = args.output / f"workload-b{batch}-o{args.tokens}.jsonl"
            if not workload.exists():
                workload.write_text("".join(json.dumps(row) + "\n" for row in make_workload(
                    batch, [args.context], [args.tokens], 0, 20261016)))
            segment = dict(key=f"b{batch}", batch=batch, workload=str(workload),
                           output=str(args.output / (destination.name + f"-b{batch}")))
            segment.update(pp_segment_override(trial["servers"][0], batch, caps[0]))
            if method.startswith("tknp-tp8"):
                segment["graph_size"] = point_layouts[batch]
            segments.append(segment)
        smoke_segments = []
        if args.smoke_tokens:
            # Exercise all layouts and a return to a prior layout before
            # collecting ANY primary throughput point in this engine.
            for index, original in enumerate(segments + segments[:1]):
                batch = original["batch"]
                workload = args.output / f"workload-b{batch}-smoke-o{args.smoke_tokens}.jsonl"
                if not workload.exists():
                    workload.write_text("".join(json.dumps(row) + "\n" for row in make_workload(
                        batch, [args.context], [args.smoke_tokens], 0, 20261016)))
                key = f"smoke-b{batch}-r{index}"
                smoke_segments.append(dict(original, key=key, workload=str(workload),
                    smoke_output_tokens=args.smoke_tokens,
                    output=str(args.output / (destination.name + "-" + key))))
        all_segments = smoke_segments + segments
        point_config = dict(config, trials=[trial], client_sweep=all_segments)
        validate_matrix(point_config, gpu_count)
        (args.output / (destination.name + "-matrix.json")).write_text(json.dumps(point_config, indent=2) + "\n")
        print("START", destination.name, "batches", batches, "caps", caps, flush=True)
        results = run_trial(trial, point_config, Path(all_segments[0]["workload"]), destination)
        for segment, result in zip(smoke_segments, results):
            audit(Path(segment["output"]), result["trial"], segment["batch"], args.smoke_tokens)
        if method.startswith("tknp-tp8"):
            from tknp_graph_sweep import audit_sweep
            audit_sweep(dict(point_config, output_tokens=args.tokens), segments)
            if smoke_segments:
                audit_sweep(dict(point_config, output_tokens=args.smoke_tokens), smoke_segments)
        for segment, result in zip(segments, results[len(smoke_segments):]):
            row = summarize(Path(segment["output"]), method, segment["batch"], args.context, args.tokens, result["trial"])
            row.update(engine_reuse_group=str(destination),
                       workload_sha256=hashlib.sha256(Path(segment["workload"]).read_bytes()).hexdigest())
            state["trials"].append(row)
            persist()
            print("PASS", destination.name, segment["key"], "TPS", row["output_tokens_per_second"], flush=True)
    for method in args.methods:
        if method in state["capacities"]:
            continue
        try:
            probe_batch = 4 * len(args.nodes) if method.endswith("-dbo") else max(4, len(args.nodes))
            probe_caps = ([2] + [6] * (len(args.nodes) - 1) if method.endswith("-dbo")
                else [2] * len(args.nodes) if method in ("tknp-tp8", "dp-tp8") else [probe_batch])
            row, trial = execute(method, probe_batch, 16, probe_caps, "probe-" + method)
            # Probes reserve 1024 output tokens even though they execute only 16.
            caps = measured_caps(Path(row["path"]), args.context, args.tokens, trial)
            state["capacities"][method] = caps
            persist()
        except Exception:
            state["failures"].append(dict(stage="probe", method=method, error=traceback.format_exc()))
            persist()
            raise
    if not args.fresh_engines:
        for method in args.methods:
            batches = [batch for batch in args.batches if not any(
                row["method"] == method and row["batch"] == batch for row in state["trials"])]
            if batches:
                try:
                    execute_group(method, batches, state["capacities"][method])
                except Exception:
                    state["failures"].append(dict(stage="persistent_measurement", method=method,
                        batches=batches, error=traceback.format_exc()))
                    persist()
                    raise
    for batch in args.batches:
        order = args.methods if args.batches.index(batch) % 2 == 0 else list(reversed(args.methods))
        for method in order:
            if any(row["method"] == method and row["batch"] == batch for row in state["trials"]):
                continue
            try:
                caps = state["capacities"][method]
                row, trial = execute(method, batch, args.tokens, caps, f"b{batch}-" + method)
                if method.startswith("tknp-tp8") and tknp_capacity_rebalance_needed(trial, row["capacities"], batch):
                    state["failures"].append(dict(stage="capacity_replan", method=method, batch=batch,
                        path=row["path"], actual_capacities=row["capacities"]))
                    caps = row["capacities"]
                    row, trial = execute(method, batch, args.tokens, caps, f"b{batch}-{method}-rebalanced")
                state["trials"].append(row)
                persist()
            except Exception:
                state["failures"].append(dict(stage="measurement", method=method, batch=batch, error=traceback.format_exc()))
                persist()
                raise
    state["finished_time"] = time.time()
    persist()
    print("FP8_DECODE_SWEEP_COMPLETE", len(state["trials"]), flush=True)


if __name__ == "__main__":
    # Batch-shell background children can inherit ignored SIGINT. Restore a
    # handler so the job's shutdown trap reaches run_trial's coordinated finally.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
