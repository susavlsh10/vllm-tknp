#!/usr/bin/env python3
"""Decode-only H100 efficiency investigation in ONE existing EOS allocation.

Real BF16 physical KV, dummy BF16 weights, no prefill or checkpoint download.
Performance measurements and profiler runs are separate. No allocation here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

from e1_70b_suite import build_trial, measured_caps, quotas, topology
from e2e_matrix import run_trial, validate_matrix
from e2e_workload import make_workload, percentile
from fp8_decode_sweep import dbo_layout

HERE = Path(__file__).resolve().parent
METHODS = ("tp", "pp-pipelined", "dp-lifetime", "tknp", "tknp-tp2",
           "tp2-pp4", "dp-tp2", "tknp-dbo", "tknp-tp2-dbo")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def configuration(job, node, output, context, tokens):
    repo = str(HERE.parents[2])
    frozen = Path(repo)/".research-source.json"
    commit = (json.loads(frozen.read_text())["commit"] if frozen.exists() else
              subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip())
    return dict(model="meta-llama/Llama-3.1-8B-Instruct", load_format="dummy",
        require_bf16=True, preinitialize_random_kv=True, kv_cache_dtype="auto",
        max_model_len=context + tokens, max_num_seqs=256, max_num_batched_tokens=512,
        gpu_memory_utilization=0.9, warmup_requests=8, warmup_output_tokens=8,
        timeout=1200, routing="least-work", slurm_job_id=str(job), slurm_nodes=[node],
        gpus_per_node=8, port_base=8880, nixl_port_base=21200, rdzv_port_base=26500,
        cache_root=str(output / "cache"), triton_root=str(output / "triton"),
        runtime=dict(repo=repo, source_commit=commit,
            python="/home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3",
            image="/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh",
            container_name="tknp_efficiency" + str(job),
            env=dict(VLLM_ATTENTION_BACKEND="FLASHINFER", OMP_NUM_THREADS="1",
                TORCHINDUCTOR_CACHE_DIR="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/inductor_torch291")),
        scope="Single-node 8B BF16 dummy-weight/random-physical-KV cached decode; queueing included, prefill/init/profiling excluded from performance")


def make_trial(method, mode, offered, caps, context, tokens, root_weight_aware=False):
    if method not in METHODS or mode not in ("eager", "graph"):
        raise ValueError("Unknown efficiency method/mode")
    if method.endswith("-dbo") and mode != "graph":
        raise ValueError("Current two-cohort prototype requires CUDA graphs")
    trial = build_trial(method, context, tokens, offered, list(range(8)), caps, "decode")
    trial["name"] = method + "-" + mode
    for role in trial["servers"]:
        role["cuda_graph"] = mode == "graph"
        role["disable_combo_kernels"] = False
        if mode == "eager":
            role.pop("owner_counts", None)
            role.pop("graph_slots", None)
            role["native_graph_shapes"] = False
            role["env"].pop("VLLM_TKNP_REQUIRE_CUDAGRAPH", None)
        if role.get("tknp", 1) > 1 and method.endswith("-dbo"):
            size, counts = dbo_layout(offered, caps)
            role.update(graph_batch_size=size, max_num_seqs=size, owner_counts=counts)
            role["env"].update(VLLM_TKNP_DBO_ROOT_REQUESTS=str(counts[0]),
                VLLM_TKNP_DBO_LOOKAHEAD="0")
        if root_weight_aware and role.get("tknp",1)>1:
            if mode!="graph":
                raise ValueError("Explicit root-aware ownership currently requires graph slots")
            from tknp_bandwidth_layout import bandwidth_aware_counts
            size=role["graph_batch_size"];degree=role["tknp"]
            dbo=role.get("enable_dbo",False)
            role["owner_counts"]=bandwidth_aware_counts(caps,size,
                root_dense_passes=2 if dbo else 1,minimum_peer=4 if dbo else 2,
                root_maximum=size//2-2*(degree-1) if dbo else None)
            if dbo:role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"]=str(role["owner_counts"][0])
    validate_matrix(dict(trials=[trial]), 8)
    return trial


def audit_point(path, trial, batch, tokens, profiled=False, context=32768):
    data = json.loads((path / "trial.json").read_text())
    client = json.loads((path / "client/summary.json").read_text())
    if (client["failed"] or client["completed"] != batch
            or client["output_tokens"] != batch * tokens):
        raise RuntimeError("Incomplete decode requests/tokens")
    if data["engine_stage_latency"]["prefill"]["steps"] or data["engine_stage_latency"]["mixed"]["steps"]:
        raise RuntimeError("Unexpected prefill/mixed work in a decode-only experiment")
    resources = {}
    for index, role in enumerate(trial["servers"]):
        key = "server" + str(index)
        manifest = json.loads((path / key / "manifest.json").read_text())
        rs = manifest["worker_resources"]
        records = data["worker_execution"][key]
        expected = list(range(len(role["gpus"])))
        if sorted(r["rank"] for r in records) != expected or sorted(r["rank"] for r in rs) != expected:
            raise RuntimeError("Missing worker precision/execution evidence")
        mode = "FULL" if role["cuda_graph"] else "NONE"
        for r in records:
            c = r["counts"]
            if (c.get("decode:" + mode, 0) <= 0
                    or any(v for k, v in c.items() if k.startswith("decode:") and k != "decode:" + mode)
                    or bool(c.get("graph:FULL_REPLAY", 0)) != role["cuda_graph"]
                    or r.get("per_request_kv_fill_calls", 0)):
                raise RuntimeError("Eager/graph execution or no-timed-KV-fill check failed")
        for r in rs:
            kv = r["synthetic_kv"]
            if (r["model_dtype"] != "torch.bfloat16" or r["quantization"] is not None
                    or r["kv_storage_dtypes"] != ["torch.bfloat16"]
                    or kv["std"] <= 0 or not kv["preinitialized_bytes"]):
                raise RuntimeError("BF16 weights/random physical KV validation failed")
        resources[key] = rs
    if any(t["preemptions_measured"] or t["shutdown"]["remaining_requests"]
           or not t["shutdown"]["all_kv_free"] for t in data["telemetry"].values()):
        raise RuntimeError("Preemption or request/KV leak")
    measured = {json.loads(x)["server_request_id"] for x in (path / "client/requests.jsonl").read_text().splitlines()}
    grouped, peaks, first, queues = {}, {}, {}, []
    for journal in path.glob("server*/rank*.jsonl"):
        admitted = {}
        for line in journal.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "admit":
                admitted[e["request_id"]] = e["time"]
            elif e["event"] == "first_compute" and e["request_id"] in measured:
                first.setdefault((journal.parent.name, e["request_id"]), e["time"])
                if journal.stem == "rank0":
                    queues.append(e["time"] - admitted[e["request_id"]])
            elif e["event"] == "step" and measured.intersection(e.get("scheduled_request_ids", [])):
                key = (journal.parent.name, e["step"])
                if key not in grouped or e["seconds"] > grouped[key]["seconds"]:
                    grouped[key] = e
                peaks[journal.parent.name] = max(peaks.get(journal.parent.name, 0), e["running"])
    sequences = {}
    for (server, _), e in sorted(grouped.items()):
        sequences.setdefault((server, tuple(sorted(e["scheduled_request_ids"]))), []).append(e)
    steady = [e["seconds"] for values in sequences.values() for e in values[4:-4]]
    engines = {}
    for (server, ids), values in sequences.items():
        for e in values[4:-4]:
            if e["running"] != peaks[server]:
                continue
            row = engines.setdefault(server,dict(tokens=0,seconds=0,steps=0))
            row["tokens"] += len(ids)
            row["seconds"] += e["seconds"]
            row["steps"] += 1
    for row in engines.values():
        row["output_tps"] = row["tokens"]/row["seconds"]
    starts, waves = {}, {}
    for (server, _), timestamp in first.items():
        starts.setdefault(server, []).append(timestamp)
    for server, timestamps in starts.items():
        groups = []
        for t in sorted(timestamps):
            if not groups or t - groups[-1][-1] > 2:
                groups.append([])
            groups[-1].append(t)
        waves[server] = list(map(len, groups))
    gated=client.get("admission_gated_decode")
    if gated:
        admitted=set()
        queues=[]
        for journal in path.glob("server*/rank0.jsonl"):
            events=[json.loads(line) for line in journal.read_text().splitlines()]
            holds=[e for e in events if e["event"]=="decode_gate" and e["action"]=="hold" and e.get("key")==gated["gate_key"]]
            releases=[e for e in events if e["event"]=="decode_gate" and e["action"]=="release" and e.get("key")==gated["gate_key"]]
            if len(holds)!=1 or len(releases)!=1:raise RuntimeError("Missing cached decode gate evidence")
            hold,release=holds[0],releases[0]
            if hold["compute_steps"]!=release["compute_steps"] or hold["kv_free_blocks"]!=release["kv_free_blocks"]:
                raise RuntimeError("GPU work or KV admission occurred before the decode timer")
            local={e["request_id"] for e in events if e["event"]=="admit" and e["request_id"] in measured and e["time"]<release["time"]}
            if len(local)!=release["remaining_requests"]:raise RuntimeError("CPU requests not fully admitted at release")
            admitted.update(local)
            queues.extend(e["time"]-release["time"] for e in events
                if e["event"]=="first_compute" and e["request_id"] in measured)
        if admitted!=measured:raise RuntimeError("Decode gate did not include every measured request")
    return dict(batch=batch, output_tokens=batch*tokens, performance_valid=not profiled,
        client=client, output_tps=gated["output_tps"] if gated else client["system_output_tps"],
        burst_output_tps=client["system_output_tps"],admission_gated=bool(gated),
        engine_step_p50_ms=1000*percentile(steady, .5) if steady else None,
        engine_step_p95_ms=1000*percentile(steady, .95) if steady else None,
        steady_engine_output_tps=sum(row["output_tps"] for row in engines.values()),
        steady_engine_details=engines,
        steady_scope="Peak-resident waves; trimmed per request set, slowest replicated rank; sums independent DP engine rates, PP uses actual scheduled microbatch tokens; excludes frontend dispatch/streaming and is NOT GPU-only time",
        capacities=measured_caps(path, context=context, tokens=tokens, trial=trial),
        peak_resident=peaks, admission_wave_sizes=waves, queue_p95_s=percentile(queues, .95),
        mode_verified="FULL" if trial["servers"][0]["cuda_graph"] else "NONE",
        all_ranks_verified=True, measured_preemptions=0, path=str(path))


def workload(output, context, batch, tokens):
    path = output / f"workload-c{context}-b{batch}-o{tokens}.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps(r) + "\n" for r in make_workload(batch, [context], [tokens], 0, 20261018)))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["probe", "matrix", "profile", "all"], default="all")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--profile-batch", type=int, default=128)
    p.add_argument("--capacity-source",type=Path,
                   help="Reuse only measured capacity discovery, never measurements")
    p.add_argument("--flashinfer-tensor-cores",choices=["0","1"],
                   help="Explicit attention-kernel A/B, applied to every method")
    p.add_argument("--metadata-cache",choices=["0","1"],
                   help="Matched TKNP immutable-index cache A/B")
    p.add_argument("--root-weight-aware",action="store_true",
                   help="Opt-in graph ownership accounting for root dense bandwidth")
    p.add_argument("--cpu-kv-control",choices=["0","1"],
                   help="A/B CPU-only TKNP admission-counter collectives")
    p.add_argument("--gpu-memory-utilization",type=float,default=0.9,
                   help="Physical memory budget, recorded separately for headroom controls")
    p.add_argument("--nccl-debug",choices=["WARN","INFO"],
                   help="Initialization diagnostics; do not use INFO timing as primary data")
    p.add_argument("--warmup-cohort-p2p",choices=["0","1"],
                   help="A/B root-star NCCL link initialization before KV sizing")
    p.add_argument("--startup-timeout",type=float,default=900,
                   help="Bound initialization failures independently of workload timeout")
    p.add_argument("--private-tp-custom",choices=["0","1"],
                   help="A/B native-priority private custom TP reductions for DBO")
    p.add_argument("--admission-gated-decode",action="store_true",
                   help="Time cached decode and natural KV waves after CPU request preparation")
    p.add_argument("--prompt-array-cache",choices=["0","1"],
                   help="A/B immutable CPU prompt arrays when PP rotates request sets")
    p.add_argument("--pp-compact-cached-tokens",choices=["0","1"],
                   help="A/B unused full-prompt IPC payload in synchronous PP")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = configuration(args.job, args.node, args.output, args.context, args.tokens)
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between zero and one")
    config["gpu_memory_utilization"]=args.gpu_memory_utilization
    if args.startup_timeout<=0:p.error("--startup-timeout must be positive")
    config["startup_timeout"]=args.startup_timeout
    config["root_weight_aware"]=args.root_weight_aware
    config["admission_gated_decode"]=args.admission_gated_decode
    if args.prompt_array_cache is not None:
        config["runtime"]["env"]["VLLM_CACHE_PROMPT_TOKEN_IDS"]=args.prompt_array_cache
    if args.pp_compact_cached_tokens is not None:
        config["runtime"]["env"]["VLLM_PP_COMPACT_CACHED_TOKENS"]=args.pp_compact_cached_tokens
    if args.nccl_debug:
        config["runtime"]["env"]["NCCL_DEBUG"]=args.nccl_debug
    if args.warmup_cohort_p2p is not None:
        config["runtime"]["env"]["VLLM_TKNP_DBO_WARMUP_P2P"]=args.warmup_cohort_p2p
    if args.private_tp_custom is not None:
        config["runtime"]["env"]["VLLM_TKNP_DBO_INDEPENDENT_TP_CUSTOM"]=args.private_tp_custom
    if args.flashinfer_tensor_cores is not None:
        config["runtime"]["env"]["VLLM_FLASHINFER_DECODE_USE_TENSOR_CORES"] = args.flashinfer_tensor_cores
    if args.metadata_cache is not None:
        config["runtime"]["env"]["VLLM_TKNP_CACHE_INDEX_TENSORS"] = args.metadata_cache
    if args.cpu_kv_control is not None:
        config["runtime"]["env"]["VLLM_TKNP_CPU_KV_CONTROL"] = args.cpu_kv_control
    state_path = args.output / "summary.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        config=config, capacities={}, trials=[], profiles=[], failures=[],
        dcp="Nontrivial native DCP is unsupported for 8-KV-head Llama 8B at TP<=8; DCP1 is identical to TP8")
    if args.capacity_source:
        source=json.loads((args.capacity_source/"summary.json").read_text())
        if source["config"]["model"]!=config["model"] or source["config"]["max_model_len"]!=config["max_model_len"]:
            raise ValueError("Capacity source must match model, context, output reservation")
        for method,caps in source["capacities"].items():
            state["capacities"].setdefault(method,caps)
        state["capacity_source"]=str(args.capacity_source)
    write_json(args.output / "plan.json", dict(config=config, methods=args.methods, modes=args.modes,
        batches=args.batches, context=args.context, tokens=args.tokens, repeats=args.repeats))
    def persist():
        write_json(state_path, state)
    def run_one(method, mode, batch, caps, tag, profile=False):
        trial = make_trial(method, mode, batch, caps, args.context, args.tokens,args.root_weight_aware)
        if profile:
            trial["servers"][0].update(profile_steps=8, profile_after_step=16)
        length = 64 if profile else 16
        cfg = dict(config, trials=[trial])
        dst = args.output / tag
        if dst.exists():
            raise RuntimeError("Preserve an incomplete attempt and choose a new output directory: " + str(dst))
        print("START", tag, flush=True)
        run_trial(trial, cfg, workload(args.output, args.context, batch, length), dst)
        row = audit_point(dst, trial, batch, length, profile, args.context)
        row.update(method=method, mode=mode)
        print("PASS", tag, "TPS", round(row["output_tps"], 2), "CAP", row["capacities"], flush=True)
        return row
    if args.stage in ("probe", "all"):
        for method in args.methods:
            # One graph-aware capacity discovery per topology. Eager uses the
            # same conservative admission budget in the matched measurements.
            if method in state["capacities"]:
                continue
            tp, pp, tk, dcp, replicas = topology(method, 8)
            batch = 4*tk if method.endswith("-dbo") else max(16, 4*tk)
            caps = ([2] + [6]*(tk-1) if method.endswith("-dbo") else
                    [max(4, batch//replicas//tk)] * (replicas*tk))
            row = run_one(method, "graph", batch, caps, "probe-" + method)
            # Probe executes 16 tokens but reserves the full primary lifetime.
            trial = make_trial(method, "graph", batch, caps, args.context, args.tokens,args.root_weight_aware)
            state["capacities"][method] = measured_caps(Path(row["path"]), args.context, args.tokens, trial)
            persist()
    if args.stage in ("matrix", "all"):
        for method in args.methods:
            for mode in args.modes:
                if method.endswith("-dbo") and mode == "eager":
                    continue
                if all(any(r["method"]==method and r["mode"]==mode and r["batch"]==b and r["repeat"]==rep
                           for r in state["trials"]) for b in args.batches for rep in range(args.repeats)):
                    continue
                caps = state["capacities"][method]
                trial = make_trial(method, mode, max(args.batches), caps, args.context, args.tokens,args.root_weight_aware)
                layouts, sizes = {}, {}
                role = trial["servers"][0]
                if role.get("tknp", 1)>1 and mode=="graph":
                    for batch in args.batches:
                        point = make_trial(method, mode, batch, caps, args.context, args.tokens,args.root_weight_aware)["servers"][0]
                        layouts[point["graph_batch_size"]] = point["owner_counts"]
                        sizes[batch] = point["graph_batch_size"]
                    role.pop("owner_counts")
                    role["graph_layouts"] = layouts
                destination = args.output / (method + "-" + mode)
                attempt = 0
                while destination.exists():
                    attempt += 1
                    destination = args.output / f"{method}-{mode}-retry{attempt}"
                segments = []
                for rep in range(args.repeats):
                    for batch in args.batches if rep%2==0 else reversed(args.batches):
                        key=f"b{batch}-r{rep}"
                        segment=dict(key=key, batch=batch, repeat=rep,
                            workload=str(workload(args.output,args.context,batch,args.tokens)),
                            output=str(args.output/(destination.name+"-"+key)))
                        if sizes:
                            segment["graph_size"] = sizes[batch]
                        if role.get("pp",1)>1:
                            segment["pp_decode_microbatch"] = max(1, math.ceil(min(batch,caps[0])/role["pp"]))
                        segments.append(segment)
                smokes=[]
                for index, segment in enumerate(segments[:len(args.batches)]):
                    key="smoke-"+segment["key"]
                    smokes.append(dict(segment,key=key,smoke_output_tokens=4,
                        workload=str(workload(args.output,args.context,segment["batch"],4)),
                        output=str(args.output/(destination.name+"-"+key))))
                cfg=dict(config,trials=[trial],client_sweep=smokes+segments)
                print("START_GROUP", destination.name, "CAP", caps, flush=True)
                write_json(args.output/(destination.name+"-matrix.json"),cfg)
                run_trial(trial,cfg,Path(smokes[0]["workload"]),destination)
                for segment in smokes:
                    audit_point(Path(segment["output"]),trial,segment["batch"],4,context=args.context)
                if sizes:
                    from tknp_graph_sweep import audit_sweep
                    audit_sweep(dict(cfg,output_tokens=args.tokens),segments)
                for segment in segments:
                    row=audit_point(Path(segment["output"]),trial,segment["batch"],args.tokens,context=args.context)
                    row.update(method=method,mode=mode,repeat=segment["repeat"],
                        layout_policy="root-bandwidth-proxy" if args.root_weight_aware else "capacity-balanced",
                        workload_sha256=hashlib.sha256(Path(segment["workload"]).read_bytes()).hexdigest())
                    state["trials"].append(row);persist()
                    print("PASS_GROUP",destination.name,segment["key"],"TPS",round(row["output_tps"],2),flush=True)
    if args.stage in ("profile", "all"):
        for method in [m for m in args.methods if m.startswith("tknp")]:
            for mode in args.modes:
                if method.endswith("-dbo") and mode=="eager":continue
                if any(r["method"]==method and r["mode"]==mode for r in state["profiles"]):continue
                row=run_one(method,mode,args.profile_batch,state["capacities"][method],
                    "profile-"+method+"-"+mode,True)
                state["profiles"].append(row);persist()
    state["last_stage_completed"]=args.stage
    state["updated_time"]=time.time();persist()
    print("EFFICIENCY_STAGE_COMPLETE",args.stage,"POINTS",len(state["trials"]),flush=True)


if __name__ == "__main__":
    main()
