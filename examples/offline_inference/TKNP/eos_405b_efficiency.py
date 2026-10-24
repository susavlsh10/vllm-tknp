#!/usr/bin/env python3
"""405B efficiency controls inside one existing four-node EOS allocation.

No allocation or downloads. Fixed recipe, real random BF16 KV, dummy mixed
FP8/BF16 weights. Admit CPU requests before timing, but include all GPU KV
admission, decode and natural capacity waves. Profile separately from timing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from e1_70b_suite import measured_caps
from e2e_matrix import run_trial, validate_matrix
from e2e_workload import make_workload, percentile
from eos_efficiency import write_json
from fp8_decode_sweep import FOUR_NODE_METHODS, audit, configuration, trial_for, pp_segment_override
from tknp_graph_sweep import audit_sweep


def validate_capacity_source(source_arguments, arguments):
    keys=("model","context","tokens","attention_backend","query_groups",
          "tensor_cores","gpu_memory_utilization")
    changed=[key for key in keys if source_arguments.get(key)!=arguments.get(key)]
    if changed:
        raise ValueError("Capacity source needs fresh probes; incompatible "+", ".join(changed))


def make_trial(method, batch, caps, context, tokens, root_requests=None,max_resident=None,pp_baseline="standard"):
    planned=min(batch,max_resident) if max_resident and method.startswith("tknp") else batch
    root_empty = root_requests==0 and method.endswith("-dbo")
    planner_caps=list(caps)
    if root_empty:
        if min(caps[1:])<4 or sum(caps[1:])//len(caps)*len(caps)<16:
            raise ValueError("Projection-only root needs supported nonempty peer cohorts")
        planned=min(planned,sum(caps[1:])//len(caps)*len(caps))
        # The legacy helper constructs only positive-root plans. Its
        # temporary root quota is replaced below, never admitted or captured.
        planner_caps[0]=max(2,planner_caps[0])
    trial = trial_for(method, context, tokens, planned, planner_caps, 32, pp_baseline)
    trial.update(offered_batch=batch,configured_admission_target=planned)
    if root_empty:
        from e1_70b_suite import quotas
        role=trial["servers"][0]
        role["owner_counts"]=[0]+quotas(caps[1:],role["graph_batch_size"])
        role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"]="0"
        return trial
    if root_requests is not None and method.startswith("tknp"):
        from e1_70b_suite import quotas
        role = trial["servers"][0]
        size = role["graph_batch_size"]
        minimum = 4 if role.get("enable_dbo") else 1
        maximum = min(caps[0], size - minimum * (len(caps) - 1))
        if role.get("enable_dbo"):
            maximum = min(maximum, size // 2 - 2 * (len(caps) - 1))
        # Keep the supported non-singleton root cohort shape. Zero-root
        # throughout both cohorts is a separate unvalidated implementation.
        lower = max(2 if role.get("enable_dbo") else 1, size - sum(caps[1:]))
        root = min(max(root_requests, lower), maximum)
        if root < lower:
            raise ValueError("Requested ownership cannot fit the graph")
        counts = [root] + quotas(caps[1:], size - root)
        if sum(counts) != size or min(counts[1:]) < minimum:
            raise ValueError("Unsupported owner layout")
        role["owner_counts"] = counts
        if role.get("enable_dbo"):
            role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"] = str(root)
    return trial


def summarize(path, trial, batch, tokens, context, profiled=False):
    evidence = audit(path, trial, batch, tokens)
    stages=json.loads((path/"trial.json").read_text())["engine_stage_latency"]
    if stages["prefill"]["steps"] or stages["mixed"]["steps"]:
        raise RuntimeError("Prefill/mixed work in a decode-only measurement")
    client = evidence["client"]
    gated = client.get("admission_gated_decode")
    if not gated:
        raise RuntimeError("Missing decode-only admission gate")
    measured = {json.loads(line)["server_request_id"]
                for line in (path / "client/requests.jsonl").read_text().splitlines()}
    admitted, queue, grouped, peaks = set(), [], {}, {}
    for journal in path.glob("server*/rank*.jsonl"):
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        if journal.stem == "rank0":
            holds = [e for e in events if e["event"] == "decode_gate"
                     and e["action"] == "hold" and e.get("key") == gated["gate_key"]]
            releases = [e for e in events if e["event"] == "decode_gate"
                        and e["action"] == "release" and e.get("key") == gated["gate_key"]]
            if len(holds) != 1 or len(releases) != 1:
                raise RuntimeError("Missing or repeated gate boundary")
            hold, release = holds[0], releases[0]
            if (hold["compute_steps"] != release["compute_steps"]
                    or hold["kv_free_blocks"] != release["kv_free_blocks"]):
                raise RuntimeError("GPU work/KV admission before the timer")
            local = {e["request_id"] for e in events if e["event"] == "admit"
                     and e["request_id"] in measured and e["time"] < release["time"]}
            if len(local) != release["remaining_requests"]:
                raise RuntimeError("Incomplete CPU admission before release")
            admitted.update(local)
            queue.extend(e["time"] - release["time"] for e in events
                         if e["event"] == "first_compute" and e["request_id"] in measured)
        for e in events:
            if e["event"] != "step" or not measured.intersection(e.get("scheduled_request_ids", [])):
                continue
            key = journal.parent.name, e["step"]
            if key not in grouped or e["seconds"] > grouped[key]["seconds"]:
                grouped[key] = e
            peaks[journal.parent.name] = max(peaks.get(journal.parent.name, 0), e["running"])
    if admitted != measured:
        raise RuntimeError("Gate omitted measured requests")
    segments = {}
    for (server, _), e in sorted(grouped.items()):
        segments.setdefault((server, tuple(sorted(e["scheduled_request_ids"]))), []).append(e)
    steady, engines = [], {}
    for (server, _), events in segments.items():
        for e in events[4:-4]:
            steady.append(e["seconds"])
            if e["running"] == peaks[server]:
                row = engines.setdefault(server, dict(tokens=0, seconds=0, steps=0))
                row["tokens"] += len(e["scheduled_request_ids"])
                row["seconds"] += e["seconds"]
                row["steps"] += 1
    for row in engines.values():
        row["output_tps"] = row["tokens"] / row["seconds"]
    return dict(batch=batch, context=context, output_length=tokens,
        output_tokens=client["output_tokens"], output_tps=gated["output_tps"],
        burst_output_tps=client["system_output_tps"], client=client,
        capacities=measured_caps(path, context, tokens, trial), peak_resident=peaks,
        queue_p50_s=percentile(queue, .5), queue_p95_s=percentile(queue, .95),
        engine_step_p50_ms=1000*percentile(steady, .5) if steady else None,
        engine_step_p95_ms=1000*percentile(steady, .95) if steady else None,
        steady_engine_output_tps=sum(r["output_tps"] for r in engines.values()),
        steady_engine_details=engines, performance_valid=not profiled,
        full_graph_all_ranks=True, measured_preemptions=0, path=str(path),
        scope="Admission-gated cached decode completion, natural capacity waves included; CPU request preparation/init/prefill excluded; slowest-rank engine steps are a separate diagnostic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--nodes", nargs=4, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["probe", "matrix", "profile", "all"], default="all")
    p.add_argument("--methods", nargs="+", choices=FOUR_NODE_METHODS, default=list(FOUR_NODE_METHODS))
    p.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 96, 128])
    p.add_argument("--extra-capture-batches",nargs="*",type=int,default=[],
                   help="Additional TKNP graph layouts, not additional measured workloads (e.g. tail 32/64)")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--pp-baseline",choices=["standard","tuned"],default="standard")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--smoke-tokens", type=int, default=8)
    p.add_argument("--capacity-source", type=Path)
    p.add_argument("--root-requests", type=int,
                   help="Desired root quota, clipped to supported shapes and physical capacity")
    p.add_argument("--max-tknp-resident",type=int,
                   help="Explicit admission/performance tradeoff, not a claim of physical KV capacity")
    p.add_argument("--tensor-cores", choices=["0", "1", "auto"], default="auto")
    p.add_argument("--query-groups",choices=["0","2"],default="0")
    p.add_argument("--tail-downsize",action="store_true")
    p.add_argument("--dbo-threshold",type=int,default=2,
                   help="Capture/replay native graphs below this size, cohort graphs at/above it")
    p.add_argument("--private-tp-custom",choices=["0","1"],default="0")
    p.add_argument("--private-tp-custom-bytes",type=int)
    p.add_argument("--attention-backend",choices=["FLASHINFER","FLASH_ATTN"],default="FLASH_ATTN")
    p.add_argument("--nccl-max-ctas",type=int)
    p.add_argument("--p2p-net-ll",choices=["0","1"])
    p.add_argument("--p2p-ll-threshold",type=int)
    p.add_argument("--nccl-graph-register",choices=["0","1"])
    p.add_argument("--gpu-memory-utilization", type=float, default=.9)
    p.add_argument("--profile-batch", type=int, default=64)
    p.add_argument("--profile-after-step",type=int,default=16)
    p.add_argument("--profile-steps",type=int,default=8)
    p.add_argument("--profile-all-methods",action="store_true")
    args = p.parse_args()
    if (min(args.context, args.tokens, args.repeats, args.smoke_tokens, *args.batches,*args.extra_capture_batches) < 1
            or len(set(args.batches)) != len(args.batches)
            or any(batch>max(args.batches) for batch in args.extra_capture_batches)
            or args.dbo_threshold<2
            or args.profile_after_step<0 or args.profile_steps<1
            or not 0 < args.gpu_memory_utilization < 1
            or args.root_requests is not None and args.root_requests < 0
            or args.max_tknp_resident is not None and (args.max_tknp_resident<16 or args.max_tknp_resident%4)):
        p.error("Invalid workload/ownership budget")
    args.output.mkdir(parents=True, exist_ok=True)
    config = configuration(args.job, args.nodes, args.model, args.context, args.tokens)
    config.update(admission_gated_decode=True, timeout=3600, startup_timeout=1200,
                  gpu_memory_utilization=args.gpu_memory_utilization)
    config["runtime"]["env"]["VLLM_FLASHINFER_DECODE_USE_TENSOR_CORES"] = args.tensor_cores
    config["runtime"]["env"]["VLLM_FLASHINFER_GQA16_QUERY_GROUPS"] = args.query_groups
    config["runtime"]["env"]["VLLM_TKNP_GRAPH_TAIL_DOWNSIZE"] = "1" if args.tail_downsize else "0"
    config["runtime"]["env"]["VLLM_TKNP_DBO_DECODE_THRESHOLD"] = str(args.dbo_threshold)
    config["runtime"]["env"]["VLLM_TKNP_DBO_INDEPENDENT_TP_CUSTOM"] = args.private_tp_custom
    config["runtime"]["env"]["VLLM_TKNP_DBO_PRIVATE_TP_MAX_BYTES"] = "0"
    if args.private_tp_custom_bytes is not None:
        if args.private_tp_custom!="1" or not 256*1024<=args.private_tp_custom_bytes<=8*1024**2:
            p.error("Private TP byte override requires custom=1 and 256 KiB through 8 MiB")
        config["runtime"]["env"]["VLLM_TKNP_DBO_PRIVATE_TP_MAX_BYTES"]=str(args.private_tp_custom_bytes)
    config["runtime"]["env"]["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    config["runtime"]["env"]["VLLM_RESEARCH_RANK_TRITON_CACHE"] = "1"
    config["runtime"]["env"]["VLLM_RESEARCH_RANK_INDUCTOR_CACHE"] = "1"
    if args.nccl_max_ctas is not None:
        if not 1<=args.nccl_max_ctas<=64:p.error("NCCL_MAX_CTAS must be between 1 and 64")
        config["runtime"]["env"]["NCCL_MAX_CTAS"]=str(args.nccl_max_ctas)
    for flag,value in (("NCCL_ALLOC_P2P_NET_LL_BUFFERS",args.p2p_net_ll),
                       ("NCCL_P2P_LL_THRESHOLD",args.p2p_ll_threshold),
                       ("NCCL_GRAPH_REGISTER",args.nccl_graph_register)):
        if value is not None:
            if flag=="NCCL_P2P_LL_THRESHOLD" and value<1:p.error("Positive P2P LL byte threshold required")
            config["runtime"]["env"][flag]=str(value)
    # NCCL warmup is a memory-accounting/stability control, not a new timing path.
    config["runtime"]["env"]["VLLM_TKNP_DBO_WARMUP_P2P"] = "1"
    # launch_role deliberately forwards scoped inherited VLLM/NCCL settings.
    # Explicit experiment flags must win over a stale login-shell setting.
    for key,value in config["runtime"]["env"].items():
        if key.startswith(("VLLM_","NCCL_")):
            os.environ[key]=str(value)
    write_json(args.output / "plan.json", dict(config=config, arguments={
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    state_path = args.output / "summary.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(capacities={}, trials=[], profiles=[], failures=[])
    if args.capacity_source:
        source_arguments=json.loads((args.capacity_source/"plan.json").read_text())["arguments"]
        validate_capacity_source(source_arguments,vars(args))
        if "tp8-pp4" in args.methods and source_arguments.get("pp_baseline","tuned") != args.pp_baseline:
            raise ValueError("PP baseline changed: fresh PP capacity probes required")
        source = json.loads((args.capacity_source / "summary.json").read_text())
        for method in args.methods:
            if method in source["capacities"]:
                state["capacities"].setdefault(method, source["capacities"][method])
    def persist():
        write_json(state_path, state)
    def workload(batch, tokens):
        target = args.output / f"workload-c{args.context}-b{batch}-o{tokens}.jsonl"
        if not target.exists():
            target.write_text("".join(json.dumps(r)+"\n" for r in make_workload(
                batch, [args.context], [tokens], 0, 20261019)))
        return target
    def one(method, batch, caps, tag, profiled=False):
        trial = make_trial(method, batch, caps, args.context, args.tokens, args.root_requests,args.max_tknp_resident,args.pp_baseline)
        if profiled:
            for role in trial["servers"]:
                role.update(profile_steps=args.profile_steps, profile_after_step=args.profile_after_step)
            if method.startswith("tknp") and args.extra_capture_batches:
                role=trial["servers"][0]
                layouts={role["graph_batch_size"]:role.pop("owner_counts")}
                for extra in args.extra_capture_batches:
                    point=make_trial(method,extra,caps,args.context,args.tokens,args.root_requests,args.max_tknp_resident)["servers"][0]
                    layouts[point["graph_batch_size"]]=point["owner_counts"]
                role["graph_layouts"]=layouts
        destination = args.output / tag
        cfg = dict(config, trials=[trial])
        validate_matrix(cfg, 32)
        write_json(args.output / (tag+"-matrix.json"), cfg)
        print("START", tag, "CAP", caps, flush=True)
        count = args.tokens if profiled else 16
        run_trial(trial, cfg, workload(batch, count), destination)
        result = summarize(destination, trial, batch, count, args.context, profiled)
        print("PASS", tag, result["output_tps"], result["capacities"], flush=True)
        return result, trial
    try:
        for method in args.methods:
            if method in state["capacities"]:
                continue
            batch = 16 if method.endswith("-dbo") else 4
            caps = ([2, 6, 6, 6] if method.endswith("-dbo") else
                    [2]*4 if method in ("tknp-tp8", "dp-tp8") else [batch])
            row, trial = one(method, batch, caps, "probe-"+method)
            state["capacities"][method] = measured_caps(Path(row["path"]), args.context, args.tokens, trial)
            persist()
        if args.stage in ("all", "matrix"):
            for method in args.methods:
                completed = {(r["batch"], r["repeat"]) for r in state["trials"] if r["method"] == method}
                if all((b,r) in completed for b in args.batches for r in range(args.repeats)):
                    continue
                caps = state["capacities"][method]
                trial = make_trial(method, max(args.batches), caps, args.context, args.tokens, args.root_requests,args.max_tknp_resident,args.pp_baseline)
                role = trial["servers"][0]
                sizes, layouts = {}, {}
                if method.startswith("tknp"):
                    for batch in dict.fromkeys(args.batches+args.extra_capture_batches):
                        point = make_trial(method, batch, caps, args.context, args.tokens, args.root_requests,args.max_tknp_resident)["servers"][0]
                        sizes[batch] = point["graph_batch_size"]
                        layouts[sizes[batch]] = point["owner_counts"]
                    role.pop("owner_counts")
                    role["graph_layouts"] = layouts
                destination = args.output / ("persistent-"+method)
                segments = []
                for rep in range(args.repeats):
                    for batch in args.batches if rep%2 == 0 else reversed(args.batches):
                        if (batch,rep) in completed:
                            continue
                        key = f"b{batch}-r{rep}"
                        segment = dict(key=key, batch=batch, repeat=rep,
                            workload=str(workload(batch,args.tokens)),
                            output=str(args.output/(destination.name+"-"+key)))
                        if sizes:
                            segment["graph_size"] = sizes[batch]
                        segment.update(pp_segment_override(role,batch,caps[0]))
                        segments.append(segment)
                smokes = []
                for segment in segments[:len(args.batches)]:
                    key = "smoke-"+segment["key"]
                    smokes.append(dict(segment,key=key,smoke_output_tokens=args.smoke_tokens,
                        workload=str(workload(segment["batch"],args.smoke_tokens)),
                        output=str(args.output/(destination.name+"-"+key))))
                cfg = dict(config,trials=[trial],client_sweep=smokes+segments)
                validate_matrix(cfg,32)
                write_json(args.output/(destination.name+"-matrix.json"),cfg)
                print("START_GROUP", method, "CAP", caps, "LAYOUTS", layouts, flush=True)
                results = run_trial(trial,cfg,Path(smokes[0]["workload"]),destination)
                if sizes:
                    audit_sweep(dict(cfg,output_tokens=args.tokens),segments)
                for segment,result in zip(smokes+segments,results):
                    n = segment.get("smoke_output_tokens",args.tokens)
                    row = summarize(Path(segment["output"]),result["trial"],segment["batch"],n,args.context)
                    if "smoke_output_tokens" in segment:
                        continue
                    row.update(method=method,repeat=segment["repeat"],pp_baseline=args.pp_baseline if role.get("pp",1)>1 else None,
                        tensor_cores=args.tensor_cores,root_requests=args.root_requests,
                        tuning=dict(dbo_threshold=args.dbo_threshold,tail_downsize=args.tail_downsize,
                            max_tknp_resident=args.max_tknp_resident,private_tp_custom=args.private_tp_custom,
                            private_tp_custom_bytes=args.private_tp_custom_bytes,nccl_max_ctas=args.nccl_max_ctas,
                            p2p_net_ll=args.p2p_net_ll,p2p_ll_threshold=args.p2p_ll_threshold,
                            extra_capture_batches=args.extra_capture_batches),
                        workload_sha256=hashlib.sha256(Path(segment["workload"]).read_bytes()).hexdigest())
                    state["trials"].append(row)
                    persist()
                    print("PASS_GROUP",method,segment["key"],round(row["output_tps"],2),flush=True)
        if args.stage in ("all","profile"):
            for method in args.methods:
                if not method.startswith("tknp") and not args.profile_all_methods:
                    continue
                row,_ = one(method,args.profile_batch,state["capacities"][method],"profile-"+method,True)
                state["profiles"].append(dict(row,method=method))
                persist()
        state["finished_time"] = time.time()
        persist()
        print("405B_EFFICIENCY_COMPLETE",args.stage,len(state["trials"]),flush=True)
    except BaseException:
        state["failures"].append(dict(time=time.time(),error=traceback.format_exc()))
        persist()
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
