#!/usr/bin/env python3
"""Resumable 70B matrix in an existing SLURM allocation, with per-point failures.

Real NIXL E2E and synthetic-KV decode controls are separate experiment families.
No allocations, credentials, downloads, or retrospective point selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from e2e_matrix import run_trial, validate_matrix
from e1_pilot_report import analyze
from e2e_slurm import client_command

HERE = Path(__file__).resolve().parent
MODEL = "/lustre/fsw/hw_nresearch_snoise/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
GIB = 1024 ** 3


def topology(method, count):
    if method in {"tknp", "tknp-dbo"}:
        return 1, 1, count, 1, 1
    if method.startswith("tknp-tp"):
        tp = int(method.removeprefix("tknp-tp").removesuffix("-dbo"))
        return tp, 1, count // tp, 1, 1
    if method in {"dp-native", "dp-lifetime"}:
        return 1, 1, 1, 1, count
    if method.startswith("dp-tp"):
        tp = int(method.removeprefix("dp-tp"))
        return tp, 1, 1, 1, count // tp
    if method == "tp":
        return count, 1, 1, 1, 1
    if method in {"pp", "pp-pipelined"}:
        return 1, count, 1, 1, 1
    if method.startswith("tp") and "-pp" in method:
        tp_text, pp_text = method.removesuffix("-pipelined").split("-pp")
        tp, pp = int(tp_text[2:]), int(pp_text)
        if tp * pp != count:
            raise ValueError(f"{method} needs {tp * pp} GPUs, not {count}")
        return tp, pp, 1, 1, 1
    if method == "dcp":
        if count < 16:
            raise ValueError("70B DCP requires TP>8 KV heads: minimum TP16+DCP2 (16 GPUs)")
        return count, 1, 1, count // 8, 1
    raise ValueError((method, count))


def validate_paper_plan(methods, decoder_count, family, prefill_tp):
    """Keep historical low-TP results readable, but never collect them anew."""
    if family == "e2e" and prefill_tp != 8:
        raise ValueError("Llama paper collection requires TP8 prefill")
    for method in methods:
        tp, pp, tk, dcp, replicas = topology(method, decoder_count)
        if tp * pp * tk * replicas != decoder_count:
            raise ValueError("Topology does not consume the complete decode GPU budget")
        if method == "dcp":
            if family != "decode":
                raise ValueError("DCP real KV handoff is unsupported; use the labeled decode-only family")
        elif tp != 8 or pp not in (1, 2) or pp == 2 and tk * replicas != 1:
            raise ValueError("Paper collection requires TP8 first and at most the TP8xPP2 baseline")
        if method.startswith("tknp") and tk < 2:
            raise ValueError("Token parallel collection requires at least two TP8 groups")


def quotas(caps, offered):
    """Balanced attention load subject to physical lifetime capacity."""
    result = [0] * len(caps)
    for _ in range(min(offered, sum(caps))):
        eligible = [i for i, cap in enumerate(caps) if result[i] < cap]
        owner = min(eligible, key=lambda i: (result[i], i))
        result[owner] += 1
    return result


def best_tknp_capacity(capacities, gpu_count):
    values = []
    for method, caps in capacities.items():
        if method.startswith("tknp"):
            degree = topology(method, gpu_count)[2]
            values.append(sum(caps) // degree * degree)
    return max(values) if values else sum(next(iter(capacities.values())))


def measurement_order(capacities, frontier_control=False):
    """Cover sharded baselines before potentially long native-DP recomputation.

    This changes trial order only, never workloads, admission or measurements.
    Both independent-replica controls remain scheduled and resumable.
    """
    controls = {"dp-native", "dp-lifetime"}
    ordered = ([m for m in capacities if m not in controls] +
               [m for m in capacities if m in controls])
    if frontier_control:
        # Under a bounded extra-control budget, cover every method family
        # before spending time on additional within-family configurations.
        priority = ["dcp", "tp", "tknp-tp8", "dp-tp8", "tp8-pp2", "tp4-pp4", "dp-native"]
        ordered.sort(key=lambda m: priority.index(m) if m in priority else len(priority))
    return ordered


def retryable_port_startup(error, logs):
    return ("Server exited during initialization" in error and
            any("EADDRINUSE" in log or "address already in use" in log.lower() for log in logs))


def measurement_passes(contexts, repeats):
    """Finish cross-context sharded coverage before slow replica controls.

    Probes still include every configuration, so both passes derive identical
    offered batches. Single-context paired repeats retain pair-wise reversal.
    """
    if len(contexts) < 2 or repeats > 1:
        return [(context, None) for context in contexts]
    return ([(context, False) for context in contexts] +
            [(context, True) for context in reversed(contexts)])


def tknp_capacity_rebalance_needed(trial, actual, offered):
    """A graph-sized profile can move a slot between root and attention pools.

    Replan only if physically achievable, aligned lanes exceed the current
    usable owner quotas. Never round DOWN a plan that already uses more lanes.
    Otherwise one overestimated root slot can create an entire tail wave while
    an attention GPU still has unused capacity.
    """
    role = next(r for r in trial["servers"] if r["role"] == "decode")
    degree = role.get("tknp", 1)
    if degree < 2:
        return False
    usable = sum(min(q, cap) for q, cap in zip(role["owner_counts"], actual))
    achievable = min(math.ceil(offered / degree) * degree, sum(actual) // degree * degree)
    return achievable > usable


def build_trial(method, context, tokens, offered, decoder_gpus, caps, family, prefill_gpus=None, prefill_tp=1):
    tp, pp, tk, dcp, replicas = topology(method, len(decoder_gpus))
    servers = []
    if family == "e2e":
        pool = prefill_gpus or [0]
        if len(pool) % prefill_tp:
            raise ValueError("Prefill GPU pool must contain complete TP groups")
        servers.extend(dict(role="prefill", tp=prefill_tp, gpus=pool[i:i+prefill_tp], connector="nixl",
                            env={"VLLM_TKNP_LIFETIME_ADMISSION": "1"})
                       for i in range(0, len(pool), prefill_tp))
    if method == "dcp" and family == "e2e":
        raise ValueError("Real KV handoff to DCP is unsupported; never relabel synthetic KV")
    for replica in range(replicas):
        width = tp * pp * tk
        own_gpus = decoder_gpus[replica * width:(replica + 1) * width]
        # Native DP must be allowed prompt-only admission and recomputation;
        # do not cap max_num_seqs at the lifetime-aware baseline's slot count.
        # DCP also keeps native admission: the optional ordinary-P/D lifetime
        # guard counts physical block_size, not DCP's enlarged logical pages.
        # Applying it unchanged would artificially halve DCP2's KV capacity.
        native = method in {"dp-native", "dcp"}
        lane_caps = caps[replica * tk:(replica + 1) * tk]
        count = math.ceil(offered / replicas)
        if tk > 1:
            target = min(math.ceil(count / tk) * tk, sum(lane_caps) // tk * tk)
            owner_counts = quotas(lane_caps, target)
        else:
            owner_counts = None
        lanes = sum(owner_counts) if owner_counts else count if native else min(count, lane_caps[0])
        if lanes < 1:
            raise ValueError("A decoder cannot fit one full request lifetime")
        env = {"VLLM_KV_CACHE_LAYOUT": "HND", "VLLM_TKNP_LIFETIME_ADMISSION": "0" if native else "1"}
        if tk > 1:
            env.update(VLLM_TKNP_REQUIRE_CUDAGRAPH="1",
                       VLLM_TKNP_CAPTURE_REBALANCE="1",
                       VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(context + tokens))
        server = dict(role="decode", gpus=own_gpus, tp=tp, pp=pp, tknp=tk, dcp=dcp,
                      connector="nixl" if family == "e2e" else "synthetic",
                      cuda_graph=True, disable_combo_kernels=True,
                      native_graph_shapes=tk == 1,
                      graph_batch_size=lanes, max_num_seqs=lanes, env=env)
        if tk > 1:
            server.update(placement="cost", graph_slots=True, owner_counts=owner_counts,
                          enable_dbo=method.endswith("-dbo"))
            if method.endswith("-dbo"):
                env.update(VLLM_TKNP_DBO_ZERO_ROOT_C1="1", VLLM_TKNP_DBO_INDEPENDENT_COMM="1",
                    VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE="1", VLLM_TKNP_DBO_INDEPENDENT_TP="1",
                    VLLM_TKNP_DBO_LAYOUT="asymmetric", VLLM_TKNP_DBO_LOOKAHEAD="1",
                    VLLM_TKNP_DBO_SYMM_MEM="0", VLLM_TKNP_DBO_C0_FRACTION="0.5",
                    VLLM_TKNP_DBO_ROOT_REQUESTS=str(owner_counts[0]))
        if pp > 1:
            microbatch = max(1, math.ceil(lanes / pp))
            server.update(pp_decode_microbatch=microbatch, graph_batch_size=microbatch)
        servers.append(server)
    return {"name": f"{family}-c{context}-b{offered}-o{tokens}-{method}", "servers": servers}


def measured_caps(path, context, tokens, trial):
    result = []
    for index, role in enumerate(trial["servers"]):
        if role["role"] != "decode":
            continue
        manifest = json.loads((path / f"server{index}/manifest.json").read_text())
        resources = manifest["worker_resources"]
        # A physical DCP page covers dcp times as many logical sequence tokens.
        # PP ranks each store only their layer interval; take the tightest rank.
        values = [max(0, (r["kv_blocks"] - 1) // math.ceil(
            (context + tokens) / (r["block_tokens"] * role.get("dcp", 1)))) for r in resources]
        tp = role.get("tp", 1)
        for owner in range(role.get("tknp", 1)):
            members = values[owner * tp:(owner + 1) * tp] if role.get("tknp", 1) > 1 else values
            result.append(min(members))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--runtime-json", type=Path, help="Explicit container/Python paths; EOS readiness matrix runtime object")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--prefill-tp", type=int, default=8)
    parser.add_argument("--job", required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--decoder-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--prefill-gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536, 98304])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--family", choices=["e2e", "decode"], default="e2e")
    parser.add_argument("--tokens", type=int, default=1025)
    parser.add_argument("--batches", type=int, nargs="+", help="Otherwise capacity-driven three-point sweep")
    parser.add_argument("--regimes", nargs="+", choices=["below", "crossover", "overload"],
                        default=["below", "crossover", "overload"])
    parser.add_argument("--max-hours", type=float, default=3.7)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=1800,
                        help="Client request deadline in seconds; does not validate partial runs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="Recorded per-server memory budget; reserve headroom for wide NCCL fanout")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Matched unprofiled repeats; reverse configuration order on alternate pairs")
    args = parser.parse_args()
    if args.methods is None:
        args.methods = ["tknp-tp8", "dp-tp8", "tp8-pp2"] + (["dcp"] if args.family == "decode" else [])
    validate_paper_plan(args.methods, len(args.decoder_gpus), args.family, args.prefill_tp)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, load_format="dummy", kv_cache_dtype="auto", require_bf16=True,
                  max_num_seqs=128, max_num_batched_tokens=1024,
                  gpu_memory_utilization=args.gpu_memory_utilization, warmup_requests=0, routing="least-work", timeout=args.request_timeout,
                  slurm_job_id=args.job, slurm_nodes=args.nodes,
                  gpus_per_node=args.gpus_per_node,
                  decoder_gpus=args.decoder_gpus,
                  cache_root=str(args.output.parent / "shared_cache"),
                  triton_root=str(args.output.parent / "shared_triton"),
                  scope=("70B BF16 architecture proxy, dummy weights; " +
                         ("real model-computed BF16 KV and NIXL P/D transfer" if args.family == "e2e" else
                          "synthetic BF16 KV decode-only control; NO prefill/transfer measurement")))
    if args.runtime_json:
        config["runtime"] = json.loads(args.runtime_json.read_text())
    status_path = args.output / "suite.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"config": config, "attempts": []}
    state["config"] = config
    state.pop("finished_time", None)
    state.pop("paused_reason", None)
    deadline = time.monotonic() + args.max_hours * 3600
    def persist():
        status_path.write_text(json.dumps(state, indent=2) + "\n")
    persist()
    def execute(trial, context, tokens, offered, startup_retries=1):
        requested_trial = trial
        config["max_model_len"] = context + args.tokens
        directory = args.output / trial["name"]
        old = next((a for a in state["attempts"] if a["status"] == "completed" and
                    (a["name"] == trial["name"] or a["name"].startswith(trial["name"] + "-retry"))), None)
        if old:
            return old
        if time.monotonic() > deadline:
            state["paused_reason"] = "Suite wall-time budget reached; resume the same output directory"
            state["paused_time"] = time.time()
            persist()
            raise TimeoutError("Suite wall-time budget reached; completed trials retained for resume")
        if directory.exists():
            trial = dict(trial, name=trial["name"] + f"-retry{len(state['attempts'])}")
            directory = args.output / trial["name"]
        workload = args.output / f"workload-c{context}-b{offered}-o{tokens}.jsonl"
        if not workload.exists():
            command = [sys.executable, str(HERE / "e2e_workload.py"), "generate", "--natural",
                       "--tokenizer", args.model, "--output", str(workload), "--count", str(offered),
                       "--lengths", str(context), "--output-lengths", str(tokens)]
            subprocess.run(client_command(config, command), check=True, timeout=120)
        validate_matrix(dict(config, trials=[trial]), len(args.nodes) * args.gpus_per_node)
        (args.output / (trial["name"] + "-config.json")).write_text(json.dumps(dict(config, trials=[trial]), indent=2) + "\n")
        row = dict(name=trial["name"], status="running", context=context, tokens=tokens, offered=offered,
                   workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(), start_time=time.time())
        state["attempts"].append(row)
        persist()
        print("START", row["name"], flush=True)
        try:
            run_trial(trial, config, workload, directory)
            record = analyze(directory)
            caps = measured_caps(directory, context, args.tokens, trial)
            if not record["all_decode_workers_full_graph"] or not record["all_servers_kv_drained"]:
                raise RuntimeError("Missing full decode graph replay or KV-drain evidence")
            row.update(status="completed", client=record["client"], capacities=caps,
                       preemptions_zero=record["all_servers_zero_measured_preemptions"],
                       full_graph=True, kv_drained=True, end_time=time.time())
            (directory / "analysis.json").write_text(json.dumps(record, indent=2) + "\n")
            print("DONE", row["name"], "TPS", record["client"]["system_output_tps"], "CAP", caps, flush=True)
        except Exception:
            row.update(status="failed", error=traceback.format_exc(), end_time=time.time())
            print("FAILED", row["name"], row["error"], flush=True)
        persist()
        if (row["status"] == "failed" and startup_retries and retryable_port_startup(
                row["error"], [p.read_text(errors="replace") for p in directory.glob("server*.log")])):
            print("RETRY_PORT_STARTUP", row["name"], flush=True)
            return execute(requested_trial, context, tokens, offered, startup_retries - 1)
        return row
    for context, control_pass in measurement_passes(args.contexts, args.repeats):
        capacities = {}
        for method in args.methods:
            try:
                tp, pp, tk, dcp, replicas = topology(method, len(args.decoder_gpus))
            except ValueError as exc:
                state.setdefault("unsupported", {})[f"c{context}-{method}"] = str(exc)
                persist()
                continue
            base_name = f"probe-{args.family}-c32768-{method}"
            base = next((r for r in state["attempts"] if r["status"] == "completed"
                         and (r["name"] == base_name or r["name"].startswith(base_name + "-retry"))), None)
            if context != 32768 and base:
                # Physical BF16 pools are measured, not guessed from advertised
                # HBM. Recompute page-rounded lifetime slots for this context;
                # every real trial still records its fresh runtime pool and
                # undersized admission is recalibrated below.
                path = args.output / base["name"]
                source_trial = json.loads((path / "trial.json").read_text())["trial"]
                capacities[method] = measured_caps(path, context, args.tokens, source_trial)
                state.setdefault("capacity_probe_sources", {})[f"c{context}-{method}"] = str(path)
                persist()
                continue
            probe_count = max(8, tk, replicas)
            probe_caps = [1 if tk > 1 else 8] * tk * replicas
            if method.endswith("-dbo"):
                # Keep both cohorts away from changing singleton position
                # shapes while exercising the real pipelined schedule.
                probe_count = 8 if tk == 2 else 8 * tk
                probe_caps = [2, 6] if tk == 2 else [8] * tk
            trial = build_trial(method, context, args.tokens, probe_count, args.decoder_gpus,
                                probe_caps, args.family, args.prefill_gpus, args.prefill_tp)
            trial["name"] = f"probe-{args.family}-c{context}-{method}"
            row = execute(trial, context, 17, probe_count)
            if row["status"] == "completed":
                capacities[method] = row["capacities"]
        state.setdefault("capacities", {})[str(context)] = capacities
        persist()
        if args.probe_only:
            continue
        if not capacities:
            continue
        tk_capacity = best_tknp_capacity(capacities, len(args.decoder_gpus))
        dp_capacity = sum(capacities.get("dp-tp8", capacities.get("dp-lifetime",
            capacities.get("dp-native", [max(1, tk_capacity // 3)]))))
        regime_batches = dict(below=max(1, dp_capacity), crossover=max(1, tk_capacity),
                              overload=max(tk_capacity + 2, math.ceil(tk_capacity * 1.25)))
        batches = args.batches or sorted({regime_batches[r] for r in args.regimes})
        state.setdefault("batches", {})[str(context)] = batches
        persist()
        for batch_index, offered in enumerate(batches):
            for repeat_index in range(args.repeats):
                methods = measurement_order(capacities, frontier_control=(
                    len(args.decoder_gpus) == 16 and args.regimes in [["below"], ["overload"]]))
                if control_pass is not None:
                    methods = [m for m in methods if (m in {"dp-native", "dp-lifetime"}) == control_pass]
                if (batch_index + repeat_index) % 2:
                    methods.reverse()
                for method in methods:
                    trial = build_trial(method, context, args.tokens, offered,
                                        args.decoder_gpus, capacities[method], args.family, args.prefill_gpus, args.prefill_tp)
                    if repeat_index:
                        trial["name"] += f"-repeat{repeat_index}"
                    row = execute(trial, context, args.tokens, offered)
                    actual = row.get("capacities")
                    old = capacities[method]
                    rebalance = (row["status"] == "completed" and
                                 tknp_capacity_rebalance_needed(trial, actual, offered))
                    growth = (row["status"] == "completed" and not method.startswith("tknp")
                              and actual != old and all(a >= b for a, b in zip(actual, old))
                              and offered > sum(old) and method not in {"dp-native", "dcp"})
                    if rebalance or growth:
                        # Never manufacture a capacity win by retaining an
                        # undersized max_num_seqs from a tiny startup probe.
                        state.setdefault("superseded_trials", []).append(row["name"])
                        state.setdefault("recalibration_notes", {})[row["name"]] = dict(
                            reason="runtime owner-pool rebalance" if rebalance else "runtime pool capacity growth",
                            probe_capacities=old, runtime_capacities=actual,
                            original_servers=trial["servers"])
                        capacities[method] = actual
                        trial = build_trial(method, context, args.tokens, offered,
                                            args.decoder_gpus, actual, args.family, args.prefill_gpus, args.prefill_tp)
                        if repeat_index:
                            trial["name"] += f"-repeat{repeat_index}"
                        trial["name"] += "-capacity-recalibrated"
                        execute(trial, context, args.tokens, offered)
    state["finished_time"] = time.time()
    persist()
    print("SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        print("SUITE_PAUSED", exc, flush=True)
