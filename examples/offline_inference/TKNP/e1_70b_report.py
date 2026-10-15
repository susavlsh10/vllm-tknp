#!/usr/bin/env python3
"""Audit completed suite points and export matched comparisons (no GPU use)."""
import argparse
from bisect import bisect_right
import json
import os
import re
from pathlib import Path
from statistics import mean, stdev
from e1_capacity_report import snapshots
from e2e_workload import percentile
from e2e_server import owner_graphs_consistent


def canonical_attempt(name):
    return re.sub(r"-retry\d+", "", name).replace("-capacity-recalibrated", "")


def stage_latencies(requests, real_prefill):
    """Client stage boundaries; transfer/admission are not pure kernel time."""
    fields = {
        "prefill_queue_and_compute_s": [r["prefill_s"] or 0 for r in requests],
        "decode_admission_handoff_to_first_token_s": [
            r["ttft_from_dispatch_s"] - (r["prefill_s"] or 0) for r in requests],
        "decode_admission_handoff_and_generation_s": [
            r["e2e_from_dispatch_s"] - (r["prefill_s"] or 0) for r in requests],
        "per_request_tpot_s": [r["tpot_s"] for r in requests],
    }
    return dict(
        percentiles={name: {"p50": percentile(values, .5), "p95": percentile(values, .95),
                             "p99": percentile(values, .99)} for name, values in fields.items()},
        scope=("Per-request client measurements, not sums of GPU kernel times. Prefill includes producer queuing; "
               "post-prefill stages include handoff and decode admission. " +
               ("Real computed KV transferred through NIXL." if real_prefill else
                "No real prefill: post-prefill stage includes synthetic KV initialization.")))


def resident_common_window(requests):
    """Observed output rate after every request warmed up, before any drains.

    This excludes synthetic KV initialization when the burst fits resident.
    A missing common window is not itself proof of a memory-capacity failure:
    real prefill can also stagger arrivals too much.
    """
    if not requests or any(len(r.get("token_times_s", [])) < 10 for r in requests):
        return None
    times = [[r["arrival_s"] + r["dispatch_lateness_s"] + t for t in r["token_times_s"]]
             for r in requests]
    start, end = max(t[4] for t in times), min(t[-5] for t in times)
    if end <= start:
        return None
    count = sum(bisect_right(t, end) - bisect_right(t, start) for t in times)
    return dict(seconds=end-start, tokens=count, output_tps=count/(end-start),
                scope="client-observed common resident window after every request's fifth token and before any request's fifth-from-last token; excludes initial KV setup, includes CPU/HTTP overhead")


def summarize(root):
    records, failures = [], []
    suites = []
    for base, directories, files in os.walk(root):
        directories[:] = [d for d in directories if not d.startswith(".") and d not in
                          {"cache", "triton", "shared_cache", "shared_triton"}]
        if "suite.json" in files:
            suites.append(Path(base) / "suite.json")
    for suite in sorted(suites):
        state = json.loads(suite.read_text())
        for row in state["attempts"]:
            if row["status"] != "completed":
                resolutions = [r["name"] for r in state["attempts"] if r["status"] == "completed"
                               and canonical_attempt(r["name"]) == canonical_attempt(row["name"])]
                failures.append(dict(suite=str(suite), resolved_by=resolutions, **row))
                continue
            if row["name"].startswith("probe-") or row["name"] in state.get("superseded_trials", []):
                continue
            path = suite.parent / row["name"]
            analysis = json.loads((path / "analysis.json").read_text())
            trial = analysis["configuration"]
            decoders = [f"server{i}" for i, r in enumerate(trial["servers"]) if r["role"] == "decode"]
            producers = [f"server{i}" for i, r in enumerate(trial["servers"]) if r["role"] == "prefill"]
            name = row["name"].split("-o" + str(row["tokens"]) + "-", 1)[1]
            method = name.split("-capacity-recalibrated")[0].split("-retry")[0].split("-repeat")[0]
            family = "e2e" if producers else "synthetic-decode"
            manifests = [json.loads((path / n / "manifest.json").read_text()) for n in producers + decoders]
            execution = analysis["execution"]
            decode_prefill_steps = sum(count for n in decoders for rank in execution.get(n, [])
                                       for key, count in rank["counts"].items() if key.startswith("prefill:"))
            decoder_gpu_count = sum(len(r["gpus"]) for r in trial["servers"] if r["role"] == "decode")
            client = row["client"]
            requests = [json.loads(line) for line in (path / "client/requests.jsonl").open()]
            token_count_ok = len(requests) == row["offered"] and all(
                not r.get("error") and len(r["token_ids"]) == row["tokens"] for r in requests)
            records.append(dict(name=row["name"], path=str(path), family=family, method=method,
                context=row["context"], output_tokens=row["tokens"], offered_requests=row["offered"],
                decoder_gpus=decoder_gpu_count, producer_gpus=sum(len(r["gpus"]) for r in trial["servers"] if r["role"] == "prefill"),
                slurm_job_ids=sorted({str(m.get("slurm_job_id")) for m in manifests}),
                worker_hosts=sorted({resource["hostname"] for m in manifests for resource in m["worker_resources"]}),
                runtime_lifetime_capacities=row["capacities"], client=client,
                kv_gib=sum(r["kv_allocated_bytes"] for n in decoders for r in analysis["resources"][n]) / 2**30,
                decode_queue=snapshots([path / n / "rank0.jsonl" for n in decoders]),
                prefill_queue=snapshots([path / n / "rank0.jsonl" for n in producers]),
                decoder_prefill_or_recompute_steps=decode_prefill_steps,
                decoder_preemptions=sum(analysis["telemetry"].get(n + "/rank0", {}).get(
                    "preemptions_measured", 0) for n in decoders),
                decode_full_graph=row["full_graph"], all_kv_drained=row["kv_drained"],
                zero_preemptions=row["preemptions_zero"], token_count_ok=token_count_ok,
                resident_common_window=resident_common_window(requests),
                stage_latencies=stage_latencies(requests, bool(producers)),
                dummy_weights=all(not m["real_weights"] for m in manifests),
                synthetic_kv=any(m["synthetic_kv"] for m in manifests),
                performance_valid=all(m["performance_valid"] for m in manifests),
                bf16_storage_verified=all(
                    resource["model_dtype"] == "torch.bfloat16" and resource["quantization"] is None
                    and resource["kv_storage_dtypes"] == ["torch.bfloat16"]
                    for m in manifests for resource in m["worker_resources"]),
                graph_owner_layout_consistent=all(m["arguments"]["tknp"] <= 1 or
                    owner_graphs_consistent(m["worker_resources"]) for m in manifests),
                engine_batch_queue_sizes=[m.get("engine_batch_queue_size") for m in manifests],
                actual_decode_configurations=[m["arguments"] for m in manifests if m["arguments"]["cuda_graph"]],
                external_pp_caveat=any(m["arguments"]["pp"] > 1 and m.get("engine_batch_queue_size") == 1
                                      for m in manifests),
                workload_sha256=row["workload_sha256"]))
    groups = {}
    for row in records:
        row["comparison_eligible"] = all(row[key] for key in (
            "token_count_ok", "decode_full_graph", "all_kv_drained", "performance_valid",
            "bf16_storage_verified", "graph_owner_layout_consistent"))
        if not row["comparison_eligible"]:
            continue
        key = (row["family"], row["decoder_gpus"], row["context"], row["output_tokens"], row["offered_requests"], row["producer_gpus"])
        groups.setdefault(key, []).append(row)
    comparisons, leaders = [], []
    for key, rows in sorted(groups.items()):
        if len({row["workload_sha256"] for row in rows}) != 1:
            raise RuntimeError(f"Mismatched workloads: {key}")
        methods = {}
        for row in rows:
            methods.setdefault(row["method"], []).append(row)
        aggregates = {method: mean(r["client"]["system_output_tps"] for r in values)
                      for method, values in methods.items()}
        statistics = {}
        for method, values in methods.items():
            rates = [r["client"]["system_output_tps"] for r in values]
            statistics[method] = dict(trials=len(rates), mean_tps=mean(rates),
                                      stdev_tps=stdev(rates) if len(rates) > 1 else None,
                                      minimum_tps=min(rates), maximum_tps=max(rates))
        families = {"tknp": [m for m in methods if m.startswith("tknp")],
                    "dp": [m for m in methods if m.startswith("dp")],
                    "tp": [m for m in methods if m == "tp"],
                    "pp": [m for m in methods if "pp" in m],
                    "dcp": [m for m in methods if m == "dcp"]}
        best = {family: max(names, key=lambda name: aggregates[name])
                for family, names in families.items() if names}
        resident = {method: mean(r["resident_common_window"]["output_tps"] for r in values
                                if r["resident_common_window"] is not None)
                    for method, values in methods.items() if any(r["resident_common_window"] is not None for r in values)}
        resident_best = {family: max([m for m in names if m in resident], key=lambda m: resident[m])
                         for family, names in families.items() if any(m in resident for m in names)}
        leaders.append(dict(family=key[0], decoder_gpus=key[1], context=key[2], output_tokens=key[3],
                            offered_requests=key[4], producer_gpus=key[5], best_methods=best,
                            configuration_statistics=statistics,
                            best_tps={f: aggregates[m] for f, m in best.items()},
                            tknp_over_best_dp=(aggregates[best["tknp"]] / aggregates[best["dp"]]
                                               if "tknp" in best and "dp" in best else None),
                            resident_best_methods=resident_best,
                            resident_best_tps={f: resident[m] for f, m in resident_best.items()},
                            measurement="mean of available trials per configuration; best TESTED configuration per family; exploratory single trials unless repeated"))
        for tk in [r for r in rows if r["method"].startswith("tknp")]:
            for baseline in [r for r in rows if not r["method"].startswith("tknp")]:
                comparisons.append(dict(family=key[0], decoder_gpus=key[1], context=key[2], output_tokens=key[3],
                    offered_requests=key[4], tknp_method=tk["method"], baseline=baseline["method"],
                    producer_gpus=key[5],
                    throughput_ratio=tk["client"]["system_output_tps"] / baseline["client"]["system_output_tps"],
                    tknp_tps=tk["client"]["system_output_tps"], baseline_tps=baseline["client"]["system_output_tps"]))
    return dict(scope="Dummy BF16 70B architecture performance proxy; not trained-checkpoint quality validation. E2E and synthetic KV are never pooled.",
                queue_metrics_scope=("Timestamp-aligned WAITING-state snapshots, excluding WAITING_FOR_REMOTE_KVS. "
                                     "These include admission/concurrency/token-budget waiting, not an exclusive KV-memory cause label; "
                                     "prefill waiting also includes ordinary producer backlog."),
                cuda_graph_audit_scope=("Decode-only model calls must replay FULL graphs on every worker with no eager decode-only fallback. "
                                        "Mixed prefill/recompute calls are not FULL graph calls under FULL_DECODE_ONLY; "
                                        "native DP preemptions and resulting recompute calls are retained as baseline costs, not silently excluded."),
                audit=dict(completed_points=len(records), eligible_points=sum(r["comparison_eligible"] for r in records),
                           total_requests=sum(r["offered_requests"] for r in records),
                           total_output_tokens=sum(r["offered_requests"] * r["output_tokens"] for r in records),
                           all_completed_points_eligible=all(r["comparison_eligible"] for r in records),
                           failed_or_incomplete_attempts=len(failures),
                           resolved_attempts=sum(bool(r["resolved_by"]) for r in failures),
                           unresolved_attempts=sum(not r["resolved_by"] for r in failures)),
                trials=records, comparisons=comparisons, leaders=leaders, incomplete_or_failed_attempts=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for r in report["trials"]:
        print(r["family"], r["decoder_gpus"], r["context"], r["offered_requests"], r["method"],
              "TPS", round(r["client"]["system_output_tps"], 2), "preempts0", r["zero_preemptions"],
              "FULL", r["decode_full_graph"], "tokens", r["token_count_ok"])
    print("completed", len(report["trials"]), "failed/incomplete", len(report["incomplete_or_failed_attempts"]))


if __name__ == "__main__":
    main()
