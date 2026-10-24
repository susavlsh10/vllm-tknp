#!/usr/bin/env python3
"""Run a complete decode batch sweep with one loaded TKNP engine.

Supply an e2e_matrix JSON containing ONE decode role with graph_layouts.
Works locally inside a node or through the existing SLURM runtime launcher.
Never allocates nodes or downloads weights. Switching live requests is forbidden.
"""
import argparse
import json
from pathlib import Path

from e2e_matrix import run_trial, validate_matrix
from e1_pilot_report import analyze
from e2e_workload import make_workload


def make_segments(config, batches, output):
    trial = config["trials"][0]
    if len(config["trials"]) != 1 or len(trial["servers"]) != 1:
        raise ValueError("Supply one TKNP decode server topology per sweep")
    role = trial["servers"][0]
    if role.get("role") != "decode" or role.get("tknp", 1) < 2 or not role.get("graph_layouts"):
        raise ValueError("TKNP decode graph_layouts are required")
    sizes = sorted(map(int, role["graph_layouts"]))
    context = config["context"]
    tokens = config["output_tokens"]
    segments = []
    for index, batch in enumerate(batches):
        if type(batch) is not int or batch < 1:
            raise ValueError("Positive offered batch size required")
        size = next((size for size in sizes if size >= batch), sizes[-1])
        key = f"b{batch}-r{index}"
        workload = output / f"workload-b{batch}.jsonl"
        if not workload.exists():
            workload.write_text("".join(json.dumps(row) + "\n" for row in
                make_workload(batch, [context], [tokens], 0, config.get("workload_seed", 20261016))))
        segments.append(dict(key=key, batch=batch, graph_size=size, workload=str(workload),
                             output=str(output / key)))
    return segments


def audit_sweep(config, segments):
    rows = []
    references = {}
    repeated = {}
    for segment in segments:
        path = Path(segment["output"])
        evidence = analyze(path)
        if not all(evidence[key] for key in ("all_decode_workers_full_graph",
                "all_servers_zero_measured_preemptions", "all_servers_kv_drained")):
            raise RuntimeError("Graph, preemption or KV-drain verification failed")
        client = evidence["client"]
        if client["completed"] != segment["batch"] or client["output_tokens"] != segment["batch"] * config["output_tokens"]:
            raise RuntimeError("Incomplete request/output token counts")
        if segment["batch"] > segment["graph_size"] and not any(
                row["max_waiting"] > 0 for row in evidence["telemetry"].values()):
            raise RuntimeError("Offered batch exceeded graph capacity but no queue/wave was observed")
        for server, records in evidence["execution"].items():
            if sorted(record["rank"] for record in records) != list(range(len(config["trials"][0]["servers"][0]["gpus"]))):
                raise RuntimeError("Missing execution evidence for an assigned rank")
            for record in records:
                if record["counts"].get("graph:FULL_REPLAY", 0) <= 0:
                    raise RuntimeError("No actual full CUDA graph replay")
                if record["active_graph_size"] != segment["graph_size"]:
                    raise RuntimeError("Wrong active graph layout")
                identity = record["graph_reuse_identity"]
                key = (server, record["rank"])
                initial = next(item["graph_reuse_identity"] for item in evidence["resources"][server]
                               if item["rank"] == record["rank"])
                if len(identity["graphs"]) < len(config["trials"][0]["servers"][0]["graph_layouts"]) or identity != initial:
                    raise RuntimeError("Model/KV pointers, worker PID, or captured graphs changed after initialization")
                if key in references and identity != references[key]:
                    raise RuntimeError("Engine or cached graph reinitialized during sweep")
                references[key] = identity
                if (config.get("preinitialize_random_kv") or config.get("preinitialize_constant_kv")) and record["per_request_kv_fill_calls"]:
                    raise RuntimeError("Unexpected timed per-request synthetic KV fill")
        requests = [json.loads(line) for line in (path / "client/requests.jsonl").read_text().splitlines()]
        outputs = {request["request_id"]: request["token_ids"] for request in requests}
        if config.get("verify_repeated_outputs"):
            if segment["batch"] in repeated and outputs != repeated[segment["batch"]]:
                raise RuntimeError("Repeated identical constant-KV workload changed token outputs after layout switches")
            repeated[segment["batch"]] = outputs
        rows.append(dict(batch=segment["batch"], graph_size=segment["graph_size"],
            output_tps=client["system_output_tps"], full_graph_all_ranks=True,
            engine_and_graphs_reused=True, path=str(path), client=client))
    return dict(scope=config.get("scope"), passed=True, trials=rows,
                distinct_worker_processes=len(references), engine_initializations=1,
                repeated_outputs_checked=bool(config.get("verify_repeated_outputs")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32, 48, 8])
    args = parser.parse_args()
    config = json.loads(args.matrix.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    trial = config["trials"][0]
    validate_matrix(config, config.get("gpu_count", max(gpu for role in trial["servers"] for gpu in role["gpus"]) + 1))
    segments = make_segments(config, args.batches, args.output)
    config["client_sweep"] = segments
    (args.output / "matrix.json").write_text(json.dumps(config, indent=2) + "\n")
    run_trial(trial, config, Path(segments[0]["workload"]), args.output / "engine")
    result = audit_sweep(config, segments)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["trials"]:
        print("PASS", row["batch"], "graph", row["graph_size"], "TPS", round(row["output_tps"], 2), flush=True)
    print("TKNP_GRAPH_SWEEP_PASS", len(segments), "ENGINE_INITIALIZATIONS=1", flush=True)


if __name__ == "__main__":
    main()
