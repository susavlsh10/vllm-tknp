"""Summarize matched, unprofiled A/B measurements and request-ordered tokens."""
import argparse
import json
import statistics
from pathlib import Path


def load_point(directory):
    metrics = [json.loads(path.read_text())
               for path in sorted(directory.glob("steady_rank*.json"))]
    if not metrics or any(point["profiled"] for point in metrics):
        raise ValueError(f"Missing unprofiled timing: {directory}")
    configs = {(point["batch"], point["context"], point["steps"], point["enable_dbo"])
               for point in metrics}
    if len(configs) != 1:
        raise ValueError(f"Rank configuration disagreement: {directory}")
    # Synchronous inference runs at the slowest rank's measured rate.
    slowest = max(metrics, key=lambda point: point["wall_step_ms_mean"])
    tokens = json.loads((directory / "output_token_ids.json").read_text())
    return slowest, tokens, len(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path,
                        help="baseline candidate [baseline candidate ...]")
    args = parser.parse_args()
    if len(args.directories) % 2:
        parser.error("Expected baseline/candidate pairs")
    pairs = []
    for baseline_dir, candidate_dir in zip(args.directories[::2], args.directories[1::2]):
        baseline, baseline_tokens, ranks = load_point(baseline_dir)
        candidate, candidate_tokens, candidate_ranks = load_point(candidate_dir)
        if candidate_ranks != ranks:
            raise ValueError("Different GPU counts")
        if baseline["enable_dbo"] or not candidate["enable_dbo"]:
            raise ValueError("Expected microbatching off/on")
        for field in ("batch", "context", "steps", "model", "load_format",
                      "kv_cache_dtype", "tensor_parallel_size", "token_parallel_size",
                      "pipeline_parallel_size", "requested_decode_tokens", "cuda_graph_mode"):
            if baseline.get(field) != candidate.get(field):
                raise ValueError(f"Mismatched {field}")
        exact = baseline_tokens == candidate_tokens
        if not exact:
            raise ValueError(f"Output token mismatch: {candidate_dir}")
        pairs.append({
            "baseline": str(baseline_dir.resolve()),
            "candidate": str(candidate_dir.resolve()),
            "ranks": ranks,
            "batch": baseline["batch"],
            "context": baseline["context"],
            "measured_steps": baseline["steps"],
            "baseline_step_ms": baseline["wall_step_ms_mean"],
            "candidate_step_ms": candidate["wall_step_ms_mean"],
            "baseline_tps": baseline["steady_decode_tps"],
            "candidate_tps": candidate["steady_decode_tps"],
            "throughput_gain_percent": 100 * (
                candidate["steady_decode_tps"] / baseline["steady_decode_tps"] - 1),
            "exact_output_tokens": exact,
            "checked_tokens": sum(len(sequence)
                                  for request in baseline_tokens
                                  for sequence in request["output_token_ids"]),
        })
    mean_baseline = statistics.mean(pair["baseline_tps"] for pair in pairs)
    mean_candidate = statistics.mean(pair["candidate_tps"] for pair in pairs)
    print(json.dumps({
        "timing": "unprofiled engine.step, slowest rank, steady full batch",
        "pairs": pairs,
        "mean_baseline_tps": mean_baseline,
        "mean_candidate_tps": mean_candidate,
        "mean_throughput_gain_percent": 100 * (mean_candidate / mean_baseline - 1),
    }, indent=2))


if __name__ == "__main__":
    main()
