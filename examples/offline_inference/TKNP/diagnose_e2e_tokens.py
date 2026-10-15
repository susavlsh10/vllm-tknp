#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First-divergence top-k diagnostics. Never overrides exact correctness gates."""
import argparse
import json
import math
from pathlib import Path


def bf16_ulp(value):
    return 2.0 ** (math.frexp(abs(value))[1] - 8) if value else 2.0 ** -133


def compare(baseline, candidate, baseline_logits, candidate_logits):
    details = []
    for req, a in baseline.items():
        b = candidate[req]
        if a["token_ids"] == b["token_ids"]:
            continue
        step = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y), None)
        item = {"request_id": req, "first_divergence": step,
                "prompt_tokens": a["prompt_tokens"], "consistent_with_bf16_near_tie": False}
        if step is not None:
            logs = [baseline_logits.get((a["server_request_id"], step)),
                    candidate_logits.get((b["server_request_id"], step))]
            item["baseline_topk"], item["candidate_topk"] = logs
            if all(logs):
                values = [dict(zip(log["ids"], log["values"])) for log in logs]
                selected = [a["token_ids"][step], b["token_ids"][step]]
                ulp = max(bf16_ulp(max(log["values"])) for log in logs)
                common = values[0].keys() & values[1].keys()
                error = max((abs(values[0][token] - values[1][token]) for token in common), default=math.inf)
                item["bf16_ulp"] = ulp
                item["common_topk_max_abs_error"] = error
                item["consistent_with_bf16_near_tie"] = (
                    all(token in row for row in values for token in selected)
                    and all(abs(row[selected[0]] - row[selected[1]]) <= ulp for row in values)
                    and error <= ulp)
        details.append(item)
    return {"exact_requests": len(baseline) - len(details), "total_requests": len(baseline),
            "scope": "Top-k at the first divergent token after an identical prefix; NOT a full-logit/KV correctness proof or an accepted tolerance gate",
            "divergences": details}


def analyze(directory):
    def rows(role):
        return {row["request_id"]: row for row in
                map(json.loads, (directory / role / "requests.jsonl").read_text().splitlines())}
    def logits(role):
        return {(row["request_id"], row["step"]): row for row in
                map(json.loads, (directory / role / "logits-rank0.jsonl").read_text().splitlines())}
    return compare(rows("colocated"), rows("disaggregated"), logits("reference"), logits("decode"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
