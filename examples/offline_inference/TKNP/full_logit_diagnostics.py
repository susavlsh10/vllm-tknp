#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full-vocabulary diagnostics on identical prefixes, never hides mismatches."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(directory):
    import torch
    def rows(label):
        return {row["request_id"]: row for row in map(json.loads,
                (directory / label / "requests.jsonl").read_text().splitlines())}
    baseline, candidate = rows("colocated"), rows("disaggregated")
    if baseline.keys() != candidate.keys():
        raise ValueError("Request sets differ")
    records = []
    for key, reference in baseline.items():
        other = candidate[key]
        limit = min(len(reference["token_ids"]), len(other["token_ids"]))
        for step in range(limit):
            if reference["token_ids"][:step] != other["token_ids"][:step]:
                break
            tensors = []
            for role, row in (("reference", reference), ("decode", other)):
                digest = hashlib.sha256(row["server_request_id"].encode()).hexdigest()
                path = directory / role / "full-logits-rank0" / f"{digest}-{step}.pt"
                record = torch.load(path, map_location="cpu", weights_only=True)
                tensors.append(record["logits"].float())
            a, b = tensors
            if a.shape != b.shape:
                raise ValueError("Vocabulary/logit shapes differ")
            error = b - a
            top2 = a.topk(2).values
            records.append({"request_id": key, "step": step,
                            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                            "max_abs_error": error.abs().max().item(),
                            "rmse": error.square().mean().sqrt().item(),
                            "relative_l2_error": (error.norm() / a.norm().clamp_min(1e-12)).item(),
                            "reference_top2_margin": (top2[0] - top2[1]).item(),
                            "argmax_matches": bool(a.argmax() == b.argmax())})
            kv = []
            for role, row in (("reference", reference), ("decode", other)):
                digest = hashlib.sha256(row["server_request_id"].encode()).hexdigest()
                paths = sorted((directory / role).glob(f"kv-samples-rank*/{digest}-{step}.pt"),
                               key=lambda p: int(p.parent.name.rsplit("rank", 1)[1]))
                shards = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
                if shards:
                    if any(s["positions"] != shards[0]["positions"] for s in shards):
                        raise ValueError("TP shards disagree on sampled positions")
                    kv.append((shards[0]["positions"], torch.cat([s["kv"].float() for s in shards], dim=-2)))
            if len(kv) == 2:
                if kv[0][0] != kv[1][0] or kv[0][1].shape != kv[1][1].shape:
                    raise ValueError("KV sample positions/head shapes differ")
                ka, kb = kv[0][1], kv[1][1]
                records[-1].update(kv_positions=kv[0][0],
                                   kv_finite=bool(torch.isfinite(ka).all() and torch.isfinite(kb).all()),
                                   kv_max_abs_error=(ka - kb).abs().max().item(),
                                   kv_relative_l2_error=((ka - kb).norm() / ka.norm().clamp_min(1e-12)).item())
    return {"scope": "full vocabulary on identical prefixes; diagnostic errors, NOT an automatically chosen tolerance",
            "compared_rows": len(records), "finite": bool(records) and all(r["finite"] for r in records),
            "exact_greedy_requests": sum(baseline[k]["token_ids"] == candidate[k]["token_ids"] for k in baseline),
            "requests": len(baseline),
            "max_abs_error": max((r["max_abs_error"] for r in records), default=None),
            "max_relative_l2_error": max((r["relative_l2_error"] for r in records), default=None),
            "kv_compared_rows": sum("kv_finite" in r for r in records),
            "kv_finite": all(r.get("kv_finite", True) for r in records),
            "kv_max_relative_l2_error": max((r["kv_relative_l2_error"] for r in records if "kv_relative_l2_error" in r), default=None),
            "rows": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    if args.output:
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if not result["finite"]:
        raise SystemExit(1)
