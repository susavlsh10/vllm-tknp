"""Numerical A/B diagnostic; reports errors without declaring correctness."""
import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    records = [torch.load(path, map_location="cpu", weights_only=True)
               for path in (args.baseline, args.candidate)]
    logits = []
    ordinals = []
    for record in records:
        ids = torch.tensor(record["request_ordinals"])
        order = ids.argsort()
        ordinals.append(ids[order])
        logits.append(record["logits"][order].float())
    assert torch.equal(ordinals[0], ordinals[1]), "Different requests"
    baseline, candidate = logits
    assert baseline.shape == candidate.shape
    error = candidate - baseline
    baseline_top = baseline.topk(2, dim=-1)
    disagree = baseline.argmax(-1) != candidate.argmax(-1)
    max_row_error = error.abs().amax(-1)
    margins = baseline_top.values[:, 0] - baseline_top.values[:, 1]
    result = {
        "shape": list(baseline.shape),
        "finite": bool(torch.isfinite(baseline).all() and torch.isfinite(candidate).all()),
        "max_abs_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "relative_l2_error": (error.norm() / baseline.norm()).item(),
        "argmax_disagreements": disagree.sum().item(),
        "rows": baseline.shape[0],
        "disagreement_details": [{
            "request_ordinal": ordinals[0][i].item(),
            "baseline_top2_margin": margins[i].item(),
            "max_row_abs_error": max_row_error[i].item(),
        } for i in disagree.nonzero().flatten().tolist()],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
