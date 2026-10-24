#!/usr/bin/env python3
"""Report full-vocabulary errors before generation prefixes diverge.

No automatically selected tolerance and no claim of trained-model accuracy.
The original strict token-equality failure remains preserved in the run log.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-relative-l2",type=float,
                   help="Explicit engineering bound; never inferred from the observed errors")
    p.add_argument("--require-tied-divergences",action="store_true")
    args=p.parse_args()
    root=args.directory
    records=[];requests=exact=0
    for segment in ("b16-r0","b32-r1","b96-r2","b32-r3"):
        policies=[]
        for name in ("dbo1-tail0","dbo1-tail1"):
            policy=root/name
            rows={r["request_id"]:r for r in map(json.loads,
                (policy/segment/"client/requests.jsonl").read_text().splitlines())}
            policies.append((policy,rows))
        assert policies[0][1].keys()==policies[1][1].keys()
        for key in policies[0][1]:
            pairs=[rows[key] for _,rows in policies]
            requests+=1;exact+=pairs[0]["token_ids"]==pairs[1]["token_ids"]
            for step in range(min(len(r["token_ids"]) for r in pairs)):
                if pairs[0]["token_ids"][:step]!=pairs[1]["token_ids"][:step]:break
                tensors=[]
                for (policy,_),row in zip(policies,pairs):
                    digest=hashlib.sha256(row["server_request_id"].encode()).hexdigest()
                    item=torch.load(policy/"engine/server0/full-logits-rank0"/f"{digest}-{step}.pt",
                                    map_location="cpu",weights_only=True)
                    tensors.append(item["logits"].float())
                a,b=tensors
                assert a.shape==b.shape
                delta=b-a
                top=a.topk(2).values
                records.append(dict(segment=segment,request_id=key,step=step,
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    max_abs_error=float(delta.abs().max()),
                    rmse=float(delta.square().mean().sqrt()),
                    relative_l2_error=float(delta.norm()/a.norm().clamp_min(1e-12)),
                    top2_margin=float(top[0]-top[1]),
                    argmax_matches=bool(a.argmax()==b.argmax())))
    summary=dict(scope=__doc__,directory=str(root),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=requests,exact_greedy_requests=exact,
        compared_full_vocabulary_rows=len(records),
        finite=all(r["finite"] for r in records),
        max_abs_error=max(r["max_abs_error"] for r in records),
        max_relative_l2_error=max(r["relative_l2_error"] for r in records),
        max_rmse=max(r["rmse"] for r in records),rows=records,
        performance_valid=False,trained_model_accuracy_valid=False)
    divergent=[r for r in records if not r["argmax_matches"]]
    summary.update(argmax_mismatched_rows=len(divergent),
        all_divergences_reference_tied=all(r["top2_margin"]==0 for r in divergent),
        max_relative_l2_bound=args.max_relative_l2)
    if args.max_relative_l2 is not None:
        summary["numerical_bound_passed"]=(summary["finite"]
            and summary["max_relative_l2_error"]<=args.max_relative_l2
            and (not args.require_tied_divergences or summary["all_divergences_reference_tied"]))
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2),flush=True)
    if not summary["finite"]:raise RuntimeError("Nonfinite diagnostic logits")
    if summary.get("numerical_bound_passed") is False:raise RuntimeError("Explicit numerical bound failed")


if __name__=="__main__":main()
