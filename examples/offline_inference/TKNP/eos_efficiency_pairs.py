#!/usr/bin/env python3
"""Three matched unprofiled legacy/fixed pairs; reverse the second pair.

Uses one existing node and sequential engines, never simultaneous workloads.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--capacity-source",type=Path,required=True)
    p.add_argument("--kind",choices=["attention","cpu-kv"],default="attention")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"pairs.json"
    state=json.loads(path.read_text()) if path.exists() else dict(trials=[])
    for pair,order in enumerate((("legacy","fixed"),("fixed","legacy"),("legacy","fixed"))):
        for variant in order:
            if any(r["pair"]==pair and r["variant"]==variant for r in state["trials"]):continue
            out=args.output/f"pair{pair}-{variant}"
            cmd=[sys.executable,str(Path(__file__).with_name("eos_efficiency.py")),
                "--job",args.job,"--node",args.node,"--output",str(out),
                "--capacity-source",str(args.capacity_source),"--stage","matrix",
                "--methods","tknp","--modes","graph","--batches","128","--repeats","1"]
            if args.kind=="attention":
                cmd += ["--flashinfer-tensor-cores","1" if variant=="legacy" else "0",
                        "--metadata-cache","0" if variant=="legacy" else "1"]
            else:
                cmd += ["--cpu-kv-control","0" if variant=="legacy" else "1",
                        "--metadata-cache","1"]
            print("PAIR_START",pair,variant,flush=True)
            subprocess.run(cmd,check=True)
            result=json.loads((out/"summary.json").read_text())["trials"]
            if len(result)!=1:raise RuntimeError("Expected one complete pair point")
            row=dict(result[0],pair=pair,variant=variant)
            state["trials"].append(row)
            path.write_text(json.dumps(state,indent=2)+"\n")
            print("PAIR_PASS",pair,variant,row["output_tps"],flush=True)
        if args.kind=="cpu-kv":
            outputs=[]
            for variant in ("legacy","fixed"):
                row=next(r for r in state["trials"] if r["pair"]==pair and r["variant"]==variant)
                records=[json.loads(line) for line in (Path(row["path"])/"client/requests.jsonl").read_text().splitlines()]
                outputs.append({r["request_id"]:r["token_ids"] for r in records})
            if outputs[0]!=outputs[1]:
                raise RuntimeError("CPU-only admission counters changed generated token IDs")
            print("PAIR_EXACT_OUTPUTS_VERIFIED",pair,flush=True)
    state["kind"]=args.kind
    state["exact_outputs_verified"]=args.kind=="cpu-kv"
    state["passed"]=len(state["trials"])==6
    state["paired_speedups"]=[next(r["output_tps"] for r in state["trials"] if r["pair"]==i and r["variant"]=="fixed")/
        next(r["output_tps"] for r in state["trials"] if r["pair"]==i and r["variant"]=="legacy") for i in range(3)]
    state["median_speedup"]=statistics.median(state["paired_speedups"])
    path.write_text(json.dumps(state,indent=2)+"\n")
    print("MATCHED_EFFICIENCY_PAIRS_COMPLETE",state["median_speedup"],flush=True)


if __name__=="__main__":main()
