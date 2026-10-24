#!/usr/bin/env python3
"""Cross-rank profiler compute concurrency on one host; excludes NCCL."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    boundaries=[];spans=[];bases=[]
    for path in sorted(args.traces.rglob("*.pt.trace.json*")):
        with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as f:
            data=json.load(f)
        bases.append(data.get("baseTimeNanoseconds",0))
        events=[e for e in data["traceEvents"] if e.get("ph")=="X"
                and e.get("cat")=="kernel" and "nccl" not in e["name"].lower()]
        intervals=[]
        for e in sorted(events,key=lambda e:e["ts"]):
            start=e["ts"]+bases[-1]/1000;end=start+e["dur"]
            if intervals and start<=intervals[-1][1]:
                intervals[-1][1]=max(intervals[-1][1],end)
            else:intervals.append([start,end])
        spans.append(dict(path=str(path),intervals=len(intervals),
                          active_ms=sum(b-a for a,b in intervals)/1000))
        for start,end in intervals:boundaries.extend(((start,1),(end,-1)))
    boundaries.sort();hist={};active=0;previous=boundaries[0][0]
    for t,change in boundaries:
        hist[active]=hist.get(active,0)+(t-previous)/1000
        active+=change;previous=t
    duration=(boundaries[-1][0]-boundaries[0][0])/1000
    result=dict(scope="One-host aligned profiler CUDA compute intervals, not SM occupancy or unprofiled throughput",
        base_times_ns=bases,rank_intervals=spans,duration_ms=duration,
        simultaneous_compute_ms=hist,
        average_computing_gpus=sum(k*v for k,v in hist.items())/duration)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
