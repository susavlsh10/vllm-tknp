#!/usr/bin/env python3
"""Summarize existing PyTorch CPU/CUDA traces; never runs GPU work.

Kernel-active time is NOT SM occupancy. NCCL kernels may be waiting rather
than transferring, so report them separately from useful compute kernels.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


def is_communication_kernel(name):
    return any(word in name.lower() for word in ("nccl","custom_ar::","cross_device_reduce",
        "multimem","all_reduce","allreduce","all_gather","allgather","reduce_scatter","reducescatter"))


def is_dense_kernel(name):
    # cuBLAS BF16 mm in this pinned CUDA runtime uses nvjet_tst_* names.
    return any(x in name.lower() for x in
               ("gemm","matmul","xmma","scaled_mm","cutlass","nvjet_"))


def union_us(events):
    end, total = float("-inf"), 0
    for event in sorted(events, key=lambda e: e["ts"]):
        start, stop = event["ts"], event["ts"] + event["dur"]
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def totals(events):
    times, counts = Counter(), Counter()
    for e in events:
        times[e["name"]] += e["dur"]
        counts[e["name"]] += 1
    return [dict(name=name, total_ms=value/1000, calls=counts[name], mean_us=value/counts[name])
            for name, value in times.most_common(20)]


def inter_kernel_gaps(events):
    """Attention-call gaps, NOT GPU idle: other kernels can occupy the gap."""
    gaps=[];end=None
    for e in sorted(events,key=lambda e:e["ts"]):
        if end is not None:gaps.append(max(0,e["ts"]-end))
        end=max(float("-inf") if end is None else end,e["ts"]+e["dur"])
    values=sorted(gaps)
    if not values:return dict(count=0)
    return dict(count=len(values),p50_us=values[len(values)//2],
        p95_us=values[min(len(values)-1,int(.95*len(values)))],max_us=values[-1],
        above_50us=sum(v>50 for v in values),above_100us=sum(v>100 for v in values),
        scope="Inter-main-attention-kernel gaps across the captured steps, including step boundaries; not GPU-idle time or layer labels")


def analyze(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    events = [e for e in data["traceEvents"] if e.get("ph")=="X" and "dur" in e]
    kernels = [e for e in events if e.get("cat")=="kernel"]
    if not kernels:
        raise RuntimeError("Trace has no CUDA kernel activity: " + str(path))
    nccl = [e for e in kernels if "nccl" in e["name"].lower()]
    communication = [e for e in kernels if is_communication_kernel(e["name"])]
    compute = [e for e in kernels if not is_communication_kernel(e["name"])]
    attention = [e for e in compute if any(x in e["name"].lower()
                 for x in ("fmha", "flash", "batchdecode", "batchprefill", "attention"))]
    attention_ids={id(e) for e in attention}
    dense = [e for e in compute if id(e) not in attention_ids and is_dense_kernel(e["name"])]
    main_attention=[e for e in attention if not any(word in e["name"].lower()
                    for word in ("combine","merge"))]
    api = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    cpu = [e for e in events if e.get("cat")=="cpu_op"]
    first = min(e["ts"] for e in kernels)
    last = max(e["ts"]+e["dur"] for e in kernels)
    span = last-first
    rank = re.search(r"-rank-(\d+)",path.name)
    return dict(path=str(path),rank=int(rank.group(1)) if rank else None,
        kernel_span_ms=span/1000, kernel_active_ms=union_us(kernels)/1000,
        compute_active_ms=union_us(compute)/1000, attention_active_ms=union_us(attention)/1000,
        nccl_active_ms=union_us(nccl)/1000, kernel_idle_ms=(span-union_us(kernels))/1000,
        communication_active_ms=union_us(communication)/1000,dense_active_ms=union_us(dense)/1000,
        dense_kernel_count=len(dense),dense_kernel_sum_ms=sum(e["dur"] for e in dense)/1000,
        dense_duty=union_us(dense)/span,
        main_attention_gaps=inter_kernel_gaps(main_attention),
        dense_communication_overlap_ms=(union_us(dense)+union_us(communication)-union_us(dense+communication))/1000,
        compute_duty=union_us(compute)/span, attention_duty=union_us(attention)/span,
        kernel_idle_fraction=(span-union_us(kernels))/span,
        kernel_count=len(kernels),attention_kernel_count=len(attention),
        top_kernels=totals(kernels),top_runtime=totals(api),top_cpu_ops=totals(cpu),
        categories=dict(Counter(e.get("cat") for e in events)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rows=[analyze(f) for f in sorted(args.traces.rglob("*.pt.trace.json*"))]
    if not rows:raise RuntimeError("No completed PyTorch traces")
    args.output.write_text(json.dumps(dict(
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Profiler-only, not throughput/SM occupancy. Interval unions; known NCCL/custom/symmetric collectives excluded from compute. Other kernels include staging copies. Attention/dense activity is separate.",
        traces=rows),indent=2)+"\n")
    for r in sorted(rows,key=lambda r:(r["rank"] is None,r["rank"] or 0)):
        print('RANK',r['rank'],'SPAN_MS',round(r['kernel_span_ms'],2),
              'COMPUTE_DUTY',round(100*r['compute_duty'],1),
              'ATTN_DUTY',round(100*r['attention_duty'],1),
              'IDLE_PCT',round(100*r['kernel_idle_fraction'],1))
        if r['rank'] in (0,1,8):
            print('TOP_KERNELS',[(x['name'][:100],round(x['total_ms'],2),x['calls']) for x in r['top_kernels'][:8]])
            print('TOP_API',[(x['name'],round(x['total_ms'],2),x['calls']) for x in r['top_runtime'][:8]])


if __name__=="__main__":main()
