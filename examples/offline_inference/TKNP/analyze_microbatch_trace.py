"""Summarize steady layer periods and attention gaps from saved torch traces."""
import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trace_dir", type=Path)
parser.add_argument("--layers", type=int, default=32)
parser.add_argument("--cohorts", type=int, default=2)
parser.add_argument("--root-zero", action="store_true")
parser.add_argument("--projection-only-root", action="store_true")
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--kernel-regex", default=(
    r"fmhaSm100fKernel|flash::flash_fwd|BatchDecodeWithPagedKVCacheKernel|"
    r"BatchPrefillWithPagedKVCacheKernel"))
args = parser.parse_args()
for path in sorted(args.trace_dir.glob("*.json.gz")):
    rank = int(re.search(r"rank-(\d+)", path.name).group(1))
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    attention = sorted((event for event in events if event.get("cat") == "kernel"
                        and re.search(args.kernel_regex, event["name"])),
                       key=lambda event: event["ts"])
    is_root = rank < args.tp
    if args.projection_only_root and is_root:
        if attention:
            raise RuntimeError("Projection-only root unexpectedly executed attention")
        print(json.dumps({"rank": rank, "attention_kernels": 0,
                          "projection_only": True, "trace": str(path)}))
        continue
    cohorts = 1 if args.root_zero and is_root else args.cohorts
    per_step = args.layers * cohorts
    if len(attention) % per_step:
        raise RuntimeError(f"Unexpected attention kernel count in {path}: {len(attention)}")
    periods, gaps, c0_times, c1_times = [], [], [], []
    for i in range(0, len(attention) - cohorts, cohorts):
        if (i // cohorts) % args.layers == args.layers - 1:
            continue  # exclude model boundary and host scheduling gap
        current, following = attention[i], attention[i + cohorts]
        periods.append(following["ts"] - current["ts"])
        last = attention[i + cohorts - 1]
        gaps.append(following["ts"] - last["ts"] - last["dur"])
        c0_times.append(current["dur"])
        if cohorts == 2:
            c1_times.append(last["dur"])
    if not periods:
        raise RuntimeError(f"No attention layer periods in {path}")
    print(json.dumps({"rank": rank, "attention_kernels": len(attention),
          "expected_kernels_per_step": per_step,
          "median_layer_period_us": statistics.median(periods),
          "median_interlayer_attention_gap_us": statistics.median(gaps),
          "median_c0_attention_us": statistics.median(c0_times),
          "median_c1_attention_us": statistics.median(c1_times) if c1_times else None,
          "trace": str(path)}))
