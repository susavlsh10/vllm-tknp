#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from statistics import median


POINT_RE = re.compile(r"(?P<sweep>batch|sequence)_bs(?P<batch>\d+)_seq(?P<seq>\d+)")


def read_rows(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topology in ("tp2_pp4", "tp2_tknp4"):
        topology_dir = result_root / topology
        if not topology_dir.exists():
            continue
        for point_dir in sorted(topology_dir.iterdir()):
            match = POINT_RE.fullmatch(point_dir.name)
            if match is None:
                continue
            for rep_dir in sorted(point_dir.glob("rep_*")):
                if not rep_dir.is_dir():
                    continue
                repetition = int(rep_dir.name.split("_")[-1])
                csv_files = list(rep_dir.glob("*.csv"))
                if len(csv_files) != 1:
                    continue
                with csv_files[0].open(newline="") as csv_file:
                    source_rows = list(csv.DictReader(csv_file))
                if len(source_rows) != 1:
                    continue
                source = source_rows[0]
                rows.append({
                    "topology": topology,
                    "sweep": match.group("sweep"),
                    "batch_size": int(match.group("batch")),
                    "seq_length": int(match.group("seq")),
                    "repetition": repetition,
                    "decode_time_ms": float(source["decode_time_ms"]),
                    "sys_decode_tps": float(source["sys_decode_tps"]),
                    "decode_tps_per_gpu": float(source["decode_tps_per_gpu"]),
                    "avg_decode_latency_ms": float(source["avg_decode_latency_ms"]),
                    "decode_tps_per_user": float(source["decode_tps_per_user"]),
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["topology"], row["sweep"], row["batch_size"], row["seq_length"]
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        topology, sweep, batch_size, seq_length = key
        summary.append({
            "topology": topology,
            "sweep": sweep,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "repetitions": len(group),
            "median_decode_time_ms": median(float(x["decode_time_ms"]) for x in group),
            "median_sys_decode_tps": median(float(x["sys_decode_tps"]) for x in group),
            "median_decode_tps_per_gpu": median(
                float(x["decode_tps_per_gpu"]) for x in group
            ),
            "median_avg_decode_latency_ms": median(
                float(x["avg_decode_latency_ms"]) for x in group
            ),
            "median_decode_tps_per_user": median(
                float(x["decode_tps_per_user"]) for x in group
            ),
        })
    return summary


def compare(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (row["topology"], row["sweep"], row["batch_size"], row["seq_length"]): row
        for row in summary
    }
    comparisons: list[dict[str, object]] = []
    point_keys = sorted(
        {
            (row["sweep"], row["batch_size"], row["seq_length"])
            for row in summary
        }
    )
    for sweep, batch_size, seq_length in point_keys:
        pp = indexed.get(("tp2_pp4", sweep, batch_size, seq_length))
        tknp = indexed.get(("tp2_tknp4", sweep, batch_size, seq_length))
        if pp is None or tknp is None:
            continue
        pp_tps = float(pp["median_sys_decode_tps"])
        tknp_tps = float(tknp["median_sys_decode_tps"])
        pp_latency = float(pp["median_avg_decode_latency_ms"])
        tknp_latency = float(tknp["median_avg_decode_latency_ms"])
        comparisons.append({
            "sweep": sweep,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "pp4_median_sys_decode_tps": pp_tps,
            "tknp4_median_sys_decode_tps": tknp_tps,
            "tknp_over_pp_throughput": tknp_tps / pp_tps,
            "pp4_median_latency_ms": pp_latency,
            "tknp4_median_latency_ms": tknp_latency,
            "pp_over_tknp_latency": pp_latency / tknp_latency,
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.result_root)
    summary = summarize(rows)
    comparisons = compare(summary)
    write_csv(args.result_root / "combined_runs.csv", rows)
    write_csv(args.result_root / "summary_median.csv", summary)
    write_csv(args.result_root / "comparison.csv", comparisons)
    print(
        f"summarized runs={len(rows)} points={len(summary)} comparisons={len(comparisons)}"
    )


if __name__ == "__main__":
    main()
