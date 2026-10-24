#!/usr/bin/env python3
"""CPU-only equivalence check for scoped journal postprocessing.

Read a completed persistent-engine matrix. Preserve all raw inputs and
recompute each point in a new directory, requiring exact result equality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from e2e_matrix import collect_trial_result,split_sweep_journals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.matrix.read_text());segments=cfg["client_sweep"]
    args.output.mkdir(parents=True,exist_ok=False)
    views={i:split_sweep_journals(args.engine/f"server{i}",[s["key"] for s in segments],
        args.output/"views"/f"server{i}") for i in range(len(cfg["trials"][0]["servers"]))}
    rows=[]
    for number,segment in enumerate(segments):
        old=Path(segment["output"])
        new=args.output/f"point-{number:04d}";new.mkdir()
        for name in ("client","launches.json"):
            (new/name).symlink_to(os.path.relpath((old/name).resolve(),new))
        for i in views:
            server=new/f"server{i}";server.mkdir()
            for name in ("manifest.json","execution.json"):
                (server/name).symlink_to(os.path.relpath((old/f"server{i}"/name).resolve(),server))
            for journal in views[i][segment["key"]].glob("rank*.jsonl"):
                (server/journal.name).symlink_to(os.path.relpath(journal,server))
        expected=json.loads((old/"trial.json").read_text())
        actual=collect_trial_result(expected["trial"],cfg,Path(segment["workload"]),new)
        differences=[key for key in expected if expected[key]!=actual.get(key)]
        if differences:raise RuntimeError(f"Result changed for {segment['key']}: {differences}")
        rows.append(dict(key=segment["key"],exact_result_match=True))
        print("JOURNAL_VIEW_EQUIVALENT",segment["key"],flush=True)
    source=Path(__file__).with_name("e2e_matrix.py")
    (args.output/"summary.json").write_text(json.dumps(dict(passed=True,points=rows,
        postprocessor_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="CPU-only exact equivalence to original full-journal results; no GPU rerun or timing change"),indent=2)+"\n")


if __name__=="__main__":main()
