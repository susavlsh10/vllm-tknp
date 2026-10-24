#!/usr/bin/env python3
"""Profile repaired PP after all unprofiled controls drain; no new allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--job",required=True)
    p.add_argument("--node",required=True)
    args=p.parse_args();root=args.root
    sys.path.insert(0,str(args.source/"examples/offline_inference/TKNP"))
    from eos_efficiency import configuration,make_trial,workload,audit_point,write_json
    from eos_fp8_four_node_suite import verify_snapshot
    from e2e_matrix import run_trial
    deadline=time.monotonic()+4200
    while True:
        try:prior=json.loads((root/"final_controls.json").read_text())
        except (OSError,ValueError):prior={}
        if prior.get("finished_time"):break
        if any(s.get("return_code") for s in prior.get("steps",[])):
            raise RuntimeError("Prior control failed; do not launch profiler")
        if time.monotonic()>deadline:raise RuntimeError("Controls not drained; no profiler launched")
        time.sleep(10)
    if not prior["passed"]:raise RuntimeError("Controls must pass before profiling")
    verify_snapshot(args.source)
    config=configuration(args.job,args.node,root/"pp-profile-cache",32768,128)
    config.update(admission_gated_decode=True,startup_timeout=240)
    config["runtime"]["env"]["VLLM_CACHE_PROMPT_TOKEN_IDS"]="1"
    caps=json.loads((root/"optimized/summary.json").read_text())["capacities"]["pp-pipelined"]
    trial=make_trial("pp-pipelined","graph",128,caps,32768,128)
    trial["servers"][0].update(profile_after_step=256,profile_steps=64)
    config["trials"]=[trial]
    destination=root/"profile_pp_cached"
    if destination.exists():raise RuntimeError("Preserve existing profile; choose a new destination")
    run_trial(trial,config,workload(root,32768,128,128),destination)
    row=audit_point(destination,trial,128,128,profiled=True)
    files=sorted(destination.glob("server0/traces/*.pt.trace.json*"))
    if len(files)!=8:raise RuntimeError("Expected a completed profiler trace on all eight ranks")
    write_json(root/"pp_cached_profile_status.json",dict(passed=True,profile=row,
        traces=[str(f) for f in files],source=str(args.source),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Profiler mechanism evidence only; not unprofiled performance"))
    print("PP_CACHED_PROFILE_COMPLETE",flush=True)


if __name__=="__main__":main()
