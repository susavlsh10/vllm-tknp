#!/usr/bin/env python3
"""Profile a preserved sweep point without changing its engine/graph plan.

The pinned runtime snapshot stays unchanged. Profiling is a separate run, never
throughput evidence, and retains the original max_num_seqs and all graph layouts
so its schedule and compiler-cache identity match the measured engine.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from e2e_matrix import run_trial, validate_matrix
from eos_405b_efficiency import summarize
from eos_efficiency import write_json


def profile_plan(matrix, key, output, after=32, steps=8):
    cfg=copy.deepcopy(matrix)
    if len(cfg["trials"])!=1:
        raise ValueError("Exactly one preserved engine plan required")
    points=[p for p in cfg["client_sweep"] if p["key"]==key]
    if len(points)!=1 or points[0].get("smoke_output_tokens"):
        raise ValueError("Select one measured (not smoke) sweep segment")
    point=points[0]
    point["key"]="profile-"+key
    point["output"]=str(output/"point")
    cfg["client_sweep"]=[point]
    for role in cfg["trials"][0]["servers"]:
        role.update(profile_after_step=after,profile_steps=steps)
    return cfg,point


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--key",default="b96-r0")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--after",type=int,default=32)
    p.add_argument("--steps",type=int,default=8)
    args=p.parse_args()
    if args.after<0 or args.steps<1:p.error("Invalid profile window")
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,point=profile_plan(json.loads(args.matrix.read_text()),args.key,args.output,args.after,args.steps)
    for key,value in cfg["runtime"].get("env",{}).items():
        if key.startswith(("VLLM_","NCCL_")):os.environ[key]=str(value)
    trial=cfg["trials"][0]
    validate_matrix(cfg,len(cfg["slurm_nodes"])*cfg["gpus_per_node"])
    write_json(args.output/"matrix.json",cfg)
    write_json(args.output/"provenance.json",dict(
        parent_matrix=str(args.matrix),parent_sha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_source=cfg["runtime"]["repo"],performance_valid=False))
    results=run_trial(trial,cfg,Path(point["workload"]),args.output/"engine")
    rows=[json.loads(line) for line in Path(point["workload"]).read_text().splitlines()]
    n=rows[0]["max_tokens"]
    context=len(rows[0]["prompt_token_ids"])
    if len(rows)!=point["batch"] or any(r["max_tokens"]!=n or len(r["prompt_token_ids"])!=context for r in rows):
        raise ValueError("This cached-decode profiler summary requires uniform lengths")
    result=summarize(Path(point["output"]),results[0]["trial"],point["batch"],n,context,True)
    traces=list((args.output/"engine").rglob("*.pt.trace.json*"))
    expected=sum(len(role["gpus"]) for role in trial["servers"])
    if len(traces)!=expected:raise RuntimeError("Incomplete per-rank profiler traces")
    result.update(trace_count=len(traces),scope="Profiler diagnostic only; not throughput evidence")
    write_json(args.output/"summary.json",result)
    print("DECODE_MATRIX_PROFILE_COMPLETE",len(traces),flush=True)


if __name__=="__main__":main()
