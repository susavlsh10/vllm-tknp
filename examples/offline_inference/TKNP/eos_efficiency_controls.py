#!/usr/bin/env python3
"""Sequential optional efficiency controls in an existing single-node job.

Run only after the main matrix has stopped. Preserve failed probes and never
change a running/frozen source tree. No allocation or submission is performed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from eos_efficiency import configuration, write_json
from e2e_slurm import client_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args();root=args.root
    state_path=root/"controls.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(steps=[])
    here=Path(__file__).resolve().parent
    def run(name,command):
        old=next((s for s in state["steps"] if s["name"]==name),None)
        if old:return old["return_code"]==0
        start=time.time();print("CONTROL_START",name,flush=True)
        with (root/(name+".log")).open("x") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        state["steps"].append(dict(name=name,command=command,return_code=result.returncode,
            start=start,stop=time.time(),log=str(root/(name+".log"))))
        write_json(state_path,state)
        print("CONTROL_FINISH",name,result.returncode,flush=True)
        return result.returncode==0
    def sweep(name,output,*extra):
        return run(name,[sys.executable,str(here/"eos_efficiency.py"),"--job",args.job,
            "--node",args.node,"--output",str(root/output),"--startup-timeout","240",*extra])
    config=configuration(args.job,args.node,root/"unit-control",32768,128)
    command=client_command(config,[sys.executable,"-m","pytest","--noconftest","-q",
        "tests/v1/worker/test_eos_efficiency.py","tests/v1/worker/test_fp8_decode_sweep.py",
        "tests/v1/worker/test_tknp_graph_sweep.py"])
    if not run("control-units",command):raise RuntimeError("Stop before GPU work: unit checks failed")
    # Warm peer links before sizing KV, not inside a full-memory CUDA capture.
    micro_ok=sweep("control-micro-probes","micro_probes_warm", "--stage","probe",
        "--methods","tknp-dbo","tknp-tp2-dbo","--warmup-cohort-p2p","1","--nccl-debug","INFO")
    if micro_ok:
        sweep("control-micro-matrix","microbatch_warm","--stage","matrix",
            "--methods","tknp-dbo","tknp-tp2-dbo","--capacity-source",str(root/"micro_probes_warm"),
            "--warmup-cohort-p2p","1")
    sweep("control-root-layout","root_bandwidth","--stage","matrix","--methods","tknp","tknp-tp2",
        "--modes","graph","--capacity-source",str(root/"optimized"),"--root-weight-aware","--repeats","2")
    run("control-cpu-kv-pairs",[sys.executable,str(here/"eos_efficiency_pairs.py"),"--job",args.job,
        "--node",args.node,"--output",str(root/"cpu_kv_pairs"),"--capacity-source",str(root/"optimized"),
        "--kind","cpu-kv"])
    state["finished_time"]=time.time();write_json(state_path,state)
    print("EFFICIENCY_CONTROLS_FINISHED",flush=True)


if __name__=="__main__":main()
