#!/usr/bin/env python3
"""Validate private native TP reductions after prior single-node work drains."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
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
    from eos_efficiency import configuration,write_json
    from eos_fp8_four_node_suite import verify_snapshot
    from e2e_slurm import client_command
    verify_snapshot(args.source)
    deadline=time.monotonic()+3600
    print("WAITING_FOR_ROOT_MICROBATCH_DRAIN",flush=True)
    while True:
        try:prior=json.loads((root/"root_microbatch_status.json").read_text())
        except (OSError,ValueError):prior={}
        if prior.get("finished_time"):break
        if time.monotonic()>deadline:raise RuntimeError("Previous workload not finished; no GPU work launched")
        time.sleep(10)
    if prior["return_code"]:raise RuntimeError("Root microbatch control must pass before this extension")
    state_path=root/"native_tp_controls.json"
    state=dict(source=str(args.source),steps=[],
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def run(name,command):
        print("NATIVE_TP_START",name,flush=True)
        start=time.time()
        with (root/(name+".log")).open("x") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        state["steps"].append(dict(name=name,command=command,return_code=result.returncode,
            started=start,finished=time.time()))
        write_json(state_path,state)
        print("NATIVE_TP_FINISH",name,result.returncode,flush=True)
        return result.returncode==0
    config=configuration(args.job,args.node,root/"native-tp-cache",32768,128)
    tests=client_command(config,[sys.executable,"-m","pytest","--noconftest","-q",
        "tests/v1/worker/test_eos_efficiency.py","tests/v1/worker/test_fp8_decode_sweep.py",
        "tests/v1/worker/test_tknp_graph_sweep.py"])
    if not run("native-tp-units",tests):raise RuntimeError("Native TP unit tests failed")
    for world in (2,8):
        command=client_command(config,[sys.executable,"-m","torch.distributed.run","--standalone",
            "--nproc-per-node",str(world),str(args.source/"examples/offline_inference/TKNP/test_private_cohort_custom_allreduce.py")])
        if not run("native-tp-component"+str(world),["timeout","--signal=TERM","--kill-after=20s","240s",*command]):
            raise RuntimeError("Concurrent custom all-reduce correctness failed")
    script=str(args.source/"examples/offline_inference/TKNP/eos_efficiency.py")
    common=[sys.executable,script,"--job",args.job,"--node",args.node,
        "--capacity-source",str(root/"micro_probes_warm"),"--modes","graph",
        "--warmup-cohort-p2p","1","--root-weight-aware","--startup-timeout","240"]
    if not run("native-tp-root-matrix",[*common,"--output",str(root/"native_tp_root"),
        "--stage","matrix","--methods","tknp-tp2-dbo","--private-tp-custom","1","--repeats","2"]):
        raise RuntimeError("Private native TP full-model validation failed")
    candidates=[]
    for directory,private in (("root_microbatch",False),("native_tp_root",True)):
        rows=json.loads((root/directory/"summary.json").read_text())["trials"]
        for method in sorted({r["method"] for r in rows}):
            selected=[r for r in rows if r["method"]==method and r["batch"]==128]
            if len(selected)!=2:raise RuntimeError("Expected two unprofiled repeats per candidate")
            candidates.append(dict(method=method,private_tp_custom=private,
                median_engine_tps=statistics.median(r["steady_engine_output_tps"] for r in selected),
                median_burst_tps=statistics.median(r["output_tps"] for r in selected),source=str(root/directory)))
    best=max(candidates,key=lambda r:r["median_engine_tps"])
    state["candidates"]=candidates;state["selected_microbatch"]=best
    write_json(root/"selected_microbatch.json",dict(best,candidates=candidates,
        selection="Exploratory B128/32K best microbatch engine rate, two repeats; not a paper-wide universal winner"))
    if not run("profile-selected-microbatch",[*common,"--output",str(root/"profile_micro_selected"),
        "--stage","profile","--methods",best["method"],"--private-tp-custom","1" if best["private_tp_custom"] else "0"]):
        raise RuntimeError("Selected microbatch profiler capture failed")
    state.update(passed=True,finished_time=time.time());write_json(state_path,state)
    print("NATIVE_TP_CONTROLS_COMPLETE",json.dumps(best),flush=True)


if __name__=="__main__":main()
