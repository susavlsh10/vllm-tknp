#!/usr/bin/env python3
"""Serial final controls inside the existing single-node allocation only."""
import argparse
import json
from pathlib import Path
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
    deadline=time.monotonic()+2400
    while True:
        try:
            state=json.loads((root/"gated_standard/summary.json").read_text())
            log=(root/"gated-full.log").read_text()
        except (OSError,ValueError):state={};log=""
        if "EFFICIENCY_STAGE_COMPLETE matrix POINTS 28" in log:break
        if "Traceback (most recent call last)" in log:raise RuntimeError("Prior matrix failed; preserve it")
        if time.monotonic()>deadline:raise RuntimeError("Prior matrix not drained; no GPU controls launched")
        time.sleep(10)
    status=dict(source=str(args.source),steps=[],passed=False)
    def run(name,command):
        print("FINAL_START",name,flush=True)
        start=time.time()
        with (root/(name+".log")).open("x") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        status["steps"].append(dict(name=name,command=command,return_code=result.returncode,
            started=start,finished=time.time()))
        write_json(root/"final_controls.json",status)
        print("FINAL_FINISH",name,result.returncode,flush=True)
        if result.returncode:raise RuntimeError("Failed final control: "+name)
    config=configuration(args.job,args.node,root/"final-cache",32768,128)
    run("final-units",client_command(config,[sys.executable,"-m","pytest","--noconftest","-q",
        "tests/v1/worker/test_eos_efficiency.py","tests/v1/worker/test_fp8_decode_sweep.py",
        "tests/v1/worker/test_tknp_graph_sweep.py"]))
    run("prompt-copy-micro",client_command(config,[sys.executable,
        str(args.source/"examples/offline_inference/TKNP/check_prompt_array_cache.py"),
        "--output",str(root/"prompt_copy_micro.json")]))
    script=str(args.source/"examples/offline_inference/TKNP/eos_efficiency.py")
    common=[sys.executable,script,"--job",args.job,"--node",args.node,
        "--stage","matrix","--capacity-source",str(root/"optimized"),
        "--admission-gated-decode","--startup-timeout","240"]
    # Alternating request-set prompt rebuilding affects PP, not stable TKNP/TP.
    for flag in ("0","1"):
        run("prompt-cache-pair"+flag,[*common,"--output",str(root/("prompt_cache_pair"+flag)),
            "--methods","pp-pipelined","--modes","graph","--batches","128",
            "--repeats","2","--prompt-array-cache",flag,"--pp-compact-cached-tokens",flag])
    run("gated-pp-cached",[*common,"--output",str(root/"gated_pp_cached"),
        "--methods","pp-pipelined","tp2-pp4","--modes","eager","graph",
        "--batches","32","64","128","256","--prompt-array-cache","1"])
    run("gated-micro",[sys.executable,script,"--job",args.job,"--node",args.node,
        "--stage","matrix","--output",str(root/"gated_micro"),
        "--capacity-source",str(root/"micro_probes_warm"),
        "--admission-gated-decode","--startup-timeout","240",
        "--methods","tknp-dbo","tknp-tp2-dbo","--modes","graph",
        "--batches","128","256","--repeats","2",
        "--warmup-cohort-p2p","1","--root-weight-aware"])
    status.update(passed=True,finished_time=time.time())
    write_json(root/"final_controls.json",status)
    print("FINAL_CONTROLS_COMPLETE",flush=True)


if __name__=="__main__":main()
