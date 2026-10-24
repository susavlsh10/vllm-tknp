#!/usr/bin/env python3
"""Freeze and submit five independent Nemotron4 method jobs AFTER readiness.

Each owns four nodes and collects both32K and64K contexts. No dependencies,
TKNP-v1, extra baselines, checkpoint downloads, or development allocation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--readiness',type=Path,required=True)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    from nemotron4_sweep import METHODS,BATCHES
    from eos_fp8_four_node_suite import freeze_source,verify_snapshot,write_json
    repo=Path(__file__).resolve().parents[3]
    if not args.submit:
        print(json.dumps(dict(methods=METHODS,contexts=[32768,65536],batches=BATCHES,
            output_length=1024,nodes_per_job=4,max_concurrent_nodes=20,dependencies=None),indent=2))
        return
    gate=json.loads(args.readiness.read_text())
    if not gate['passed'] or not gate.get('correctness'):
        raise RuntimeError('Passed single-node correctness and full-model gate required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('Preserve/commit reviewed source before collection')
    for name,digest in gate['source_sha256'].items():
        import hashlib
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Readiness GPU source changed: '+name)
    # Do not let the development allocation consume extra nodes during collection.
    active=subprocess.check_output(['squeue','-h','-j',str(gate['job']),'-o','%T'],text=True).strip()
    if active:
        raise RuntimeError('Release the interactive development allocation before collection: '+active)
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.output/'source'
    snapshot=freeze_source(repo,source);verify_snapshot(source)
    state=dict(source=str(source),source_commit=snapshot['commit'],readiness=str(args.readiness),
        parent=str(args.parent),contexts=[32768,65536],batches=BATCHES,output_length=1024,
        dummy_weights=True,kv_dtype='bfloat16',context_extension='linear32 from4096 to131072',
        nodes_per_job=4,max_concurrent_nodes=20,dependencies=None,jobs=[])
    write_json(args.output/'submission.json',state)
    script=source/'examples/offline_inference/TKNP/jobs/eos_nemotron4_v2.sbatch'
    for method in METHODS:
        output=args.output/method
        command=['sbatch','--parsable','--dependency=',
            '--job-name=hw_nresearch_snoise-snoise:nemo4-'+method,
            '--output='+str(args.output/(method+'-%j.log')),str(script),str(source),
            str(args.parent),str(args.readiness),str(output),method]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Unknown submission outcome; inspect before retry')
        state['jobs'].append(dict(method=method,job=job,output=str(output),command=command,
                                  submitted_at=time.time()))
        write_json(args.output/'submission.json',state)
        print('SUBMITTED',method,job,flush=True)


if __name__=='__main__':main()
