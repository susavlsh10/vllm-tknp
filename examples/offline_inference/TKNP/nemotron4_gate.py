#!/usr/bin/env python3
"""Fail-closed single-node Nemotron4 validation in an interactive allocation.

Never allocates nodes or submits paper sweeps. A failed stage stops here and
retains its artifacts. Resuming reuses only successful stages with matching
GPU-source hashes; source repairs create distinct attempt directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


GPU_FILES = ['vllm/model_executor/models/nemotron.py',
    'vllm/distributed/tknp_layer_ownership.py',
    'vllm/model_executor/layers/token_parallel_linear.py',
    'vllm/v1/worker/gpu_model_runner.py',
    'vllm/v1/attention/backends/flash_attn.py',
    'vllm/attention/ops/common.py',
    'examples/offline_inference/TKNP/e2e_server.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source','parent','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--node',required=True)
    args=p.parse_args()
    here=args.source/'examples/offline_inference/TKNP'
    sys.path.insert(0,str(here))
    from eos_efficiency import write_json
    from nemotron4_models import TOKENIZER_SOURCE, TOKENIZER_REVISION, TOKENIZER_SHA256
    for name,digest in TOKENIZER_SHA256.items():
        assert hashlib.sha256((args.parent/'tokenizer'/name).read_bytes()).hexdigest()==digest
    hashes={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in GPU_FILES}
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(job=args.job,node=args.node,stages={})
    state.update(passed=False,source_sha256=hashes,tokenizer_provenance=dict(
        source=TOKENIZER_SOURCE,revision=TOKENIZER_REVISION,sha256=TOKENIZER_SHA256))
    def persist():write_json(state_path,state)
    persist()
    runtime='/home/sshrestha/workspace/2026/vllm-tknp-eos-venv/bin/python3'
    image='/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh'
    launch=['srun','--jobid='+args.job,'--overlap','--nodes=1','--ntasks=1',
        '--nodelist='+args.node,'--mpi=pmix','--container-image='+image,
        '--container-name=nemotron4_'+args.job,
        '--container-mounts=/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise']
    subprocess.run(launch+['/bin/true'],check=True)
    test_log=args.output/('targeted_tests_'+str(time.time_ns())+'.log')
    with test_log.open('x') as log:
        subprocess.run(launch+['env','PYTHONPATH='+str(args.source),
            'NEMOTRON4_TOKENIZER='+str(args.parent/'tokenizer'),
            'NEMOTRON4_MODELS='+str(args.parent/'models'),runtime,'-m','pytest',
            '--noconftest','-q',str(args.source/'tests/v1/worker/test_nemotron4_v2.py'),
            str(args.source/'tests/v1/worker/test_nemotron4_v2_gpu.py'),
            str(args.source/'tests/v1/worker/test_tknp_v2_ownership.py'),
            str(args.source/'tests/v1/worker/test_tknp_customop_contract.py')],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    state['targeted_tests']=str(test_log);persist()
    common=[sys.executable,'-u',str(here/'nemotron4_stage.py'),'--source',str(args.source),
        '--models',str(args.parent/'models'),'--tokenizer',str(args.parent/'tokenizer'),
        '--job',args.job,'--nodes',args.node]
    def stage(name,options):
        # This kernel is used only by DCP. Preserve validated non-DCP stages
        # when repairing it, but pin it in the final full-source readiness hash.
        stage_hashes={k:v for k,v in hashes.items()
                      if k!='vllm/attention/ops/common.py' or '--dcp' in options}
        old=state['stages'].get(name,{})
        if old.get('passed') and old.get('source_sha256')==stage_hashes:
            print('STAGE_REUSE',name,flush=True);return Path(old['path'])
        dest=args.output/name
        attempt=0
        while dest.exists():
            attempt+=1;dest=args.output/(name+'-attempt'+str(attempt))
        state['stages'][name]=dict(path=str(dest),passed=False,source_sha256=stage_hashes)
        persist();print('STAGE_START',name,flush=True)
        with dest.with_suffix('.log').open('x') as log:
            subprocess.run(common+['--output',str(dest)]+options,check=True,
                           stdout=log,stderr=subprocess.STDOUT)
        assert json.loads((dest/'summary.json').read_text())['passed']
        state['stages'][name]['passed']=True;persist()
        print('STAGE_PASS',name,flush=True);return dest
    check=['--tiny','--context','128','--tokens','16','--constant-kv','--logits',
           '--kv-bytes',str(256*1024**2)]
    bf_ref=stage('bf16_reference',check+['--bf16','--tp','2','--slots','4','--batches','4'])
    bf_v2=stage('bf16_v2',check+['--bf16','--tp','2','--tknp','4','--slots','16',
                                '--layouts','4','8','16','--batches','4','8','16','20'])
    fp_ref=stage('fp8_reference',check+['--tp','2','--slots','4','--batches','4'])
    fp_v2=stage('fp8_v2',check+['--tp','2','--tknp','4','--slots','16',
                               '--layouts','4','8','16','--batches','4','8','16','20'])
    tp_ref=stage('fp8_tp8_reference',check+['--tp','8','--slots','4','--batches','4'])
    dcp=stage('fp8_dcp4',check+['--tp','8','--dcp','4','--slots','4','--batches','4'])
    pp=stage('fp8_pp4',check+['--tp','2','--pp','4','--slots','8','--batches','4'])
    dp=stage('fp8_dp4',check+['--tp','2','--dp','4','--slots','2','--batches','4'])
    for name,reference,candidate in [('bf16_v2',bf_ref,bf_v2),('fp8_v2',fp_ref,fp_v2),
                                    ('fp8_dcp4',tp_ref,dcp),('fp8_pp4',fp_ref,pp),('fp8_dp4',fp_ref,dp)]:
        output=args.output/(name+'_logits_'+str(time.time_ns())+'.json')
        subprocess.run(launch+['env','PYTHONPATH='+str(args.source),runtime,
            str(here/'tknp_v2_compare.py'),'--reference',str(reference),'--candidate',str(candidate),
            '--batch','4','--output',str(output)],check=True)
        state.setdefault('correctness',{})[name]=json.loads(output.read_text());persist()
    for ref,candidate in [(bf_ref,bf_v2),(fp_ref,fp_v2)]:
        r,c=[json.loads((x/'summary.json').read_text())['trials'][0]['resources'] for x in [ref,candidate]]
        assert sum(x['parameter_storage_bytes'] for x in r)==sum(x['parameter_storage_bytes'] for x in c)
    long=['--tiny','--context','65536','--tokens','32','--reserve-tokens','1024',
          '--kv-bytes',str(2*1024**3),'--batches','8']
    for name,options in [('long_tp',['--tp','8','--slots','4']),
                         ('long_dcp',['--tp','8','--dcp','4','--slots','4']),
                         ('long_dp',['--tp','2','--dp','4','--slots','2']),
                         ('long_pp',['--tp','2','--pp','4','--slots','8'])]:
        stage(name,long+options)
    for context in [32768,65536]:
        stage('long_v2_'+str(context),['--tiny','--context',str(context),'--tokens','32',
            '--reserve-tokens','1024','--tp','2','--tknp','4','--slots','16',
            '--layouts','4','8','16','--batches','4','8','16','20','--kv-bytes',str(2*1024**3)])
    stage('full340b_tp8',['--context','65536','--tokens','32','--reserve-tokens','1024',
                        '--tp','8','--slots','2','--batches','2'])
    stage('full340b_v2',['--context','65536','--tokens','32','--reserve-tokens','1024',
                       '--tp','4','--tknp','2','--slots','2','--batches','2','4'])
    state.update(passed=True,finished_time=time.time(),source_sha256=hashes,
        scope='Single-node implementation/precision/graph/long-context validation; dummy weights, '
              'not trained-model accuracy or proof of 128K model quality')
    persist();write_json(args.output/'readiness_gate.json',state)
    print('NEMOTRON4_SINGLE_NODE_GATE_PASS',flush=True)


if __name__=='__main__':main()
