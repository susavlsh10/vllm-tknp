#!/usr/bin/env python3
"""Capacity-probed Nemotron ModelOpt FP8 sweep in an existing 32-GPU job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

METHODS = {'dp': ['--tp','8','--dp','4'],
           'tknp': ['--tp','8','--tknp','4'],
           'tp': ['--tp','32'],
           'dcp': ['--tp','32','--dcp','4'],
           'pp': ['--tp','8','--pp','4']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--nodes',nargs=4,required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--container-name',required=True)
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--batches',nargs='+',type=int,default=[8,16,32,64,96,128,160,192,256])
    args=p.parse_args()
    if any(b<8 or b%4 for b in args.batches):
        p.error('Use batches >=8 divisible by4 for the supported four-owner layouts')
    sys.path.insert(0,str(args.source/'examples/offline_inference/TKNP'))
    from e1_70b_suite import quotas
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(job=args.job,nodes=args.nodes,model='nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8',
        context=98304,output_tokens=1024,batches=args.batches,stages={},capacities={},trials=[],
        source=str(args.source),scope='Admission-gated synthetic cached decode; dummy ModelOpt FP8 weights/activations, FP8 KV; no accuracy/prefill/arrival validation',
        caveats=['PP is scheduling-only; both CPU fast paths disabled; native concurrency4.',
                 'TKNP uses one fixed ownership layout per engine; equal layouts reuse the model.',
                 'TKNP reserves one request per owner for graph growth; native baselines use full probed ceilings.',
                 'Natural waves and tail padding are included in completion throughput. No TKNP DBO.'])

    def save():
        tmp=args.output/'summary.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(args.output/'summary.json')

    def run(name,method,batches,graph,tokens,owners=None):
        output=args.output/name
        cmd=[sys.executable,str(args.source/'examples/offline_inference/TKNP/nemotron_ultra_readiness.py'),
             '--source',str(args.source),'--output',str(output),'--job',args.job,
             '--nodes',*args.nodes,'--modelopt','--container-name',args.container_name,
             '--tokenizer',args.tokenizer,'--context','98304','--tokens',str(tokens),
             '--gpu-memory-utilization','0.90','--graph-batch-size',str(graph),
             '--batches',*map(str,batches),*METHODS[method]]
        if owners is not None:cmd+=['--owner-counts',*map(str,owners)]
        print('STAGE_START',name,flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        state['stages'][name]=dict(method=method,returncode=rc,command=cmd)
        save()
        if rc:
            print('STAGE_FAILED',name,rc,flush=True)
            return None
        data=json.loads((output/'summary.json').read_text())
        assert data['passed']
        if tokens==1024:
            for row in data['trials']:
                state['trials'].append(dict(method=method,batch=row['batch'],
                    output_tps=row['output_tps'],client=row['client'],
                    capacity=row['capacity96k_osl1024'],summary=str(output/'summary.json'),
                    key=row['key'],pp_baseline='scheduling-only' if method=='pp' else None,
                    graph_ceiling=graph,owner_counts=owners))
        save()
        print('STAGE_PASS',name,flush=True)
        return data

    for method in args.methods:
        probe=run('probe_'+method,method,[8],2 if method=='dp' else 8,32,
                  [2]*4 if method=='tknp' else None)
        if probe is None:continue
        caps=probe['trials'][0]['capacity96k_osl1024']
        state['capacities'][method]=caps
        save()
        if method=='tknp':
            usable=[max(0,c-1) for c in caps]
            if min(usable)<2:raise RuntimeError('Insufficient TKNP owner capacity')
            groups={}
            for batch in args.batches:
                counts=tuple(quotas(usable,min(batch,sum(usable))))
                groups.setdefault(counts,[]).append(batch)
            for index,(counts,batches) in enumerate(groups.items()):
                run('decode_tknp_layout'+str(index),method,batches,sum(counts),1024,list(counts))
        else:
            run('decode_'+method,method,args.batches,min(max(args.batches),min(caps)),1024)
    state['complete']=(len(state['trials'])==len(args.methods)*len(args.batches)
                       and all(s['returncode']==0 for s in state['stages'].values()))
    save()
    print('NEMOTRON_FOUR_NODE_COMPLETE',state['complete'],flush=True)
    return 0 if state['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
