#!/usr/bin/env python3
"""One method's 32K/64K Nemotron4 paper sweeps in an existing four-node job.

Methods run in independent allocations; model/layout reuse is within each
context. Each context has its own fresh lifetime-capacity probe and engine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


METHODS = {'tknp-v2-tp8': ['--tp','8','--tknp','4'],
           'dp-tp8': ['--tp','8','--dp','4'],
           'tp8-pp4': ['--tp','8','--pp','4'],
           'tp': ['--tp','32'], 'dcp': ['--tp','32','--dcp','4']}
BATCHES = [4,8,16,32,64,96,128]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','parent','output','readiness']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--nodes',nargs=4,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    args=p.parse_args()
    here=args.source/'examples/offline_inference/TKNP'
    sys.path.insert(0,str(here))
    from eos_fp8_four_node_suite import verify_snapshot
    from eos_efficiency import write_json
    from eos_405b_efficiency import summarize
    from tknp_v2_405b_sweep import equal_layouts
    from nemotron4_capacity import smaller_equal_capacity
    snapshot=verify_snapshot(args.source)
    gate=json.loads(args.readiness.read_text())
    assert gate['passed']
    for name,digest in gate['source_sha256'].items():
        assert snapshot['source_sha256'][name]==digest, 'Readiness source mismatch: '+name
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(model='nvidia/Nemotron-4-340B-Instruct architecture',method=args.method,
        job=args.job,nodes=args.nodes,source_commit=snapshot['commit'],source=str(args.source),
        model_config_sha256=hashlib.sha256((args.parent/'models/340b_fp8/config.json').read_bytes()).hexdigest(),
        tokenizer=str(args.parent/'tokenizer'),input_lengths=[32768,65536],output_length=1024,
        batches=BATCHES,capacities={},probe_capacities={},trials=[],
        caveats=['Custom Llama-derived mixed FBGEMM FP8/BF16 dummy recipe, BF16 random KV.',
                 'Original model is4K; linear32 RoPE extension is a synthetic systems proxy, not128K accuracy.',
                 'Admission-gated completion includes natural waves; initialization/prefill excluded.',
                 'Equal owner layouts are fixed within tail waves.',
                 'PP is scheduling-only, native concurrency4, both CPU fast paths0.'])
    write_json(args.output/'summary.json',state)
    for context in [32768,65536]:
        directory=args.output/('c'+str(context));directory.mkdir()
        common=[sys.executable,'-u',str(here/'nemotron4_stage.py'),
            '--source',str(args.source),'--models',str(args.parent/'models'),
            '--tokenizer',str(args.parent/'tokenizer'),'--job',args.job,'--nodes',*args.nodes,
            '--context',str(context),*METHODS[args.method]]
        def run(name,options):
            target=directory/name
            with (directory/(name+'.log')).open('x') as log:
                subprocess.run(common+['--output',str(target)]+options,
                               check=True,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((target/'summary.json').read_text());assert result['passed']
            return result
        probe_slots=8 if args.method=='tknp-v2-tp8' else (4 if args.method=='tp8-pp4' else 1)
        probe_batch=8 if args.method=='tknp-v2-tp8' else 4
        print('PROBE_START',args.method,context,flush=True)
        probe=run('probe',['--tokens','16','--reserve-tokens','1024',
                          '--slots',str(probe_slots),'--batches',str(probe_batch)])
        caps=probe['trials'][0]['capacity']
        if min(caps)<1:raise RuntimeError('No physical lifetime capacity')
        state['capacities'][str(context)]=caps
        state['probe_capacities'][str(context)]=caps
        options=['--tokens','1024','--batches',*map(str,BATCHES)]
        if args.method=='tknp-v2-tp8':
            slots,layouts=equal_layouts(BATCHES,caps)
            options+=['--slots',str(slots),'--layouts',*map(str,layouts)]
            plan=dict(physical_owner_capacities=caps,equal_graph_ceiling=slots,layouts=layouts,
                      unused_slots=sum(caps)-slots)
        elif args.method=='dp-tp8':
            ceilings=[min(32,c) for c in caps]
            options+=['--slots',str(max(ceilings)),'--replica-slots',*map(str,ceilings)]
            plan=dict(physical_replica_capacities=caps,graph_ceilings=ceilings)
        else:
            options+=['--slots',str(min(max(BATCHES),caps[0]))]
            plan=dict(physical_capacity=caps,graph_ceiling=min(max(BATCHES),caps[0]))
        write_json(directory/'capacity_plan.json',plan)
        print('SWEEP_START',args.method,context,'capacity',caps,flush=True)
        sweep_name='sweep'
        if args.method=='tknp-v2-tp8':
            plan['probe_physical_owner_capacities']=caps
            plan['startup_attempts']=[]
            block_sizes={r['block_tokens'] for r in probe['trials'][0]['resources']}
            assert len(block_sizes)==1
            base_options=['--tokens','1024','--batches',*map(str,BATCHES)]
            for attempt in range(3):
                sweep_name='sweep' if attempt==0 else 'sweep-startup-retry'+str(attempt)
                options=base_options+['--slots',str(slots),'--layouts',*map(str,layouts)]
                try:
                    result=run(sweep_name,options)
                except subprocess.CalledProcessError:
                    if attempt==2:
                        raise
                    evidence=smaller_equal_capacity(directory/sweep_name,slots,4,
                        context,1024,next(iter(block_sizes)))
                    plan['startup_attempts'].append(evidence)
                    slots,layouts=equal_layouts(BATCHES,
                        [evidence['equal_graph_ceiling']//4]*4)
                    plan.update(equal_graph_ceiling=slots,layouts=layouts)
                    write_json(directory/'capacity_plan.json',plan)
                    print('STARTUP_CAPACITY_RETRY',context,sweep_name,evidence,flush=True)
                else:
                    break
            actual=result['trials'][0]['capacity']
            assert 4*min(actual)>=slots
            plan.update(physical_owner_capacities=actual,actual_physical_owner_capacities=actual,
                        successful_sweep=sweep_name,unused_slots=sum(actual)-slots)
            state['capacities'][str(context)]=actual
            write_json(directory/'capacity_plan.json',plan)
        else:
            result=run(sweep_name,options)
        for point in result['trials']:
            path=directory/sweep_name/('b'+str(point['batch']))
            raw=json.loads((path/'trial.json').read_text())
            row=summarize(path,raw['trial'],point['batch'],1024,context)
            row.update(method=args.method,pp_baseline='scheduling-only' if args.method=='tp8-pp4' else None,
                resources=point['resources'],source_commit=snapshot['commit'],job=args.job,
                nodes=args.nodes,workload_sha256=raw['workload_sha256'],
                tknp_microbatching=False,tknp_v2=args.method=='tknp-v2-tp8',
                sweep_attempt=sweep_name,
                pp_scheduler_cap_policy='ceil(min(offered,resident_capacity)/4)' if args.method=='tp8-pp4' else None,
                native_pipeline_concurrency=4 if args.method=='tp8-pp4' else None)
            state['trials'].append(row);write_json(args.output/'summary.json',state)
            print('POINT_PASS',args.method,context,point['batch'],row['output_tps'],flush=True)
    assert len(state['trials'])==14
    state['passed']=True;write_json(args.output/'summary.json',state)
    print('NEMOTRON4_METHOD_SWEEP_PASS',flush=True)


if __name__=='__main__':main()
