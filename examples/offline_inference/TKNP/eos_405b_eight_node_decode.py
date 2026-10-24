#!/usr/bin/env python3
"""Sequential equal-budget 64-H100 cached-decode sweep; no allocation here."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--nodes',nargs=8,required=True)
    p.add_argument('--methods',nargs='+',choices=['dp-tp8','tknp-tp8','dcp','tp','tp8-pp8'],
                   default=['dp-tp8','tknp-tp8','dcp','tp','tp8-pp8'])
    args=p.parse_args()
    sys.path.insert(0,str(args.source/'examples/offline_inference/TKNP'))
    from eos_fp8_four_node_suite import verify_snapshot
    from fp8_decode_sweep import configuration,trial_for
    from eos_405b_efficiency import summarize
    from eos_efficiency import write_json,workload
    from e1_70b_suite import measured_caps
    from e2e_matrix import run_trial,validate_matrix
    verify_snapshot(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    batches=[8,16,32,64,96,128]
    cfg=configuration(args.job,args.nodes,str(args.parent/'model405b'),32768,1024)
    cfg.update(admission_gated_decode=True,startup_timeout=1200,timeout=3600)
    cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN',
        VLLM_PP_COMPACT_CACHED_TOKENS='0',VLLM_CACHE_PROMPT_TOKEN_IDS='0',
        VLLM_RESEARCH_RANK_TRITON_CACHE='1',VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1',
        VLLM_TKNP_DBO_WARMUP_P2P='1',HF_HUB_OFFLINE='1')
    for k,v in cfg['runtime']['env'].items():
        if k.startswith(('VLLM_','NCCL_')):os.environ[k]=str(v)
    state=dict(nodes=args.nodes,job=args.job,gpu_budget=64,batches=batches,
        context=32768,output_length=1024,source=str(args.source),trials=[],failures=[],
        caveats=['Dummy FP8-recipe weights and physical BF16 random KV; not accuracy validation.',
                 'TP-only is TP64, not TP32, to match the64GPU budget.',
                 'Batch sweep starts at8 so every DP replica can receive at least one request.'])
    def trial(method,batch,caps):
        t=trial_for(method,32768,1024,batch,caps,64,'tuned' if method=='tp8-pp8' else 'standard')
        if method=='tp8-pp8':
            r=t['servers'][0];r['pp_baseline']='scheduling-only'
            r['env'].update(VLLM_PP_COMPACT_CACHED_TOKENS='0',VLLM_CACHE_PROMPT_TOKEN_IDS='0')
        return t
    for method in args.methods:
        print('METHOD_START',method,flush=True)
        dest=args.output/method;dest.mkdir()
        try:
            probe_batch=16 if method=='tknp-tp8' else 8
            initial=[2]*8 if method in ('dp-tp8','tknp-tp8') else [probe_batch]
            pt=trial(method,probe_batch,initial)
            pc=dict(cfg,trials=[pt]);validate_matrix(pc,64)
            run_trial(pt,pc,workload(dest,32768,probe_batch,16),dest/'probe')
            summarize(dest/'probe',pt,probe_batch,16,32768)
            caps=measured_caps(dest/'probe',32768,1024,pt)
            t=trial(method,max(batches),caps)
            sizes={}
            if method=='tknp-tp8':
                role=t['servers'][0];layouts={}
                for b in batches:
                    r=trial(method,b,caps)['servers'][0]
                    sizes[b]=r['graph_batch_size'];layouts[sizes[b]]=r['owner_counts']
                role.pop('owner_counts');role['graph_layouts']=layouts
            segments=[]
            for b in batches:
                seg=dict(key=f'b{b}',batch=b,workload=str(workload(dest,32768,b,1024)),output=str(dest/f'b{b}'))
                if sizes:seg['graph_size']=sizes[b]
                if method=='tp8-pp8':seg['pp_decode_microbatch']=math.ceil(min(b,caps[0])/8)
                segments.append(seg)
            smokes=[dict(s,key='smoke-'+s['key'],workload=str(workload(dest,32768,s['batch'],8)),
                         output=str(dest/('smoke-'+s['key'])),smoke_output_tokens=8) for s in segments]
            config=dict(cfg,trials=[t],client_sweep=smokes+segments)
            validate_matrix(config,64);write_json(dest/'plan.json',config)
            results=run_trial(t,config,Path(smokes[0]['workload']),dest/'persistent')
            for seg,result in zip(smokes+segments,results):
                row=summarize(Path(seg['output']),result['trial'],seg['batch'],seg.get('smoke_output_tokens',1024),32768)
                if 'smoke_output_tokens' in seg:continue
                row.update(method=method,pp_baseline='scheduling-only' if method=='tp8-pp8' else None,
                           configured_graph_size=seg.get('graph_size'),pp_decode_microbatch=seg.get('pp_decode_microbatch'))
                state['trials'].append(row)
                write_json(args.output/'summary.json',state)
                print('POINT_PASS',method,seg['batch'],row['output_tps'],flush=True)
            print('METHOD_PASS',method,flush=True)
        except Exception as exc:
            traceback.print_exc()
            state['failures'].append(dict(method=method,error=repr(exc)))
            write_json(args.output/'summary.json',state)
    write_json(args.output/'summary.json',state)
    print('EIGHT_NODE_COMPLETE failures=',len(state['failures']),flush=True)
    if state['failures']:raise SystemExit(1)


if __name__=='__main__':main()
