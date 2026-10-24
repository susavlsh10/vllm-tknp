#!/usr/bin/env python3
"""One Llama TKNP-v2 stage in an existing EOS allocation (no allocation)."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--nodes',nargs='+',required=True)
    p.add_argument('--model',default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--tokenizer',default='/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results/nemotron_modelopt_20261023/tokenizer')
    p.add_argument('--tp',type=int,default=1)
    p.add_argument('--tknp',type=int,default=2)
    p.add_argument('--legacy',action='store_true')
    p.add_argument('--eager',action='store_true')
    p.add_argument('--constant-kv',action='store_true')
    p.add_argument('--fp8',action='store_true')
    p.add_argument('--logits',action='store_true',help='Correctness-only full logits')
    p.add_argument('--profile-steps',type=int,default=0)
    p.add_argument('--profile-after-step',type=int,default=8)
    p.add_argument('--standard-cpu',action='store_true',
                   help='Match paper baseline CPU flags (both optimizations off)')
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--reserve-tokens',type=int,
                   help='Keep production max length / capture reservation for short probes')
    p.add_argument('--batches',nargs='+',type=int,default=[4])
    p.add_argument('--slots',type=int,default=4)
    p.add_argument('--layout-sizes',nargs='+',type=int,
                   help='Equal-owner FULL graph layouts, changed only after drain')
    p.add_argument('--kv-bytes',type=int)
    p.add_argument('--container-name',required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.source/'examples/offline_inference/TKNP'))
    from eos_efficiency import configuration,workload,write_json
    from e2e_matrix import run_trial,validate_matrix
    from e1_70b_suite import measured_caps
    args.output.mkdir(parents=True,exist_ok=False)
    reserve=args.reserve_tokens or args.tokens
    if reserve<args.tokens:raise ValueError('Reservation shorter than output')
    cfg=configuration(args.job,args.nodes[0],args.output,args.context,reserve)
    cfg.update(slurm_nodes=args.nodes,model=args.model,tokenizer=args.tokenizer,
        admission_gated_decode=True,startup_timeout=1800,timeout=1800,
        warmup_requests=1,warmup_output_tokens=4,require_bf16=not args.fp8,
        require_fbgemm_fp8=args.fp8,dummy_fbgemm_fp8=args.fp8,
        scope=f'{args.model}, {args.tp*args.tknp} GPUs; dummy weights, physical BF16 synthetic KV; '
              'admission-gated decode including waves, no measured prefill or accuracy claim')
    if args.constant_kv:
        cfg.update(preinitialize_random_kv=False,preinitialize_constant_kv=True)
    cfg['runtime']['container_name']=args.container_name
    cfg['runtime']['env'].update(VLLM_TKNP_V2='0' if args.legacy or args.tknp==1 else '1',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN',HF_HUB_OFFLINE='1',
        VLLM_TKNP_RECORD_GRAPH_IDENTITY='1',VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='1',
        VLLM_TKNP_CPU_KV_CONTROL='1',VLLM_TKNP_ROOT_BIAS_TOKENS='0',
        VLLM_TKNP_BALANCED_WINDOW='0',VLLM_RESEARCH_RANK_TRITON_CACHE='1',
        VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1')
    if args.standard_cpu:
        cfg['runtime']['env'].update(VLLM_PP_COMPACT_CACHED_TOKENS='0',
                                     VLLM_CACHE_PROMPT_TOKEN_IDS='0')
    counts=[args.slots//args.tknp]*args.tknp
    if args.slots%args.tknp:raise ValueError('Equal graph slots required')
    role=dict(role='decode',gpus=list(range(args.tp*args.tknp)),tp=args.tp,pp=1,
        tknp=args.tknp,dcp=1,connector='synthetic',cuda_graph=not args.eager,
        native_graph_shapes=args.tknp==1,graph_batch_size=args.slots,max_num_seqs=args.slots,
        placement='count',env={'VLLM_TKNP_LIFETIME_ADMISSION':'1'})
    if args.logits:role.update(record_full_logits=True,skip_diagnostic_kv=True)
    if args.profile_steps:role.update(profile_steps=args.profile_steps,
                                     profile_after_step=args.profile_after_step)
    if args.kv_bytes:role['kv_cache_memory_bytes']=args.kv_bytes
    if args.tknp>1 and not args.eager:
        sizes=sorted(set(args.layout_sizes or [args.slots]))
        if sizes[-1]!=args.slots or any(s<args.tknp or s%args.tknp for s in sizes):
            raise ValueError('Layouts must be positive equal-owner shapes ending at --slots')
        role.update(graph_slots=True,graph_layouts={s:[s//args.tknp]*args.tknp for s in sizes})
        role['env'].update(VLLM_TKNP_REQUIRE_CUDAGRAPH='1',VLLM_TKNP_CAPTURE_REBALANCE='1',
            VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(args.context+reserve))
    trial=dict(name='tknp-v2-readiness',servers=[role])
    segments=[dict(key='b'+str(b),batch=b,workload=str(workload(args.output,args.context,b,args.tokens)),
                   output=str(args.output/('b'+str(b)))) for b in args.batches]
    if args.tknp>1 and not args.eager:
        for segment in segments:
            segment['graph_size']=next((s for s in sizes if s>=segment['batch']),sizes[-1])
    config=dict(cfg,trials=[trial],client_sweep=segments)
    validate_matrix(config,8*len(args.nodes));write_json(args.output/'plan.json',config)
    results=run_trial(trial,config,Path(segments[0]['workload']),args.output/'engine')
    state=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=[])
    for seg,result in zip(segments,results):
        point=Path(seg['output']);raw=json.loads((point/'trial.json').read_text())
        c=json.loads((point/'client/summary.json').read_text())
        assert c['completed']==seg['batch'] and c['failed']==0 and c['output_tokens']==seg['batch']*args.tokens
        assert all(not t['preemptions_measured'] and t['shutdown']['all_kv_free'] for t in raw['telemetry'].values())
        records=raw['worker_execution']['server0']
        assert len(records)==args.tp*args.tknp
        mode='NONE' if args.eager else 'FULL'
        assert all(r['counts'].get('decode:'+mode,0)>0 for r in records)
        if not args.eager:assert all(r['counts'].get('graph:FULL_REPLAY',0)>0 for r in records)
        m=json.loads((point/'server0/manifest.json').read_text())
        if not args.legacy and args.tknp>1:
            plans=[r['tknp_v2'] for r in m['worker_resources']]
            assert all(p['layer_owners']==plans[0]['layer_owners'] for p in plans)
            if not args.eager:
                splits=m['worker_resources'][0]['graph_splits']
                assert all(len(set(c))==1 for c in splits.values()),splits
        state['trials'].append(dict(batch=seg['batch'],performance_valid=m['performance_valid'],
            output_tps=c['admission_gated_decode']['output_tps'],
            client=c,resources=m['worker_resources'],capacity=measured_caps(point,args.context,reserve,result['trial'])))
    state['passed']=True
    write_json(args.output/'summary.json',state)
    print('TKNP_V2_STAGE_PASS',flush=True)


if __name__=='__main__':main()
