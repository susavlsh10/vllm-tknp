#!/usr/bin/env python3
"""One explicitly selected Nemotron4 stage in an existing allocation."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source','models','output','tokenizer']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs='+', required=True)
    for key in ['tp','pp','tknp','dcp','dp']:
        p.add_argument('--'+key, type=int, default=1)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--constant-kv', action='store_true')
    p.add_argument('--logits', action='store_true')
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--reserve-tokens', type=int)
    p.add_argument('--batches', type=int, nargs='+', default=[4])
    p.add_argument('--slots', type=int, default=4)
    p.add_argument('--replica-slots', type=int, nargs='+')
    p.add_argument('--layouts', type=int, nargs='+')
    p.add_argument('--kv-bytes', type=int)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if args.tp*args.pp*args.tknp*args.dp > 8*len(args.nodes) or args.tp % args.dcp:
        p.error('Invalid allocation/topology')
    if sum(x > 1 for x in [args.pp,args.tknp,args.dcp,args.dp]) > 1:
        p.error('Use one outer parallelism dimension at a time')
    sys.path.insert(0,str(args.source/'examples/offline_inference/TKNP'))
    from eos_efficiency import configuration, workload, write_json
    from e2e_matrix import run_trial, validate_matrix
    from e1_70b_suite import measured_caps
    reserve = args.reserve_tokens or args.tokens
    if reserve < args.tokens:
        p.error('Reservation smaller than output length')
    args.output.mkdir(parents=True,exist_ok=False)
    model = args.models/(('tiny' if args.tiny else '340b')+('_bf16' if args.bf16 else '_fp8'))
    cfg = configuration(args.job,args.nodes[0],args.output,args.context,reserve)
    cfg.update(slurm_nodes=args.nodes, model=str(model), tokenizer=str(args.tokenizer),
        admission_gated_decode=True,startup_timeout=1800,timeout=3600,
        warmup_requests=1,warmup_output_tokens=4,require_bf16=args.bf16,
        require_fbgemm_fp8=not args.bf16,dummy_fbgemm_fp8=not args.bf16,
        scope='Nemotron4 research FP8/BF16 dummy recipe; BF16 synthetic physical KV; '
              'linear32 extended context is a systems proxy, not model accuracy/prefill validation')
    if args.constant_kv:
        cfg.update(preinitialize_random_kv=False,preinitialize_constant_kv=True)
    cfg['runtime']['container_name']='nemotron4_'+args.job
    cfg['runtime']['env'].update(VLLM_TKNP_V2='1' if args.tknp>1 else '0',
        VLLM_ATTENTION_BACKEND='FLASH_ATTN', HF_HUB_OFFLINE='1',
        VLLM_TKNP_RECORD_GRAPH_IDENTITY='1',
        VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='0' if args.pp>1 else '1',
        VLLM_TKNP_CPU_KV_CONTROL='1',VLLM_TKNP_ROOT_BIAS_TOKENS='0',
        VLLM_TKNP_BALANCED_WINDOW='0',VLLM_RESEARCH_RANK_TRITON_CACHE='1',
        VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1', VLLM_PP_COMPACT_CACHED_TOKENS='0',
        VLLM_CACHE_PROMPT_TOKEN_IDS='0')
    role = dict(role='decode',gpus=list(range(args.tp*args.pp*args.tknp)),
        tp=args.tp,pp=args.pp,tknp=args.tknp,dcp=args.dcp,
        connector='synthetic',cuda_graph=not args.eager,native_graph_shapes=args.tknp==1,
        graph_batch_size=args.slots,max_num_seqs=args.slots,placement='count',
        env={'VLLM_TKNP_LIFETIME_ADMISSION':'1'})
    if args.kv_bytes:
        role['kv_cache_memory_bytes']=args.kv_bytes
    if args.logits:
        role.update(record_full_logits=True,skip_diagnostic_kv=True)
    layouts = sorted(set(args.layouts or [args.slots]))
    if args.tknp>1 and not args.eager:
        if layouts[-1]!=args.slots or any(s<args.tknp or s%args.tknp for s in layouts):
            p.error('Equal positive graph layouts ending at --slots required')
        role.update(graph_slots=True,graph_layouts={s:[s//args.tknp]*args.tknp for s in layouts})
        role['env'].update(VLLM_TKNP_REQUIRE_CUDAGRAPH='1',VLLM_TKNP_CAPTURE_REBALANCE='1',
                          VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(args.context+reserve))
    if args.pp>1:
        role.update(pp_baseline='scheduling-only',
                    pp_decode_microbatch=max(1,math.ceil(args.slots/args.pp)),
                    graph_batch_size=max(1,math.ceil(args.slots/args.pp)))
    roles=[]
    width=args.tp*args.pp*args.tknp
    for replica in range(args.dp):
        r=copy.deepcopy(role);r['gpus']=list(range(replica*width,(replica+1)*width))
        if args.replica_slots:
            if len(args.replica_slots)!=args.dp or min(args.replica_slots)<1:
                p.error('One positive admission/graph ceiling per DP replica required')
            r.update(max_num_seqs=args.replica_slots[replica],graph_batch_size=args.replica_slots[replica])
        roles.append(r)
    trial=dict(name='nemotron4-v2-readiness',servers=roles)
    segments=[]
    for b in args.batches:
        segment=dict(key='b'+str(b),batch=b,
            workload=str(workload(args.output,args.context,b,args.tokens,bos_token_id=2)),
            output=str(args.output/('b'+str(b))))
        if args.tknp>1 and not args.eager:
            segment['graph_size']=next((s for s in layouts if s>=b),layouts[-1])
        if args.pp>1:
            segment['pp_decode_microbatch']=max(1,math.ceil(min(b,args.slots)/args.pp))
        segments.append(segment)
    config=dict(cfg,trials=[trial],client_sweep=segments)
    validate_matrix(config,8*len(args.nodes));write_json(args.output/'plan.json',config)
    if args.plan_only:
        print('NEMOTRON4_PLAN_PASS',flush=True);return
    print('NEMOTRON4_STAGE_START',vars(args),flush=True)
    results=run_trial(trial,config,Path(segments[0]['workload']),args.output/'engine')
    state=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=[])
    graph_identity_reference=None
    for segment,result in zip(segments,results):
        point=Path(segment['output']);raw=json.loads((point/'trial.json').read_text())
        client=json.loads((point/'client/summary.json').read_text())
        assert client['completed']==segment['batch'] and client['failed']==0
        assert client['output_tokens']==segment['batch']*args.tokens
        assert not raw['engine_stage_latency']['prefill']['steps'] and not raw['engine_stage_latency']['mixed']['steps']
        assert all(not t['preemptions_measured'] and t['shutdown']['all_kv_free']
                   for t in raw['telemetry'].values())
        resources=[]
        for i,r in enumerate(roles):
            manifest=json.loads((point/f'server{i}/manifest.json').read_text())
            records=raw['worker_execution'][f'server{i}']
            assert len(records)==len(r['gpus'])
            mode='NONE' if args.eager else 'FULL'
            assert all(x['counts'].get('decode:'+mode,0)>0 for x in records)
            if not args.eager:
                assert all(x['counts'].get('graph:FULL_REPLAY',0)>0 for x in records)
            rs=manifest['worker_resources']
            for resource in rs:
                assert resource['model_dtype']=='torch.bfloat16'
                assert resource['quantization']==(None if args.bf16 else 'fbgemm_fp8')
                assert resource['kv_storage_dtypes']==['torch.bfloat16']
                assert not resource['synthetic_kv']['per_request_fill_calls']
                if args.tknp>1:
                    assert len(resource['kv_attention_layer_names'])==(8 if args.tiny else 96)
                    assert resource['tknp_v2']['equal_attention_ownership']
                    owners=resource['tknp_v2']['layer_owners']
                    assert len(owners)==(8 if args.tiny else 96)
                    assert resource['tknp_v2']['owned_dense_layers']==[
                        layer for layer,owner in enumerate(owners)
                        if owner==resource['tknp_v2']['tknp_rank']]
            resources.extend(rs)
        if args.tknp>1 and not args.eager:
            identities=[r['graph_reuse_identity'] for r in resources]
            assert all(identity['model_parameters'] and identity['kv_pointers'] and identity['graphs']
                       for identity in identities)
            if graph_identity_reference is None:
                graph_identity_reference=identities
            else:
                assert identities==graph_identity_reference, 'Model/KV/graph identity changed between layouts'
        state['trials'].append(dict(batch=segment['batch'],client=client,
            resources=resources,capacity=measured_caps(point,args.context,reserve,result['trial']),
            output_tps=client['admission_gated_decode']['output_tps'],
            performance_valid=not(args.constant_kv or args.logits),full_graph=not args.eager,
            path=str(point)))
    state['graph_identity_verified']=graph_identity_reference is not None
    state['passed']=True;write_json(args.output/'summary.json',state)
    print('NEMOTRON4_STAGE_PASS',flush=True)


if __name__=='__main__':
    main()
