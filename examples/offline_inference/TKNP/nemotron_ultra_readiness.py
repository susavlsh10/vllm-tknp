#!/usr/bin/env python3
"""Run ONE explicitly selected stage in an existing interactive allocation."""
import argparse
import json
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--node')
    p.add_argument('--nodes',nargs='+')
    p.add_argument('--tp', type=int, default=8)
    p.add_argument('--tknp', type=int, default=1)
    p.add_argument('--dp', type=int, default=1)
    p.add_argument('--dcp',type=int,default=1)
    p.add_argument('--pp',type=int,default=1)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--fp8', action='store_true')
    p.add_argument('--modelopt', action='store_true', help='Official ModelOpt FP8 weights and FP8 KV')
    p.add_argument('--container-name')
    p.add_argument('--tokenizer',default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--disable-compile', action='store_true',
                   help='Diagnostic FULL graph capture without Inductor; not a paper timing')
    p.add_argument('--constant-kv', action='store_true', help='Identical cache correctness control, not timing')
    p.add_argument('--context', type=int, default=98304)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--batches', type=int, nargs='+', default=[1,2])
    p.add_argument('--owner-counts', type=int, nargs='+')
    p.add_argument('--gpu-memory-utilization', type=float, default=.9)
    p.add_argument('--graph-batch-size', type=int, help='Explicit graph ceiling independent of offered batch')
    args = p.parse_args()
    nodes=args.nodes or ([args.node] if args.node else [])
    if not nodes or (args.nodes and args.node):
        p.error('Choose --node or --nodes')
    if args.tp*args.tknp*args.dp*args.pp>8*len(nodes) or args.tp%args.dcp:
        p.error('Topology exceeds allocation or DCP does not divide TP')
    if args.tknp>1 and args.dcp>1:
        p.error('TKNP+DCP combination is outside this readiness study')
    if args.dp>1 and args.tknp>1:
        raise ValueError('This readiness harness does not combine DP and TKNP')
    if args.pp>1 and (args.tknp>1 or args.dcp>1 or args.dp>1):
        p.error('PP baseline must be a standalone TP x PP configuration')
    sys.path.insert(0, str(args.source/'examples/offline_inference/TKNP'))
    from eos_efficiency import configuration, audit_point, write_json, workload
    from e2e_matrix import run_trial, validate_matrix
    from fp8_decode_sweep import audit as audit_fp8
    from e1_70b_suite import measured_caps
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args.job, nodes[0], args.output, args.context, 1024)
    cfg['slurm_nodes']=nodes
    cfg['scope'] = ('Nemotron NAS synthetic decode only; dummy recipe weights, '
                    f'{len(nodes)} EOS nodes, FULL CUDA graphs; no prefill or accuracy claim')
    name = 'nemotron_nas_tiny' if args.tiny else 'nemotron_ultra_253b'
    if args.modelopt:
        name += '_modelopt'
    elif args.fp8:
        name += '_fp8'
    cfg.update(model=str(args.source/'examples/offline_inference/TKNP/model_configs'/name),
               tokenizer=args.tokenizer,
               admission_gated_decode=True, startup_timeout=1800, timeout=1200,
               warmup_requests=1, warmup_output_tokens=4,
               require_bf16=not (args.fp8 or args.modelopt), require_fbgemm_fp8=args.fp8 and not args.modelopt,
               dummy_fbgemm_fp8=args.fp8)
    if args.modelopt:
        cfg.update(require_modelopt_fp8=True,dummy_modelopt_fp8=True,dummy_fbgemm_fp8=False,
                   kv_cache_dtype='fp8')
    cfg['gpu_memory_utilization'] = args.gpu_memory_utilization
    if args.constant_kv:
        cfg.update(preinitialize_random_kv=False,preinitialize_constant_kv=True)
    cfg['runtime']['container_name'] = args.container_name or 'nemotron_runtime'+args.job
    cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN', HF_HUB_OFFLINE='1',
        VLLM_TKNP_RECORD_GRAPH_IDENTITY='1', VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='1',
        VLLM_TKNP_VALIDATE_HETERO_METADATA='0', VLLM_TKNP_BALANCED_WINDOW='0',
        VLLM_TKNP_ROOT_BIAS_TOKENS='0', VLLM_RESEARCH_RANK_TRITON_CACHE='1',
        VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1', VLLM_TKNP_CPU_KV_CONTROL='1',
        VLLM_RESEARCH_STACK_DUMP='1')
    if args.pp>1:
        cfg['runtime']['env'].update(VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='0',
            VLLM_PP_COMPACT_CACHED_TOKENS='0',VLLM_CACHE_PROMPT_TOKEN_IDS='0')
    maximum = args.graph_batch_size or math.ceil(max(args.batches)/args.dp)
    role = dict(role='decode', gpus=list(range(args.tp*args.tknp*args.pp)), tp=args.tp,
                pp=args.pp, tknp=args.tknp, dcp=args.dcp, connector='synthetic', cuda_graph=True,
                native_graph_shapes=args.tknp==1, graph_batch_size=maximum,
                max_num_seqs=maximum, disable_combo_kernels=False,
                disable_compile=args.disable_compile,
                env=dict(VLLM_TKNP_LIFETIME_ADMISSION='1'))
    if args.pp>1:
        role.update(pp_baseline='scheduling-only',pp_decode_microbatch=math.ceil(maximum/args.pp),
                    graph_batch_size=math.ceil(maximum/args.pp))
    if args.constant_kv:
        role['record_logit_topk'] = True
    if args.tiny:
        role['kv_cache_memory_bytes'] = 512*1024**2
    if args.tknp>1:
        counts = args.owner_counts or [maximum//args.tknp]*args.tknp
        if len(counts)!=args.tknp or sum(counts)!=maximum:
            raise ValueError('Explicit owners must sum to largest batch')
        role.update(placement='cost', graph_slots=True, graph_layouts={maximum:counts})
        role['env'].update(VLLM_TKNP_REQUIRE_CUDAGRAPH='1', VLLM_TKNP_CAPTURE_REBALANCE='1',
                           VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(args.context+1024))
    roles=[]
    for replica in range(args.dp):
        own=json.loads(json.dumps(role))
        width=args.tp*args.tknp*args.pp
        own['gpus']=list(range(replica*width,(replica+1)*width))
        roles.append(own)
    trial = dict(name='nemotron-readiness', servers=roles)
    segments=[]
    for repeat in range(args.repeats):
        for b in args.batches:
            key=f'b{b}' if args.repeats==1 else f'b{b}-r{repeat}'
            segments.append(dict(key=key,batch=b,workload=str(workload(args.output,args.context,b,args.tokens)),
                                 output=str(args.output/key)))
            if args.pp>1:
                segments[-1]['pp_decode_microbatch']=math.ceil(min(b,maximum)/args.pp)
    config = dict(cfg,trials=[trial],client_sweep=segments)
    validate_matrix(config,8*len(nodes))
    write_json(args.output/'plan.json',config)
    if args.plan_only:
        print('NEMOTRON_PLAN_VALIDATED',flush=True)
        return
    print('NEMOTRON_STAGE_START',vars(args),flush=True)
    results = run_trial(trial,config,Path(segments[0]['workload']),args.output/'engine')
    state = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=[])
    for segment,result in zip(segments,results):
        point = Path(segment['output'])
        if args.constant_kv:
            client=json.loads((point/'client/summary.json').read_text())
            data=json.loads((point/'trial.json').read_text())
            assert client['completed']==segment['batch'] and not client['failed']
            assert client['output_tokens']==segment['batch']*args.tokens
            assert all(not x['preemptions_measured'] and x['shutdown']['all_kv_free']
                       for x in data['telemetry'].values())
            executions=[r for values in data['worker_execution'].values() for r in values]
            assert len(executions)==args.tp*args.tknp*args.dp*args.pp
            assert all(r['counts'].get('graph:FULL_REPLAY',0)>0 for r in executions)
            row=dict(batch=segment['batch'],client=client,performance_valid=False,
                     diagnostic=True,all_ranks_verified=True)
        elif args.modelopt:
            from fp8_decode_sweep import analyze
            data=analyze(point)
            client=data['client']
            assert client['completed']==segment['batch'] and not client['failed']
            assert client['output_tokens']==segment['batch']*args.tokens
            for gate in ('all_decode_workers_full_graph','all_servers_zero_measured_preemptions','all_servers_kv_drained'):
                assert data[gate],gate
            raw=json.loads((point/'trial.json').read_text())
            assert not raw['engine_stage_latency']['prefill']['steps']
            assert not raw['engine_stage_latency']['mixed']['steps']
            records=[r for rows in raw['worker_execution'].values() for r in rows]
            assert len(records)==args.tp*args.tknp*args.dp*args.pp
            assert all(r['counts'].get('graph:FULL_REPLAY',0)>0 and not r['per_request_kv_fill_calls'] for r in records)
            assert all(r['counts'].get('finite_logits_checks',0)>0 for r in records
                       if (r['rank']>=args.tp*(args.pp-1) if args.pp>1 else r['rank']<args.tp))
            row=dict(batch=segment['batch'],client=client,output_tps=client['admission_gated_decode']['output_tps'],
                     performance_valid=not args.disable_compile,all_ranks_verified=True)
        elif args.fp8:
            data = audit_fp8(point,result['trial'],segment['batch'],args.tokens)
            row = dict(batch=segment['batch'],output_tokens=segment['batch']*args.tokens,
                       output_tps=data['client']['admission_gated_decode']['output_tps'],
                       client=data['client'],performance_valid=not args.disable_compile,all_ranks_verified=True)
        else:
            row = audit_point(point,result['trial'],segment['batch'],args.tokens,context=args.context)
            if args.disable_compile:
                row['performance_valid'] = False
        resources=[]
        for path in sorted(point.glob('server*/manifest.json')):
            manifest=json.loads(path.read_text())
            resources.extend(dict(r,server=path.parent.name) for r in manifest['worker_resources'])
        expected_layers = 2 if args.tiny else 64
        if args.pp==1:
            assert all(len(r['kv_attention_layer_names'])==expected_layers for r in resources)
        else:
            assert sum(len(r['kv_attention_layer_names']) for r in resources)==expected_layers*args.tp
            manifest=json.loads((point/'server0/manifest.json').read_text())
            assert manifest['engine_batch_queue_size']==args.pp
            assert all(manifest['runtime_env'][k]=='0' for k in
                       ('VLLM_PP_COMPACT_CACHED_TOKENS','VLLM_CACHE_PROMPT_TOKEN_IDS'))
        for r in resources:
            # Per-rank page accounting must agree with active attention layers,
            # local KV heads and BF16 storage, not nominal block-position count.
            # TP greater than the KV-head count replicates KV heads; DCP
            # instead partitions sequence pages, not this physical head count.
            local_heads = max(1,(2 if args.tiny else 8)//args.tp)
            head_dim = 128
            active_layers=len(r['kv_attention_layer_names'])
            expected = r['kv_blocks']*r['block_tokens']*active_layers*2*local_heads*head_dim*(1 if args.modelopt else 2)
            assert r['kv_allocated_bytes']==expected,(r['rank'],r['kv_allocated_bytes'],expected)
            if args.modelopt:
                assert r['quantization']=='modelopt' and r['model_dtype']=='torch.bfloat16'
                assert r['kv_cache_dtype'] in ('fp8','fp8_e4m3')
                assert r['kv_storage_dtypes'] in (['torch.uint8'],['torch.float8_e4m3fn'])
                assert r['synthetic_kv']['preinitialized_bytes']==r['kv_allocated_bytes']
                assert len(r['kv_scales'])==active_layers
                assert all(x['k']>0 and x['v']>0 and x['q']>0 for x in r['kv_scales'])
                if r['rank']<args.tp:
                    assert r['model_storage_bytes'].get('torch.float8_e4m3fn',0)>0
        row.update(key=segment['key'],path=str(point))
        row.update(resources=resources,capacity96k_osl1024=measured_caps(point,98304,1024,result['trial']))
        state['trials'].append(row)
    state['passed']=len(state['trials'])==len(segments)
    write_json(args.output/'summary.json',state)
    print('NEMOTRON_STAGE_COMPLETE',state['passed'],flush=True)


if __name__=='__main__':
    main()
