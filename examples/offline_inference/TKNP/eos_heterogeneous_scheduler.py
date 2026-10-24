#!/usr/bin/env python3
"""Single-node8B heterogeneous-context readiness, never submits multi-node work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--node',required=True)
    p.add_argument('--variant',choices=['fifo','window','root-aware'],required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.source/'examples/offline_inference/TKNP'))
    from eos_efficiency import configuration,make_trial,audit_point,write_json
    from e2e_matrix import run_trial,validate_matrix
    from eos_fp8_four_node_suite import verify_snapshot
    from heterogeneous_decode import DISTRIBUTIONS,make_mixed_workload,workload_description,summarize_mixture
    verify_snapshot(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(job=args.job,node=args.node,model='Llama3.1 8B BF16 dummy',kv='BF16 random physical',
        max_context=98304,graph_resident_request_slots=24,output_lengths=[128,1024],trials=[],diagnostics=[],
        failures=[],scope='Single-node scheduler comparison; fixed24 graph lanes deliberately force B48 waves, not a physical KV capacity result.',
        dcp='Nontrivial DCP is not supported for8B8KVheads on8GPUs; no DCP result is fabricated.',
        future='Four-node405B only after user review; no multi-node jobs are submitted.')
    def run(name,method,cases,diagnostic=False):
        cfg=configuration(args.job,args.node,args.output/name,98304,1024)
        cfg.update(admission_gated_decode=True,startup_timeout=600,max_num_seqs=24,
                   warmup_requests=4,warmup_output_tokens=4)
        cfg['runtime']['env'].update(VLLM_TKNP_BALANCED_WINDOW='0' if args.variant=='fifo' else '24', VLLM_TKNP_ROOT_BIAS_TOKENS='65536' if args.variant=='root-aware' else '0')
        cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN',HF_HUB_OFFLINE='1',
            HF_HOME='/lustre/fsw/hw_nresearch_snoise/hf_cache',
            VLLM_TKNP_RECORD_GRAPH_IDENTITY='1',
            VLLM_TKNP_VALIDATE_HETERO_METADATA='1' if diagnostic else '0')
        t=make_trial(method,'graph',24,[24]*4,98304,1024)
        for role in t['servers']:
            if role.get('tknp',1)>1:
                role['graph_layouts']={role['graph_batch_size']:role.pop('owner_counts')}
        segments=[];workloads={}
        for case_index,(distribution,batch,tokens,order) in enumerate(cases):
            key=f'r{case_index}-{distribution}-b{batch}-o{tokens}-{order}'
            rows=make_mixed_workload(distribution,batch,tokens,order=order)
            path=args.output/(key+'.jsonl')
            serialized=''.join(json.dumps(r)+'\n' for r in rows)
            if path.exists():assert path.read_text()==serialized
            else:path.write_text(serialized)
            workloads[key]=path
            segments.append(dict(key=key,workload=str(path),output=str(args.output/(name+'-'+key)),
                                 batch=batch,tokens=tokens,distribution=distribution,order=order))
        config=dict(cfg,trials=[t],client_sweep=segments)
        validate_matrix(config,8);write_json(args.output/(name+'-plan.json'),config)
        print('ENGINE_START',name,flush=True)
        results=run_trial(t,config,Path(segments[0]['workload']),args.output/name)
        identities=None
        for seg,result in zip(segments,results):
            point=Path(seg['output'])
            row=audit_point(point,t,seg['batch'],seg['tokens'],profiled=diagnostic,context=98304)
            workload_rows=[json.loads(line) for line in workloads[seg['key']].read_text().splitlines()]
            stats=summarize_mixture(point,workload_rows)
            assert stats['scheduler_queue_observed_requests']==seg['batch']
            counters=[];current={}
            for f in point.glob('server*/execution.json'):
                records=json.loads(f.read_text());counters.extend(records)
                for record in records:
                    identity=record.get('graph_reuse_identity')
                    assert identity, 'Missing stable model/KV/graph identity'
                    current[(f.parent.name,record['rank'])]=identity
            assert len(current)==8
            if identities is not None:assert current==identities,'Model/KV/graph addresses changed across mixed workloads'
            identities=current
            if diagnostic:
                assert all(r['counts'].get('hetero_metadata_checks',0)>0 for r in counters)
                assert any(r['counts'].get('hetero_mixed_length_checks',0)>0 for r in counters)
            waiting=0
            for f in point.glob('server*/rank*.jsonl'):
                for line in f.read_text().splitlines():
                    event=json.loads(line)
                    if event['event']=='step':waiting=max(waiting,event.get('waiting_capacity',0))
            if seg['batch']==48:assert waiting>0,'Overflow workload did not exercise queueing'
            row.update(method=method,distribution=seg['distribution'],output_length=seg['tokens'],
                       order=seg['order'],diagnostic=diagnostic,peak_waiting_per_controller=waiting,
                       workload_sha256=hashlib.sha256(Path(seg['workload']).read_bytes()).hexdigest(),
                       heterogeneous=stats,stable_model_kv_graph_addresses=True)
            state['diagnostics' if diagnostic else 'trials'].append(row)
            write_json(args.output/'summary.json',state)
            print('POINT_PASS',name,seg['key'],'tps',round(row['output_tps'],2),'waiting',waiting,flush=True)
    try:
        cases=[]
        for repeat in range(3):
            points=[(d,b,128,'shuffled') for b in (24,48) for d in ('uniform64k','balanced32_64_96k','skewed32_96k')]
            cases.extend(points if repeat%2==0 else reversed(points))
        cases.append(('balanced32_64_96k',24,1024,'shuffled'))
        state['variant']=args.variant
        run(args.variant,'tknp-tp2',cases)
        state['passed']=True;state['finished_time']=time.time()
        write_json(args.output/'summary.json',state)
        print('HETEROGENEOUS_VALIDATION_COMPLETE',flush=True)
    except Exception as exc:
        state['failures'].append(repr(exc));write_json(args.output/'summary.json',state)
        raise


if __name__=='__main__':main()
