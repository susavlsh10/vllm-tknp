#!/usr/bin/env python3
"""405B TP8/PP4 scheduling-only sweep in an existing four-node job."""
import argparse
import json
import math
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs=4, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[32,64,96,128])
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.source/'examples/offline_inference/TKNP'))
    from eos_fp8_four_node_suite import verify_snapshot
    from eos_405b_efficiency import make_trial, summarize
    from fp8_decode_sweep import configuration
    from eos_efficiency import write_json, workload
    from e1_70b_suite import measured_caps
    from e2e_matrix import run_trial, validate_matrix
    verify_snapshot(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args.job, args.nodes, str(args.parent/'model405b'), 32768, 128)
    cfg.update(admission_gated_decode=True, startup_timeout=1200, timeout=1800)
    cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN',
        VLLM_PP_COMPACT_CACHED_TOKENS='0', VLLM_CACHE_PROMPT_TOKEN_IDS='0',
        VLLM_RESEARCH_RANK_TRITON_CACHE='1', VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1',
        HF_HUB_OFFLINE='1')
    for k,v in cfg['runtime']['env'].items():
        if k.startswith(('VLLM_','NCCL_')):
            os.environ[k] = str(v)
    def trial(batch, caps):
        t = make_trial('tp8-pp4', batch, caps, 32768, 128, pp_baseline='tuned')
        for role in t['servers']:
            role['pp_baseline'] = 'scheduling-only'
            role['env'].update(VLLM_PP_COMPACT_CACHED_TOKENS='0', VLLM_CACHE_PROMPT_TOKEN_IDS='0')
        return t
    probe = trial(4, [4])
    probe_path = args.output/'probe'
    print('PROBE_START', flush=True)
    run_trial(probe, dict(cfg,trials=[probe]), workload(args.output,32768,4,16), probe_path)
    summarize(probe_path, probe, 4, 16, 32768)
    caps = measured_caps(probe_path,32768,128,probe)
    t = trial(max(args.batches),caps)
    role = t['servers'][0]
    segments = []
    for repeat in range(args.repeats):
        for batch in (args.batches if repeat%2==0 else reversed(args.batches)):
            key = f'b{batch}-r{repeat}'
            segments.append(dict(key=key,batch=batch,repeat=repeat,
                workload=str(workload(args.output,32768,batch,128)),
                output=str(args.output/key),
                pp_decode_microbatch=math.ceil(min(batch,caps[0])/4)))
    config = dict(cfg,trials=[t],client_sweep=segments)
    validate_matrix(config,32)
    write_json(args.output/'plan.json',config)
    print('SWEEP_START capacities=',caps,flush=True)
    results = run_trial(t,config,Path(segments[0]['workload']),args.output/'persistent')
    state = dict(method='tp8-pp4',pp_baseline='scheduling-only',capacities=caps,
        nodes=args.nodes,job=args.job,source=str(args.source),trials=[],failures=[])
    for segment,result in zip(segments,results):
        point = Path(segment['output'])
        row = summarize(point,result['trial'],segment['batch'],128,32768)
        manifest = json.loads((point/'server0/manifest.json').read_text())
        assert manifest['engine_batch_queue_size']==4
        assert manifest['arguments']['pp_decode_microbatch']==role['pp_decode_microbatch']
        assert all(manifest['runtime_env'][k]=='0' for k in
            ('VLLM_PP_COMPACT_CACHED_TOKENS','VLLM_CACHE_PROMPT_TOKEN_IDS'))
        row.update(method='tp8-pp4',pp_baseline='scheduling-only',repeat=segment['repeat'],
            pp_decode_microbatch=segment['pp_decode_microbatch'])
        state['trials'].append(row)
        write_json(args.output/'summary.json',state)
        print('POINT_PASS',segment['key'],row['output_tps'],flush=True)
    print('PP_SCHEDULING_ONLY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
