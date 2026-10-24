#!/usr/bin/env python3
"""Quick eight-way PP ablation; existing one-node allocation only."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--node', required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source / 'examples/offline_inference/TKNP'))
    from eos_efficiency import configuration, make_trial, workload, audit_point, write_json
    from e2e_matrix import run_trial
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(source=str(args.source), job=args.job, node=args.node,
                 model='Llama-3.1-8B', tp=2, pp=4, context=32768, batch=64,
                 output_length=128, repeats=3, trials=[], failures=[])
    cases = [(s,c,a,16 if s else None) for s,c,a in
             [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
    # A small scheduling-cap sensitivity after the complete factorial.
    cases += [(1,1,1,8), (1,1,1,32)]
    for scheduling, compact, array, cap in cases:
        name = f's{scheduling}-c{compact}-a{array}-cap{cap or 512}'
        print('CASE_START', name, flush=True)
        cfg = configuration(args.job, args.node, args.output / name, 32768, 128)
        cfg.update(admission_gated_decode=True, startup_timeout=360,
                   max_num_seqs=64, warmup_requests=4, warmup_output_tokens=8)
        cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN',
            VLLM_PP_COMPACT_CACHED_TOKENS=str(compact),
            VLLM_CACHE_PROMPT_TOKEN_IDS=str(array),
            HF_HUB_OFFLINE='1', HF_HOME='/lustre/fsw/hw_nresearch_snoise/hf_cache')
        trial = make_trial('tp2-pp4', 'graph', 64, [64], 32768, 128)
        role = trial['servers'][0]
        role['env'].update(VLLM_PP_COMPACT_CACHED_TOKENS=str(compact),
                           VLLM_CACHE_PROMPT_TOKEN_IDS=str(array))
        role.update(max_num_seqs=64, graph_batch_size=cap or 64)
        if cap:
            role['pp_decode_microbatch'] = cap
        else:
            role.pop('pp_decode_microbatch', None)
        w = workload(args.output, 32768, 64, 128)
        destination = args.output / name
        cfg['trials'] = [trial]
        cfg['client_sweep'] = [dict(key=f'r{r}', workload=str(w),
            output=str(args.output / f'{name}-r{r}')) for r in range(3)]
        write_json(args.output / f'{name}-plan.json', cfg)
        try:
            run_trial(trial, cfg, w, destination)
            rows = []
            for repeat in range(3):
                point = args.output / f'{name}-r{repeat}'
                row = audit_point(point, trial, 64, 128, context=32768)
                manifest = json.loads((point/'server0/manifest.json').read_text())
                assert manifest['engine_batch_queue_size'] == 4
                assert manifest['runtime_env']['VLLM_PP_COMPACT_CACHED_TOKENS'] == str(compact)
                assert manifest['runtime_env']['VLLM_CACHE_PROMPT_TOKEN_IDS'] == str(array)
                assert manifest['arguments'].get('pp_decode_microbatch') == cap
                row.update(case=name, scheduling=scheduling, compact=compact,
                           prompt_array=array, cap=cap, repeat=repeat)
                rows.append(row)
            state['trials'].extend(rows)
            print('CASE_PASS', name, 'median_tps', statistics.median(r['output_tps'] for r in rows), flush=True)
        except Exception as exc:
            state['failures'].append(dict(case=name,error=repr(exc)))
            write_json(args.output/'summary.json', state)
            raise
        write_json(args.output/'summary.json', state)
    state['finished_time'] = time.time()
    write_json(args.output/'summary.json', state)
    print('PP_ABLATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
