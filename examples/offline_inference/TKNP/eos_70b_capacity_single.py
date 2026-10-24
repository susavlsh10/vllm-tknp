#!/usr/bin/env python3
"""One-node BF16 70B96K capacity/crossover study, never allocates nodes."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback


def capacity_batches(capacities):
    """Even global batches keep the two DP replicas equally offered work."""
    dp = sum(capacities['dp-tp4'])
    tk = sum(capacities['tknp-tp4'])//2*2
    tp = capacities['tp'][0]
    candidates = [4, 8, dp//2, dp-2, dp, dp+2, tk, tk+2, tp, tp+2,
                  2*max(dp, tk, tp)]
    return sorted({max(4, math.ceil(v/2)*2) for v in candidates})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--node', required=True)
    p.add_argument('--methods', nargs='+', choices=['dp-tp4', 'tknp-tp4', 'tp'])
    p.add_argument('--capacity-source', type=Path)
    p.add_argument('--isolated-layouts', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.source/'examples/offline_inference/TKNP'))
    from eos_fp8_four_node_suite import verify_snapshot
    from eos_efficiency import configuration, audit_point, write_json, workload
    from e1_70b_suite import build_trial, measured_caps
    from e2e_matrix import run_trial, validate_matrix
    verify_snapshot(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    context, tokens = 98304, 1024
    cfg = configuration(args.job, args.node, args.output, context, tokens)
    cfg.update(model=str(args.source/'examples/offline_inference/TKNP/model_configs/llama31_70b'),
               tokenizer='meta-llama/Llama-3.1-8B-Instruct', admission_gated_decode=True,
               startup_timeout=1200, timeout=2400, warmup_requests=4, warmup_output_tokens=8,
               scope='Single-node70B BF16 dummy weights/BF16 random physicalKV,96K cached decode; no prefill/accuracy claim')
    cfg['runtime']['env'].update(VLLM_ATTENTION_BACKEND='FLASH_ATTN', HF_HUB_OFFLINE='1',
        VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='1', VLLM_TKNP_BALANCED_WINDOW='0',
        VLLM_TKNP_ROOT_BIAS_TOKENS='0', VLLM_TKNP_VALIDATE_HETERO_METADATA='0',
        VLLM_PP_COMPACT_CACHED_TOKENS='0', VLLM_CACHE_PROMPT_TOKEN_IDS='0')
    methods = args.methods or ['dp-tp4', 'tknp-tp4', 'tp']
    state = dict(job=args.job, node=args.node, source=str(args.source),
                 model='Llama3.1 70B BF16 dummy', kv='BF16 preinitialized random physical',
                 context=context, output_length=tokens, capacities={}, probes=[], trials=[], failures=[],
                 notes=['Eight H100 GPUs; DP=TP4xDP2, TKNP=TP4xTKNP2, TP=TP8.',
                        'All requests queued before decode gate; admission/waves included.',
                        'Capacity-derived graphs, actual per-request lifetime KV reservations.',
                        'No microbatching; one observation per workload, same allocation.',
                        'No prefill, online-arrival, or trained-model accuracy claims.'])
    if args.capacity_source:
        previous = json.loads(args.capacity_source.read_text())
        assert previous['context'] == context and previous['output_length'] == tokens
        state['capacities'] = previous['capacities']
        state['capacity_source'] = str(args.capacity_source)

    def trial(method, batch, caps):
        t = build_trial(method, context, tokens, batch, list(range(8)), caps, 'decode')
        for role in t['servers']:
            role['disable_combo_kernels'] = False
            role['env'].update(VLLM_TKNP_LIFETIME_ADMISSION='1')
        validate_matrix(dict(trials=[t]), 8)
        return t

    try:
        for method in methods:
            dest = args.output/method
            dest.mkdir()
            if method in state['capacities']:
                continue
            t = trial(method, 4, [2, 2] if method != 'tp' else [4])
            print('CAPACITY_PROBE_START', method, flush=True)
            config = dict(cfg, trials=[t])
            run_trial(t, config, workload(dest, context, 4, 16), dest/'probe')
            row = audit_point(dest/'probe', t, 4, 16, context=context)
            caps = measured_caps(dest/'probe', context, tokens, t)
            if min(caps) < 1:
                raise RuntimeError('A role cannot fit one96K+1024 request')
            state['capacities'][method] = caps
            state['probes'].append(dict(method=method, evidence=row, lifetime_capacity=caps))
            write_json(args.output/'summary.json', state)
            print('CAPACITY', method, caps, flush=True)
        batches = capacity_batches(state['capacities'])
        state['batches'] = batches
        for method in methods:
            dest = args.output/method
            caps = state['capacities'][method]
            t = trial(method, max(batches), caps)
            sizes = {}
            if method == 'tknp-tp4':
                role = t['servers'][0]
                layouts = {}
                for batch in batches:
                    r = trial(method, batch, caps)['servers'][0]
                    sizes[batch] = r['graph_batch_size']
                    layouts[sizes[batch]] = r['owner_counts']
                role.pop('owner_counts')
                role['graph_layouts'] = layouts
            segments = []
            for batch in batches:
                segment = dict(key=f'b{batch}', batch=batch,
                               workload=str(workload(dest, context, batch, tokens)),
                               output=str(dest/f'b{batch}'))
                if sizes:
                    segment['graph_size'] = sizes[batch]
                segments.append(segment)
            config = dict(cfg, trials=[t], client_sweep=segments)
            validate_matrix(config, 8)
            write_json(dest/'plan.json', config)
            print('SWEEP_START', method, batches, flush=True)
            if args.isolated_layouts and method == 'tknp-tp4':
                # This70B compiled path specializes ownership-dependent tensor
                # extents. Avoid reusing one compiled graph across layouts:
                # load one engine per exact layout, reuse it for overflow batches.
                results = []
                for size in sorted(set(sizes.values())):
                    group = [s for s in segments if s['graph_size'] == size]
                    gt = trial(method, size, caps)
                    role = gt['servers'][0]
                    role['graph_layouts'] = {size: role.pop('owner_counts')}
                    gc = dict(cfg, trials=[gt], client_sweep=group,
                              cache_root=str(dest/f'layout{size}-cache'))
                    write_json(dest/f'layout{size}-plan.json', gc)
                    results.extend(run_trial(gt, gc, Path(group[0]['workload']), dest/f'layout{size}'))
                segments = sorted(segments, key=lambda s: (s['graph_size'], s['batch']))
            else:
                results = run_trial(t, config, Path(segments[0]['workload']), dest/'persistent')
            assert len(results) == len(segments)
            for segment, result in zip(segments, results):
                point = Path(segment['output'])
                row = audit_point(point, result['trial'], segment['batch'], tokens, context=context)
                actual_caps = measured_caps(point, context, tokens, result['trial'])
                row.update(method=method, physical_lifetime_capacity=actual_caps,
                           context=context, output_length=tokens,
                           configured_graph_size=segment.get('graph_size'))
                # Max concurrency is limited by actual KV, not a hidden
                # worst-case request-count clamp or intentionally small graph.
                for i, role in enumerate(result['trial']['servers']):
                    m = json.loads((point/f'server{i}/manifest.json').read_text())
                    assert m['scheduler_max_num_running_reqs'] == role['max_num_seqs']
                    assert m['runtime_env']['VLLM_TKNP_LIFETIME_ADMISSION'] == '1'
                state['trials'].append(row)
                write_json(args.output/'summary.json', state)
                print('POINT_PASS', method, segment['batch'], row['output_tps'],
                      'physicalcaps', actual_caps, 'peak', row['peak_resident'], flush=True)
        state['passed'] = len(state['trials']) == len(methods)*len(batches)
        write_json(args.output/'summary.json', state)
        print('SINGLE_NODE_70B_CAPACITY_COMPLETE', state['passed'], flush=True)
    except Exception as exc:
        traceback.print_exc()
        state['failures'].append(repr(exc))
        write_json(args.output/'summary.json', state)
        raise


if __name__ == '__main__':
    main()
