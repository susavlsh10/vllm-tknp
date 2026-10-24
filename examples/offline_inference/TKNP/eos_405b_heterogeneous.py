#!/usr/bin/env python3
"""Six-method four-node cached-decode heterogeneous-context comparison."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs=4, required=True)
    p.add_argument('--methods', nargs='+', choices=['tknp-tp8', 'tknp-balanced', 'dp-tp8', 'dcp', 'tp8-pp4', 'tp'])
    p.add_argument('--actual-length-admission', action='store_true')
    p.add_argument('--reference-summary', type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.source/'examples/offline_inference/TKNP'))
    from eos_fp8_four_node_suite import verify_snapshot
    from fp8_decode_sweep import configuration, trial_for
    from eos_405b_efficiency import summarize
    from eos_efficiency import write_json, workload
    from e1_70b_suite import measured_caps
    from e2e_matrix import run_trial, validate_matrix
    from heterogeneous_decode import make_mixed_workload, summarize_mixture
    verify_snapshot(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    batches = [24, 48, 96]
    distributions = ['uniform64k', 'balanced32_64_96k', 'uniform48k', 'skewed32_96k']
    methods = args.methods or ['tknp-tp8', 'tknp-balanced', 'dp-tp8', 'dcp', 'tp8-pp4', 'tp']
    if args.actual_length_admission and not set(methods) <= {'dp-tp8', 'tp'}:
        raise ValueError('Actual-length retry is scoped to DP and TP only')
    cfg = configuration(args.job, args.nodes, str(args.parent/'model405b'), 98304, 1024)
    cfg.update(admission_gated_decode=True, startup_timeout=1200, timeout=3600)
    cfg['runtime']['env'].update(
        VLLM_ATTENTION_BACKEND='FLASH_ATTN', HF_HUB_OFFLINE='1',
        VLLM_PP_COMPACT_CACHED_TOKENS='0', VLLM_CACHE_PROMPT_TOKEN_IDS='0',
        VLLM_TKNP_BALANCED_WINDOW='0', VLLM_TKNP_ROOT_BIAS_TOKENS='0',
        VLLM_TKNP_VALIDATE_HETERO_METADATA='0',
        VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION='1' if args.actual_length_admission else '0',
        VLLM_RESEARCH_RANK_TRITON_CACHE='1', VLLM_RESEARCH_RANK_INDUCTOR_CACHE='1')
    for key, value in cfg['runtime']['env'].items():
        if key.startswith(('VLLM_', 'NCCL_')):
            os.environ[key] = str(value)
    state = dict(job=args.job, nodes=args.nodes, source=str(args.source), batches=batches,
                 distributions=distributions, output_length=1024, trials=[], failures=[],
                 caveats=['Dummy mixed FP8/BF16 weights, physical BF16 random KV; no accuracy claim.',
                          'Gated decode bursts, not prefill or online arrivals.',
                          ('32K-safe graph ceiling; actual prompt+output KV reservations govern admission.'
                           if args.actual_length_admission else
                           '96K lifetime probes bound graph slots conservatively; not maximum mixed KV capacity.'),
                          'One measurement per point; sequential method order.'])
    shared_tknp_caps = None
    hashes = {}
    if args.reference_summary:
        reference = json.loads(args.reference_summary.read_text())
        for row in reference['trials']:
            hashes[f"{row['distribution']}-b{row['batch']}"] = row['workload_sha256']
        state['reference_summary'] = str(args.reference_summary)
        state['cross_allocation_comparison'] = True

    def trial(method, batch, caps):
        base = 'tknp-tp8' if method == 'tknp-balanced' else method
        t = trial_for(base, 98304, 1024, batch, caps, 32,
                      'tuned' if base == 'tp8-pp4' else 'standard')
        for role in t['servers']:
            role.setdefault('env', {}).update(VLLM_TKNP_BALANCED_WINDOW='24' if method == 'tknp-balanced' else '0',
                                             VLLM_TKNP_ROOT_BIAS_TOKENS='0')
            if base == 'tp8-pp4':
                role['pp_baseline'] = 'scheduling-only'
                role['env'].update(VLLM_PP_COMPACT_CACHED_TOKENS='0', VLLM_CACHE_PROMPT_TOKEN_IDS='0')
        return t

    for method in methods:
        dest = args.output/method
        dest.mkdir()
        print('METHOD_START', method, flush=True)
        try:
            probe_batch = 8 if method.startswith('tknp') else 4
            initial = [2]*4 if method.startswith(('tknp', 'dp')) else [probe_batch]
            pt = trial(method, probe_batch, initial)
            pc = dict(cfg, trials=[pt])
            validate_matrix(pc, 32)
            run_trial(pt, pc, workload(dest, 98304, probe_batch, 16), dest/'probe')
            summarize(dest/'probe', pt, probe_batch, 16, 98304)
            caps = measured_caps(dest/'probe', 98304, 1024, pt)
            if args.actual_length_admission:
                caps = measured_caps(dest/'probe', 32768, 1024, pt)
            if method == 'tknp-tp8':
                shared_tknp_caps = caps
            elif method == 'tknp-balanced':
                if shared_tknp_caps is None:
                    raise RuntimeError('Missing matched native TKNP capacity')
                if any(a < b for a, b in zip(caps, shared_tknp_caps)):
                    raise RuntimeError('Balanced engine capacity below matched native layout')
                caps = shared_tknp_caps
            t = trial(method, max(batches), caps)
            sizes = {}
            if method.startswith('tknp'):
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
                for distribution in distributions:
                    key = f'{distribution}-b{batch}'
                    path = dest/(key+'.jsonl')
                    rows = make_mixed_workload(distribution, batch, 1024)
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    segment = dict(key=key, batch=batch, distribution=distribution,
                                   workload=str(path), output=str(dest/key))
                    if sizes:
                        segment['graph_size'] = sizes[batch]
                    if method == 'tp8-pp4':
                        segment['pp_decode_microbatch'] = math.ceil(min(batch, caps[0])/4)
                    segments.append(segment)
            config = dict(cfg, trials=[t], client_sweep=segments)
            validate_matrix(config, 32)
            write_json(dest/'plan.json', config)
            results = run_trial(t, config, Path(segments[0]['workload']), dest/'persistent')
            if len(results) != len(segments):
                raise RuntimeError('Incomplete persistent sweep')
            for segment, result in zip(segments, results):
                point = Path(segment['output'])
                row = summarize(point, result['trial'], segment['batch'], 1024, 98304)
                rows = [json.loads(line) for line in Path(segment['workload']).read_text().splitlines()]
                stats = summarize_mixture(point, rows)
                assert stats['scheduler_queue_observed_requests'] == segment['batch']
                digest = row['client']['workload_sha256']
                key = segment['key']
                if key in hashes:
                    assert hashes[key] == digest, 'Unmatched workloads'
                hashes[key] = digest
                if method == 'tp8-pp4':
                    manifest = json.loads((point/'server0/manifest.json').read_text())
                    assert manifest['engine_batch_queue_size'] == 4
                    assert all(manifest['runtime_env'].get(k) == '0' for k in
                               ['VLLM_PP_COMPACT_CACHED_TOKENS', 'VLLM_CACHE_PROMPT_TOKEN_IDS'])
                    # Persistent manifests describe startup; segment journals
                    # are the evidence for the dynamically changed cap.
                    assert manifest['arguments']['pp_decode_microbatch'] == t['servers'][0]['pp_decode_microbatch']
                    scheduled = []
                    for journal in point.glob('server*/rank*.jsonl'):
                        for line in journal.read_text().splitlines():
                            event = json.loads(line)
                            if event['event'] == 'step' and event.get('scheduled_tokens', 0):
                                scheduled.append(event['scheduled_tokens'])
                    assert scheduled and max(scheduled) <= segment['pp_decode_microbatch']
                row.update(method=method, distribution=segment['distribution'],
                           actual_length_admission=args.actual_length_admission,
                           heterogeneous=stats, workload_sha256=digest,
                           configured_graph_size=segment.get('graph_size'),
                           pp_decode_microbatch=segment.get('pp_decode_microbatch'),
                           pp_baseline='scheduling-only' if method == 'tp8-pp4' else None)
                state['trials'].append(row)
                if args.actual_length_admission:
                    for index, role in enumerate(result['trial']['servers']):
                        manifest = json.loads((point/f'server{index}/manifest.json').read_text())
                        assert manifest['scheduler_max_num_running_reqs'] == role['max_num_seqs']
                        assert manifest['runtime_env']['VLLM_TKNP_LIFETIME_ADMISSION'] == '1'
                    if segment['distribution'] == 'uniform48k' and method == 'dp-tp8':
                        assert max(row['peak_resident'].values()) > 2, 'Old DP clamp is still active'
                write_json(args.output/'summary.json', state)
                print('POINT_PASS', method, key, row['output_tps'], flush=True)
        except Exception as exc:
            traceback.print_exc()
            state['failures'].append(dict(method=method, error=repr(exc)))
            write_json(args.output/'summary.json', state)
    state['passed'] = not state['failures'] and len(state['trials']) == len(methods)*len(batches)*len(distributions)
    write_json(args.output/'summary.json', state)
    print('HETEROGENEOUS_405B_COMPLETE', state['passed'], flush=True)
    if not state['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
