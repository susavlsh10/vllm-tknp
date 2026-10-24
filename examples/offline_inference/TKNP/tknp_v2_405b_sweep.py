#!/usr/bin/env python3
"""TKNP-v2 only: cross-node readiness then a matched 405B decode sweep.

Runs inside an existing four/eight-node allocation; no checkpoint downloads. The
single-node readiness gate must pass before this script is submitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def equal_layouts(batches, capacities):
    if not batches or min(batches) < 1 or not capacities or min(capacities) < 1:
        raise ValueError('Positive batch sizes and physical owner capacities required')
    degree = len(capacities)
    maximum = min(min(capacities)*degree, ((max(batches)+degree-1)//degree)*degree)
    return maximum, sorted({min(maximum, ((b+degree-1)//degree)*degree) for b in batches})


def baseline_rows(baseline, context, tokens, batches):
    """Accept original summaries or the portable eight-node merged PP index."""
    if 'results' in baseline:
        actual_context = baseline['input_tokens']
        actual_tokens = baseline['output_tokens_per_request']
        rows = baseline['results']
    else:
        actual_context = baseline['context']
        actual_tokens = baseline['output_length']
        rows = baseline['trials']
    if (actual_context, actual_tokens) != (context, tokens):
        raise ValueError('Baseline context/output length mismatch')
    selected = [r for r in rows if r['batch'] in batches and r.get('pp_baseline') != 'tuned']
    if not selected or any(not r['performance_valid'] or not r['full_graph_all_ranks']
                           or r['measured_preemptions'] for r in selected):
        raise ValueError('Missing or invalid preserved baseline')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs='+', required=True)
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--batches', type=int, nargs='+', default=[4, 8, 16, 32, 64, 96, 128])
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--readiness', type=Path, required=True)
    p.add_argument('--profile-steps', type=int, default=0,
                   help='Separate profiled run, never a throughput measurement')
    p.add_argument('--profile-after-step', type=int, default=32)
    args = p.parse_args()
    degree = len(args.nodes)
    if degree not in (4, 8) or len(set(args.nodes)) != degree:
        p.error('Require four or eight distinct eight-GPU nodes')
    if (min(args.context, args.tokens, *args.batches) < 1
            or len(set(args.batches)) != len(args.batches)
            or any(b < degree or b % degree for b in args.batches)):
        p.error('Batches must be distinct positive multiples of the TKNP degree')
    if args.profile_steps and (degree != 4 or len(args.batches) != 1
                              or args.profile_after_step + args.profile_steps >= args.tokens):
        p.error('Profile one four-node workload with enough output steps')
    here = args.source / 'examples/offline_inference/TKNP'
    sys.path.insert(0, str(here))
    from eos_fp8_four_node_suite import verify_snapshot
    from eos_efficiency import write_json
    from eos_405b_efficiency import summarize

    snapshot = verify_snapshot(args.source)
    readiness = json.loads(args.readiness.read_text())
    assert readiness['passed']
    for name, digest in readiness['source_sha256'].items():
        if snapshot['source_sha256'].get(name) != digest:
            raise RuntimeError('Single-node validated source mismatch: ' + name)
    baseline = json.loads(args.baseline.read_text())
    old = baseline_rows(baseline, args.context, args.tokens, args.batches)
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-u', str(here / 'tknp_v2_readiness.py'),
              '--source', str(args.source), '--job', args.job, '--nodes', *args.nodes,
              '--container-name', 'tknpv2_' + args.job, '--tp', '8', '--tknp', str(degree),
              '--model', str(args.parent / 'model405b'), '--fp8', '--standard-cpu',
              '--context', str(args.context)]

    def run(stage, options):
        with (args.output / (stage + '.log')).open('x') as log:
            subprocess.run(common + ['--output', str(args.output / stage)] + options,
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads((args.output / stage / 'summary.json').read_text())
        assert state['passed']
        return state

    # Verify every worker, weight placement and real FULL replay before the
    # long sweep. The probe retains the production maximum sequence length.
    probe = run('cross_node_probe', ['--tokens', '16', '--reserve-tokens', str(args.tokens),
                                    '--slots', str(2*degree), '--batches', str(2*degree)])
    caps = probe['trials'][0]['capacity']
    assert len(caps) == degree and min(caps) >= 2, caps
    maximum, sizes = equal_layouts(args.batches, caps)
    write_json(args.output / 'capacity_plan.json', dict(physical_capacity=caps,
        equal_attention_slots=maximum, layout_sizes=sizes,
        unused_physical_slots=sum(caps)-maximum,
        tail_policy='Fixed layout within each natural wave; no active-tail downsizing'))
    stage = 'profile' if args.profile_steps else 'sweep'
    options = ['--tokens', str(args.tokens), '--slots', str(maximum),
                         '--layout-sizes', *map(str, sizes),
                         '--batches', *map(str, args.batches)]
    clocks = []
    if args.profile_steps:
        from tknp_trace_merge import calibrate_nodes
        clocks.append(calibrate_nodes(args.job, args.nodes, here / 'tknp_trace_merge.py'))
        write_json(args.output / 'clock_before.json', clocks[-1])
        options += ['--profile-steps', str(args.profile_steps),
                    '--profile-after-step', str(args.profile_after_step)]
    state = run(stage, options)
    if args.profile_steps:
        clocks.append(calibrate_nodes(args.job, args.nodes, here / 'tknp_trace_merge.py'))
        write_json(args.output / 'clock_after.json', clocks[-1])
    rows = []
    for point in state['trials']:
        path = args.output / stage / ('b' + str(point['batch']))
        raw = json.loads((path / 'trial.json').read_text())
        trial = raw['trial']
        row = summarize(path, trial, point['batch'], args.tokens, args.context,
                        profiled=bool(args.profile_steps))
        row.update(method='tknp-v2-tp8', resources=point['resources'],
                   physical_capacity=point['capacity'], equal_attention=True,
                   microbatching=False, source_commit=snapshot['commit'],
                   workload_sha256=raw['workload_sha256'])
        if bool(point['performance_valid']) == bool(args.profile_steps):
            raise RuntimeError('Profiling/throughput validity mismatch')
        rows.append(row)
        print('POINT_PASS', point['batch'], row['output_tps'], flush=True)
    for previous in old:
        evidence = Path(previous['path']) / 'trial.json'
        if evidence.exists():
            old_hash = json.loads(evidence.read_text())['workload_sha256']
            row = next(r for r in rows if r['batch'] == previous['batch'])
            if old_hash != row['workload_sha256']:
                raise RuntimeError('Baseline workload differs: ' + str(evidence))
    comparison = dict(passed=True, performance_valid=not bool(args.profile_steps),
        context=args.context, output_length=args.tokens, gpu_budget=8*degree,
        job=args.job, nodes=args.nodes, source=str(args.source),
        single_node_readiness=str(args.readiness),
        model_config_sha256=hashlib.sha256((args.parent/'model405b/config.json').read_bytes()).hexdigest(),
        trials=rows, preserved_baseline_trials=old, baseline_path=str(args.baseline),
        baseline_job=baseline.get('job'), baseline_nodes=baseline['nodes'],
        baseline_provenance=baseline.get('provenance'),
        caveats=['Dummy mixed FP8/BF16 weights, physical BF16 random KV; not accuracy or prefill.',
                 'Baselines are preserved measurements from a different allocation/source version.',
                 'Admission-gated completion throughput includes queueing and natural waves.',
                 'Equal owner graph slots can leave a few physical slots unused.',
                 'Fixed graph shape retained inside each tail wave; no automatic tail downsizing.'])
    write_json(args.output / 'comparison.json', comparison)
    if args.profile_steps:
        from tknp_trace_merge import merge_traces
        resources = state['trials'][0]['resources']
        hosts = {r['rank']: r['hostname'] for r in resources}
        merge_traces(args.output / stage / 'engine/server0/traces',
                     args.output / 'all_ranks.chrome.json', 8*degree, hosts, clocks)
        print('ALL_RANKS_CHROME_TRACE_PASS', flush=True)
    print('TKNP_V2_405B_SWEEP_PASS', flush=True)


if __name__ == '__main__':
    main()
