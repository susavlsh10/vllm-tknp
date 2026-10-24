#!/usr/bin/env python3
"""TKNP-v2 only: cross-node readiness then a matched 405B decode sweep.

Runs inside an existing four-node allocation; no checkpoint downloads. The
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs=4, required=True)
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--batches', type=int, nargs='+', default=[4, 8, 16, 32, 64, 96, 128])
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--readiness', type=Path, required=True)
    args = p.parse_args()
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
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-u', str(here / 'tknp_v2_readiness.py'),
              '--source', str(args.source), '--job', args.job, '--nodes', *args.nodes,
              '--container-name', 'tknpv2_' + args.job, '--tp', '8', '--tknp', '4',
              '--model', str(args.parent / 'model405b'), '--fp8', '--standard-cpu',
              '--context', str(args.context)]

    def run(stage, options):
        with (args.output / (stage + '.log')).open('x') as log:
            subprocess.run(common + ['--output', str(args.output / stage)] + options,
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads((args.output / stage / 'summary.json').read_text())
        assert state['passed']
        return state

    # Verify all 32 workers, weight placement and real FULL replay before the
    # long sweep. The probe retains the production maximum sequence length.
    probe = run('cross_node_probe', ['--tokens', '16', '--reserve-tokens', str(args.tokens),
                                    '--slots', '8', '--batches', '8'])
    caps = probe['trials'][0]['capacity']
    assert len(caps) == 4 and min(caps) >= 2, caps
    maximum, sizes = equal_layouts(args.batches, caps)
    write_json(args.output / 'capacity_plan.json', dict(physical_capacity=caps,
        equal_attention_slots=maximum, layout_sizes=sizes,
        unused_physical_slots=sum(caps)-maximum,
        tail_policy='Fixed layout within each natural wave; no active-tail downsizing'))
    state = run('sweep', ['--tokens', str(args.tokens), '--slots', str(maximum),
                         '--layout-sizes', *map(str, sizes),
                         '--batches', *map(str, args.batches)])
    rows = []
    for point in state['trials']:
        path = args.output / 'sweep' / ('b' + str(point['batch']))
        raw = json.loads((path / 'trial.json').read_text())
        trial = raw['trial']
        row = summarize(path, trial, point['batch'], args.tokens, args.context)
        row.update(method='tknp-v2-tp8', resources=point['resources'],
                   physical_capacity=point['capacity'], equal_attention=True,
                   microbatching=False, source_commit=snapshot['commit'],
                   workload_sha256=raw['workload_sha256'])
        rows.append(row)
        print('POINT_PASS', point['batch'], row['output_tps'], flush=True)
    baseline = json.loads(args.baseline.read_text())
    assert baseline['context'] == args.context and baseline['output_length'] == args.tokens
    old = [r for r in baseline['trials'] if r['batch'] in args.batches
           and r.get('pp_baseline') != 'tuned']
    for previous in old:
        evidence = Path(previous['path']) / 'trial.json'
        if evidence.exists():
            old_hash = json.loads(evidence.read_text())['workload_sha256']
            row = next(r for r in rows if r['batch'] == previous['batch'])
            if old_hash != row['workload_sha256']:
                raise RuntimeError('Baseline workload differs: ' + str(evidence))
    comparison = dict(passed=True, context=args.context, output_length=args.tokens,
        job=args.job, nodes=args.nodes, source=str(args.source),
        single_node_readiness=str(args.readiness),
        model_config_sha256=hashlib.sha256((args.parent/'model405b/config.json').read_bytes()).hexdigest(),
        trials=rows, preserved_baseline_trials=old, baseline_path=str(args.baseline),
        baseline_job=baseline['job'], baseline_nodes=baseline['nodes'],
        caveats=['Dummy mixed FP8/BF16 weights, physical BF16 random KV; not accuracy or prefill.',
                 'Baselines are preserved measurements from a different allocation/source version.',
                 'Admission-gated completion throughput includes queueing and natural waves.',
                 'Equal owner graph slots can leave a few physical slots unused.',
                 'Fixed graph shape retained inside each tail wave; no automatic tail downsizing.'])
    write_json(args.output / 'comparison.json', comparison)
    print('TKNP_V2_405B_SWEEP_PASS', flush=True)


if __name__ == '__main__':
    main()
