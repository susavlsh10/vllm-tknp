#!/usr/bin/env python3
"""Submit three independent TKNP-v2 sweeps and one all-rank profiling job.

Only this explicit launcher allocates nodes. Each batch script releases its
own allocation on exit. No baseline reruns, downloads, or job dependencies.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def specifications(root):
    common = [4, 8, 16, 32, 64, 96, 128]
    return [
        dict(key='4node_64k', nodes=4, context=65536, batches=common,
             baseline=str(root/'decode405b_4node_c65536_o1024_20261020_job6299012/summary.json')),
        dict(key='4node_96k', nodes=4, context=98304, batches=common,
             baseline=str(root/'decode405b_4node_c98304_o1024_20261020_job6299013/summary.json')),
        dict(key='8node_32k', nodes=8, context=32768, batches=[8, 16, 32, 64, 96, 128],
             baseline=str(root/'405b_8node_decode_batch_sweep_combined.json')),
        dict(key='profile_4node_64k_b96', nodes=4, context=65536, batches=[96],
             profile_steps=16, profile_after_step=32,
             baseline=str(root/'decode405b_4node_c65536_o1024_20261020_job6299012/summary.json'))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--readiness', type=Path, required=True)
    p.add_argument('--results-root', type=Path,
        default=Path('/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_results'))
    p.add_argument('--submit', action='store_true', help='Without this flag only print the plan')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    specs = specifications(args.results_root)
    if not args.submit:
        print(json.dumps(specs, indent=2))
        return
    from eos_fp8_four_node_suite import freeze_source, verify_snapshot, write_json
    from tknp_v2_405b_sweep import baseline_rows
    gate = json.loads(args.readiness.read_text())
    if not gate['passed']:
        raise RuntimeError('Single-node readiness gate failed')
    for spec in specs:
        baseline_rows(json.loads(Path(spec['baseline']).read_text()), spec['context'], 1024,
                      spec['batches'])
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Commit/preserve source changes before freezing the study')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/'source_v2_study'
    snapshot = freeze_source(repo, source)
    verify_snapshot(source)
    for name, digest in gate['source_sha256'].items():
        if snapshot['source_sha256'].get(name) != digest:
            raise RuntimeError('Validated GPU source changed: ' + name)
    submission = dict(created_at=time.time(), source=str(source), source_commit=snapshot['commit'],
        readiness=str(args.readiness), account='hw_nresearch_snoise',
        total_requested_nodes=20, simultaneous_jobs_authorized=True,
        dependencies=None, output_length=1024, methods=['TP8 x TKNP-v2'],
        precision='FBGEMM FP8/BF16 recipe weights; BF16 random physical KV',
        cpu_fast_paths=False, microbatching=False, jobs=[])
    index = args.output/'submission.json'
    write_json(index, submission)
    script = source/'examples/offline_inference/TKNP/jobs/eos_tknp_v2_405b.sbatch'
    for spec in specs:
        output = args.output/spec['key']
        command = ['sbatch', '--parsable', '--nodes='+str(spec['nodes']), '--dependency=',
                   '--output='+str(args.output/(spec['key']+'-%j.log')),
                   '--job-name=hw_nresearch_snoise-snoise:v2-'+spec['key'], str(script),
                   str(source), str(output), str(args.readiness),
                   '--context', str(spec['context']), '--tokens', '1024',
                   '--batches', *map(str, spec['batches']), '--baseline', spec['baseline']]
        if spec.get('profile_steps'):
            command += ['--profile-steps', str(spec['profile_steps']),
                        '--profile-after-step', str(spec['profile_after_step'])]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unrecognized sbatch response; inspect before retrying')
        submission['jobs'].append(dict(spec, job=job, command=command, output=str(output),
                                       submitted_at=time.time()))
        write_json(index, submission)  # Preserve partial submission if a later job is rejected.
        print('SUBMITTED', spec['key'], job, flush=True)


if __name__ == '__main__':
    main()
