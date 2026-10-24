#!/usr/bin/env python3
"""Short capacity-informed validation inside one EXISTING two-node allocation.

Never submits work or allocates nodes. Failed stages remain intact. This is
synthetic systems validation, not checkpoint accuracy or a throughput frontier.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--nodes', nargs=2)
    p.add_argument('--wait-existing-allocation', action='store_true')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--container-name', required=True)
    args = p.parse_args()
    if not args.nodes and not args.wait_existing_allocation:
        p.error('Supply --nodes or --wait-existing-allocation')
    if args.wait_existing_allocation:
        while True:
            value = subprocess.check_output(['squeue','-h','-j',args.job,'-o','%T|%N'],text=True).strip()
            if not value:
                raise RuntimeError('Existing allocation disappeared; no replacement will be submitted')
            status, node_spec = value.split('|',1)
            if status == 'RUNNING':
                args.nodes = subprocess.check_output(['scontrol','show','hostnames',node_spec],text=True).split()
                if len(args.nodes) != 2:
                    raise RuntimeError('Expected exactly two nodes')
                break
            if status != 'PENDING':
                raise RuntimeError('Unexpected allocation state: '+status)
            time.sleep(20)
        mounts = '/home/sshrestha:/home/sshrestha,/lustre/fsw/hw_nresearch_snoise:/lustre/fsw/hw_nresearch_snoise'
        # One extraction per node, before any DP replicas can race Pyxis.
        subprocess.run(['srun','--jobid='+args.job,'--overlap','-N2','-n2',
                        '--ntasks-per-node=1','--kill-on-bad-exit=1',
                        '--container-image=/lustre/fsw/hw_nresearch_snoise/sshrestha/vllm-tknp-v1.sqsh',
                        '--container-name='+args.container_name,'--container-mounts='+mounts,
                        'nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(job=args.job, nodes=args.nodes, context=98304,
                 recipe='ModelOpt FP8 weights/activations and FP8 E4M3 KV; dummy scales',
                 stages={}, scope='Two-node decode-only readiness; no accuracy claim')

    def save():
        target = args.output/'validation.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(target)

    methods = {
        'dp': ['--tp','8','--dp','2'],
        'tknp': ['--tp','8','--tknp','2'],
        'tp16': ['--tp','16'],
        'dcp2': ['--tp','16','--dcp','2'],
    }

    def stage(name, method, batches, graph, tokens, owners=None):
        output = args.output/name
        command = [sys.executable, str(args.source/'examples/offline_inference/TKNP/nemotron_ultra_readiness.py'),
                   '--source',str(args.source),'--output',str(output),
                   '--job',args.job,'--nodes',*args.nodes,'--modelopt',
                   '--container-name',args.container_name,'--tokenizer',args.tokenizer,
                   '--context','98304','--tokens',str(tokens),
                   '--gpu-memory-utilization','0.90','--graph-batch-size',str(graph),
                   '--batches',*map(str,batches),*methods[method]]
        if owners is not None:
            command += ['--owner-counts',*map(str,owners)]
        print('STAGE_START',name,flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state['stages'][name] = dict(returncode=result.returncode, command=command,
                                    summary=str(output/'summary.json'))
        save()
        if result.returncode:
            print('STAGE_FAILED',name,result.returncode,flush=True)
            return None
        data = json.loads((output/'summary.json').read_text())
        assert data['passed']
        print('STAGE_COMPLETE',name,flush=True)
        return data

    caps = {}
    for method in methods:
        data = stage('probe_'+method,method,[4],2 if method=='dp' else 4,32,
                     [2,2] if method=='tknp' else None)
        if data:
            caps[method] = data['trials'][0]['capacity96k_osl1024']
        state['probe_capacities'] = caps
        save()
    if 'dp' not in caps or 'tknp' not in caps:
        raise RuntimeError('DP/TKNP capacity probe failed; preserve attempts and diagnose interactively')
    # Native methods retain their full measured ceilings and actual-length
    # lifetime admission guard. TKNP's explicit fixed ownership needs one
    # slot/owner of headroom for the larger graph's memory footprint.
    usable = {method:[max(1,c-(1 if method=='tknp' else 0)) for c in values]
              for method,values in caps.items()}
    owners = usable['tknp']
    batches = sorted(set([sum(caps['dp'])+4, sum(owners)+4]))
    state.update(configured_slots=usable,batches=batches,output_tokens=1024,
                 admission_note='Actual physical lifetime reservations remain enabled. Native methods use full probe ceilings; TKNP reserves one request per owner for graph growth. DP graph ceiling is per replica.')
    save()
    for method in methods:
        if method not in usable:
            continue
        graph = sum(owners) if method=='tknp' else min(usable[method])
        stage('decode_'+method,method,batches,graph,1024,
              owners if method=='tknp' else None)
    state['complete'] = all(state['stages'].get('decode_'+m,{}).get('returncode')==0 for m in methods)
    save()
    print('VALIDATION_COMPLETE',state['complete'],flush=True)
    return 0 if state['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
