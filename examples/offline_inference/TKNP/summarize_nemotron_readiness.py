#!/usr/bin/env python3
"""Create a portable index of the one-node Nemotron readiness evidence."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--modelopt',action='store_true')
    args = p.parse_args()
    result = dict(
        model='nvidia/Llama-3_1-Nemotron-Ultra-253B-v1',
        scope='One EOS node, eight H100 GPUs; synthetic decode only',
        parameters=253401268224, active_attention_layers=64,
        block_positions=162, active_ffn_layers=81, kv_heads=8, head_dim=128,
        kv_dtype='BF16', global_kv_bytes_per_token=262144,
        kv_bytes_per_96k_plus_1024_request=26038239232,
        fp8_recipe='Synthetic FBGEMM FP8 linear weights; BF16 embeddings/head and KV; not an official NVIDIA quantized checkpoint',
        caveats=[
            'Dummy weights and synthetic KV do not validate trained-model accuracy or real prefill.',
            'Capacity depends on graph shapes, memory utilization and collective/workspace headroom.',
            'Calculated 32K/64K capacities are not measured resident peaks.',
            'OSL128 bring-up timings are not comparable to OSL1024 timings.',
            'No multi-node, PP, DCP or microbatch validation is implied.',
        ], stages=[], incomplete_attempts=[])
    if args.modelopt:
        result.update(model='nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8',
            kv_dtype='FP8 E4M3',global_kv_bytes_per_token=131072,
            kv_bytes_per_96k_plus_1024_request=13019119616,
            fp8_recipe='Official ModelOpt FP8 weights/activations, FP8 KV, lm_head excluded; dummy positive scales, not trained calibration')
    failures = {
        'fp8_tp4_tknp2_probe': 'Stalled before first decode; stopped interactively',
        'fp8_tp4_tknp2_cpucontrol': 'First sampled-token NCCL broadcast failed with CUDA out of memory at memory utilization 0.95',
        'fp8_tp4_tknp2_u90': 'Asymmetric singleton layout (1,3) did not complete graph construction; stopped interactively',
        'fp8_tp4_tknp2_96k_u94': 'Corrected: slow compilation finished; all eight ranks replayed FULL graph and batch4 drained. Run stopped before remaining points, so no complete audited sweep',
        'tiny_modelopt_tp2': 'Startup failed because named container could not see shared cached tokenizer; retry uses explicit visible cached files',
    }
    for stage in sorted(args.root.iterdir()):
        if not stage.is_dir() or stage.name.startswith('source'):
            continue
        summary = stage/'summary.json'
        if not summary.exists():
            if (stage/'plan.json').exists():
                result['incomplete_attempts'].append(dict(stage=stage.name,
                    performance_valid=False, reason=failures.get(stage.name,
                        'No completed audited summary; inspect stage log')))
            continue
        data = json.loads(summary.read_text())
        item = dict(stage=stage.name, passed=data['passed'], arguments=data['arguments'], points=[])
        for row in data['trials']:
            point=stage/row.get('key',f"b{row['batch']}")
            raw=json.loads((point/'trial.json').read_text())
            identities_stable=True
            for server, records in raw['worker_execution'].items():
                manifest=json.loads((point/server/'manifest.json').read_text())
                initial={r['rank']:r.get('graph_reuse_identity') for r in manifest['worker_resources']}
                for record in records:
                    current=record.get('graph_reuse_identity')
                    if current is not None and current!=initial[record['rank']]:
                        identities_stable=False
            measured={json.loads(line)['server_request_id'] for line in
                      (point/'client/requests.jsonl').read_text().splitlines()}
            peaks={}
            for journal in point.glob('server*/rank0.jsonl'):
                events=[json.loads(line) for line in journal.read_text().splitlines()]
                peaks[journal.parent.name]=max((e['running'] for e in events
                    if e['event']=='step' and measured.intersection(e.get('scheduled_request_ids',[]))),default=0)
            groups = {}
            for r in row['resources']:
                tp = int(data['arguments']['tp'])
                key = (r.get('server', 'server0'), int(r['rank'])//tp)
                page_tokens = r['block_tokens'] * int(data['arguments'].get('dcp',1))
                cap = {str(k):(r['kv_blocks']-1)//((k*1024+1024+page_tokens-1)//page_tokens) for k in (32,64,96)}
                entry = dict(rank=r['rank'], model_memory_bytes=r['model_memory_bytes'],
                    parameter_storage_bytes=r['parameter_storage_bytes'], kv_bytes=r['kv_allocated_bytes'],
                    kv_blocks=r['kv_blocks'], block_tokens=r['block_tokens'],
                    active_kv_layers=len(r['kv_attention_layer_names']),
                    physical_request_capacity_osl1024=cap)
                groups.setdefault(str(key), []).append(entry)
            client=row['client']
            item['points'].append(dict(batch=row['batch'],
                performance_valid=row['performance_valid'],
                peak_resident_per_server=peaks,
                all_ranks_full_graph_verified=row['all_ranks_verified'],
                graph_model_kv_identity_stable=identities_stable,
                measured_preemptions=sum(t['preemptions_measured'] for t in raw['telemetry'].values()),
                all_kv_drained=all(t['shutdown']['all_kv_free'] for t in raw['telemetry'].values()),
                key=row.get('key',f"b{row['batch']}"),
                output_tps=row.get('output_tps'), client=client, rank_groups=groups,
                trial=str((point/'trial.json').relative_to(args.root))))
        # Keep the original summary unmodified; arguments contain provenance.
        item['summary']=str(summary.relative_to(args.root))
        result['stages'].append(item)
    if args.modelopt:
        controls=[args.root/x/'b4/client/requests.jsonl' for x in
            ('tiny_modelopt_tp2_retry','tiny_modelopt_tp2_tknp2')]
        if all(x.exists() for x in controls):
            tokens=[{r['request_id']:r['token_ids'] for r in map(json.loads,x.read_text().splitlines())}
                    for x in controls]
            result['identical_kv_correctness_control']=dict(requests=len(tokens[0]),
                exact_token_matches=sum(tokens[0][k]==tokens[1].get(k) for k in tokens[0]),
                output_tokens_per_request=32,scope='Small NAS FP8 control, not trained-model accuracy')
        points=[point for stage in result['stages'] for point in stage['points']]
        result['validation']=dict(completed_stages=len(result['stages']),validated_points=len(points),
            all_stages_passed=all(stage['passed'] for stage in result['stages']),
            all_points_full_graph=all(point['all_ranks_full_graph_verified'] for point in points),
            all_graph_model_kv_identities_stable=all(point['graph_model_kv_identity_stable'] for point in points),
            measured_preemptions=sum(point['measured_preemptions'] for point in points),
            all_kv_drained=all(point['all_kv_drained'] for point in points))
    target=args.root/'comparison.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(target)


if __name__=='__main__':
    main()
