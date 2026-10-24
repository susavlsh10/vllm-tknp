#!/usr/bin/env python3
"""Aggregate the single-node v2 correctness, graph, memory and decode gates."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def audit(root, source):
    def read(name):
        return json.loads((root / name).read_text())

    checks = {}
    for key in ('bf16', 'fp8'):
        report = read(key + '_logit_equivalence.json')
        assert report['passed'] and report['all_tokens_equal'] and report['logit_vectors'] == 64
        checks[key + '_logits'] = report
    for reference, candidate in [('reference_tp2_logits', 'v2_four_owners_retry'),
                                 ('fp8_reference', 'fp8_four_owners')]:
        a, b = [read(stage + '/summary.json') for stage in (reference, candidate)]
        assert a['passed'] and b['passed']
        ar, br = [s['trials'][0]['resources'] for s in (a, b)]
        expected = sum(r['parameter_storage_bytes'] for r in ar)
        assert sum(r['parameter_storage_bytes'] for r in br) == expected
        assert len(br) == 8 and all(r['tknp_v2']['owned_dense_layers'] for r in br)
        assert all(len(r['kv_attention_layer_names']) == 32 for r in br)
        checks[candidate + '_storage'] = dict(total_parameter_bytes=expected,
            per_rank_parameter_bytes=[r['parameter_storage_bytes'] for r in br],
            layer_owners=br[0]['tknp_v2']['layer_owners'])
    graphs = read('v2_four_owners_retry/summary.json')
    assert [p['batch'] for p in graphs['trials']] == [4, 8, 16, 20]
    large = read('fp8_long_context/summary.json')
    assert large['passed'] and [p['batch'] for p in large['trials']] == [32, 64, 128, 160]
    assert all(p['performance_valid'] and len(p['resources']) == 8 for p in large['trials'])
    checks['large_decode'] = [dict(batch=p['batch'], output_tps=p['output_tps'],
        physical_capacity=p['capacity'], client_itl=p['client']['intertoken_s']) for p in large['trials']]
    profile = read('fp8_standard_cpu_profile/summary.json')
    assert profile['passed']
    manifest = read('fp8_standard_cpu_profile/b8/server0/manifest.json')
    assert all(manifest['runtime_env'].get(k) == '0' for k in
               ('VLLM_PP_COMPACT_CACHED_TOKENS', 'VLLM_CACHE_PROMPT_TOKEN_IDS'))
    checks['standard_cpu_profile'] = dict(passed=True, ranks=8, performance_valid=False)
    traces = list((root/'fp8_standard_cpu_profile/engine/server0/traces').glob('*.gz'))
    assert len(traces) == 8
    kernels = []
    for path in traces:
        with gzip.open(path, 'rt') as stream:
            events = json.load(stream)['traceEvents']
        count = sum(e.get('cat') == 'kernel' for e in events)
        assert count > 0
        kernels.append(dict(path=str(path), kernel_events=count))
    checks['standard_cpu_profile']['traces'] = kernels
    test_log = (root/'targeted_tests_retry.log').read_text()
    passed = re.search(r'(\d+) passed', test_log)
    assert passed and 'FAILED' not in test_log
    checks['targeted_tests'] = dict(passed=int(passed.group(1)),
        scope='Scoped pytest --noconftest; unrelated repository conftest requires missing tblib')
    legacy = read('fp8_legacy_b64/summary.json')
    assert legacy['passed'] and legacy['trials'][0]['performance_valid']
    before = legacy['trials'][0]['output_tps']
    after = next(p['output_tps'] for p in large['trials'] if p['batch'] == 64)
    checks['matched_8b_b64_control'] = dict(legacy_tps=before, v2_tps=after,
        v2_over_legacy=after/before, active_owner_counts=[16]*4,
        v1_max_graph_slots=64, v2_max_graph_slots=128,
        cpu_flags='Default CPU fast paths on for both; paper sweep explicitly turns both off')
    files = ['vllm/distributed/tknp_layer_ownership.py',
             'vllm/model_executor/layers/token_parallel_linear.py',
             'vllm/model_executor/models/llama.py', 'vllm/v1/worker/gpu_model_runner.py',
             'examples/offline_inference/TKNP/e2e_server.py']
    return dict(passed=True, checks=checks, source_sha256={
        name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
        scope='Single-node dummy-weight decode implementation validation; not trained-model accuracy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.source)
    with (args.root/'readiness_gate.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
