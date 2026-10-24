#!/usr/bin/env python3
"""Compare request-aligned synthetic decode logits; does not allocate GPUs."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(reference, candidate, batch):
    import torch

    def requests(stage):
        return {r['request_id']: r for r in map(json.loads,
            (stage / f'b{batch}/client/requests.jsonl').read_text().splitlines())}

    a, b = requests(reference), requests(candidate)
    assert set(a) == set(b) and len(a) == batch
    worst_error = worst_relative = 0.0
    minimum_cosine = 1.0
    compared = 0
    for key in a:
        assert a[key]['token_ids'] == b[key]['token_ids'], key
        for step in range(len(a[key]['token_ids'])):
            tensors = []
            for stage, request in ((reference, a[key]), (candidate, b[key])):
                digest = hashlib.sha256(request['server_request_id'].encode()).hexdigest()
                path = stage / 'engine/server0/full-logits-rank0' / f'{digest}-{step}.pt'
                record = torch.load(path, map_location='cpu', weights_only=True)
                assert record['step'] == step and record['request_id'] == request['server_request_id']
                tensor = record['logits'].float()
                assert bool(torch.isfinite(tensor).all())
                tensors.append(tensor)
            x, y = tensors
            relative = float((x-y).norm() / x.norm().clamp_min(1e-12))
            cosine = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
            assert relative < 0.02 and cosine > 0.999, (key, step, relative, cosine)
            assert int(x.argmax()) == int(y.argmax()), (key, step)
            worst_error = max(worst_error, float((x-y).abs().max()))
            worst_relative = max(worst_relative, relative)
            minimum_cosine = min(minimum_cosine, cosine)
            compared += 1
    return dict(passed=True, batch=batch, logit_vectors=compared,
                max_absolute_error=worst_error, max_relative_l2=worst_relative,
                min_cosine=minimum_cosine, all_tokens_equal=True,
                reference=str(reference), candidate=str(candidate),
                scope='Synthetic weights/constant KV implementation equivalence, not model accuracy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.batch)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
