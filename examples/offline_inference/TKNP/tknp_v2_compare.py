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
    compared = replicated_copies_checked = non_output_copies = 0
    manifests = {}
    for key in a:
        assert a[key]['token_ids'] == b[key]['token_ids'], key
        for step in range(len(a[key]['token_ids'])):
            tensors = []
            for stage, request in ((reference, a[key]), (candidate, b[key])):
                digest = hashlib.sha256(request['server_request_id'].encode()).hexdigest()
                paths = sorted((stage/'engine').glob(f'server*/full-logits-rank*/{digest}-{step}.pt'))
                assert paths, (stage, key, step, paths)
                tensor = None
                for path in paths:
                    server = path.parents[1].name
                    manifest_path = stage/f'b{batch}'/server/'manifest.json'
                    if manifest_path not in manifests:
                        manifests[manifest_path] = {r['rank']: r for r in json.loads(
                            manifest_path.read_text())['worker_resources']}
                    rank = int(path.parent.name.removeprefix('full-logits-rank'))
                    owner = manifests[manifest_path][rank].get('tknp_v2')
                    if owner and owner['tknp_rank'] != owner['control_rank']:
                        # Non-output TKNP ranks expose identity-head placeholders.
                        # Their status is verified from recorded topology, not values.
                        non_output_copies += 1
                        continue
                    record = torch.load(path, map_location='cpu', weights_only=True)
                    assert record['step'] == step and record['request_id'] == request['server_request_id']
                    copy = record['logits'].float()
                    assert bool(torch.isfinite(copy).all())
                    if tensor is None:
                        tensor = copy
                    else:
                        # TP ranks may each record the same all-gathered logits.
                        # Validate every replica rather than picking one blindly.
                        assert torch.equal(tensor, copy), ('Replicated TP logits disagree', path)
                        replicated_copies_checked += 1
                assert tensor is not None, ('No real output-owner logits', stage, key, step)
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
                replicated_logit_copies_checked=replicated_copies_checked,
                non_output_rank_placeholders_excluded=non_output_copies,
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
