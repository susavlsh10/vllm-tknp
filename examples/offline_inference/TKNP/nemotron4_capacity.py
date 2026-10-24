"""Fail-closed startup calibration; never resize graphs with active requests."""
import math
from pathlib import Path
import re


def smaller_equal_capacity(failed_stage, previous_slots, degree, context,
                           output_tokens, block_tokens):
    failed_stage = Path(failed_stage)
    if any(failed_stage.rglob('requests.jsonl')):
        raise ValueError('Cannot retry after client requests have started')
    logs = list((failed_stage/'engine').glob('server*.log'))
    if not logs:
        raise ValueError('Missing failed startup worker log')
    text = '\n'.join(path.read_text(errors='replace') for path in logs)
    markers = ('Actual KV pools cannot fit the requested graph lanes',
               'TKNP v2 FULL graphs require equal positive owner counts')
    if not any(marker in text for marker in markers):
        raise ValueError('Failure is not a known startup graph-capacity rejection')
    values = [int(value.replace(',', '')) for value in re.findall(
        r'GPU KV cache size:\s*([\d,]+) tokens', text)]
    if not values or min(degree, context, output_tokens, block_tokens) < 1:
        raise ValueError('Missing valid actual KV-pool capacity evidence')
    blocks_per_request = math.ceil((context+output_tokens)/block_tokens)
    # Match fit_capture_split's one-block safety margin. The minimum over
    # every reported pool is conservative and preserves equal owner counts.
    per_owner = (min(values)//block_tokens-1)//blocks_per_request
    slots = min(per_owner*degree, previous_slots-degree)
    if slots < degree or previous_slots % degree:
        raise ValueError('No smaller positive equal graph layout is feasible')
    return dict(equal_graph_ceiling=slots, min_reported_kv_tokens=min(values),
                observed_pool_records=len(values), block_tokens=block_tokens,
                lifetime_tokens=context+output_tokens,
                actual_min_owner_capacity=per_owner,
                failed_stage=str(failed_stage),
                scope='Startup-only full-configuration calibration; no timed requests were run')
