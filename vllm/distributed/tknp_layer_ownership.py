# SPDX-License-Identifier: Apache-2.0
"""Static layer ownership for TKNP v2; control/KV ownership stays unchanged."""
import os
import re
from functools import lru_cache


def enabled():
    return os.environ.get('VLLM_TKNP_V2', '0') == '1'


def partition_layers(costs, endpoint_cost, degree):
    """Minimax contiguous blocks, including fixed rank-zero endpoint weights."""
    costs = tuple(int(c) for c in costs)
    if degree < 1 or len(costs) < degree or min(costs) <= 0 or endpoint_cost < 0:
        raise ValueError('Invalid TKNP v2 layer partition')

    def split(limit):
        cuts = [0]
        begin = 0
        for owner in range(degree):
            budget = limit - (endpoint_cost if owner == 0 else 0)
            end, used = begin, 0
            last = len(costs) - (degree-owner-1)
            while end < last and used+costs[end] <= budget:
                used += costs[end]
                end += 1
            if end == begin:
                return None
            cuts.append(end)
            begin = end
        return cuts if begin == len(costs) else None

    low = max(max(costs), endpoint_cost+costs[0])
    high = sum(costs)+endpoint_cost
    while low < high:
        mid = (low+high)//2
        if split(mid) is None:
            low = mid+1
        else:
            high = mid
    # Many cuts attain the minimax bound. Greedily filling each earlier owner
    # can leave the final block unnecessarily light. Among optimal solutions,
    # minimize squared storage (equivalently variance), including endpoints.
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1]+cost)
    states = {(0, 0): (0, None)}
    for count in range(1, degree+1):
        for end in range(count, len(costs)-(degree-count)+1):
            best = None
            for begin in range(count-1, end):
                previous = states.get((count-1, begin))
                used = prefix[end]-prefix[begin]+(endpoint_cost if count == 1 else 0)
                if previous is None or used > low:
                    continue
                candidate = (previous[0]+used*used, begin)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                states[count, end] = best
    cuts = [len(costs)]
    for count in range(degree, 0, -1):
        cuts.append(states[count, cuts[-1]][1])
    cuts.reverse()
    return tuple(owner for owner in range(degree)
                 for _ in range(cuts[owner], cuts[owner+1]))


@lru_cache(maxsize=16)
def llama_owners(layers, hidden, intermediate, heads, kv_heads, head_dim,
                 vocab, tied, weight_bytes, degree, ignored=()):
    costs = llama_layer_costs(layers, hidden, intermediate, heads, kv_heads,
                             head_dim, weight_bytes, ignored)
    # Embedding, output head and final norm remain on control TP group zero.
    endpoints = (1 if tied else 2)*vocab*hidden*2 + hidden*2
    return partition_layers(costs, endpoints, degree)


def llama_layer_costs(layers, hidden, intermediate, heads, kv_heads, head_dim,
                      weight_bytes, ignored=()):
    """Account for recipe-exempt BF16 projections, including fused modules.

    The 405B recipe leaves *all* attention projections and the endpoint MLPs
    in BF16. Treating every linear as FP8 would misbalance its block ownership.
    Quantization scale buffers are small; actual storage is audited at load.
    """
    def width(index, names):
        prefixes = [f'model.layers.{index}.{name}' for name in names]
        return 2 if any(p == skip or p.startswith(skip+'.')
                        for p in prefixes for skip in ignored) else weight_bytes
    return tuple(
        hidden*(heads+2*kv_heads)*head_dim*width(i, ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'])
        + hidden*heads*head_dim*width(i, ['self_attn.o_proj'])
        + 2*hidden*intermediate*width(i, ['mlp.gate_proj', 'mlp.up_proj'])
        + hidden*intermediate*width(i, ['mlp.down_proj']) + 2*hidden*2
        for i in range(layers))


def current_owners():
    if not enabled():
        return None
    from vllm.config import get_current_vllm_config
    cfg = get_current_vllm_config()
    pc = cfg.parallel_config
    if pc.pipeline_parallel_size != 1 or pc.decode_context_parallel_size != 1 or pc.enable_dbo:
        raise ValueError('TKNP v2 requires PP1/DCP1 and DBO disabled')
    if cfg.cache_config.cache_dtype not in ('auto', 'bfloat16'):
        raise ValueError('First TKNP v2 implementation requires model-precision/BF16 KV')
    hf = cfg.model_config.hf_config
    if hf.model_type != 'llama':
        raise ValueError('First TKNP v2 implementation supports Llama only')
    return llama_owners(hf.num_hidden_layers, hf.hidden_size, hf.intermediate_size,
                        hf.num_attention_heads, hf.num_key_value_heads,
                        getattr(hf, 'head_dim', hf.hidden_size//hf.num_attention_heads),
                        hf.vocab_size, hf.tie_word_embeddings,
                        1 if cfg.model_config.quantization in ('fbgemm_fp8','modelopt') else 2,
                        pc.token_parallel_size,
                        tuple((getattr(hf, 'quantization_config', None) or {}).get(
                            'modules_to_not_convert', [])))


def owner_for_prefix(prefix):
    owners = current_owners()
    match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', prefix)
    return owners[int(match.group(1))] if owners is not None and match else 0
