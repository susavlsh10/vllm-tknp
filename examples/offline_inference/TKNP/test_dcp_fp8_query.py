"""Regression tests for DCP's cached-FP8/current-BF16 attention boundary."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.backends import flash_attn as backend


@pytest.mark.parametrize('fp8', [False, True])
def test_current_query_dtype_and_scale(fp8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.arange(256, device=device, dtype=torch.float32).reshape(2, 1, 128) / 64
    q = q.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    kv = torch.zeros((2, 1, 128), device=device, dtype=torch.bfloat16)
    cache = kv.to(torch.float8_e4m3fn) if fp8 else kv
    descale = torch.full((2, 1), .25, device=device)
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 2],device=device,dtype=torch.int32),
        max_query_len=1, block_table=None, dcp_context_kv_lens=None,
        max_dcp_context_kv_len=32, scheduler_metadata=None, max_num_splits=1, causal=True)
    impl = object.__new__(backend.FlashAttentionImpl)
    impl.scale = .125
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0.
    impl.vllm_flash_attn_version = 3
    calls = []

    def flash(**kwargs):
        calls.append(kwargs)
        assert kwargs['q'].dtype == kwargs['k'].dtype == kwargs['v'].dtype
        return torch.zeros_like(kv), torch.zeros((1, 2),device=device)

    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor)
    with patch.object(backend,'get_dcp_group',return_value=group), \
         patch.object(backend,'flash_attn_varlen_func',side_effect=flash), \
         patch.object(backend,'cp_lse_ag_out_rs',side_effect=lambda out,lse,*a,**kw:(out,lse)), \
         patch.object(backend,'merge_attn_states'):
        impl._forward_with_dcp(q,kv,kv,cache,cache,torch.empty_like(kv),metadata,
                               q_descale=descale,k_descale=descale,v_descale=descale)
    assert len(calls) == 2
    assert calls[0]['q'] is q and calls[0]['k'] is cache
    assert calls[0]['q_descale'] is descale
    if fp8:
        torch.testing.assert_close(calls[1]['q'], q.to(kv.dtype)*.25)
        assert all(calls[1][k] is None for k in ('q_descale','k_descale','v_descale'))
    else:
        assert calls[1]['q'] is q

