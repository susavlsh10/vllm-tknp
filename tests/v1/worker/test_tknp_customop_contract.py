# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator contract check")
def test_communication_ops_do_not_return_mutated_input_aliases(monkeypatch):
    from vllm.model_executor.layers import token_parallel_linear as linear

    # Check functionalization/fake schemas without needing a distributed job.
    monkeypatch.setattr(linear, "_is_tknp_dummy_run", lambda: True)
    send = torch.randn(2, 16, device="cuda")
    receive = torch.empty(1, 16, device="cuda")
    assert torch.ops.vllm.tknp_scatter(send, receive, [1, 1], 0, "unused", 0) is None
    token = torch.ops.vllm.tknp_gather(send, receive, [1, 1], 0, "unused", 0)
    assert token.numel() == 1 and token.data_ptr() != receive.data_ptr()
    for operation in (torch.ops.vllm.tknp_scatter, torch.ops.vllm.tknp_gather):
        torch.library.opcheck(operation, (send, receive, [1, 1], 0, "unused", 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator contract check")
@pytest.mark.parametrize("rank,local_rows", [(0, 2), (1, 2), (2, 0)])
def test_v2_handoff_fresh_outputs_and_empty_attention_lane(monkeypatch, rank, local_rows):
    from types import SimpleNamespace
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import token_parallel_linear as linear

    monkeypatch.setattr(linear, "_is_tknp_dummy_run", lambda: True)
    monkeypatch.setattr(linear, "_get_tknp_group_by_name", lambda _: None)
    monkeypatch.setattr(linear, "_get_pynccl_comm", lambda _: SimpleNamespace(rank=rank))
    monkeypatch.setattr(parallel_state, "get_tknp_rank", lambda: rank)
    hidden = torch.randn(8, 16, device="cuda")
    residual = torch.randn_like(hidden)
    dependency = torch.zeros(1, device="cuda")
    arguments = (hidden, residual, dependency, 8, local_rows, 0, 1, "unused")
    outputs = torch.ops.vllm.tknp_handoff(*arguments)
    expected = 8 if rank == 1 else local_rows
    assert all(t.shape == (expected, 16) for t in outputs)
    assert all(t.numel() == 0 or t.data_ptr() not in (hidden.data_ptr(), residual.data_ptr())
               for t in outputs)
    torch.library.opcheck(torch.ops.vllm.tknp_handoff, arguments)
