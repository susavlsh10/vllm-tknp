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
