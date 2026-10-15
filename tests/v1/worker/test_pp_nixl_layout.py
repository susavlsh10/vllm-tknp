# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest


def test_pp_reader_selects_only_its_global_layer_interval(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.pp_nixl_connector import (
        NixlConnectorWorker, PPNixlWorker,
    )

    # This layout-only fixture deliberately does not create a NIXL agent.
    monkeypatch.setattr(PPNixlWorker, "shutdown", lambda self: None)
    worker = object.__new__(PPNixlWorker)
    worker.world_size = 1
    worker.pp_start, worker.pp_end = 16, 32
    worker.num_layers = 16
    worker.block_window_per_layer = []
    worker.model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: 32)
    metadata = SimpleNamespace(kv_caches_base_addr=list(range(32)), block_lens=[1024] * 32)
    captured = []
    monkeypatch.setattr(NixlConnectorWorker, "add_remote_agent",
                        lambda self, meta, rank, size: captured.append(meta) or "agent")
    assert worker.add_remote_agent(metadata) == "agent"
    assert captured[0].kv_caches_base_addr == list(range(16, 32))
    assert captured[0].block_lens == [1024] * 16
    assert metadata.kv_caches_base_addr == list(range(32))  # producer remains intact
    with pytest.raises(ValueError):
        worker.add_remote_agent(metadata, remote_tp_size=2)
    metadata.kv_caches_base_addr = list(range(16))
    with pytest.raises(ValueError):
        worker.add_remote_agent(metadata)
