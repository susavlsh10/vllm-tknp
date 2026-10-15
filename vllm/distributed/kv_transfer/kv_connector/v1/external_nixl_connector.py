# SPDX-License-Identifier: Apache-2.0
"""Replicated-scheduler NIXL completion for external TP/PP launchers.

Native NIXL assumes one centralized scheduler fed by MultiprocExecutor's
worker-output aggregator. A torchrun engine has a scheduler on every rank:
each must see identical completion only after EVERY model shard is ready.
This adapter does not change KV addressing or transfer granularity.
"""
from collections import defaultdict
import torch.distributed as dist
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
from vllm.distributed.kv_transfer.kv_connector.v1.pp_nixl_connector import PPNixlConnector


class ExternalCompletion:
    def __init__(self, vllm_config, role, kv_cache_config=None):
        if vllm_config.parallel_config.distributed_executor_backend != "external_launcher":
            raise ValueError("External NIXL adapter requires a replicated external launcher")
        self._completed_shards = defaultdict(set)
        self._published = set()
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_scheduler is not None:
            self.connector_scheduler.side_channel_port += dist.get_rank()

    def set_xfer_handshake_metadata(self, metadata):
        group = get_world_group().cpu_group
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, metadata, group=group)
        super().set_xfer_handshake_metadata({key: value for item in gathered for key, value in item.items()})

    def get_finished_count(self):
        return 1  # This adapter aggregates, not the one-worker external executor.

    def get_finished(self, finished_req_ids):
        sent, received = super().get_finished(finished_req_ids)
        group = get_world_group().cpu_group
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, (sent, received), group=group)
        ready = set()
        for rank, (sending, receiving) in enumerate(gathered):
            if sending:
                raise RuntimeError("External TP/PP adapter is decode-only; use PP1 native prefill")
            for req_id in receiving:
                if req_id not in self._published:
                    self._completed_shards[req_id].add(rank)
                    if len(self._completed_shards[req_id]) == len(gathered):
                        ready.add(req_id)
        for req_id in ready:
            del self._completed_shards[req_id]
        self._published.update(ready)
        self._published.difference_update(finished_req_ids)
        return set(), ready

    def get_block_ids_with_load_errors(self):
        errors = super().get_block_ids_with_load_errors()
        group = get_world_group().cpu_group
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, bool(errors), group=group)
        if any(gathered):
            raise RuntimeError("NIXL transfer failed on a TP/PP shard; do not silently diverge schedulers")
        return errors


class ExternalNixlConnector(ExternalCompletion, NixlConnector):
    pass


class ExternalPPNixlConnector(ExternalCompletion, PPNixlConnector):
    pass
