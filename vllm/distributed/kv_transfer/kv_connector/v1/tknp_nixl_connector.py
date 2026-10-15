# SPDX-License-Identifier: Apache-2.0
"""NIXL pull into the sticky KV owner of an external-launcher TKNP engine.

The ordinary NIXL connector assumes every worker has blocks for every request.
TKNP does not: only one TP group owns each request. Never issue a NIXL read on
a non-owner. Publish completion to *all* replicated schedulers only after all
TP shards of the owner have completed. This is a research external-launcher
adapter, not support for the stock centralized multiprocess TKNP executor.
"""

import copy
from collections import defaultdict

import torch.distributed as dist

from vllm.distributed.parallel_state import (
    get_tknp_rank,
    get_tknp_world_size,
    get_tp_group,
    get_world_group,
    is_tknp_initialized,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector


class TKNPNixlConnector(NixlConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        self._tknp = is_tknp_initialized() and get_tknp_world_size() > 1
        self._completion_ranks = defaultdict(set)
        self._published = set()
        if self._tknp:
            if vllm_config.parallel_config.distributed_executor_backend != "external_launcher":
                raise ValueError("TKNPNixlConnector requires external_launcher")
            if vllm_config.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("TKNP NIXL adapter does not support PP")
            # Each owner TP group has a distinct registered-memory identity.
            vllm_config = copy.copy(vllm_config)
            vllm_config.kv_transfer_config = copy.copy(vllm_config.kv_transfer_config)
            vllm_config.kv_transfer_config.engine_id += f"-owner{get_tknp_rank()}"
        super().__init__(vllm_config, role, kv_cache_config)
        if self._tknp and self.connector_scheduler is not None:
            # External launcher has one scheduler per process. Avoid duplicate
            # ZMQ binds; no shared scheduler or implicit rank-zero listener.
            self.connector_scheduler.side_channel_port += dist.get_rank()

    def set_xfer_handshake_metadata(self, metadata):
        if self._tknp:
            gathered = [None] * dist.get_world_size(get_tp_group().cpu_group)
            dist.all_gather_object(gathered, metadata, group=get_tp_group().cpu_group)
            metadata = {key: value for item in gathered for key, value in item.items()}
        super().set_xfer_handshake_metadata(metadata)

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        if self._tknp:
            allocations = scheduler_output.token_parallel_allocations
            assert allocations is not None
            owners = allocations.req_to_tknp_rank
            metadata.reqs_to_recv = {
                req: value for req, value in metadata.reqs_to_recv.items()
                # An aborted, never-admitted request has no owner/blocks.
                # One TP group must still issue the zero-byte read notification
                # so the producer can release its exported KV. Never read KV
                # on a nonowner for a live request.
                if owners.get(req, 0 if not value.local_block_ids else -1) == get_tknp_rank()
            }
        return metadata

    def get_finished_count(self):
        # We already aggregate the owner TP shards, and each external executor
        # returns one worker result to its local scheduler.
        return 1 if self._tknp else super().get_finished_count()

    def get_finished(self, finished_req_ids):
        sending, receiving = super().get_finished(finished_req_ids)
        if not self._tknp:
            return sending, receiving
        group = get_world_group().cpu_group
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, (sending, receiving), group=group)
        tp_size = self.connector_worker.world_size
        done = set()
        for rank, (sent, received) in enumerate(gathered):
            if sent:
                raise RuntimeError("TKNP adapter is decode-only; use ordinary NIXL for prefill")
            for req in received:
                if req not in self._published:
                    self._completion_ranks[req].add(rank)
                    if len(self._completion_ranks[req]) == tp_size:
                        done.add(req)
        for req in done:
            del self._completion_ranks[req]
        self._published.update(done)
        self._published.difference_update(finished_req_ids)
        return set(), done

    def get_block_ids_with_load_errors(self):
        errors = super().get_block_ids_with_load_errors()
        if self._tknp:
            # Physical block IDs are owner-local and cannot be interpreted by
            # another rank's scheduler. Fail the run instead of silently
            # recomputing an unvalidated TKNP prefill on a subset of ranks.
            group = get_world_group().cpu_group
            gathered = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered, bool(errors), group=group)
            if any(gathered):
                raise RuntimeError("TKNP NIXL transfer failed on an owner shard")
        return errors
