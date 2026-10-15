# SPDX-License-Identifier: Apache-2.0
"""Research Llama PP decoder reads its layer interval from a PP1 producer.

The stock connector treats all remote layers as local, so it must not be used
unchanged for PP1 -> PPn. This adapter slices *registered memory descriptors*,
not tensor data, and tells the producer to await all PP consumers' reads.
Only full-attention Llama, FlashInfer HND VRAM, PP1 prefill and P.TP <= D.TP
are supported. DCP, MLA, layer sharing, sliding windows and PP prefill are not.
"""
import copy

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector, NixlConnectorScheduler, NixlConnectorWorker,
)


class PPNixlWorker(NixlConnectorWorker):
    def __init__(self, config, engine_id):
        super().__init__(config, engine_id)
        parallel = config.parallel_config
        if (config.model_config.architecture != "LlamaForCausalLM"
                or self.use_mla or self.use_host_buffer or self.backend_name != "FLASHINFER"
                or self.kv_cache_layout != "HND" or parallel.decode_context_parallel_size != 1):
            raise ValueError("PP NIXL research adapter supports full-attention Llama HND VRAM only")
        self.pp_start, self.pp_end = config.model_config.get_layers_start_end_indices(parallel)
        # A PP1 producer must hold its blocks until every reader (TP x PP)
        # has completed, rather than releasing after the first PP stage reads.
        self.notification_consumer_count = self.world_size * parallel.pipeline_parallel_size

    def add_remote_agent(self, metadata, remote_tp_rank=0, remote_tp_size=1):
        if remote_tp_size > self.world_size:
            raise ValueError("PP NIXL adapter requires P.TP <= D.TP")
        total = self.model_config.get_total_num_hidden_layers()
        if (len(metadata.kv_caches_base_addr) != total or len(metadata.block_lens) != total
                or self.num_layers != self.pp_end - self.pp_start or self.block_window_per_layer):
            raise ValueError("PP NIXL requires a PP1 producer and one unshared global-attention KV region per layer")
        metadata = copy.copy(metadata)
        metadata.kv_caches_base_addr = metadata.kv_caches_base_addr[self.pp_start:self.pp_end]
        metadata.block_lens = metadata.block_lens[self.pp_start:self.pp_end]
        return super().add_remote_agent(metadata, remote_tp_rank, remote_tp_size)


class PPNixlConnector(NixlConnector):
    def __init__(self, config, role, kv_cache_config=None):
        KVConnectorBase_V1.__init__(self, config, role, kv_cache_config)
        self.engine_id = config.kv_transfer_config.engine_id
        if config.parallel_config.pipeline_parallel_size <= 1:
            raise ValueError("PPNixlConnector is for PP decoders, not prefill")
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorScheduler(config, self.engine_id)
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            self.connector_worker = PPNixlWorker(config, self.engine_id)
