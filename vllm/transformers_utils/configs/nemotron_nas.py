# SPDX-License-Identifier: Apache-2.0
"""Native configuration adapter for the published Llama Nemotron NAS layout.

Keep block dictionaries JSON/pickle compatible while exposing the attributes
used by vLLM's existing DeciLM implementation. No remote modeling code executes.
"""
from transformers import LlamaConfig


class NASBlockConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _block_attributes(value):
    if isinstance(value, dict):
        return NASBlockConfig({k: _block_attributes(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_block_attributes(v) for v in value]
    return value


class NemotronNASConfig(LlamaConfig):
    model_type = "nemotron-nas"

    def __init__(self, block_configs=None, **kwargs):
        super().__init__(**kwargs)
        if block_configs is not None and len(block_configs) != self.num_hidden_layers:
            raise ValueError("One NAS block configuration is required per layer")
        self.block_configs = _block_attributes(block_configs)
        self.intermediate_size = None
        self.num_key_value_heads = None
