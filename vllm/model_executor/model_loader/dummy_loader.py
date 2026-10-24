# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights


def initialize_fbgemm_fp8_dummy_weights(model, low=-1e-3, high=1e-3, seed=1234):
    """Opt-in synthetic FP8 initialization, also used by spawned MP workers."""
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8LinearMethod
    quantized = {id(module.weight): module for module in model.modules()
                 if isinstance(getattr(module, "quant_method", None), FBGEMMFp8LinearMethod)}
    overrides = {}
    for module in quantized.values():
        overrides[id(module.weight_scale)] = 1e-3 / 224.0
        overrides[id(module.input_scale_ub)] = module.quant_method.quant_config.input_scale_ub
    with torch.no_grad():
        for tensor in model.parameters():
            if not torch.is_floating_point(tensor):
                continue
            if id(tensor) in overrides:
                tensor.fill_(overrides[id(tensor)])
                continue
            generator = torch.Generator(device=tensor.device).manual_seed(seed)
            if id(tensor) in quantized:
                flat = tensor.view(-1)
                chunk = 8 * 1024 * 1024
                for begin in range(0, flat.numel(), chunk):
                    count = min(chunk, flat.numel() - begin)
                    scratch = torch.empty(count, dtype=torch.float16, device=tensor.device)
                    scratch.uniform_(-224, 224, generator=generator)
                    flat[begin:begin + count].copy_(scratch.to(tensor.dtype))
                del scratch
            else:
                tensor.uniform_(low, high, generator=generator)


def initialize_modelopt_fp8_dummy_weights(model, seed=1234):
    """Synthetic ModelOpt per-tensor FP8 weights/scales, never accuracy evidence."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8LinearMethod, ModelOptFp8KVCacheMethod)
    quantized, scales = set(), {}
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, ModelOptFp8LinearMethod):
            weight = getattr(module, "weight", None)
            if weight is not None:
                quantized.add(id(weight))
            for name, value in (("weight_scale", 1e-3 / 224), ("input_scale", 1e-4)):
                tensor = getattr(module, name, None)
                if tensor is not None:
                    scales[id(tensor)] = value
        elif isinstance(method, ModelOptFp8KVCacheMethod):
            for name, value in (("k_scale", 1.0), ("v_scale", 1.0),
                                ("q_scale", 1e-4), ("prob_scale", 1/448)):
                tensor = getattr(module, name, None)
                if tensor is not None:
                    scales[id(tensor)] = value
    with torch.no_grad():
        for tensor in model.parameters():
            if not torch.is_floating_point(tensor):
                continue
            if id(tensor) in scales:
                tensor.fill_(scales[id(tensor)])
                continue
            generator = torch.Generator(device=tensor.device).manual_seed(seed)
            if id(tensor) in quantized:
                flat = tensor.view(-1)
                for begin in range(0, flat.numel(), 8*1024*1024):
                    count = min(8*1024*1024, flat.numel()-begin)
                    scratch = torch.empty(count, dtype=torch.float16, device=tensor.device)
                    scratch.uniform_(-224, 224, generator=generator)
                    flat[begin:begin+count].copy_(scratch.to(tensor.dtype))
                del scratch
            else:
                tensor.uniform_(-1e-3, 1e-3, generator=generator)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self.fbgemm_fp8_dummy = extra == {"fbgemm_fp8_dummy": True}
        self.modelopt_fp8_dummy = extra == {"modelopt_fp8_dummy": True}
        if extra and not (self.fbgemm_fp8_dummy or self.modelopt_fp8_dummy):
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        if self.fbgemm_fp8_dummy:
            if model_config.quantization != "fbgemm_fp8":
                raise ValueError("FP8 dummy initialization requires fbgemm_fp8 quantization")
            initialize_fbgemm_fp8_dummy_weights(model)
        elif self.modelopt_fp8_dummy:
            if model_config.quantization != "modelopt":
                raise ValueError("ModelOpt dummy initialization requires modelopt quantization")
            initialize_modelopt_fp8_dummy_weights(model)
        else:
            initialize_dummy_weights(model)
