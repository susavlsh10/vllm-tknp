"""Recipe and synthetic-weight invariants for the bounded EOS FP8 probe."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from fp8_405b_smoke import checkpoint_counts, initialize_fp8_dummy_weights


def test_official_405b_tensor_census():
    config = dict(hidden_size=16384, intermediate_size=53248, vocab_size=128256,
                  num_hidden_layers=126, num_key_value_heads=8, head_dim=128)
    assert checkpoint_counts(config) == {
        "BF16": 81314922496, "F8_E4M3": 324538466304, "F32": 15237120}
    config["num_key_value_heads"] = 16
    assert checkpoint_counts(config)["BF16"] != 81314922496


@pytest.mark.parametrize("initializer", ["smoke", "native_worker"])
def test_fp8_dummy_has_positive_scales_and_nonzero_weights(initializer):
    import torch
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8LinearMethod
    from vllm.model_executor.model_loader.dummy_loader import initialize_fbgemm_fp8_dummy_weights
    initialize = (initialize_fp8_dummy_weights if initializer == "smoke"
                  else initialize_fbgemm_fp8_dummy_weights)
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.float8_e4m3fn),
                                      requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(64, 1), requires_grad=False)
    layer.input_scale_ub = torch.nn.Parameter(torch.empty(()), requires_grad=False)
    method = FBGEMMFp8LinearMethod.__new__(FBGEMMFp8LinearMethod)
    method.quant_config = SimpleNamespace(input_scale_ub=1200.0)
    layer.quant_method = method
    model.add_module("quantized", layer)
    initialize(model)
    weight = layer.weight.float()
    assert torch.isfinite(weight).all()
    assert (weight != 0).float().mean() > 0.99
    assert (layer.weight_scale > 0).all()
    assert layer.input_scale_ub.item() == 1200.0
    assert (weight * layer.weight_scale).abs().max() <= 0.00101
    saved = weight.clone()
    initialize(model)
    torch.testing.assert_close(layer.weight.float(), saved, rtol=0, atol=0)
