#!/usr/bin/env python3
"""Nemotron-4 architecture-faithful dummy configs; no checkpoint conversion.

Source: NVIDIA Nemotron-4-340B-Instruct/model_config.yaml (96/18432/73728,
96 query heads,8 KV heads,192 head dim,LayerNorm1P,ReLU2,half-dimension RoPE).
FP8 is our Llama-derived research recipe, NOT an official Nemotron FP8 artifact.
Linear RoPE interpolation extends4096 by32 to131072 for systems tests only.
"""
import argparse
import json
from pathlib import Path

SOURCE = 'https://huggingface.co/nvidia/Nemotron-4-340B-Instruct/blob/main/model_config.yaml'
TOKENIZER_SOURCE = 'mgoin/Nemotron-4-340B-Instruct-hf'
TOKENIZER_REVISION = '9a5732d1245b6ce5bb07b268c2de8ddb3058af2d'
TOKENIZER_SHA256 = {
    'tokenizer.model': '6dfd8b970f437002fc445214304969fe59e64d4f48500bd0b77ba55340f2d811',
    'tokenizer_config.json': 'bc74336a12cd4fc0e45ec274015fcee3ecd745d85c3d5438ba23eb10c120d707',
    'special_tokens_map.json': '6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0'}


def model_config(tiny=False, fp8=False):
    config = dict(architectures=['NemotronForCausalLM'], model_type='nemotron',
        hidden_size=4608 if tiny else 18432, intermediate_size=18432 if tiny else 73728,
        num_hidden_layers=8 if tiny else 96, num_attention_heads=24 if tiny else 96,
        num_key_value_heads=2 if tiny else 8, head_dim=192,
        vocab_size=32768 if tiny else 256000, hidden_act='relu2',
        norm_eps=1e-5, attention_bias=False, mlp_bias=False,
        attention_dropout=0.0, tie_word_embeddings=False, initializer_range=0.0063,
        max_position_embeddings=4096, bos_token_id=2, eos_token_id=3,
        torch_dtype='bfloat16', use_cache=True,
        rope_parameters=dict(rope_type='linear', factor=32.0, rope_theta=10000.0,
                             partial_rotary_factor=0.5),
        research_provenance=dict(source=SOURCE, original_context=4096,
            effective_context=131072, context_extension_validated_for_accuracy=False,
            dummy_weights=True, tiny_architecture_proxy=tiny,
            recipe='Llama-derived FBGEMM FP8/BF16 policy' if fp8 else 'BF16 reference'))
    if fp8:
        ignored = ['lm_head']
        for i in range(config['num_hidden_layers']):
            ignored.extend(f'model.layers.{i}.self_attn.{p}_proj' for p in ['q','k','v','o'])
        for i in [0, config['num_hidden_layers']-1]:
            ignored.extend(f'model.layers.{i}.mlp.{p}_proj' for p in ['up','down'])
        config['quantization_config'] = dict(quant_method='fbgemm_fp8',
            activation_scale_ub=1200.0, modules_to_not_convert=ignored)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for tiny in (False, True):
        for fp8 in (False, True):
            name = ('tiny' if tiny else '340b') + ('_fp8' if fp8 else '_bf16')
            directory = args.output/name
            directory.mkdir(exist_ok=False)
            (directory/'config.json').write_text(json.dumps(model_config(tiny, fp8), indent=2)+'\n')
            print(name, directory, flush=True)


if __name__ == '__main__':
    main()
