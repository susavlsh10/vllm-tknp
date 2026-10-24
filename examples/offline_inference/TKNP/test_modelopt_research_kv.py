"""CPU checks for official recipe metadata and finite encoded synthetic FP8 KV."""
import json
from pathlib import Path
import unittest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import initialize_research_kv


class ModelOptResearchKVTest(unittest.TestCase):
    def test_published_metadata(self):
        root=Path(__file__).parent/'model_configs/nemotron_ultra_253b_modelopt'
        quant=json.loads((root/'hf_quant_config.json').read_text())['quantization']
        self.assertEqual(quant,dict(quant_algo='FP8',kv_cache_quant_algo='FP8',exclude_modules=['lm_head']))
        blocks=json.loads((root/'config.json').read_text())['block_configs']
        self.assertEqual(sum(not b['attention']['no_op'] for b in blocks),64)
        self.assertEqual(64*2*8*128*(98304+1024),13019119616)

    def test_fp8_encoding_finite_and_repeatable(self):
        a=torch.empty((2,8,16,2,128),dtype=torch.uint8).transpose(1,2)
        b=torch.empty_like(a)
        for t in (a,b):
            initialize_research_kv(t,cache_dtype='fp8',mean=.015,std=.01,seed=123)
        self.assertTrue(torch.equal(a,b))
        decoded=a.contiguous().view(torch.float8_e4m3fn).float()
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertGreater(decoded.std().item(),.005)
        self.assertLess(abs(decoded.mean().item()-.015),.001)

    def test_bytes_require_explicit_fp8(self):
        with self.assertRaises(ValueError):
            initialize_research_kv(torch.empty(32,dtype=torch.uint8),
                cache_dtype='auto',mean=.015,std=.01,seed=123)

    def test_bf16_path_is_unchanged(self):
        a=torch.empty(1024,dtype=torch.bfloat16)
        initialize_research_kv(a,cache_dtype='auto',mean=.015,std=.01,seed=123)
        generator=torch.Generator().manual_seed(123)
        b=torch.empty_like(a).normal_(.015,.01,generator=generator)
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__':
    unittest.main()
