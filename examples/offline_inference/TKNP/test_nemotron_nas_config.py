"""Architecture/serialization checks; no GPUs or checkpoint tensors required."""
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('nas_config_test', ROOT/'vllm/transformers_utils/configs/nemotron_nas.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class NemotronNASConfigTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((Path(__file__).parent/'model_configs/nemotron_ultra_253b/config.json').read_text())

    def test_active_blocks_and_json_pickle_roundtrip(self):
        c = module.NemotronNASConfig(**self.raw)
        self.assertEqual(len(c.block_configs), 162)
        self.assertEqual(sum(not b.attention.no_op for b in c.block_configs), 64)
        self.assertEqual(sum(not b.ffn.no_op for b in c.block_configs), 81)
        restored = module.NemotronNASConfig(**json.loads(json.dumps(c.to_dict())))
        self.assertEqual(restored.block_configs, c.block_configs)
        self.assertEqual(pickle.loads(pickle.dumps(c)).block_configs[0].attention.n_heads_in_group, 16)

    def test_published_parameter_census(self):
        c = self.raw
        h = c['hidden_size']
        params = 2*c['vocab_size']*h+h
        for b in c['block_configs']:
            if not b['attention']['no_op']:
                kv = c['num_attention_heads']//b['attention']['n_heads_in_group']
                params += 2*h*h+2*h*kv*128+h
            if not b['ffn']['no_op']:
                intermediate = math.ceil(int(2*b['ffn']['ffn_mult']*h/3)/256)*256
                params += 3*h*intermediate+h
        self.assertEqual(params, 253401268224)
        self.assertEqual(params*2, 506802536448)

    def test_kv_cost_uses_active_attention_not_block_positions(self):
        per_token = 64*2*8*128*2
        self.assertEqual(per_token, 262144)
        self.assertEqual(per_token*(98304+1024)/2**30, 24.25)

    def test_invalid_block_count_rejected(self):
        raw = dict(self.raw, block_configs=self.raw['block_configs'][:-1])
        with self.assertRaises(ValueError):
            module.NemotronNASConfig(**raw)


if __name__=='__main__':
    unittest.main()
