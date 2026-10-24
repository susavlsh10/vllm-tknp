"""CPU-only Nemotron architecture/storage planner regression checks."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


plan = module('nemotron4_plan', 'vllm/distributed/tknp_layer_ownership.py')
models = module('nemotron4_models', 'examples/offline_inference/TKNP/nemotron4_models.py')


class NemotronTests(unittest.TestCase):
    def test_official_dimensions_and_explicit_context_extrapolation(self):
        c = models.model_config()
        self.assertEqual((c['num_hidden_layers'], c['hidden_size'], c['intermediate_size']),
                         (96, 18432, 73728))
        self.assertEqual(c['num_attention_heads']*c['head_dim'], c['hidden_size'])
        self.assertEqual(c['num_key_value_heads'], 8)
        self.assertEqual(c['hidden_act'], 'relu2')
        self.assertEqual(c['max_position_embeddings']*c['rope_parameters']['factor'], 131072)
        self.assertEqual(c['rope_parameters']['partial_rotary_factor'], .5)
        self.assertFalse(c['research_provenance']['context_extension_validated_for_accuracy'])

    def test_recipe_keeps_all_attention_and_endpoints_bf16(self):
        c = models.model_config(fp8=True)
        ignored = c['quantization_config']['modules_to_not_convert']
        self.assertEqual(len(ignored), 389)
        self.assertIn('model.layers.95.self_attn.v_proj', ignored)
        self.assertIn('model.layers.0.mlp.up_proj', ignored)
        self.assertIn('model.layers.95.mlp.down_proj', ignored)
        self.assertNotIn('model.layers.1.mlp.up_proj', ignored)
        self.assertFalse(any('gate' in name for name in ignored))

    def test_nemotron_cost_has_two_not_three_mlp_matrices(self):
        kwargs = dict(layers=1, hidden=768, intermediate=3072, heads=4,
                      kv_heads=1, head_dim=192, weight_bytes=2)
        llama, nemo = plan.llama_layer_costs(**kwargs), plan.nemotron_layer_costs(**kwargs)
        self.assertEqual(llama[0]-nemo[0], 768*3072*2-4*768)

    def test_full_recipe_contiguous_all_owners_and_endpoint_budget(self):
        c = models.model_config(fp8=True)
        for degree in (2, 4, 8):
            owners = plan.nemotron_owners(96, 18432, 73728, 96, 8, 192, 256000,
                False, 1, degree, tuple(c['quantization_config']['modules_to_not_convert']))
            self.assertEqual(list(owners), sorted(owners))
            self.assertEqual(set(owners), set(range(degree)))
            self.assertLess(owners.count(0), owners.count(1))

    def test_native_and_v2_use_same_nemotron_operations(self):
        tree = ast.parse((ROOT/'vllm/model_executor/models/nemotron.py').read_text())
        classes = {n.name:n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        source = ast.get_source_segment((ROOT/'vllm/model_executor/models/nemotron.py').read_text(),
                                        classes['NemotronAttention'])
        self.assertIn('TokenParallelQKVLinear(', source)
        self.assertIn('TokenParallelRowLinear(', source)
        self.assertIn('self.rotary_emb(positions, q, k)', source)
        self.assertIn('self.attn(q, k, v)', source)
        self.assertEqual(len(classes['NemotronMLP'].decorator_list), 1)


if __name__ == '__main__':
    unittest.main()
