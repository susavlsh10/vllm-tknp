"""CPU-only tests for bounded, evidence-based startup capacity calibration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[3]/'examples/offline_inference/TKNP/nemotron4_capacity.py'
spec = importlib.util.spec_from_file_location('nemotron4_capacity', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)
        (self.stage/'engine').mkdir()

    def write(self, text):
        (self.stage/'engine/server0.log').write_text(text)

    def fit(self, **changes):
        options = dict(previous_slots=96, degree=4, context=32768,
                       output_tokens=1024, block_tokens=16)
        options.update(changes)
        return module.smaller_equal_capacity(self.stage, **options)

    def test_real_failed_pool_shrinks_96_to_92(self):
        self.write('GPU KV cache size: 822,640 tokens\nGPU KV cache size: 807,696 tokens\n'
                   'ValueError: Actual KV pools cannot fit the requested graph lanes')
        result = self.fit()
        self.assertEqual(result['equal_graph_ceiling'], 92)
        self.assertEqual(result['actual_min_owner_capacity'], 23)
        self.assertEqual(result['observed_pool_records'], 2)

    def test_safety_block_and_request_rounding(self):
        self.write('GPU KV cache size: 128 tokens\nActual KV pools cannot fit the requested graph lanes')
        self.assertEqual(self.fit(previous_slots=16, context=16, output_tokens=1)
                         ['equal_graph_ceiling'], 12)

    def test_equal_layout_rejection_can_calibrate(self):
        self.write('GPU KV cache size: 807696 tokens\n'
                   'TKNP v2 FULL graphs require equal positive owner counts')
        self.assertEqual(self.fit()['equal_graph_ceiling'], 92)

    def test_not_after_measurement(self):
        self.write('GPU KV cache size: 807696 tokens\nActual KV pools cannot fit the requested graph lanes')
        (self.stage/'b4/client').mkdir(parents=True)
        (self.stage/'b4/client/requests.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'client requests'):
            self.fit()

    def test_does_not_mask_non_capacity_errors(self):
        self.write('GPU KV cache size: 807696 tokens\nCUDA out of memory')
        with self.assertRaisesRegex(ValueError, 'not a known'):
            self.fit()

    def test_requires_actual_pool_evidence(self):
        self.write('Actual KV pools cannot fit the requested graph lanes')
        with self.assertRaisesRegex(ValueError, 'capacity evidence'):
            self.fit()

    def test_never_retries_same_shape(self):
        self.write('GPU KV cache size: 999999 tokens\nActual KV pools cannot fit the requested graph lanes')
        self.assertEqual(self.fit()['equal_graph_ceiling'], 92)

    def test_no_positive_layout(self):
        self.write('GPU KV cache size: 16 tokens\nActual KV pools cannot fit the requested graph lanes')
        with self.assertRaisesRegex(ValueError, 'No smaller'):
            self.fit()


if __name__ == '__main__':
    unittest.main()
