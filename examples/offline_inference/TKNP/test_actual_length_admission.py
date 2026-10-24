"""CPU regression tests for the opt-in FP8 benchmark admission clamp."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ActualLengthAdmissionTest(unittest.TestCase):
    def run_clamp(self, enabled, pp=1, lifetime='1'):
        source = Path(__file__).with_name('e2e_server.py')
        tree = ast.parse(source.read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'args.require_fbgemm_fp8 and args.tknp == 1')
        scheduler = SimpleNamespace(max_num_running_reqs=6)
        scope = dict(os=os, args=SimpleNamespace(require_fbgemm_fp8=True,
                     tknp=1, pp=pp, max_model_len=99328, dcp=1),
                     resources=[dict(kv_blocks=13816, block_tokens=16)],
                     core=SimpleNamespace(scheduler=scheduler))
        with patch.dict(os.environ, VLLM_RESEARCH_ACTUAL_LENGTH_ADMISSION=enabled,
                        VLLM_TKNP_LIFETIME_ADMISSION=lifetime):
            exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec'), scope)
        return scheduler.max_num_running_reqs

    def test_default_retains_old_clamp(self):
        self.assertEqual(self.run_clamp('0'), 2)

    def test_actual_lengths_keep_graph_ceiling(self):
        self.assertEqual(self.run_clamp('1'), 6)

    def test_no_reservations_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_clamp('1', lifetime='0')

    def test_unvalidated_pp_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_clamp('1', pp=4)


if __name__ == '__main__':
    unittest.main()
