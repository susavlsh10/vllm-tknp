"""CPU-only checks for exact mixtures and deterministic request generation."""
import unittest
from heterogeneous_decode import make_mixed_workload,workload_description


class Workloads(unittest.TestCase):
    def test_balanced_mean(self):
        a=workload_description(make_mixed_workload('balanced32_64_96k',24,64))
        b=workload_description(make_mixed_workload('uniform64k',24,64))
        self.assertEqual(a['total_prompt_tokens'],b['total_prompt_tokens'])
        self.assertEqual(a['context_histogram'],{32768:8,65536:8,98304:8})

    def test_skew_mean(self):
        a=workload_description(make_mixed_workload('skewed32_96k',24,64))
        b=workload_description(make_mixed_workload('uniform48k',24,64))
        self.assertEqual(a['total_prompt_tokens'],b['total_prompt_tokens'])
        self.assertEqual(a['context_histogram'],{32768:18,98304:6})

    def test_reproducible(self):
        self.assertEqual(make_mixed_workload('balanced32_64_96k',12,8),make_mixed_workload('balanced32_64_96k',12,8))

    def test_order_and_constraints(self):
        rows=make_mixed_workload('balanced32_64_96k',12,8,order='long-first')
        lengths=[len(r['prompt_token_ids']) for r in rows]
        self.assertEqual(lengths,sorted(lengths,reverse=True))
        self.assertEqual(len({r['request_id'] for r in rows}),12)
        self.assertTrue(all(r['arrival_s']==0 and r['max_tokens']==8 for r in rows))
        shuffled=make_mixed_workload('balanced32_64_96k',12,8)
        self.assertEqual({r['request_id']:r for r in rows},{r['request_id']:r for r in shuffled})
        with self.assertRaises(ValueError):make_mixed_workload('balanced32_64_96k',8,8)


if __name__=='__main__':unittest.main()
