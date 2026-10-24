"""CPU-only launch/grid and all-rank Chrome trace contract tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).parents[3] / 'examples/offline_inference/TKNP'


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load('tknp_v2_405b_sweep')
merge = load('tknp_trace_merge')


class StudyTests(unittest.TestCase):
    def test_eight_node_equal_capacity(self):
        maximum, sizes = sweep.equal_layouts([8, 16, 32, 64, 96, 128], [31]*7+[30])
        self.assertEqual(maximum, 128)
        self.assertEqual(sizes, [8, 16, 32, 64, 96, 128])
        maximum, sizes = sweep.equal_layouts([4, 8, 16, 32, 64, 96, 128], [8, 9, 9, 8])
        self.assertEqual((maximum, sizes), (32, [4, 8, 16, 32]))

    def test_portable_baseline_and_tuned_exclusion(self):
        rows = [dict(batch=8, performance_valid=True, full_graph_all_ranks=True,
                     measured_preemptions=0, pp_baseline=x) for x in ['scheduling-only', 'tuned']]
        original = dict(context=32768, output_length=1024, trials=rows)
        portable = dict(input_tokens=32768, output_tokens_per_request=1024, results=rows)
        self.assertEqual(sweep.baseline_rows(original, 32768, 1024, [8]), rows[:1])
        self.assertEqual(sweep.baseline_rows(portable, 32768, 1024, [8]), rows[:1])
        with self.assertRaises(ValueError):
            sweep.baseline_rows(original, 65536, 1024, [8])

    def test_offset_interpolation_and_bounds(self):
        clocks = [dict(hosts={'h': dict(controller_mid_ns=t, offset_ns=o, uncertainty_ns=u)})
                  for t, o, u in [(1000, 100, 10), (2000, 300, 20)]]
        self.assertEqual(merge.host_correction('h', 1600, clocks), (200, 20))
        self.assertEqual(merge.host_correction('h.domain', 0, clocks), (100, 20))
        self.assertEqual(merge.host_correction('h', 10000, clocks), (300, 20))

    @staticmethod
    def trace(rank, base=1000000000):
        return dict(baseTimeNanoseconds=base,
            distributedInfo=dict(rank=rank, world_size=2), traceEvents=[
                dict(ph='M', name='process_name', pid=0, tid=0, args=dict(name='GPU')),
                dict(ph='X', name='attention', cat='kernel', pid=0, tid=7,
                     ts=100+rank*2, dur=5),
                dict(ph='X', name='cudaGraphLaunch', cat='cuda_runtime', pid=100,
                     tid=7, ts=99+rank*2, dur=1),
                dict(ph='s', name='flow', id=1, pid=0, tid=7, ts=100+rank*2),
                dict(ph='f', name='flow', id=1, pid=100, tid=7, ts=101+rank*2)])

    def test_merge_preserves_offsets_durations_and_unique_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in range(2):
                (root / f'rank{rank}.pt.trace.json').write_text(json.dumps(self.trace(rank)))
            clocks = [dict(hosts={host: dict(controller_mid_ns=1000000000,
                      offset_ns=rank*1000, uncertainty_ns=20)
                      for rank, host in enumerate(['a', 'b'])})]
            result = merge.merge_traces(root, root/'merged.json', 2, {0:'a', 1:'b'}, clocks)
            data = json.loads((root/'merged.json').read_text())
            kernels = [e for e in data['traceEvents'] if e.get('cat') == 'kernel']
            self.assertEqual([e['dur'] for e in kernels], [5, 5])
            self.assertEqual(kernels[1]['ts']-kernels[0]['ts'], 1)
            self.assertNotEqual(kernels[0]['pid'], kernels[1]['pid'])
            self.assertEqual({e['id'] for e in data['traceEvents'] if 'id' in e}, {'r0:1', 'r1:1'})
            self.assertEqual(len(result['ranks']), 2)
            self.assertFalse(result['performance_valid'])
            self.assertTrue((root/'merged.json.gz').exists())

    def test_fail_closed_on_missing_duplicate_or_empty_gpu_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'r.pt.trace.json'
            path.write_text(json.dumps(self.trace(0)))
            with self.assertRaises(ValueError):
                merge.merge_traces(root, root/'merged.json', 2)
            (root/'duplicate.pt.trace.json').write_text(path.read_text())
            with self.assertRaises(ValueError):
                merge.merge_traces(root, root/'merged.json', 2)
            (root/'duplicate.pt.trace.json').unlink()
            second = self.trace(1)
            second['traceEvents'] = []
            (root/'second.pt.trace.json').write_text(json.dumps(second))
            with self.assertRaises(ValueError):
                merge.merge_traces(root, root/'merged.json', 2)


if __name__ == '__main__':
    unittest.main()
