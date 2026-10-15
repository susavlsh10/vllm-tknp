"""CPU-only launcher contract checks; no torch, SLURM or cluster needed."""
import os
from pathlib import Path
import subprocess
import unittest


LAUNCHER = (Path(__file__).resolve().parents[3] /
            "examples/offline_inference/TKNP/jobs/eos_microbatch_worker.sh")


class EOSLauncherTest(unittest.TestCase):
    def launch(self, variant="pipelined", **overrides):
        env = dict(os.environ, TKNP_WORKTREE="/repo", TKNP_PYTHON="/venv/bin/python3",
                   TKNP_MODEL="/cached/model", TKNP_TP_SIZE="8", TKNP_WORLD_SIZE="2",
                   TKNP_LOCAL_GPUS="8", TKNP_NNODES="2", TKNP_NODE_RANK="0",
                   MASTER_ADDR="eos0001", MASTER_PORT="29601",
                   TKNP_DEVELOPMENT_SMOKE="0", TKNP_PROFILE="0")
        env.update(overrides)
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run", "/results/new",
                               variant, "32", "32768", "128"], env=env,
                              text=True, capture_output=True)

    def test_realistic_multinode_command(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tensor-parallel-size 8", result.stdout)
        self.assertIn("--token-parallel-size 2", result.stdout)
        self.assertIn("--nnodes=2", result.stdout)
        self.assertIn("--nproc-per-node=8", result.stdout)
        self.assertIn("--enable-dbo", result.stdout)
        self.assertIn("--kv-cache-dtype auto", result.stdout)
        self.assertIn("FULL_DECODE_ONLY", result.stdout)

    def test_off_keeps_graph_but_disables_dbo(self):
        result = self.launch("off")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--enable-dbo", result.stdout)
        self.assertIn("--cuda-graph", result.stdout)

    def test_topology_must_match_all_participating_gpus(self):
        result = self.launch(TKNP_LOCAL_GPUS="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TP x TKNP", result.stderr)

    def test_low_tp_requires_explicit_development_exception(self):
        result = self.launch(TKNP_TP_SIZE="2", TKNP_LOCAL_GPUS="4", TKNP_NNODES="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Paper runs require TP8", result.stderr)
        result = self.launch(TKNP_TP_SIZE="2", TKNP_LOCAL_GPUS="4", TKNP_NNODES="1",
                             TKNP_DEVELOPMENT_SMOKE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--standalone", result.stdout)

    def test_unsupported_recipe_is_rejected(self):
        result = self.launch("legacy")
        self.assertNotEqual(result.returncode, 0)

    def test_profile_flag_is_explicit(self):
        result = self.launch(TKNP_PROFILE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--torch-profiler-dir /results/new/traces", result.stdout)


if __name__ == "__main__":
    unittest.main()
