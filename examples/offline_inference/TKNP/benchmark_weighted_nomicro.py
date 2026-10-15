"""Ablation: retain DBO's rank totals, but execute one unsplit model batch.

This only replaces the pure request-count planner in the external-launcher
benchmark process. No communication/stream/model implementation is replaced.
Keep root ownership positive; zero-root unsplit graphs are not supported by
this fixed-shape prototype. Set VLLM_TKNP_DBO_ROOT_REQUESTS to the DBO root quota.
"""
import runpy
from pathlib import Path

from vllm.forward_context import compute_dbo_tknp_split
from vllm.v1.core.sched import scheduler
from vllm.v1.worker import gpu_model_runner

scheduler.compute_capacity_aware_tknp_split = compute_dbo_tknp_split
gpu_model_runner.compute_capacity_aware_tknp_split = compute_dbo_tknp_split
runpy.run_path(str(Path(__file__).with_name("tknp_inference_benchmarks.py")),
               run_name="__main__")
