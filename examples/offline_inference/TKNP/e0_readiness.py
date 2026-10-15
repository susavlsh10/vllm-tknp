#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small single-node readiness scenarios. Never allocates or submits jobs.

Examples (inside the pinned environment):
  python e0_readiness.py --section staged --context 32768 --output NEW_DIRECTORY
  python e0_readiness.py --section queue --context 32768 --output NEW_DIRECTORY
All pool sizes below are artificial test caps, NOT headline capacity results.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def make_config(model, context, section, count=4, output_tokens=33):
    gib = 1024 ** 3
    common = {"model": model, "load_format": "auto", "kv_cache_dtype": "auto",
              "require_bf16": True,
              "max_model_len": context + output_tokens, "max_num_seqs": 16,
              "max_num_batched_tokens": 4096, "kv_cache_memory_bytes": 96 * gib,
              "warmup_requests": 0, "timeout": 900,
              "scope": "single-node real-weight E0 infrastructure test; artificially capped pools, NOT paper performance",
              "trials": []}
    producer = {"role": "prefill", "gpus": [0], "connector": "nixl",
                "kv_cache_memory_bytes": 16 * gib}
    environment = {"VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS": str(context + output_tokens),
                   "VLLM_TKNP_REQUIRE_CUDAGRAPH": "1", "VLLM_KV_CACHE_LAYOUT": "HND",
                   "VLLM_TKNP_DBO_ROOT_REQUESTS": "2", "VLLM_TKNP_DBO_ZERO_ROOT_C1": "1",
                   "VLLM_TKNP_DBO_INDEPENDENT_COMM": "1", "VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE": "1",
                   "VLLM_TKNP_DBO_LOOKAHEAD": "0", "VLLM_TKNP_DBO_LAYOUT": "asymmetric"}
    decoder = {"role": "decode", "gpus": [1, 2], "tknp": 2, "connector": "nixl",
               "placement": "cost", "cuda_graph": True, "graph_slots": True,
               "graph_batch_size": 8, "owner_counts": [2, 6], "env": environment,
               "disable_combo_kernels": True}
    if section == "staged":
        for enabled in (False, True):
            common["trials"].append({"name": "real-staged-" + ("on" if enabled else "off"),
                                     "staged_decode": True,
                                     "servers": [producer, dict(decoder, enable_dbo=enabled)]})
        common["trials"].append({"name": "synthetic-staged-on", "staged_decode": True,
                                 "servers": [dict(decoder, connector="synthetic", enable_dbo=True)]})
    elif section == "queue":
        producer = dict(producer, kv_cache_memory_bytes=64 * gib)
        common["trials"].append({"name": "real-queued-tknp-on",
                                 "prefill_barrier": True,
                                 "servers": [producer, dict(decoder, enable_dbo=True)]})
        common["trials"].append({"name": "real-queued-dp2", "prefill_barrier": True, "servers": [producer] + [
            {"role": "decode", "gpus": [gpu], "connector": "nixl", "cuda_graph": True,
             "max_num_seqs": 2, "graph_batch_size": 2, "kv_cache_memory_bytes": 8 * gib,
             "disable_combo_kernels": True} for gpu in (1, 2)]})
    elif section in {"tp", "pp"}:
        role = {"role": "decode", "gpus": [1, 2], section: 2, "connector": "nixl",
                "cuda_graph": True, "disable_combo_kernels": True,
                "graph_batch_size": 4, "max_num_seqs": 4}
        common["trials"].append({"name": "real-" + section + "2", "servers": [producer, role]})
    else:
        raise ValueError("Unknown readiness section")
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["staged", "queue", "tp", "pp", "unit", "diagnostics"], required=True)
    parser.add_argument("--context", type=int, choices=[32768, 65536, 98304], default=32768)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--count", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.section == "unit":
        args.output.mkdir(parents=True, exist_ok=False)
        subprocess.run([sys.executable, "-m", "pytest", "--noconftest", "-q",
                        "--junitxml=" + str(args.output / "unit.xml"),
                        "tests/v1/worker/test_tknp_cohort_plan.py",
                        "tests/v1/worker/test_tknp_placement.py",
                        "tests/v1/worker/test_tknp_pilot.py",
                        "tests/v1/worker/test_pp_nixl_layout.py",
                        "tests/v1/worker/test_tknp_customop_contract.py"], check=True)
        return
    if args.section == "diagnostics":
        if not args.input:
            parser.error("Full-logit diagnostics requires --input SMOKE_DIRECTORY")
        from full_logit_diagnostics import compare
        result = compare(args.input)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        if not result["finite"]:
            raise SystemExit(1)
        return
    count = args.count or (12 if args.section == "queue" else 8)
    tokens = args.output_tokens or (65 if args.section == "queue" else 33)
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    config = make_config(args.model, args.context, args.section, count, tokens)
    matrix = args.output / "matrix.json"
    matrix.write_text(json.dumps(config, indent=2) + "\n")
    workload = args.output / "workload.jsonl"
    subprocess.run([sys.executable, str(here / "e2e_workload.py"), "generate", "--natural",
                    "--output", str(workload), "--count", str(count), "--lengths", str(args.context),
                    "--output-lengths", str(tokens), "--tokenizer", args.model], check=True)
    command = [sys.executable, str(here / "e2e_matrix.py"), "--matrix", str(matrix),
               "--workload", str(workload), "--output", str(args.output / "trials")]
    if args.validate_only:
        command += ["--validate-only"]
    subprocess.run(command, check=True)
    if not args.validate_only:
        summaries = {trial["name"]: json.loads((args.output / "trials" / trial["name"] /
                                                "client/summary.json").read_text())
                     for trial in config["trials"]}
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
