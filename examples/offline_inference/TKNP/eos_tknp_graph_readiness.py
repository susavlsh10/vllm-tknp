#!/usr/bin/env python3
"""Single-node 8B validation of multi-layout TKNP graph reuse. No allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def configuration(context, tokens, tp, tknp, dbo, random_kv):
    layouts = ({4: [2, 2], 8: [2, 6], 16: [4, 12], 32: [8, 24]}
               if tknp == 2 else {4: [1, 1, 1, 1], 8: [2, 2, 2, 2],
                   16: [2, 5, 5, 4], 32: [4, 10, 9, 9], 64: [8, 19, 19, 18]})
    if dbo:
        layouts.pop(4, None)
        if tknp == 4:
            layouts.pop(8, None)  # Minimum validated two-cohort shape: no changing singleton shards.
    env = dict(VLLM_KV_CACHE_LAYOUT="HND", VLLM_TKNP_LIFETIME_ADMISSION="1",
               VLLM_TKNP_REQUIRE_CUDAGRAPH="1", VLLM_TKNP_CAPTURE_REBALANCE="1",
               VLLM_TKNP_CUDAGRAPH_REQUEST_TOKENS=str(context + tokens))
    if dbo:
        env.update(VLLM_TKNP_DBO_ROOT_REQUESTS="2", VLLM_TKNP_DBO_C0_FRACTION="0.5",
            VLLM_TKNP_DBO_ZERO_ROOT_C1="1", VLLM_TKNP_DBO_INDEPENDENT_COMM="1",
            VLLM_TKNP_DBO_ROOT_SPLIT_COMPUTE="1", VLLM_TKNP_DBO_INDEPENDENT_TP="1",
            VLLM_TKNP_DBO_LAYOUT="asymmetric", VLLM_TKNP_DBO_LOOKAHEAD="0",
            VLLM_TKNP_DBO_SYMM_MEM="0")
    role = dict(role="decode", gpus=list(range(tp * tknp)), tp=tp, tknp=tknp,
        connector="synthetic", placement="cost", graph_slots=True, cuda_graph=True,
        graph_batch_size=max(layouts), graph_layouts=layouts, enable_dbo=dbo,
        max_num_seqs=max(layouts), env=env)
    result = dict(model="meta-llama/Llama-3.1-8B-Instruct", load_format="dummy",
        context=context, output_tokens=tokens, require_bf16=True, kv_cache_dtype="auto",
        preinitialize_random_kv=random_kv, verify_repeated_outputs=not random_kv,
        verify_busy_switch_rejection=True,
        max_model_len=context + tokens, max_num_seqs=max(layouts), max_num_batched_tokens=128,
        gpu_memory_utilization=0.8, warmup_requests=2, warmup_output_tokens=4,
        timeout=600, routing="least-work", port_base=8860,
        cache_root="/lustre/fsw/hw_nresearch_snoise/sshrestha/tknp_eos_cache/graph_reuse_validation",
        gpu_count=tp * tknp, trials=[dict(name=f"tp{tp}-tknp{tknp}-dbo{int(dbo)}", servers=[role])],
        scope="EOS single-node 8B BF16 dummy-weight multi-layout graph-reuse validation; not trained-model accuracy or paper throughput")
    if context < 4096:
        result["kv_cache_memory_bytes"] = 1024 ** 3
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--tknp", type=int, choices=[2, 4], default=2)
    parser.add_argument("--dbo", action="store_true")
    parser.add_argument("--random-kv", action="store_true")
    args = parser.parse_args()
    config = configuration(args.context, args.tokens, args.tp, args.tknp, args.dbo, args.random_kv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = args.output.with_suffix(".json")
    matrix.write_text(json.dumps(config, indent=2) + "\n")
    batches = ([4, 8, 16, 32, 48, 8, 32, 4] if args.tknp == 2
               else [4, 8, 16, 32, 64, 96, 32, 16, 4])
    subprocess.run([sys.executable, str(Path(__file__).with_name("tknp_graph_sweep.py")),
                    "--matrix", str(matrix), "--output", str(args.output),
                    "--batches", *map(str, batches)], check=True)


if __name__ == "__main__":
    main()
