#!/usr/bin/env python3
"""Isolate 405B TP32/DCP4's attention shape with an 8B-sized dummy model.

Existing allocation only. Diagnostic, not a model-comparison result: eight
H100s, four query heads and one replicated KV head per rank (two global KV
heads instead of Llama's eight), odd 111-token graph, physical
BF16 random KV, native admission waves. Does not download weights.
"""
import argparse
import json
import os
from pathlib import Path

from eos_efficiency import configuration,write_json
from e2e_matrix import run_trial,validate_matrix
from e2e_workload import make_workload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--eager",action="store_true")
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--slots",type=int,default=111)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=configuration(args.job,args.node,args.output,32768,16)
    template=Path("/lustre/fsw/hw_nresearch_snoise/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/config.json")
    model=json.loads(template.read_text());model["num_key_value_heads"]=2
    model_path=args.output/"synthetic-shape-config";model_path.mkdir()
    write_json(model_path/"config.json",model)
    cfg.update(tokenizer=cfg["model"],model=str(model_path))
    cfg.update(kv_cache_memory_bytes=24*1024**3,max_num_seqs=args.slots,
        max_num_batched_tokens=256,warmup_requests=4,warmup_output_tokens=4,
        admission_gated_decode=True)
    env=cfg["runtime"]["env"]
    env.update(VLLM_ATTENTION_BACKEND="FLASH_ATTN",VLLM_RESEARCH_RANK_TRITON_CACHE="1")
    # Synchronous diagnostic only; never used for performance comparisons.
    if args.eager:env["CUDA_LAUNCH_BLOCKING"]="1"
    for key,value in env.items():os.environ[key]=str(value)
    role=dict(role="decode",gpus=list(range(8)),tp=8,dcp=4,pp=1,tknp=1,
        connector="synthetic",cuda_graph=not args.eager,disable_combo_kernels=False,
        native_graph_shapes=not args.eager,max_num_seqs=args.slots,
        graph_batch_size=args.slots if not args.eager else None,
        env=dict(VLLM_TKNP_LIFETIME_ADMISSION="0"))
    trial=dict(name="dcp-odd111-graph" if not args.eager else "dcp-odd111-eager",servers=[role])
    cfg["trials"]=[trial]
    validate_matrix(cfg,8)
    workload=args.output/"workload.jsonl"
    workload.write_text("".join(json.dumps(r)+"\n" for r in make_workload(args.batch,[32768],[16],0,20261019)))
    write_json(args.output/"matrix.json",cfg)
    run_trial(trial,cfg,workload,args.output/"trial")
    client=json.loads((args.output/"trial/client/summary.json").read_text())
    if client["failed"] or client["completed"]!=args.batch:raise RuntimeError("Incomplete diagnostic")
    write_json(args.output/"summary.json",dict(passed=True,client=client,
        performance_valid=False,scope=__doc__))


if __name__=="__main__":main()
