#!/usr/bin/env python3
"""Pinned native FA3 paged-decode numerical/graph diagnostic; no model run.

Same physical BF16 KV layout as FlashInfer diagnostics, exposed as stride-only
K/V views. Capture short context and grow it, with multiple empty graph rows.
Never run concurrently with model throughput measurements.
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,is_fa_version_supported,fa_version_unsupported_reason,
        get_scheduler_metadata)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--length",type=int,default=32768)
    p.add_argument("--batches",type=int,nargs="+",default=[2,6,16,20,32,64,96])
    p.add_argument("--aot-splits",type=int,default=0)
    args=p.parse_args()
    assert is_fa_version_supported(3),fa_version_unsupported_reason(3)
    torch.manual_seed(1019)
    rows=[];page,heads,dim=16,16,128
    for batch in args.batches:
        per=(args.length+page-1)//page
        kv=torch.randn(batch*per,2,1,page,dim,device="cuda",dtype=torch.bfloat16)
        k,v=(kv[:,i].permute(0,2,1,3) for i in (0,1))
        qkv=torch.randn(batch,18,dim,device="cuda",dtype=torch.bfloat16)
        query=qkv[:,:heads]
        table=torch.randperm(batch*per,device="cuda",dtype=torch.int32).view(batch,per)
        cu=torch.arange(batch+1,device="cuda",dtype=torch.int32)
        seq=torch.empty(batch,device="cuda",dtype=torch.int32)
        metadata=None
        def run():
            return flash_attn_varlen_func(q=query,k=k,v=v,
                max_seqlen_q=1,cu_seqlens_q=cu,max_seqlen_k=args.length,
                seqused_k=seq,block_table=table,softmax_scale=dim**-.5,
                causal=False,return_softmax_lse=True,fa_version=3,
                scheduler_metadata=metadata,num_splits=args.aot_splits)
        graph=None
        for phase in range(3):
            length=args.length//4 if phase==0 else args.length
            lengths=[0 if (i+phase)%5==0 else max(1,length-(i*31+phase*7)%511)
                     for i in range(batch)]
            seq.copy_(torch.tensor(lengths,dtype=torch.int32))
            if args.aot_splits:
                new=get_scheduler_metadata(batch_size=batch,max_seqlen_q=1,
                    max_seqlen_k=args.length,num_heads_q=heads,num_heads_kv=1,
                    headdim=dim,cache_seqlens=seq,qkv_dtype=query.dtype,
                    cu_seqlens_q=cu,page_size=page,causal=False,num_splits=args.aot_splits)
                if metadata is None:metadata=new
                else:
                    assert new.shape==metadata.shape
                    metadata.copy_(new)
            qkv.normal_()
            if graph is None:
                for _ in range(3):run()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out,lse=run()
            graph.replay();torch.cuda.synchronize()
            fresh,_=run()
            active=torch.tensor([n>0 for n in lengths],device="cuda")
            torch.testing.assert_close(out[active],fresh[active],rtol=.02,atol=5e-4)
            errors=[]
            selected=sorted({next(i for i,n in enumerate(lengths) if n),
                             max(i for i,n in enumerate(lengths) if n)})
            for i in selected:
                n=lengths[i]
                physical=table[i,:(n+page-1)//page].long()
                keys=k[physical].permute(2,0,1,3).reshape(1,-1,dim)[:,:n]
                values=v[physical].permute(2,0,1,3).reshape(1,-1,dim)[:,:n]
                with sdpa_kernel(SDPBackend.MATH):
                    reference=F.scaled_dot_product_attention(query[i].float()[None,:,None,:],
                        keys.float()[None],values.float()[None],enable_gqa=True)[0,:,0]
                torch.testing.assert_close(out[i].float(),reference,rtol=.02,atol=5e-4)
                errors.append(float((out[i].float()-reference).abs().max()))
            samples=[]
            for _ in range(3):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graph.replay()
                end.record();end.synchronize();samples.append(start.elapsed_time(end)/30)
            row=dict(batch=batch,max_context=args.length,phase=phase,
                graph_ms=statistics.median(samples),repeats_ms=samples,
                max_abs_error=max(errors),graph_matches_fresh=True,
                masked_rows=sum(n==0 for n in lengths),fa_version=3,aot_splits=args.aot_splits)
            rows.append(row);print("FA3_DECODE_PASS",json.dumps(row),flush=True)
            args.output.write_text(json.dumps(dict(passed=False,rows=rows),indent=2)+"\n")
        del graph,kv,k,v,qkv,query,table,cu,seq,out,lse,fresh
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(passed=True,rows=rows,
        scope="Single-layer native FA3 paged BF16 decode with strided query/KV views, changing lengths/masked rows, graph-vs-fresh and FP32 SDPA checks; not model throughput"),indent=2)+"\n")


if __name__=="__main__":main()
