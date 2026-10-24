#!/usr/bin/env python3
"""Exact GQA16 query grouping with SHARED physical KV, isolated diagnostic.

One KV head, 16 query heads per request. Grouping query heads turns each
request into multiple logical decode rows pointing to the SAME KV pages.
No replication of physical KV or model weights. Numerical and graph checks
are required before any runtime integration. Never run alongside model jobs.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import time


def main():
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batches",type=int,nargs="+",default=[2,4,6,8,12,16,20,24,32,34])
    p.add_argument("--length",type=int,default=32768)
    p.add_argument("--groups",type=int,nargs="+",default=[1,2])
    p.add_argument("--public-plan",action="store_true")
    p.add_argument("--sdpa-reference",action="store_true")
    p.add_argument("--nonempty-padding",action="store_true",
                   help="Grouped masked rows read one borrowed page; output discarded")
    args=p.parse_args()
    torch.manual_seed(20261019)
    rows=[];page=16;dim=128;heads=16
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(dict(rows=rows,passed=False,
            scope="Isolated same-KV GQA16 head-grouping graph diagnostic; not model throughput"),indent=2)+"\n")
    for batch in args.batches:
        per=math.ceil(args.length/page)
        kv=torch.randn(batch*per,2,1,page,dim,device="cuda",dtype=torch.bfloat16)
        order=torch.randperm(kv.shape[0],device="cuda",dtype=torch.int32)
        query=torch.randn(batch,heads,dim,device="cuda",dtype=torch.bfloat16)
        reference={}
        for groups in args.groups:
            expanded=batch*groups
            workspace=torch.empty(128*1024**2,device="cuda",dtype=torch.uint8)
            indices=torch.empty(batch*per*groups,device="cuda",dtype=torch.int32)
            wrapper=BatchDecodeWithPagedKVCacheWrapper(workspace,"HND",use_cuda_graph=True,
                paged_kv_indptr_buffer=torch.zeros(expanded+1,device="cuda",dtype=torch.int32),
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=torch.zeros(expanded,device="cuda",dtype=torch.int32),
                use_tensor_cores=groups==1)
            graph=None
            for phase in (0,1):
                # Change lengths/queries and include a masked graph row.
                lengths=[max(0,args.length-(i*113+phase*7)%1024) for i in range(batch)]
                if phase:lengths[0]=0
                kernel_lengths=[max(1,n) if args.nonempty_padding and groups>1 else n for n in lengths]
                counts=[math.ceil(n/page) for n in kernel_lengths]
                pages=[]
                for i,n in enumerate(counts):
                    for _ in range(groups):
                        pages.append(order[i*per:i*per+n])
                payload=torch.cat(pages)
                indices.zero_();indices[:payload.numel()].copy_(payload)
                sizes=[n for n in counts for _ in range(groups)]
                indptr=torch.tensor([0]+list(itertools.accumulate(sizes)),dtype=torch.int32)
                last=torch.tensor([((n-1)%page+1) if n else 0 for n in kernel_lengths
                                   for _ in range(groups)],dtype=torch.int32)
                seq=torch.tensor([n for n in kernel_lengths for _ in range(groups)],dtype=torch.int32)
                if args.public_plan:
                    wrapper.plan(indptr,indices,last,heads//groups,1,dim,page,
                        q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=dim**-.5)
                else:
                    fast_plan_decode(wrapper,indptr,indices,last,seq,heads//groups,1,dim,page,
                        q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=dim**-.5)
                if graph is None:
                    for _ in range(3):wrapper.run(query.reshape(expanded,heads//groups,dim),kv)
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out=wrapper.run(query.reshape(expanded,heads//groups,dim),kv)
                # Deterministic changing queries, identical across variants.
                query.copy_(torch.arange(query.numel(),device="cuda",dtype=torch.float32)
                            .reshape_as(query).sin().mul_(phase+1).to(query.dtype))
                graph.replay();torch.cuda.synchronize()
                restored=out.reshape(batch,heads,dim)
                fresh=wrapper.run(query.reshape(expanded,heads//groups,dim),kv).reshape_as(restored)
                torch.cuda.synchronize()
                torch.testing.assert_close(restored[phase:],fresh[phase:],rtol=.02,atol=5e-4)
                if args.sdpa_reference:
                    import torch.nn.functional as F
                    from torch.nn.attention import SDPBackend,sdpa_kernel
                    refs=[]
                    for i,n in enumerate(lengths):
                        if not n:
                            refs.append(torch.zeros((heads,dim),device="cuda"))
                            continue
                        selected=order[i*per:i*per+counts[i]].long()
                        k=kv[selected,0].permute(1,0,2,3).reshape(1,-1,dim)[:,:n]
                        v=kv[selected,1].permute(1,0,2,3).reshape(1,-1,dim)[:,:n]
                        with sdpa_kernel(SDPBackend.MATH):
                            ref=F.scaled_dot_product_attention(query[i].float()[None,:,None,:],
                                k.float()[None],v.float()[None],enable_gqa=True)[0,:,0]
                        refs.append(ref)
                    reference[phase]=torch.stack(refs)
                if groups==1:
                    if not args.sdpa_reference:reference[phase]=restored.clone()
                torch.testing.assert_close(restored[phase:].float(),reference[phase][phase:].float(),rtol=.02,atol=5e-4)
                times=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):graph.replay()
                    end.record();end.synchronize();times.append(start.elapsed_time(end)/30)
                row=dict(batch=batch,length=args.length,phase=phase,groups=groups,
                    tensor_cores=groups==1,graph_ms=statistics.median(times),repeats_ms=times,
                    max_abs_error=float((restored[phase:]-reference[phase][phase:]).abs().max()),
                    graph_replayed=True,physical_kv_bytes=kv.numel()*kv.element_size(),
                    duplicated_page_indices=True,duplicated_physical_kv=False)
                row["public_plan"]=args.public_plan
                row["nonempty_padding"]=args.nonempty_padding
                rows.append(row);save();print("GQA_GROUP",json.dumps(row),flush=True)
            del graph,wrapper,workspace,indices,out
        del kv,query,reference,order
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(passed=True,rows=rows,finished_time=time.time(),
        scope="Exact query grouping compared with separately SDPA-validated native GQA16 tensor-core decoder, changing lengths/masked rows, fixed-address graph replay; no model/serving speedup claim"),indent=2)+"\n")


if __name__=="__main__":main()
