#!/usr/bin/env python3
"""Single-GPU numerical and changing-metadata graph check for decode kernels.

Run only between model experiments. Compares both FlashInfer decode paths
with FP32 PyTorch math SDPA on the SAME random physical paged KV and queries.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--q-heads",type=int,default=32)
    p.add_argument("--kv-heads",type=int,default=8)
    p.add_argument("--variants",nargs="+",choices=["tc","cc"],default=["tc","cc"])
    args=p.parse_args()
    if args.kv_heads<1 or args.q_heads<args.kv_heads or args.q_heads%args.kv_heads:
        p.error("Query heads must be a positive multiple of KV heads")
    torch.manual_seed(20261018)
    device=torch.device("cuda:0")
    batch,kv_heads,q_heads,dim,page=4,args.kv_heads,args.q_heads,128,16
    max_seq=32768
    num_pages=batch*math.ceil(max_seq/page)
    kv=torch.randn(num_pages,2,kv_heads,page,dim,device=device,dtype=torch.bfloat16)
    query=torch.randn(batch,q_heads,dim,device=device,dtype=torch.bfloat16)
    queries=torch.randn(3,batch,q_heads,dim,device=device,dtype=torch.bfloat16)
    page_order=torch.randperm(num_pages,device=device,dtype=torch.int32)
    workspace=torch.empty(128*1024**2,device=device,dtype=torch.uint8)
    results=[]
    for tensor_cores in [name=="tc" for name in args.variants]:
        indptr_buf=torch.zeros(batch+1,device=device,dtype=torch.int32)
        indices_buf=torch.zeros(num_pages,device=device,dtype=torch.int32)
        last_buf=torch.zeros(batch,device=device,dtype=torch.int32)
        wrapper=BatchDecodeWithPagedKVCacheWrapper(workspace,"HND",use_cuda_graph=True,
            paged_kv_indptr_buffer=indptr_buf,paged_kv_indices_buffer=indices_buf,
            paged_kv_last_page_len_buffer=last_buf,use_tensor_cores=tensor_cores)
        graph=None;output=None
        for phase,lengths in enumerate(([32768,24561,16385,8193],[0,32761,8195,4097],[32753,12289,0,16387])):
            pages=[math.ceil(n/page) for n in lengths]
            indptr=torch.tensor([0]+list(__import__('itertools').accumulate(pages)),dtype=torch.int32)
            last=torch.tensor([((n-1)%page+1) if n else 0 for n in lengths],dtype=torch.int32)
            seq=torch.tensor(lengths,dtype=torch.int32)
            indices_buf.zero_()
            indices_buf[:sum(pages)].copy_(page_order[:sum(pages)])
            fast_plan_decode(wrapper,indptr,indices_buf,last,seq,q_heads,kv_heads,dim,page,
                q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=dim**-.5)
            query.copy_(queries[phase])
            if graph is None:
                for _ in range(3):wrapper.run(query,kv)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=wrapper.run(query,kv)
            graph.replay();torch.cuda.synchronize()
            errors=[]
            for i,n in enumerate(lengths):
                if not n:continue
                physical=page_order[indptr[i]:indptr[i+1]].long()
                k=kv[physical,0].permute(1,0,2,3).reshape(kv_heads,-1,dim)[:,:n]
                v=kv[physical,1].permute(1,0,2,3).reshape(kv_heads,-1,dim)[:,:n]
                k=k.repeat_interleave(q_heads//kv_heads,dim=0)
                v=v.repeat_interleave(q_heads//kv_heads,dim=0)
                with sdpa_kernel(SDPBackend.MATH):
                    ref=F.scaled_dot_product_attention(query[i].float()[None,:,None,:],
                        k.float()[None],v.float()[None],is_causal=False)[0,:,0]
                torch.testing.assert_close(output[i].float(),ref,atol=5e-4,rtol=.02)
                errors.append(float((output[i].float()-ref).abs().max()))
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(30):graph.replay()
            end.record();end.synchronize()
            row=dict(tensor_cores=tensor_cores,phase=phase,lengths=lengths,
                q_heads=q_heads,kv_heads=kv_heads,
                max_abs_error=max(errors),kernel_graph_ms=begin.elapsed_time(end)/30,
                graph_replayed=True,empty_rows_present=any(n==0 for n in lengths),active_outputs_verified=True)
            results.append(row);print("DECODE_KERNEL_PASS",json.dumps(row),flush=True)
        del graph,wrapper,output
    args.output.write_text(json.dumps(dict(passed=True,tests=results,
        scope="BF16 paged KV, FP32 math SDPA reference, changing lengths/empty rows, fixed-address graph replay; single-layer diagnostic, not model throughput"),indent=2)+"\n")


if __name__=="__main__":main()
