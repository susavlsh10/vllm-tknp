#!/usr/bin/env python3
"""Isolated attention diagnostic; run between, never alongside, model runs.

Same BF16 paged KV/queries for each kernel variant. Timed CUDA graph replays,
not end-to-end model/serving throughput. Fixed-split controls are diagnostics
only: they do not establish graph correctness for changing sequence lengths.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode
    import h100_cc_fixed_plan as fixed_planner
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gqa16",action="store_true",
                   help="405B TP8 local attention shapes, public default planners only")
    args=p.parse_args()
    metadata=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        prototype_planner_sha256=hashlib.sha256(Path(fixed_planner.__file__).read_bytes()).hexdigest())
    torch.manual_seed(20261018)
    rows=[];length,page,dim=32768,16,128
    shapes=(((16,1,[2,4,5,6,8,10,12,16,20,24,28,32,34]),) if args.gqa16 else
            ((32,8,[4,8,12,13,15,16,17,18,24,32]),
             (16,4,[16,24,31,32,33,34,35,36,48,64]),
             (4,1,[64,96,120,128,131,132,133,135,136,144])))
    for qheads,kvheads,batches in shapes:
        pages_per=math.ceil(length/page)
        kv=torch.randn(max(batches)*pages_per,2,kvheads,page,dim,device="cuda",dtype=torch.bfloat16)
        indices=torch.randperm(kv.shape[0],device="cuda",dtype=torch.int32)
        for batch in batches:
            q=torch.randn(batch,qheads,dim,device="cuda",dtype=torch.bfloat16)
            indptr=torch.arange(batch+1,dtype=torch.int32)*pages_per
            last=torch.full((batch,),page,dtype=torch.int32)
            seq=torch.full((batch,),length,dtype=torch.int32)
            reference=None
            # Default tensor-core path versus default CUDA-core path first.
            variants=((False,None,None),(True,None,None)) if args.gqa16 else (
                (False,None,None),(True,None,None),(True,128,None),(True,512,None),
                (False,None,128),(False,None,256),(False,None,512))
            for tc,split,cc_split in variants:
                workspace=torch.empty(128*1024**2,device="cuda",dtype=torch.uint8)
                wrapper=BatchDecodeWithPagedKVCacheWrapper(workspace,"HND",use_cuda_graph=True,
                    paged_kv_indptr_buffer=torch.zeros(batch+1,device="cuda",dtype=torch.int32),
                    paged_kv_indices_buffer=indices[:batch*pages_per],
                    paged_kv_last_page_len_buffer=torch.zeros(batch,device="cuda",dtype=torch.int32),
                    use_tensor_cores=tc)
                fast_plan_decode(wrapper,indptr,indices[:batch*pages_per],last,seq,qheads,kvheads,dim,page,
                    q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=dim**-.5,
                    fixed_split_size=-1 if split is None else split)
                if cc_split:
                    fixed_planner.fixed_split_decode_plan(wrapper,indptr,indices[:batch*pages_per],last,
                        qheads,dim,page,cc_split,pages_per)
                for _ in range(3):wrapper.run(q,kv)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=wrapper.run(q,kv)
                graph.replay();torch.cuda.synchronize()
                if reference is None:reference=out.clone()
                torch.testing.assert_close(out,reference,rtol=.02,atol=5e-4)
                times=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):graph.replay()
                    end.record();end.synchronize();times.append(start.elapsed_time(end)/30)
                ms=statistics.median(times)
                row=dict(batch=batch,q_heads=qheads,kv_heads=kvheads,head_request_pairs=batch*kvheads,
                    tensor_cores=tc,fixed_split_pages=split,cc_split_pages=cc_split,graph_ms=ms,repeats_ms=times,
                    kv_read_gb=batch*length*kvheads*dim*4/1e9,
                    effective_kv_tb_per_s=batch*length*kvheads*dim*4/(ms*1e9),
                    max_abs_error=float((out-reference).abs().max()),plan_info=list(wrapper._plan_info))
                rows.append(row);print("BATCH_KERNEL",json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(passed=False,metadata=metadata,rows=rows),indent=2)+"\n")
                del graph,wrapper,workspace,out
            del q,reference
        del kv,indices
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(passed=True,metadata=metadata,rows=rows,
        scope="Single-layer 32K decode, same random BF16 KV/query, graph replay timing; effective KV rate is a payload estimate, not measured HBM traffic; fixed splits not approved for changing-length model graphs"),indent=2)+"\n")


if __name__=="__main__":main()
