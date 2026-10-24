"""Changing-input, multi-shape concurrent graph check for private TP scratch.

Run with torchrun and 2 or 8 local H100 ranks, between model workloads.
This is a correctness diagnostic, not a model throughput measurement.
"""
from contextlib import ExitStack
import argparse
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size",type=int,default=4096)
    parser.add_argument("--native-tp-selection",action="store_true",
        help="Check private small-message custom AR plus native symmetric-memory selection")
    args=parser.parse_args()
    if args.hidden_size<1:parser.error("Positive hidden size required")
    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    world=dist.get_world_size()
    options=dict(max_size=256*1024) if args.native_tp_selection else {}
    comms=[CustomAllreduce(dist.group.WORLD,torch.device("cuda",rank),**options) for _ in range(2)]
    assert all(not c.disabled for c in comms),"Private custom all-reduce unavailable"
    symm=([SymmMemCommunicator(dist.group.WORLD,torch.device("cuda",rank)) for _ in range(2)]
          if args.native_tp_selection else None)
    if symm is not None:assert all(not c.disabled for c in symm)
    def reduce(cohort,x):
        if comms[cohort].should_custom_ar(x):return comms[cohort].custom_all_reduce(x)
        assert symm is not None and symm[cohort].should_use_symm_mem(x)
        return symm[cohort].all_reduce(x)
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    inputs={b:[torch.zeros(b,args.hidden_size,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
            for b in ((4,48,128) if args.native_tp_selection else (8,64,128))}
    for b in inputs:
        for cohort,x in enumerate(inputs[b]):
            for _ in range(3):reduce(cohort,x)
    torch.cuda.synchronize();dist.barrier()
    graphs={};outputs={}
    with ExitStack() as contexts:
        for c in comms:contexts.enter_context(c.capture())
        for b in inputs:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=streams[0]):
                streams[1].wait_stream(streams[0])
                results=[]
                for cohort in range(2):
                    with torch.cuda.stream(streams[cohort]):
                        results.append(reduce(cohort,inputs[b][cohort]))
                streams[0].wait_stream(streams[1])
            graphs[b]=graph;outputs[b]=results
    for iteration in range(1,65):
        b=tuple(reversed(inputs))[iteration%3]
        # Keep every partial sum below 256: BF16 integer addition is then
        # exact regardless of reduction order, including at world size eight.
        for cohort,x in enumerate(inputs[b]):x.fill_(rank+iteration%16+cohort*8)
        streams[0].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):graphs[b].replay()
        streams[0].synchronize()
        for cohort,out in enumerate(outputs[b]):
            target=world*(iteration%16+cohort*8)+world*(world-1)//2
            torch.testing.assert_close(out,torch.full_like(out,target),rtol=0,atol=0)
    # Also check non-integer activations against a CPU FP32 collective.
    # These are correctness controls, never part of performance timing.
    for b in inputs:
        for x in inputs[b]:x.normal_(0,.01)
        streams[0].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):graphs[b].replay()
        streams[0].synchronize()
        for cohort,x in enumerate(inputs[b]):
            reference=x.float().cpu()
            dist.all_reduce(reference)
            torch.testing.assert_close(outputs[b][cohort].float().cpu(),reference,rtol=.02,atol=3e-4)
    dist.barrier()
    print(f"PRIVATE_COHORT_CUSTOM_PASS rank={rank} world={world} changing_replays=64 graph_shapes=3 hidden={args.hidden_size} random_fp32_checks=6 max_bytes={comms[0].max_size} native_tp_selection={args.native_tp_selection}",flush=True)
    for c in comms:c.close()
    dist.destroy_process_group()


if __name__=="__main__":main()
