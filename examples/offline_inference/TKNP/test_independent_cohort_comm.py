"""Four-GPU NCCL graph test with changing, row-distinct cohort payloads."""
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.model_executor.layers.token_parallel_linear import (
    _gather_to_root_nccl,
    _scatter_from_root_nccl,
)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    device = torch.device("cuda", rank)
    comms = [PyNcclCommunicator(dist.group.WORLD, device=device) for _ in range(2)]
    assert all(comm.available for comm in comms)
    counts = [[4, 3, 3, 2], [0, 4, 4, 4]]
    inputs = [torch.zeros((12, 128), device=device, dtype=torch.bfloat16) for _ in range(2)]
    recvs = [[torch.empty((counts[c][rank], 128), device=device, dtype=torch.bfloat16)
              for c in range(2)] for _ in range(2)]
    outputs = [[torch.empty_like(inputs[c]) for c in range(2)] for _ in range(2)]
    compute = [torch.cuda.Stream() for _ in range(2)] if rank == 0 else [torch.cuda.Stream()] * 2
    communication = [torch.cuda.Stream() for _ in range(2)]
    ready = [torch.cuda.Event() for _ in range(2)]
    received = [torch.cuda.Event() for _ in range(2)]
    attention_done = [torch.cuda.Event() for _ in range(2)]
    gathered = [torch.cuda.Event() for _ in range(2)]
    capture_stream = torch.cuda.Stream()

    def run():
        for stream in set(compute):
            stream.wait_stream(capture_stream)
        for layer in range(2):
            for cohort in range(2):
                with torch.cuda.stream(compute[cohort]):
                    source = inputs[cohort] if layer == 0 else outputs[layer - 1][cohort]
                    if layer:
                        compute[cohort].wait_event(gathered[cohort])
                    ready[cohort].record()
                with torch.cuda.stream(communication[cohort]):
                    communication[cohort].wait_event(ready[cohort])
                    _scatter_from_root_nccl(source if rank == 0 else None,
                                            recvs[layer][cohort], counts[cohort], 0, comms[cohort])
                    received[cohort].record()
                with torch.cuda.stream(compute[cohort]):
                    compute[cohort].wait_event(received[cohort])
                    recvs[layer][cohort].mul_(2)
                    attention_done[cohort].record()
                with torch.cuda.stream(communication[cohort]):
                    communication[cohort].wait_event(attention_done[cohort])
                    _gather_to_root_nccl(recvs[layer][cohort],
                                        outputs[layer][cohort] if rank == 0 else None,
                                        counts[cohort], 0, comms[cohort])
                    gathered[cohort].record()
        for stream in communication:
            capture_stream.wait_stream(stream)

    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            run()
    capture_stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run()
    for iteration in range(64):
        with torch.cuda.stream(capture_stream):
            for cohort in range(2):
                inputs[cohort].copy_(torch.arange(12, device=device).view(12, 1)
                                     + iteration + cohort)
            graph.replay()
        capture_stream.synchronize()
        if rank == 0:
            for cohort in range(2):
                torch.testing.assert_close(outputs[1][cohort], inputs[cohort] * 4,
                                           rtol=0, atol=0)
    dist.barrier()
    print(f"INDEPENDENT_COHORT_NCCL_PASS rank={rank} changing_replays=64 layers=2", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
