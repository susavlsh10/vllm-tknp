"""CUDA-graph test of private symmetric TP scratch on concurrent streams.

Run under torchrun with four local ranks (two-shot), or eight ranks across two
GB200 nodes (multimem). No model weights, requests, or performance claims.
"""
import argparse
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="Use 16/10 MiB at world 4; 64/40 MiB at world 8")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    communicators = [SymmMemCommunicator(dist.group.WORLD, device) for _ in range(2)]
    assert all(not comm.disabled for comm in communicators), "No symmetric backend"
    columns = 4096 if args.large else 1024
    rows0, rows1 = ((8192, 5120) if world == 8 else (2048, 1280)) if args.large else (512, 320)
    inputs = [torch.zeros((rows0, columns), dtype=torch.bfloat16, device=device),
              torch.zeros((rows1, columns), dtype=torch.bfloat16, device=device)]
    outputs = [torch.empty_like(inp) for inp in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    capture_stream = torch.cuda.Stream()

    def run():
        for cohort, stream in enumerate(streams):
            stream.wait_stream(capture_stream)
            with torch.cuda.stream(stream):
                assert communicators[cohort].all_reduce(inputs[cohort], out=outputs[cohort]) is not None
        for stream in streams:
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
            for cohort, inp in enumerate(inputs):
                rows = torch.arange(inp.shape[0], device=device).view(-1, 1) % 8
                inp.copy_(rows + iteration % 7 + cohort + rank * 0.25)
            graph.replay()
        capture_stream.synchronize()
        for cohort, out in enumerate(outputs):
            rows = torch.arange(out.shape[0], device=device).view(-1, 1) % 8
            expected = world * (rows + iteration % 7 + cohort) + 0.25 * world * (world - 1) / 2
            torch.testing.assert_close(out, expected.expand_as(out).to(out.dtype),
                                       rtol=0, atol=0)
    dist.barrier()
    print(f"SYMM_COHORT_ALLREDUCE_PASS rank={rank} world={world} large={args.large} changing_replays=64", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
