"""Four-rank graph-replay transport test, including zero-work root C1."""
import os
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.tknp_symmetric_memory import TKNPSymmetricMemory


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    mailbox = TKNPSymmetricMemory(
        dist.group.WORLD, 2, 32, 128, torch.bfloat16, torch.device("cuda", rank))
    print("HANDLE", rank, mailbox.handle.put_signal.__doc__, flush=True)
    inputs = [torch.arange(12 * 128, device="cuda", dtype=torch.float32).to(torch.bfloat16).view(12, 128)
              for _ in range(2)]
    counts = [[4, 3, 3, 2], [0, 4, 4, 4]]
    recvs = [torch.empty((counts[c][rank], 128), device="cuda", dtype=torch.bfloat16)
             for c in range(2)]
    outputs = [torch.empty((12, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()

    def run():
        second_stream.wait_stream(stream)
        for layer in range(2):
            for cohort in range(2):
                with torch.cuda.stream(stream if cohort == 0 else second_stream):
                    mailbox.scatter(inputs[cohort] if rank == 0 else None,
                                    recvs[cohort], counts[cohort], 0, layer, cohort)
                    attention = recvs[cohort] * 2
                    mailbox.gather(attention, outputs[cohort] if rank == 0 else None,
                                   counts[cohort], 0, layer, cohort)
        stream.wait_stream(second_stream)

    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for iteration in range(1, 65):
        with torch.cuda.stream(stream):
            for cohort in range(2):
                inputs[cohort].copy_(torch.arange(12, device="cuda").view(12, 1)
                                     + iteration + cohort)
            graph.replay()
        stream.synchronize()
        if rank == 0:
            for cohort in range(2):
                torch.testing.assert_close(outputs[cohort], inputs[cohort] * 2, rtol=0, atol=0)
    dist.barrier()
    print(f"SYMMETRIC_MAILBOX_PASS rank={rank} changing_replays=64 concurrent_cohorts=2", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
