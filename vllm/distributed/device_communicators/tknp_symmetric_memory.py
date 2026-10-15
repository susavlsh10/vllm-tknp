"""Experimental one-sided TKNP mailboxes for a direct-NVLink GPU domain.

Each layer/cohort has separate QKV and output storage. The sender writes peer
memory then publishes a release signal; the consumer waits only for its own
mailbox. No remote receive has to be posted before a producer can proceed.
Storage and signal identities are fixed before graph capture. Signal pads use
the backend's consumable semaphore protocol, not host-recorded CUDA events.
"""

import torch
import torch.distributed._symmetric_memory as symm_mem


class TKNPSymmetricMemory:
    def __init__(self, group, layers, rows, cols, dtype, device, backend="CUDA"):
        # The default CUDA allocator is device-registered on first allocation;
        # set_backend only selects the explicitly registered alternative ones.
        if backend != "CUDA":
            symm_mem.set_backend(backend)
        self.layers, self.rows, self.cols = layers, rows, cols
        self.storage = symm_mem.empty(
            (layers, 2, 2, rows * cols), dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.storage, group)
        self.rank = self.handle.rank
        self.peers = [self.handle.get_buffer(
            rank, self.storage.shape, dtype) for rank in range(self.handle.world_size)]
        self.storage.zero_()
        torch.cuda.synchronize()
        # Establish initialization visibility before any producer writes.
        self.handle.barrier()
        torch.cuda.synchronize()

    def _view(self, peer, layer, cohort, direction, rows, cols):
        if layer >= self.layers or rows > self.rows or cols > self.cols:
            raise RuntimeError("TKNP symmetric mailbox capacity exceeded")
        return self.peers[peer][layer, cohort, direction, :rows * cols].view(rows, cols)

    def scatter(self, send, recv, counts, root, layer, cohort):
        cols = recv.size(-1)
        channel = cohort
        if self.rank == root:
            offset = 0
            for peer, count in enumerate(counts):
                if count:
                    chunk = send[offset:offset + count]
                    if peer == root:
                        recv[:count].copy_(chunk)
                    else:
                        self._view(peer, layer, cohort, 0, count, cols).copy_(chunk)
                        self.handle.put_signal(peer, channel=channel)
                offset += count
        elif counts[self.rank]:
            self.handle.wait_signal(root, channel=channel)
            recv.copy_(self._view(self.rank, layer, cohort, 0, recv.size(0), cols))

    def gather(self, send, recv, counts, root, layer, cohort):
        cols = send.size(-1)
        channel = cohort + 2
        if self.rank != root and counts[self.rank]:
            offset = sum(counts[:self.rank])
            target = self._view(root, layer, cohort, 1, sum(counts), cols)
            target[offset:offset + counts[self.rank]].copy_(send)
            self.handle.put_signal(root, channel=channel)
        elif self.rank == root:
            offset = 0
            source = self._view(root, layer, cohort, 1, sum(counts), cols)
            for peer, count in enumerate(counts):
                if count:
                    if peer == root:
                        recv[offset:offset + count].copy_(send)
                    else:
                        self.handle.wait_signal(peer, channel=channel)
                        recv[offset:offset + count].copy_(source[offset:offset + count])
                offset += count
