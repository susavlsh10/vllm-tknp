# SPDX-License-Identifier: Apache-2.0
"""Deterministic, per-request lifetime KV admission (no GPU imports)."""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class PlacementRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int

    def blocks(self, block_size: int) -> int:
        return ceil((self.prompt_tokens + self.output_tokens) / block_size)


def unreserved_lifetime_blocks(available_blocks, block_size, live_allocations):
    """Physical free blocks minus growth of KV-admitted live requests.

    Entries are (prompt tokens, maximum outputs, currently allocated blocks).
    Zero-allocation waiting requests are NOT admitted and reserve nothing.
    In-flight remote KV receives have physical blocks, so their future growth
    counts even though they are not in the running batch yet.
    """
    if available_blocks < 0 or block_size <= 0:
        raise ValueError("Invalid KV capacity")
    future = sum(max(0, ceil((prompt + outputs) / block_size) - allocated)
                 for prompt, outputs, allocated in live_allocations if allocated > 0)
    return max(0, available_blocks - future)


def place_fifo(requests, available_blocks, block_size, loads=None, policy="cost", request_limits=None):
    """Reserve lifetime blocks; stop at the first non-fitting request.

    available_blocks must exclude future growth reservations of live requests.
    This planner never migrates KV. `count` balances requests, `memory` balances
    reserved blocks, and `cost` uses summed context lengths as a decode-attention
    work proxy. These are explicit experimental policies, not learned costs.
    """
    if policy not in {"count", "memory", "cost"}:
        raise ValueError(f"Unknown TKNP placement policy {policy}")
    if block_size <= 0 or not available_blocks or any(x < 0 for x in available_blocks):
        raise ValueError("Invalid KV capacity")
    remaining = list(available_blocks)
    costs = list(loads or [0] * len(remaining))
    if len(costs) != len(remaining):
        raise ValueError("One load is required per owner")
    assignments = {}
    limits = list(request_limits) if request_limits is not None else [len(requests)] * len(remaining)
    if len(limits) != len(remaining) or any(limit < 0 for limit in limits):
        raise ValueError("Invalid graph lane limits")
    for req in requests:
        if req.prompt_tokens <= 0 or req.output_tokens <= 0:
            raise ValueError("Prompt and output lengths must be positive")
        needed = req.blocks(block_size)
        feasible = [rank for rank, free in enumerate(remaining) if free >= needed and limits[rank] > 0]
        if not feasible:
            break
        rank = min(feasible, key=lambda r: (costs[r], r))
        assignments[req.request_id] = rank
        remaining[rank] -= needed
        limits[rank] -= 1
        costs[rank] += (1 if policy == "count" else needed if policy == "memory"
                        else req.prompt_tokens + req.output_tokens)
    return assignments
