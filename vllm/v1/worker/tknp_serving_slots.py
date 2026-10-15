# SPDX-License-Identifier: Apache-2.0
"""Fixed graph lanes, dynamic request occupants. No fake engine requests/KV."""
from collections import deque
from dataclasses import dataclass


def validate_capture_split(counts, total, free_blocks, blocks_per_request):
    """Explicit owner quotas for matched overlap-on/off graph ablations."""
    if (len(counts) != len(free_blocks) or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != total or blocks_per_request <= 0
            or any(n * blocks_per_request > free for n, free in zip(counts, free_blocks))):
        raise ValueError("Capture owner counts must match graph size and per-owner KV capacity")
    return list(counts)


def fit_capture_split(counts, total, free_blocks, blocks_per_request):
    """Opt-in startup repair against actual pools, retaining exact graph size.

    Profiling at a larger graph shape can move a page between owner pools.
    Preserve valid explicit quotas; otherwise move only the overflowing lanes.
    Never shrink the graph or overcommit physical KV capacity.
    """
    if (len(counts) != len(free_blocks) or sum(counts) != total
            or blocks_per_request <= 0 or any(type(n) is not int or n < 0 for n in counts)):
        return validate_capture_split(counts, total, free_blocks, blocks_per_request)
    caps = [max(0, (free - 1) // blocks_per_request) for free in free_blocks]
    result = [min(n, cap) for n, cap in zip(counts, caps)]
    for _ in range(total - sum(result)):
        eligible = [i for i, cap in enumerate(caps) if result[i] < cap]
        if not eligible:
            raise ValueError("Actual KV pools cannot fit the requested graph lanes")
        owner = min(eligible, key=lambda i: (result[i], i))
        result[owner] += 1
    return validate_capture_split(result, total, free_blocks, blocks_per_request)


@dataclass
class SlotPlan:
    owner_layout: list[int]
    request_slots: list[int]
    local_request_indices: list[int | None]


def plan_slots(request_owners, owner_layout, rank):
    """Map logical requests into captured rank/cohort lanes.

    Ownership stays sticky, but a request can move between its owner's cohort
    lanes at a step boundary. Only metadata and QKV row positions move, not KV.
    Padding lanes have no request and must use slot_mapping=-1/seq_len=0.
    """
    free = {}
    for slot, owner in enumerate(owner_layout):
        free.setdefault(owner, deque()).append(slot)
    request_slots = []
    slot_requests = {}
    for index, owner in enumerate(request_owners):
        if owner not in free or not free[owner]:
            raise ValueError(f"Owner {owner} exceeds its captured graph lanes")
        slot = free[owner].popleft()
        request_slots.append(slot)
        slot_requests[slot] = index
    local = [slot_requests.get(slot) for slot, owner in enumerate(owner_layout) if owner == rank]
    return SlotPlan(list(owner_layout), request_slots, local)
