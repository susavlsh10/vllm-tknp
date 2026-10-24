# SPDX-License-Identifier: Apache-2.0
"""Fixed graph lanes, dynamic request occupants. No fake engine requests/KV."""
from collections import deque
from dataclasses import dataclass


def normalize_capture_layouts(layouts, world_size):
    """One immutable ownership signature per global decode graph size."""
    if not isinstance(layouts, dict) or not layouts:
        raise ValueError("Capture layouts must be a nonempty size-to-owner-count mapping")
    result = {}
    for key, counts in layouts.items():
        size = int(key)
        if (str(size) != str(key) or size < world_size or size % world_size
                or size in result or not isinstance(counts, list)
                or len(counts) != world_size
                or any(type(n) is not int or n < 0 for n in counts)
                or any(n == 0 for n in counts[1:]) or sum(counts) != size):
            raise ValueError("Each layout needs an aligned size and matching nonnegative owner quotas; attention owners must be nonempty")
        result[size] = list(counts)
    return dict(sorted(result.items()))


def select_capture_layout(layouts, size, *, remaining_requests=0,
                          batch_queue_depth=0, all_kv_free=True):
    """Switch only at drained boundaries; never move live request ownership."""
    if remaining_requests or batch_queue_depth or not all_kv_free:
        raise ValueError("Drain requests, communication and KV before switching graph layout")
    if type(size) is not int or size not in layouts:
        raise ValueError("Select an exactly captured TKNP graph size")
    return list(layouts[size])


def select_resident_replay_size(layouts, ceiling, request_owners):
    """Smallest captured graph covering sticky owners, without changing admission.

    This selects a replay ONLY. It never changes ownership, KV placement,
    scheduler slot limits, or the active admission layout. Empty graph lanes
    can therefore shrink during a tail wave without moving any live KV.
    """
    if ceiling not in layouts:
        raise ValueError("Admission ceiling must be an exactly captured layout")
    counts = [0]*len(layouts[ceiling])
    for owner in request_owners:
        if type(owner) is not int or not 0<=owner<len(counts):
            raise ValueError("Invalid resident request owner")
        counts[owner] += 1
    for size, quotas in sorted(layouts.items()):
        if size<=ceiling and all(n<=q for n,q in zip(counts,quotas)):
            return size
    raise ValueError("Resident owners exceed captured replay quotas")


def cached_graph_inventory(model):
    """Inspect wrappers without forwarding __getattr__ through the model."""
    result, visited = {}, set()
    def visit(wrapper, path):
        if wrapper is None or id(wrapper) in visited:
            return
        visited.add(id(wrapper))
        attributes = vars(wrapper)
        for field in ("concrete_cudagraph_entries", "cudagraphs"):
            for key, entry in attributes.get(field, {}).items():
                graph = getattr(entry, "cudagraph", None)
                if graph is not None:
                    result[path + "/" + field + "/" + repr(key)] = id(graph)
        for field in ("cudagraph_wrapper", "runnable"):
            visit(attributes.get(field), path + "/" + field)
    visit(model, "model")
    return result


def empty_root_position_rows(local_counts):
    """Compiler padding only: keep zero-work C1's positions as wide as C0."""
    return max(1, sum(int(count or 0) for count in local_counts))


def validate_multi_layout_dbo_shapes(cohorts):
    """Keep singleton specialization out of the shared compiled DBO path."""
    root_rows = []
    for size, counts in cohorts.items():
        root_rows.append(max(1, sum(counts[0])))
        for owner, (a, b) in enumerate(counts[1:], 1):
            if min(a, b) < 2:
                raise ValueError(f"Compiled multi-layout TKNP needs >=2 attention rows per cohort: size={size}, owner={owner}, counts={(a, b)}")
    if min(root_rows) == 1 and max(root_rows) > 1:
        raise ValueError("Do not mix singleton and nonsingleton root position shapes in a compiled layout table")


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
        # A zero quota is intentional (e.g. a projection-only root), not
        # spare capacity to introduce a new attention/compile path at startup.
        eligible = [i for i, cap in enumerate(caps) if counts[i] and result[i] < cap]
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
