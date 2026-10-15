# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)
track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


def compute_capacity_aware_tknp_split(
    num_requests: int,
    blocks_per_request: int,
    free_blocks_per_rank: list[int],
) -> list[int]:
    """Return the deterministic contiguous TKNP request split.

    This is shared by the scheduler and CUDA Graph capture so a graph's
    rank-local tensor shapes exactly match the runtime request assignment.
    Rank 0 is capped at the even share (and at its smaller KV capacity),
    while any remainder is distributed over non-root ranks first.
    """
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    if blocks_per_request <= 0:
        raise ValueError(
            f"blocks_per_request must be positive, got {blocks_per_request}"
        )
    if not free_blocks_per_rank:
        raise ValueError("free_blocks_per_rank must not be empty")

    world_size = len(free_blocks_per_rank)
    capacities = [
        max(0, int(free_blocks) // blocks_per_request)
        for free_blocks in free_blocks_per_rank
    ]
    if world_size == 1:
        if capacities[0] < num_requests:
            raise RuntimeError(
                f"TKNP KV capacity is insufficient for {num_requests} requests: "
                f"capacity={capacities}"
            )
        return [num_requests]

    root_count = min(num_requests // world_size, capacities[0])
    remaining = num_requests - root_count
    non_root_base, non_root_remainder = divmod(remaining, world_size - 1)

    counts = [root_count]
    for rank in range(1, world_size):
        target = non_root_base + (1 if rank <= non_root_remainder else 0)
        counts.append(min(target, capacities[rank]))

    shortfall = num_requests - sum(counts)
    if shortfall > 0:
        for rank in range(1, world_size):
            room = capacities[rank] - counts[rank]
            give = min(room, shortfall)
            counts[rank] += give
            shortfall -= give
            if shortfall == 0:
                break

    if shortfall > 0:
        room = capacities[0] - counts[0]
        give = min(room, shortfall)
        counts[0] += give
        shortfall -= give

    if shortfall > 0:
        raise RuntimeError(
            f"TKNP KV capacity is insufficient for {num_requests} requests: "
            f"assigned={counts}, capacity={capacities}, "
            f"unassigned={shortfall}"
        )
    return counts


def tknp_cohort_split_point(total: int) -> int:
    fraction = float(os.environ.get("VLLM_TKNP_DBO_C0_FRACTION", "0.5"))
    if not 0 < fraction < 1:
        raise ValueError("TKNP C0 fraction must be strictly between zero and one")
    if os.environ.get("VLLM_TKNP_DBO_ROOT_REQUESTS") == "0" and fraction != 0.5:
        raise ValueError("Projection-only TKNP roots currently require equal cohorts")
    return max(1, min(total - 1, int(total * fraction)))


def compute_dbo_tknp_split(
    num_requests: int,
    blocks_per_request: int,
    free_blocks_per_rank: list[int],
) -> list[int]:
    """Return a capacity-safe TKNP split for asymmetric two-cohort DBO.

    Attention ranks need at least one request in both cohorts. The root's
    quota is configurable, including zero for a projection-only root. A zero
    root slice uses a padded compiler row but owns no KV or attention work.
    Spread the remaining requests over attention-only ranks so root dense
    compute can overlap the other cohort's attention.

    The helper is shared by scheduling and CUDA Graph capture; consequently
    it must remain deterministic for a fixed capacity vector.
    """
    if os.environ.get("VLLM_TKNP_DBO_LAYOUT", "asymmetric") == "balanced":
        return compute_capacity_aware_tknp_split(
            num_requests, blocks_per_request, free_blocks_per_rank
        )
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    if blocks_per_request <= 0:
        raise ValueError(
            f"blocks_per_request must be positive, got {blocks_per_request}"
        )
    if not free_blocks_per_rank:
        raise ValueError("free_blocks_per_rank must not be empty")

    world_size = len(free_blocks_per_rank)
    if world_size == 1 or num_requests < 2 * world_size:
        return compute_capacity_aware_tknp_split(
            num_requests, blocks_per_request, free_blocks_per_rank
        )

    capacities = [
        max(0, int(free_blocks) // blocks_per_request)
        for free_blocks in free_blocks_per_rank
    ]
    minimum_per_rank = 2
    root_empty = os.environ.get("VLLM_TKNP_DBO_ROOT_REQUESTS") == "0"
    if any(capacity < minimum_per_rank for capacity in capacities[1:]) or (
        not root_empty and capacities[0] < minimum_per_rank
    ):
        raise RuntimeError(
            "TKNP DBO requires capacity for one request per rank in both "
            f"cohorts: capacity={capacities}"
        )

    cohort_0_size = tknp_cohort_split_point(num_requests)
    balanced_root_c0 = max(1, cohort_0_size // world_size)
    root_count = 0 if root_empty else max(minimum_per_rank, int(os.environ.get(
        "VLLM_TKNP_DBO_ROOT_REQUESTS", str(balanced_root_c0 + 1)
    )))
    if root_count > capacities[0]:
        raise RuntimeError(
            "TKNP DBO KV capacity is insufficient on the root rank: "
            f"required={root_count}, capacity={capacities[0]}"
        )

    counts = [root_count] + [minimum_per_rank] * (world_size - 1)
    remaining = num_requests - sum(counts)
    while remaining > 0:
        candidates = [
            rank
            for rank in range(1, world_size)
            if counts[rank] < capacities[rank]
        ]
        if candidates:
            rank = min(
                candidates,
                key=lambda candidate: (counts[candidate], candidate),
            )
        elif not root_empty and counts[0] < capacities[0]:
            # Capacity pressure may require more C0 work on the root.  This is
            # preferable to rejecting a workload which still fits globally.
            rank = 0
        else:
            raise RuntimeError(
                "TKNP DBO KV capacity is insufficient: "
                f"assigned={counts}, capacity={capacities}, "
                f"unassigned={remaining}"
            )
        counts[rank] += 1
        remaining -= 1

    return counts


def build_tknp_cohort_rank_order(
    counts: list[int], num_cohorts: int = 2
) -> list[int]:
    """Build a request-to-rank order with balanced contiguous cohorts.

    TKNP normally assigns one contiguous request block to each rank.  That is
    a poor layout for microbatching because the first half of the scheduler
    batch can contain requests for only a subset of ranks.  This helper keeps
    the same total per-rank allocation while spreading every rank's requests
    as evenly as possible across ``num_cohorts`` contiguous scheduler slices.

    The returned list contains one TKNP rank for each request in scheduler
    order. Cohort sizes differ by at most one request.
    """
    if num_cohorts < 1:
        raise ValueError(f"num_cohorts must be positive, got {num_cohorts}")
    if any(count < 0 for count in counts):
        raise ValueError(f"counts must be non-negative, got {counts}")

    cohort_counts = [
        [count // num_cohorts for _ in range(num_cohorts)] for count in counts
    ]
    cohort_totals = [sum(row[i] for row in cohort_counts)
                     for i in range(num_cohorts)]

    # Place indivisible remainders in the currently smallest cohort. Rotate
    # tie-breaking by rank so rank 0 is not systematically favored.
    for rank, count in enumerate(counts):
        for remainder_idx in range(count % num_cohorts):
            smallest = min(cohort_totals)
            candidates = [
                i for i, total in enumerate(cohort_totals) if total == smallest
            ]
            cohort = candidates[(rank + remainder_idx) % len(candidates)]
            cohort_counts[rank][cohort] += 1
            cohort_totals[cohort] += 1

    order: list[int] = []
    for cohort in range(num_cohorts):
        for rank in range(len(counts)):
            order.extend([rank] * cohort_counts[rank][cohort])

    assert len(order) == sum(counts)
    assert max(cohort_totals, default=0) - min(cohort_totals, default=0) <= 1
    return order


def build_tknp_dbo_cohort_rank_order(counts: list[int]) -> list[int]:
    """Build two contiguous DBO cohorts with a near-empty root C1 slice.

    Attention ranks retain at least one request in each cohort. Rank 0's C1
    ownership is one by default, or zero with ZERO_ROOT_C1. A projection-only
    root owns neither cohort. The scheduler and capture share the same fixed
    split point, including the optional unequal asymmetric cohort sizes.
    """
    if os.environ.get("VLLM_TKNP_DBO_LAYOUT", "asymmetric") == "balanced":
        return build_tknp_cohort_rank_order(counts, num_cohorts=2)
    if len(counts) < 2:
        return build_tknp_cohort_rank_order(counts, num_cohorts=2)
    if counts[0] == 0:
        return build_tknp_cohort_rank_order(counts, num_cohorts=2)
    if any(count < 2 for count in counts):
        return build_tknp_cohort_rank_order(counts, num_cohorts=2)

    total = sum(counts)
    split_point = tknp_cohort_split_point(total)
    cohort_targets = [split_point, total - split_point]
    cohort_counts = [[0, 0] for _ in counts]
    root_c1 = 0 if os.environ.get("VLLM_TKNP_DBO_ZERO_ROOT_C1", "0") == "1" else 1
    cohort_counts[0] = [counts[0] - root_c1, root_c1]
    for rank in range(1, len(counts)):
        cohort_counts[rank] = [counts[rank] // 2, counts[rank] - counts[rank] // 2]
    if root_c1 == 0:
        # Balance the longer C1 attention stage first. Balancing C0 first
        # can turn rank-total remainders into two-request C1 imbalances.
        c1_counts = [0] * len(counts)
        remaining = cohort_targets[1]
        while remaining:
            candidates = [rank for rank in range(1, len(counts))
                          if c1_counts[rank] < counts[rank] - 1]
            if not candidates:
                raise RuntimeError("Zero-root C1 needs nonempty C0 work on attention ranks")
            rank = min(candidates, key=lambda r: (c1_counts[r], r))
            c1_counts[rank] += 1
            remaining -= 1
        cohort_counts = [[count - c1_counts[rank], c1_counts[rank]]
                         for rank, count in enumerate(counts)]

    delta = cohort_targets[0] - sum(row[0] for row in cohort_counts)
    while delta != 0:
        if delta > 0:
            candidates = [
                rank for rank in range(1, len(counts))
                if cohort_counts[rank][1] > 1
            ]
            if not candidates:
                raise RuntimeError(
                    f"Cannot balance TKNP DBO cohorts for counts={counts}"
                )
            rank = min(candidates, key=lambda candidate: (
                cohort_counts[candidate][0], candidate
            ))
            cohort_counts[rank][0] += 1
            cohort_counts[rank][1] -= 1
            delta -= 1
        else:
            candidates = [
                rank for rank in range(1, len(counts))
                if cohort_counts[rank][0] > 1
            ]
            if not candidates:
                raise RuntimeError(
                    f"Cannot balance TKNP DBO cohorts for counts={counts}"
                )
            rank = max(candidates, key=lambda candidate: (
                cohort_counts[candidate][0], -candidate
            ))
            cohort_counts[rank][0] -= 1
            cohort_counts[rank][1] += 1
            delta += 1

    order: list[int] = []
    for cohort in range(2):
        for rank in range(len(counts)):
            order.extend([rank] * cohort_counts[rank][cohort])

    assert len(order) == total
    assert sum(1 for rank in order[cohort_targets[0]:] if rank == 0) == root_c1
    return order

@dataclass
class TokenParallelMetadata:
    """Metadata for layers using Token Parallelism."""

    num_reqs: int
    num_actual_tokens: Optional[int] = None
    stage: str = "prefill"
    tknp_enabled: bool = False
    tknp_rank: Optional[int] = None
    token_parallel_world_size: int = 1
    tokens_per_rank: Optional[list[int]] = None
    rank_start_loc: Optional[list[int]] = None
    rank_end_loc: Optional[list[int]] = None
    request_to_rank: Optional[list[int]] = None
    rank_request_indices: Optional[list[list[int]]] = None
    rank_token_spans: Optional[list[list[tuple[int, int]]]] = None
    local_query_start_loc: Optional[torch.Tensor] = None
    local_seq_lens: Optional[torch.Tensor] = None
    local_num_tokens: Optional[int] = None
    root_rank: int = 0
    _dummy_run: bool = False
    is_cuda_graph: bool = False
    # Tensor whose trailing dimension is the local TKNP token count. During
    # torch.compile this supplies a symbolic local-shard size to TKNP layers,
    # allowing one process to compile several CUDA Graph batch/split variants.
    _compile_local_shape_source: Optional[torch.Tensor] = None


class BatchDescriptor(NamedTuple):
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        """
        Return a relaxed version of current batch descriptor that is still compatible
        with PIECEWISE cudagraphs (or mixed prefill-decode FA cudagraphs).
        """
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    sp_size = len(sp_tokens)

    local_size = [-1] * sp_size
    for i in range(sp_size):
        # Take into account sharding if MoE activation is sequence parallel.
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.

        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.

        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.

        `self.local_sizes` is only valid inside the context.

        Args:
            sequence_parallel_size: When Attn is TP and MoE layers are EP,
                                    we use SP between the layers to avoid
                                    redundant ops. We need this value to
                                    compute the chunked sizes.
            max_chunk_size_per_rank: The max number of tokens each rank is
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        self.local_sizes = _compute_chunked_local_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
            max_chunk_size_per_rank,
            chunk_idx,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # In this case the input to the MoEs will be distributed w.r.t both
    # DP and TP rank.
    # When sp_size==1, this is just the cummulative num tokens across DP.
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None
    
    # TKNP
    # The outer UBatchWrapper context carries one entry per microbatch. Each
    # worker thread installs its individual TokenParallelMetadata before the
    # compiled model is invoked, so model layers still see a scalar object.
    tknp_metadata: Optional[
        TokenParallelMetadata | list[TokenParallelMetadata]
    ] = None

    def __post_init__(self):
        assert self.cudagraph_runtime_mode.valid_runtime_modes(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    tknp_metadata: Optional[
        TokenParallelMetadata | list[TokenParallelMetadata]
    ] = None,
):
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        tknp_metadata=tknp_metadata,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    tknp_metadata: Optional[
        TokenParallelMetadata | list[TokenParallelMetadata]
    ] = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
        attn_metadata is not None or num_tokens is not None
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if num_tokens_across_dp is None:
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
                allow_dp_padding=False,
            )
            assert num_tokens_across_dp is not None
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        tknp_metadata,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform

            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
