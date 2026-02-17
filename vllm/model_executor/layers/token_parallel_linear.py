"""
Token Parallel Linear Layers for vLLM.

This module implements token parallel versions of linear layers used in attention.
In token parallelism, only the root rank (rank 0) in each token parallel group 
loads weights and computes projections, while other ranks focus on attention 
computation for their assigned token partitions.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass, fields

from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.distributed.parallel_state import (
    get_tknp_rank, 
    get_tknp_world_size, 
    get_tknp_group, 
    is_tknp_initialized
)

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)

from vllm.forward_context import (
    ForwardContext,
    TokenParallelMetadata,
    get_forward_context,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: query max_num_batched_tokens from the current VllmConfig
# ---------------------------------------------------------------------------

def _get_max_num_batched_tokens() -> int:
    """Return the configured max_num_batched_tokens, or a safe default."""
    try:
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if config is not None and config.scheduler_config is not None:
            return config.scheduler_config.max_num_batched_tokens
    except Exception:
        pass
    return 2048  # SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS


# ---------------------------------------------------------------------------
# Helper: extract the PyNccl communicator from a TKNP GroupCoordinator
# ---------------------------------------------------------------------------

def _get_pynccl_comm(tknp_group):
    """Extract the PyNcclCommunicator from a TKNP GroupCoordinator, if available."""
    if tknp_group is None:
        return None
    dc = getattr(tknp_group, 'device_communicator', None)
    if dc is None:
        return None
    return getattr(dc, 'pynccl_comm', None)


# ---------------------------------------------------------------------------
# Pre-allocated communication buffer pool
# ---------------------------------------------------------------------------

class _CommBufferPool:
    """
    A simple, per-layer pool of pre-allocated GPU tensors used as
    communication buffers for scatter / gather operations.

    Buffers are allocated once (at the configured *max_num_batched_tokens*
    size) and re-used on every forward call, eliminating per-iteration
    ``torch.empty`` allocations on the hot path.

    If a forward call requires a buffer larger than the pre-allocated one
    (should not happen under normal scheduling), the pool transparently
    falls back to a fresh allocation and up-sizes its internal buffer for
    future calls.
    """

    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        name: str,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a buffer of *at least* ``(rows, cols)`` elements.

        The returned tensor is a **narrow view** with the exact requested
        shape ``(rows, cols)``; the underlying storage may be larger.
        """
        key = name
        buf = self._buffers.get(key)

        # Allocate / re-allocate when the existing buffer is too small or
        # has a mismatched dtype / device.
        if (
            buf is None
            or buf.size(0) < rows
            or buf.size(1) != cols
            or buf.dtype != dtype
            or buf.device != device
        ):
            # Allocate with some headroom so we don't re-allocate for
            # small fluctuations.
            alloc_rows = max(rows, buf.size(0) if buf is not None else 0)
            new_buf = torch.empty(
                (alloc_rows, cols), dtype=dtype, device=device
            )
            self._buffers[key] = new_buf
            buf = new_buf

        # Return an exact-sized view (zero-cost, shares storage).
        return buf[:rows]


# ---------------------------------------------------------------------------
# NCCL grouped send / recv primitives
# ---------------------------------------------------------------------------

def _scatter_from_root_nccl(
    send_buf: Optional[torch.Tensor],
    recv_buf: torch.Tensor,
    tokens_per_rank: List[int],
    root_rank: int,
    pynccl_comm,
):
    """
    Scatter variable-sized chunks from root to all ranks using NCCL
    grouped send/recv.
    """
    rank = pynccl_comm.rank
    pynccl_comm.group_start()
    if rank == root_rank:
        offset = 0
        for dst_rank, count in enumerate(tokens_per_rank):
            if count == 0:
                continue
            chunk = send_buf[offset:offset + count]
            if dst_rank == root_rank:
                recv_buf.copy_(chunk)
            else:
                pynccl_comm.send(chunk, dst_rank)
            offset += count
    if rank != root_rank and tokens_per_rank[rank] > 0:
        pynccl_comm.recv(recv_buf, root_rank)
    pynccl_comm.group_end()


def _gather_to_root_nccl(
    send_buf: torch.Tensor,
    recv_buf: Optional[torch.Tensor],
    tokens_per_rank: List[int],
    root_rank: int,
    pynccl_comm,
):
    """
    Gather variable-sized chunks from all ranks to root using NCCL
    grouped send/recv.
    """
    rank = pynccl_comm.rank
    pynccl_comm.group_start()
    if rank == root_rank:
        offset = 0
        for src_rank, count in enumerate(tokens_per_rank):
            if count == 0:
                continue
            dst_slice = recv_buf[offset:offset + count]
            if src_rank == root_rank:
                dst_slice.copy_(send_buf)
            else:
                pynccl_comm.recv(dst_slice, src_rank)
            offset += count
    if rank != root_rank and tokens_per_rank[rank] > 0:
        pynccl_comm.send(send_buf, root_rank)
    pynccl_comm.group_end()


# ===================================================================
# TokenParallelQKVLinear
# ===================================================================

class TokenParallelQKVLinear(QKVParallelLinear):
    """
    QKV projection layer with Token Parallelism.

    Only root rank (rank 0) holds weights and computes the projection.
    The QKV output is then scattered to all ranks via pre-allocated buffers
    and NCCL grouped send/recv for minimal overhead.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)

        # ---- token-parallel topology ---------------------------------
        self.is_tknp_enabled = is_tknp_initialized()
        self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
        self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
        self.is_root_rank = (self.tknp_rank == 0)
        self.tknp_group = get_tknp_group() if self.is_tknp_enabled else None
        self.pg = (self.tknp_group.device_group
                   if hasattr(self.tknp_group, 'device_group')
                   else self.tknp_group)
        self.device = (torch.device('cuda', torch.cuda.current_device())
                       if torch.cuda.is_available()
                       else torch.device('cpu'))
        self.quant_method = None

        # ---- fast NCCL path ------------------------------------------
        use_pynccl = True  # Set to False to disable NCCL path and test fallback
        self._pynccl = _get_pynccl_comm(self.tknp_group) if use_pynccl else None

        self.params_dtype = kwargs.get('params_dtype', None)
        if self.params_dtype is None:
            self.params_dtype = torch.get_default_dtype()

        # ---- root vs non-root initialisation -------------------------
        if self.is_root_rank:
            super().__init__(*args, **kwargs)
        else:
            hidden_size = kwargs.get('hidden_size', args[0] if args else None)
            head_size = kwargs.get('head_size', args[1] if args else None)
            total_num_heads = kwargs.get('total_num_heads',
                                        args[2] if len(args) > 2 else None)
            total_num_kv_heads = kwargs.get('total_num_kv_heads',
                                           args[3] if len(args) > 3 else None)

            self.hidden_size = hidden_size
            self.head_size = head_size
            self.total_num_heads = total_num_heads
            self.total_num_kv_heads = total_num_kv_heads

            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.num_heads = self.total_num_heads // self.tp_size
            self.num_kv_heads = self.total_num_kv_heads // self.tp_size
            self.qkv_size_per_partition = (
                (self.num_heads + 2 * self.num_kv_heads) * self.head_size
            )

            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # ---- pre-allocate communication buffers ----------------------
        max_tokens = _get_max_num_batched_tokens()
        qkv_dim = self.qkv_size_per_partition if not self.is_root_rank else (
            (self.num_heads + 2 * self.num_kv_heads) * self.head_size
            if hasattr(self, 'num_heads') else 0
        )

        self._buf_pool = _CommBufferPool()
        if self.is_tknp_enabled and qkv_dim > 0:
            # Pre-warm the recv buffer (every rank needs one).
            self._buf_pool.get(
                "qkv_recv", max_tokens, qkv_dim,
                self.params_dtype, self.device,
            )

        logger.debug(
            "TokenParallelQKVLinear init  rank=%d  root=%s  pynccl=%s  "
            "max_tokens=%d  qkv_dim=%d",
            self.tknp_rank, self.is_root_rank,
            self._pynccl is not None, max_tokens, qkv_dim,
        )

    # ------------------------------------------------------------------ #
    #  forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_tknp_enabled or self.tknp_world_size == 1:
            return super().forward(x)

        metadata = getattr(get_forward_context(), "tknp_metadata", None)
        if (metadata is None or metadata._dummy_run
                or metadata.tokens_per_rank is None):
            return self._fallback_forward(x, metadata)

        tokens_per_rank = metadata.tokens_per_rank
        total_tokens = sum(tokens_per_rank)
        root_rank = (metadata.root_rank
                     if metadata.root_rank is not None else 0)

        # ---- validation ----------------------------------------------
        if self.is_root_rank:
            if x is None:
                raise RuntimeError(
                    "Root rank must receive hidden states for TKNP prefill.")
            if total_tokens != x.size(0):
                assert False, (
                    "TokenParallelQKVLinear: total tokens mismatch on root rank.")
        elif x is not None and total_tokens != x.size(0):
            return self._fallback_forward(x, metadata)

        # ---- compute QKV on root -------------------------------------
        if self.is_root_rank:
            qkv_full, _ = super().forward(x)
            qkv_dim = qkv_full.size(1)
            dtype = qkv_full.dtype
        else:
            qkv_full = None
            qkv_dim = self.qkv_size_per_partition
            dtype = self.params_dtype

        # ---- obtain recv buffer from the pool ------------------------
        recv_count = tokens_per_rank[self.tknp_rank]
        qkv_local = self._buf_pool.get(
            "qkv_recv", recv_count, qkv_dim, dtype, self.device,
        )

        # ---- scatter -------------------------------------------------
        if self._pynccl is not None and not self._pynccl.disabled:
            _scatter_from_root_nccl(
                send_buf=qkv_full,
                recv_buf=qkv_local,
                tokens_per_rank=tokens_per_rank,
                root_rank=root_rank,
                pynccl_comm=self._pynccl,
            )
        else:
            # Fallback: torch.distributed all_to_all_single
            if self.is_root_rank:
                send_splits = tokens_per_rank
                recv_splits = [0] * self.tknp_world_size
                recv_splits[self.tknp_rank] = recv_count
                dist.all_to_all_single(
                    qkv_local, qkv_full,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=self.pg,
                )
            else:
                send_splits = [0] * self.tknp_world_size
                recv_splits = [0] * self.tknp_world_size
                recv_splits[root_rank] = recv_count
                empty_send = torch.empty(
                    (0, qkv_dim), dtype=dtype, device=self.device,
                )
                dist.all_to_all_single(
                    qkv_local, empty_send,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=self.pg,
                )

        return qkv_local, None

    # ------------------------------------------------------------------ #
    #  fallback (dummy runs / legacy path)                                 #
    # ------------------------------------------------------------------ #

    def _fallback_forward(
        self,
        x: torch.Tensor,
        metadata: Optional[TokenParallelMetadata],
    ) -> tuple[torch.Tensor, None]:
        """Fallback path that mimics the legacy even scatter.  Dummy runs only."""
        if not metadata._dummy_run:
            logger.info(
                "Rank %d: falling back to legacy TokenParallelQKVLinear forward.",
                self.tknp_rank,
            )
        if self.is_root_rank:
            if x is None:
                raise RuntimeError("Root rank received no input tensors.")
            qkv_full, _ = super().forward(x)
            qkv_chunks = list(
                torch.chunk(qkv_full, self.tknp_world_size, dim=0))
            qkv_local = qkv_chunks[self.tknp_rank]
        else:
            total = 0
            if metadata and metadata.num_actual_tokens is not None:
                total = metadata.num_actual_tokens
            elif x is not None:
                total = x.size(0)
            base = total // self.tknp_world_size
            remainder = total % self.tknp_world_size
            local_tokens = base + (1 if self.tknp_rank < remainder else 0)
            qkv_local = self._buf_pool.get(
                "qkv_recv", local_tokens, self.qkv_size_per_partition,
                self.params_dtype, self.device,
            )
            qkv_chunks = None

        dist.scatter(
            tensor=qkv_local,
            scatter_list=qkv_chunks,
            src=get_tknp_group().first_rank,
            group=self.pg,
        )
        return qkv_local, None

    def extra_repr(self) -> str:
        if hasattr(self, 'hidden_size'):
            return f"hidden_size={self.hidden_size}, is_root_rank={self.is_root_rank}"
        return f"is_root_rank={self.is_root_rank}"


# ===================================================================
# TokenParallelRowLinear
# ===================================================================

class TokenParallelRowLinear(RowParallelLinear):
    """
    Row-parallel linear layer with Token Parallelism.

    Gather-then-compute pattern using pre-allocated buffers and NCCL
    grouped send/recv for efficiency.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)

        # ---- token-parallel topology ---------------------------------
        self.is_tknp_enabled = is_tknp_initialized()
        self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
        self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
        self.is_root_rank = (self.tknp_rank == 0)
        self.tknp_group = get_tknp_group() if self.is_tknp_enabled else None
        self.pg = (self.tknp_group.device_group
                   if hasattr(self.tknp_group, 'device_group')
                   else self.tknp_group)

        # ---- fast NCCL path ------------------------------------------
        use_pynccl = True  # Set to False to disable NCCL path and test fallback
        self._pynccl = _get_pynccl_comm(self.tknp_group) if use_pynccl else None

        # ---- base-class init (creates weights on every rank, then we
        #       drop them on non-root) --------------------------------
        super().__init__(*args, **kwargs)
        self.output_size_full = self.output_size

        if not self.is_root_rank:
            self.register_parameter('weight', None)
            if getattr(self, "bias", None) is not None:
                self.register_parameter('bias', None)

        # ---- pre-allocate communication buffer (root only) -----------
        self._buf_pool = _CommBufferPool()
        if self.is_tknp_enabled and self.is_root_rank:
            max_tokens = _get_max_num_batched_tokens()
            # The hidden_size fed into the row-linear is
            # input_size_per_partition (TP-sharded).
            hidden = self.input_size_per_partition
            self._buf_pool.get(
                "gather_recv", max_tokens, hidden,
                self.params_dtype, 
                torch.device('cuda', torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device('cpu'),
            )

        logger.debug(
            "TokenParallelRowLinear init  rank=%d  root=%s  pynccl=%s",
            self.tknp_rank, self.is_root_rank, self._pynccl is not None,
        )

    # ------------------------------------------------------------------ #
    #  forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
    ) -> Optional[Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.nn.Parameter]]]]:
        if not self.is_tknp_enabled:
            return super().forward(x)

        metadata = getattr(get_forward_context(), "tknp_metadata", None)
        if (metadata is None or not metadata.tknp_enabled
                or metadata.tokens_per_rank is None):
            return super().forward(x)

        tokens_per_rank = metadata.tokens_per_rank
        root_rank = (metadata.root_rank
                     if metadata.root_rank is not None else 0)
        local_count = tokens_per_rank[self.tknp_rank]
        hidden_size = x.size(-1)

        if x.size(0) != local_count:
            logger.warning(
                "Rank %d: expected %d tokens, got %d – falling back.",
                self.tknp_rank, local_count, x.size(0),
            )
            return super().forward(x)

        # ---- gather to root ------------------------------------------
        if self._pynccl is not None and not self._pynccl.disabled:
            if self.is_root_rank:
                total_tokens = sum(tokens_per_rank)
                gathered_input = self._buf_pool.get(
                    "gather_recv", total_tokens, hidden_size,
                    x.dtype, x.device,
                )
                _gather_to_root_nccl(
                    send_buf=x,
                    recv_buf=gathered_input,
                    tokens_per_rank=tokens_per_rank,
                    root_rank=root_rank,
                    pynccl_comm=self._pynccl,
                )
                return super().forward(gathered_input)
            else:
                _gather_to_root_nccl(
                    send_buf=x,
                    recv_buf=None,
                    tokens_per_rank=tokens_per_rank,
                    root_rank=root_rank,
                    pynccl_comm=self._pynccl,
                )
                return None, None
        else:
            # Fallback: torch.distributed all_to_all_single
            if self.is_root_rank:
                total_tokens = sum(tokens_per_rank)
                gathered_input = self._buf_pool.get(
                    "gather_recv", total_tokens, hidden_size,
                    x.dtype, x.device,
                )
                send_splits = [0] * self.tknp_world_size
                send_splits[self.tknp_rank] = local_count
                recv_splits = tokens_per_rank
                dist.all_to_all_single(
                    gathered_input, x,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=self.pg,
                )
                return super().forward(gathered_input)
            else:
                send_splits = [0] * self.tknp_world_size
                send_splits[root_rank] = local_count
                recv_splits = [0] * self.tknp_world_size
                empty_recv = torch.empty(
                    (0, hidden_size), dtype=x.dtype, device=x.device,
                )
                dist.all_to_all_single(
                    empty_recv, x,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=self.pg,
                )
                return None, None

    def extra_repr(self) -> str:
        if hasattr(self, 'input_size') and hasattr(self, 'output_size'):
            return (f"input_size={self.input_size}, output_size={self.output_size}, "
                    f"is_root_rank={self.is_root_rank}")
        return f"is_root_rank={self.is_root_rank}"


# ===================================================================
# Factory helpers
# ===================================================================

def create_token_parallel_qkv_linear(
    hidden_size: int,
    head_size: int,
    total_num_heads: int,
    total_num_kv_heads: Optional[int] = None,
    bias: bool = True,
    **kwargs,
) -> TokenParallelQKVLinear:
    """Factory function to create TokenParallelQKVLinear layer."""
    return TokenParallelQKVLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=bias,
        **kwargs,
    )


def create_token_parallel_row_linear(
    input_size: int,
    output_size: int,
    bias: bool = True,
    reduce_results: bool = True,
    **kwargs,
) -> TokenParallelRowLinear:
    """Factory function to create TokenParallelRowLinear layer."""
    return TokenParallelRowLinear(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        reduce_results=reduce_results,
        **kwargs,
    )


# ===================================================================
# init_tknp_layer decorator  (unchanged logic, kept for completeness)
# ===================================================================

def init_tknp_layer(cls_to_wrap: type) -> type:
    """
    A class decorator that replaces a module with an identity function
    on non-root ranks when token parallelism is enabled.
    """

    class TokenParallelWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

            class _TokenParallelIdentityQuantMethod:
                def __init__(self, wrapper: "TokenParallelWrapper"):
                    self.wrapper = wrapper
                    self.quant_config = None

                def apply(self,
                          layer: nn.Module,
                          x: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
                    vocab = self.wrapper.embedding_dim
                    if vocab is None:
                        vocab = x.size(-1)
                    output_shape = (x.size(0), vocab)
                    return x.new_zeros(output_shape)

            # 1. Check for token parallel init and ranks
            self.is_tknp_enabled = is_tknp_initialized()
            self.tknp_rank = get_tknp_rank() if self.is_tknp_enabled else 0
            self.tknp_world_size = get_tknp_world_size() if self.is_tknp_enabled else 1
            self.is_root_rank = (self.tknp_rank == 0)

            # 2. If self.is_root_rank is True, setup the regular class.
            if self.is_root_rank:
                self.module = cls_to_wrap(*args, **kwargs)

            # 3. If we are not the root rank, setup an identity function.
            else:
                self.module = nn.Identity()

            self.embedding_dim = kwargs.get('embedding_dim', args[1] if len(args) > 1 else None)
            self._return_tuple = None
            if "RMSNorm" in cls_to_wrap.__name__ or "LayerNorm" in cls_to_wrap.__name__:
                self._return_tuple = True
            else:
                self._return_tuple = False

            if not self.is_root_rank:
                self.quant_method = _TokenParallelIdentityQuantMethod(self)

        def forward(self, *args, **kwargs):
            # Special handling for embedding layers on non-root ranks
            if self.embedding_dim and not self.is_root_rank:
                input_tensor = args[0] if args else kwargs.get('input_', None)
                if input_tensor is not None and input_tensor.dim() == 1:
                    batch_size = input_tensor.shape[0]
                    return torch.zeros(batch_size, self.embedding_dim,
                                       dtype=torch.bfloat16,
                                       device=input_tensor.device)

            if not self.is_root_rank:
                input_tensor = args[0] if args else kwargs.get('input_', None)
                if len(args) == 1:
                    return input_tensor
                elif len(args) == 2:
                    return (input_tensor, args[1])
                else:
                    return input_tensor

            return self.module(*args, **kwargs)

        def tie_weights(self, embed_tokens: nn.Module):
            if not self.is_tknp_enabled or self.is_root_rank:
                target = getattr(self, "module", self)
                try:
                    real_embed = getattr(embed_tokens, "module", embed_tokens)
                except Exception:
                    real_embed = embed_tokens
                if hasattr(target, "tie_weights"):
                    return target.tie_weights(real_embed)
                return target
            return self

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def named_parameters(self, *args, **kwargs):
            return self.module.named_parameters(*args, **kwargs)

        def parameters(self, *args, **kwargs):
            return self.module.parameters(*args, **kwargs)

        def named_modules(self, *args, **kwargs):
            return self.module.named_modules(*args, **kwargs)

        def named_buffers(self, *args, **kwargs):
            return self.module.named_buffers(*args, **kwargs)

    return TokenParallelWrapper
