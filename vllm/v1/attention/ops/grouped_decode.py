# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hopper GQA16 decode through two exact GQA8 query groups.

Only page *indices* are duplicated. The model's physical KV storage is shared.
Graph replay uses a checked, pinned-version fixed-grid planner, because the
ordinary planner can change launch dimensions when context grows. Masked rows
borrow physical page zero and are then restored to zero output/-inf LSE.
No request or KV block is admitted/allocated for these graph lanes.
"""
import os

import torch
import triton
import triton.language as tl


def query_group_count(num_q_heads, num_kv_heads, *, is_hopper,
                      batch_invariant=False, override=None):
    policy = override if override is not None else os.environ.get(
        "VLLM_FLASHINFER_GQA16_QUERY_GROUPS", "0")
    if policy not in ("0", "2"):
        raise ValueError("VLLM_FLASHINFER_GQA16_QUERY_GROUPS must be 0 or 2")
    return (2 if policy == "2" and is_hopper and not batch_invariant
            and num_q_heads == 16 and num_kv_heads == 1 else 1)


def grouped_cpu_metadata(indptr, last, lengths, groups):
    """Repeat logical rows, retaining a one-page read for masked graph lanes."""
    counts = indptr[1:] - indptr[:-1]
    if (groups != 2 or (counts < 0).any() or len(counts) != len(last)
            or len(lengths)!=len(last)):
        raise ValueError("Invalid grouped decode metadata")
    repeated = counts.clamp_min(1).repeat_interleave(groups)
    expanded = torch.empty(len(repeated)+1, dtype=indptr.dtype)
    expanded[0] = 0
    torch.cumsum(repeated, 0, out=expanded[1:])
    return (expanded, last.clamp_min(1).repeat_interleave(groups),
            lengths.clamp_min(1).repeat_interleave(groups))


@triton.jit
def _copy_grouped_page_indices(source, target, original_indptr, expanded_indptr,
                               GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    first = tl.load(original_indptr+row)
    stop = tl.load(original_indptr+row+1)
    count = stop-first
    expanded_first = tl.load(expanded_indptr+row*GROUPS)
    effective = tl.maximum(count, 1)
    offsets = tl.arange(0, BLOCK)
    for start in tl.range(0, effective, BLOCK):
        index = start+offsets
        # Page zero is always physically present. Its result is ignored for
        # a masked lane, and it is never appended to or counted as live KV.
        values = tl.load(source+first+index, mask=index<count, other=0)
        for group in tl.static_range(GROUPS):
            tl.store(target+expanded_first+group*effective+index,
                     values, mask=index<effective)


@triton.jit
def _mask_grouped_outputs(output,lse,original_indptr,HAS_LSE:tl.constexpr,
                          HEADS:tl.constexpr,DIM:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    if tl.load(original_indptr+row+1)==tl.load(original_indptr+row):
        offsets=tl.arange(0,BLOCK)
        tl.store(output+row*HEADS*DIM+offsets,0,mask=offsets<HEADS*DIM)
        if HAS_LSE:
            tl.store(lse+row*HEADS+offsets,-float("inf"),mask=offsets<HEADS)


class QueryGroupedDecodeWrapper:
    def __init__(self, workspace, layout, *, batch_size, max_requests,
                 max_page_indices, use_cuda_graph,max_pages_per_request=None):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        self.groups = 2
        self.max_requests = batch_size if use_cuda_graph else max_requests
        self.fixed_max_pages=max_pages_per_request or max(1,max_page_indices//max_requests)
        self._planned=False
        if use_cuda_graph:
            from vllm.v1.attention.ops.fixed_decode_plan import (
                choose_fixed_chunk_pages,require_fixed_decode_abi)
            require_fixed_decode_abi()
            self.fixed_chunk_pages=choose_fixed_chunk_pages(batch_size*self.groups,1,
                self.fixed_max_pages,torch.cuda.get_device_properties(workspace.device).multi_processor_count)
        device = workspace.device
        self.original_indptr = torch.empty(self.max_requests+1, device=device,
                                           dtype=torch.int32)
        self.indices = torch.empty(max_page_indices*self.groups, device=device,
                                   dtype=torch.int32)
        self.base = BatchDecodeWithPagedKVCacheWrapper(workspace, layout,
            use_cuda_graph=use_cuda_graph, use_tensor_cores=False,
            paged_kv_indptr_buffer=(torch.empty(self.max_requests*self.groups+1,
                device=device,dtype=torch.int32) if use_cuda_graph else None),
            paged_kv_indices_buffer=self.indices if use_cuda_graph else None,
            paged_kv_last_page_len_buffer=(torch.empty(self.max_requests*self.groups,
                device=device,dtype=torch.int32) if use_cuda_graph else None))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def plan_grouped(self, indptr, indices, last, lengths, num_q_heads,
                     num_kv_heads, head_dim, page_size, **options):
        if num_q_heads != 16 or num_kv_heads != 1 or len(last)>self.max_requests:
            raise ValueError("Grouped decode requires the validated GQA16/one-KV-head shape")
        expanded, expanded_last, _ = grouped_cpu_metadata(indptr,last,lengths,self.groups)
        if int(expanded[-1]) > self.indices.numel():
            raise ValueError("Insufficient grouped page-index storage")
        self.original_indptr[:len(indptr)].copy_(indptr,non_blocking=True)
        if not self.base.is_cuda_graph_enabled or not self._planned:
            self.base.plan(expanded,self.indices,expanded_last,
                           num_q_heads//self.groups,num_kv_heads,head_dim,page_size,**options)
            self._planned=True
        if self.base.is_cuda_graph_enabled:
            from vllm.v1.attention.ops.fixed_decode_plan import fixed_split_decode_plan
            fixed_split_decode_plan(self.base,expanded,self.indices,expanded_last,
                num_q_heads//self.groups,head_dim,page_size,self.fixed_chunk_pages,self.fixed_max_pages)
        _copy_grouped_page_indices[(len(last),)](indices,self.indices,
            self.original_indptr,self.base._paged_kv_indptr_buf,
            GROUPS=self.groups,BLOCK=256)

    def run(self, query, kv, **options):
        batch,heads,dim = query.shape
        if heads != 16:
            raise ValueError("Unexpected grouped query shape")
        original_shape = query.shape
        # reshape performs a graph-captured pack when Q is a strided QKV view.
        grouped = query.reshape(batch*self.groups,heads//self.groups,dim)
        for key in ("out","lse"):
            value = options.get(key)
            if value is not None:
                if not value.is_contiguous():
                    raise ValueError("Grouped decode output buffers must be contiguous")
                options[key] = value.view(batch*self.groups,heads//self.groups,*value.shape[2:])
        result = self.base.run(grouped,kv,**options)
        values=result[0] if isinstance(result,tuple) else result
        logs=result[1] if isinstance(result,tuple) else options.get("lse")
        # Empty local DCP contexts must contribute zero/-inf, not the borrowed
        # padding page, when the other ranks merge their attention results.
        _mask_grouped_outputs[(batch,)](values,logs if logs is not None else values,
            self.original_indptr,HAS_LSE=logs is not None,HEADS=heads,DIM=dim,
            BLOCK=triton.next_power_of_2(heads*dim))
        if isinstance(result,tuple):
            return result[0].reshape(original_shape),result[1].reshape(batch,heads)
        return result.reshape(original_shape)
