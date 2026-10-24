# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only fixed-grid decode planning, NOT enabled in vLLM.

No attention arithmetic is changed. This builds FlashInfer's existing
DecodePlanInfo/partition metadata, using an explicit chunk size and a fixed
maximum grid so CUDA graph replay survives length/chunk-boundary changes.
The ten-field ABI and workspace limits are checked rather than assumed.
"""
import math


def fixed_decode_plan_layout(batch, q_heads, head_dim, max_pages, chunk_pages):
    values=(batch,q_heads,head_dim,max_pages,chunk_pages)
    if any(type(v) is not int or v<=0 for v in values):
        raise ValueError("Positive integer fixed decode dimensions required")
    padded=batch*math.ceil(max_pages/chunk_pages)
    offset=0;fields={}
    for name,size in (("requests",padded*4),("tiles",padded*4),
                      ("indptr",(padded+1)*4),("chunk",4),("mask",padded)):
        offset=(offset+15)//16*16
        fields[name]=(offset,size);offset+=size
    s_offset=(padded*q_heads*head_dim*4+15)//16*16
    return dict(padded=padded,fields=fields,int_bytes=offset,
        float_bytes=s_offset+padded*q_heads*4,
        plan=[padded,0,s_offset,fields["requests"][0],fields["tiles"][0],
              fields["indptr"][0],fields["mask"][0],fields["chunk"][0],1,1])


def fixed_split_decode_plan(wrapper,indptr_cpu,indices,last_page_len_cpu,
                            q_heads,head_dim,page_size,chunk_pages,max_pages):
    """Plan outside capture, after the library's first-call JIT warmup.

    Mirrors FlashInfer attention/scheduler.cuh DecodePlanInfo (ten fields).
    Temporary partial values and LSE are FP32, as in its native CUDA-core
    planner. Only metadata/grid selection differs; run/merge kernels are the
    installed library's unmodified kernels.
    """
    import torch
    if (wrapper.use_tensor_cores or not wrapper.is_cuda_graph_enabled
            or len(wrapper._plan_info)!=10):
        raise RuntimeError("Fixed CC planner requires graph mode and the supported ten-field ABI")
    if indptr_cpu.device.type!="cpu" or indptr_cpu.dtype!=torch.int32:
        raise ValueError("Fixed CC planner requires CPU int32 indptr")
    batch=last_page_len_cpu.numel()
    if indptr_cpu.numel()!=batch+1 or batch!=wrapper._fixed_batch_size:
        raise ValueError("Fixed CC batch shape changed")
    pages=indptr_cpu[1:]-indptr_cpu[:-1]
    if bool((pages<0).any()) or int(pages.max())>max_pages:
        raise ValueError("KV pages exceed the fixed decode grid bound")
    key=(batch,q_heads,head_dim,page_size,chunk_pages,max_pages,
         wrapper._int_workspace_buffer.data_ptr(),wrapper._float_workspace_buffer.data_ptr())
    state=getattr(wrapper,"_vllm_fixed_cc_state",None)
    if state is None or state["key"]!=key:
        layout=fixed_decode_plan_layout(batch,q_heads,head_dim,max_pages,chunk_pages)
        pin=wrapper._pin_memory_int_workspace_buffer
        if (pin.dtype!=torch.uint8 or not pin.is_pinned()
                or layout["int_bytes"]>min(pin.numel(),wrapper._int_workspace_buffer.numel())
                or layout["float_bytes"]>wrapper._float_workspace_buffer.numel()*wrapper._float_workspace_buffer.element_size()):
            raise RuntimeError("Unsupported or insufficient fixed CC planner workspace")
        # A native first-call plan may still be copying this pinned buffer.
        torch.cuda.current_stream().synchronize()
        pin[:layout["int_bytes"]].zero_()
        views={name:pin[start:start+size].view(torch.bool if name=="mask" else torch.int32)
               for name,(start,size) in layout["fields"].items()}
        state=wrapper._vllm_fixed_cc_state=dict(key=key,layout=layout,views=views,
            copy_done=torch.cuda.Event(),copy_recorded=False)
    if state["copy_recorded"]:
        # Fence ONLY the previous metadata DMA, not the following model graph,
        # before mutating host storage that CUDA may still be reading.
        state["copy_done"].synchronize()
    counts=torch.div(pages+chunk_pages-1,chunk_pages,rounding_mode="floor").clamp_min(1)
    views=state["views"];layout=state["layout"]
    views["indptr"][0]=0
    torch.cumsum(counts,0,dtype=torch.int32,out=views["indptr"][1:batch+1])
    active=int(views["indptr"][batch])
    views["requests"].zero_();views["tiles"].zero_();views["mask"].fill_(False)
    views["requests"][:active].copy_(torch.repeat_interleave(torch.arange(batch,dtype=torch.int32),counts))
    starts=torch.repeat_interleave(views["indptr"][:batch],counts)
    views["tiles"][:active].copy_(torch.arange(active,dtype=torch.int32)-starts)
    views["mask"][:active]=True
    views["chunk"][0]=chunk_pages*page_size
    wrapper._paged_kv_indptr_buf.copy_(indptr_cpu,non_blocking=True)
    wrapper._paged_kv_last_page_len_buf.copy_(last_page_len_cpu,non_blocking=True)
    if indices.data_ptr()!=wrapper._paged_kv_indices_buf.data_ptr():
        if indices.numel()>wrapper._paged_kv_indices_buf.numel():
            raise RuntimeError("Fixed CC page-index buffer exceeded")
        wrapper._paged_kv_indices_buf[:indices.numel()].copy_(indices,non_blocking=True)
    size=layout["int_bytes"]
    wrapper._int_workspace_buffer[:size].copy_(wrapper._pin_memory_int_workspace_buffer[:size],non_blocking=True)
    wrapper._plan_info=list(layout["plan"])
    state["copy_done"].record();state["copy_recorded"]=True
