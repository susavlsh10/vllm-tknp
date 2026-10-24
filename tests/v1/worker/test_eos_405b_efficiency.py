"""Plan-only four-node FP8 efficiency controls; no GPU allocation."""
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parents[3]/"examples/offline_inference/TKNP"
sys.path.insert(0,str(HERE))
from eos_405b_efficiency import make_trial
from e2e_matrix import validate_matrix
from fp8_decode_sweep import FOUR_NODE_METHODS


@pytest.mark.parametrize("batch",[4,32,64,96,128])
@pytest.mark.parametrize("mode",["standard","tuned"])
def test_pp_baselines_keep_native_concurrency_and_separate_tuning(batch,mode):
    from fp8_decode_sweep import pp_segment_override
    role=make_trial("tp8-pp4",batch,[113],32768,128,pp_baseline=mode)["servers"][0]
    assert role["pp"]==4 and role["tp"]==8 and role["native_graph_shapes"]
    assert "batch_queue_size" not in role
    flag="1" if mode=="tuned" else "0"
    assert role["env"]["VLLM_CACHE_PROMPT_TOKEN_IDS"]==flag
    assert role["env"]["VLLM_PP_COMPACT_CACHED_TOKENS"]==flag
    if mode=="standard":
        assert "pp_decode_microbatch" not in role
        assert role["graph_batch_size"]==role["max_num_seqs"]==min(batch,113)
        assert pp_segment_override(role,batch,113)=={}
    else:
        expected=(min(batch,113)+3)//4
        assert role["pp_decode_microbatch"]==expected
        assert pp_segment_override(role,batch,113)=={"pp_decode_microbatch":expected}


def test_rank_compile_cache_isolation_is_opt_in_and_idempotent(monkeypatch,tmp_path):
    import os,socket
    from vllm.v1.worker.gpu_worker import _isolate_research_compile_caches
    monkeypatch.setenv("TRITON_CACHE_DIR",str(tmp_path/"triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR",str(tmp_path/"inductor"))
    monkeypatch.setenv("VLLM_RESEARCH_RANK_TRITON_CACHE","0")
    monkeypatch.setenv("VLLM_RESEARCH_RANK_INDUCTOR_CACHE","0")
    _isolate_research_compile_caches(3)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"]==str(tmp_path/"inductor")
    monkeypatch.setenv("VLLM_RESEARCH_RANK_TRITON_CACHE","1")
    monkeypatch.setenv("VLLM_RESEARCH_RANK_INDUCTOR_CACHE","1")
    monkeypatch.setattr(socket,"gethostname",lambda:"test-node")
    _isolate_research_compile_caches(3)
    expected={"TRITON_CACHE_DIR":str(tmp_path/"triton/worker_rank_3"),
        "TORCHINDUCTOR_CACHE_DIR":str(tmp_path/"inductor/test-node/worker_rank_3")}
    assert {k:os.environ[k] for k in expected}==expected
    _isolate_research_compile_caches(3)
    assert {k:os.environ[k] for k in expected}==expected
    from torch._inductor.runtime.runtime_utils import cache_dir
    assert cache_dir()==expected["TORCHINDUCTOR_CACHE_DIR"]


@pytest.mark.parametrize("random",[False,True])
def test_preinitialized_constant_correctness_kv_is_explicit_and_exclusive(monkeypatch,tmp_path,random):
    from e2e_server import arguments
    mode="--preinitialize-random-kv" if random else "--preinitialize-constant-kv"
    base=["server","--connector","synthetic","--result-dir",str(tmp_path)]
    monkeypatch.setattr(sys,"argv",base+[mode])
    args=arguments()
    assert args.preinitialized_kv
    assert args.preinitialize_random_kv is random
    assert args.preinitialize_constant_kv is not random
    monkeypatch.setattr(sys,"argv",base+["--preinitialize-random-kv","--preinitialize-constant-kv"])
    with pytest.raises(SystemExit):arguments()


def test_profiler_never_counts_collective_wait_kernels_as_compute():
    from eos_efficiency_trace import is_communication_kernel
    for name in ("ncclDevKernel_SendRecv","custom_ar::cross_device_reduce_2stage",
                 "multimem_all_reduce_kernel","two_shot_all_reduce_kernel"):
        assert is_communication_kernel(name)
    for name in ("flash_fwd_kernel","cutlass_gemm","fused_rms_norm"):
        assert not is_communication_kernel(name)


@pytest.mark.parametrize("rank",range(4))
def test_fa3_dcp_padded_lanes_have_zero_not_negative_context(rank):
    import torch
    from types import SimpleNamespace as NS
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    builder=FlashAttentionMetadataBuilder.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule=False;builder.aot_sliding_window=(-1,-1)
    builder.use_full_cuda_graph=False
    builder.cache_config=NS(cache_dtype="auto");builder.kv_cache_dtype=torch.bfloat16
    builder.dcp_world_size=4;builder.dcp_rank=rank;builder.cp_kv_cache_interleave_size=1
    common=NS(num_reqs=2,num_actual_tokens=2,max_query_len=1,max_seq_len=32,
        query_start_loc=torch.arange(3,dtype=torch.int32),seq_lens=torch.tensor([32,0],dtype=torch.int32),
        block_table_tensor=torch.zeros(2,2,dtype=torch.int32),slot_mapping=torch.tensor([0,-1]),causal=True)
    result=builder.build(0,common)
    assert result.dcp_context_kv_lens.tolist()==[8 if rank<3 else 7,0]
    assert common.seq_lens.tolist()==[32,0]


def test_fa3_dcp_graph_lengths_reuse_updated_storage_across_batch_shapes():
    import torch
    from types import SimpleNamespace as NS
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    builder=FlashAttentionMetadataBuilder.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule=False;builder.aot_sliding_window=(-1,-1)
    builder.use_full_cuda_graph=True;builder.max_cudagraph_size=8;builder.max_num_splits=16
    builder.cache_config=NS(cache_dtype="auto");builder.kv_cache_dtype=torch.bfloat16
    builder.dcp_world_size=4;builder.dcp_rank=3;builder.cp_kv_cache_interleave_size=1
    builder.dcp_context_kv_lens_buffer=torch.zeros(8,dtype=torch.int32)
    old=None
    for lengths in ([32]*8,[100]*3,[132]*8):
        n=len(lengths)
        common=NS(num_reqs=n,num_actual_tokens=n,max_query_len=1,max_seq_len=max(lengths),
            query_start_loc=torch.arange(n+1,dtype=torch.int32),
            seq_lens=torch.tensor(lengths,dtype=torch.int32),
            block_table_tensor=torch.zeros(n,16,dtype=torch.int32),
            slot_mapping=torch.arange(n),causal=True)
        meta=builder.build(0,common)
        assert meta.dcp_context_kv_lens.data_ptr()==builder.dcp_context_kv_lens_buffer.data_ptr()
        expected=(lengths[0]-1)//4
        assert meta.dcp_context_kv_lens.tolist()==[expected]*n
        if old is not None:
            assert old[:n].tolist()==[expected]*n
        if old is None:old=meta.dcp_context_kv_lens
    assert old.tolist()==[32]*8


def test_dcp_lifetime_reservation_uses_logical_not_physical_token_span():
    from vllm.v1.core.sched.tknp_placement import unreserved_lifetime_blocks
    # DCP4 has 64 logical tokens per physical 16-token page.
    # 110 residents need two extra pages apiece over a 32K prompt for OSL128.
    assert unreserved_lifetime_blocks(400,64,[(32768,128,512)]*110)==180
    trial=make_trial("dcp",128,[111],32768,128)
    assert trial["servers"][0]["max_num_seqs"]==111
    assert trial["servers"][0]["env"]["VLLM_TKNP_LIFETIME_ADMISSION"]=="1"


@pytest.mark.parametrize("root_capacity",[0,1,5])
def test_projection_only_root_never_uses_fictitious_kv_capacity(root_capacity):
    role=make_trial("tknp-tp8-dbo",128,[root_capacity,34,34,34],32768,128,0)["servers"][0]
    assert role["owner_counts"]==[0,34,33,33]
    assert role["graph_batch_size"]==100
    assert role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"]=="0"


def test_startup_capacity_repair_preserves_explicit_zero_attention_owners():
    from vllm.v1.worker.tknp_serving_slots import fit_capture_split
    assert fit_capture_split([0,4,4,4],12,[1001,31,51,51],10)==[0,3,5,4]
    with pytest.raises(ValueError):fit_capture_split([0,4,4,4],12,[1001,31,41,41],10)


def test_capacity_source_must_match_model_precision_and_full_lifetime():
    from eos_405b_efficiency import validate_capacity_source
    args=dict(model="405b",context=32768,tokens=1024,attention_backend="FLASH_ATTN",
        query_groups="0",tensor_cores="auto",gpu_memory_utilization=.9)
    validate_capacity_source(args,dict(args))
    for key,value in (("tokens",128),("context",98304),("model","8b"),
                      ("attention_backend","FLASHINFER"),("gpu_memory_utilization",.95)):
        with pytest.raises(ValueError):validate_capacity_source(dict(args,**{key:value}),args)


def test_segment_journal_views_preserve_boundaries_and_global_shutdown(tmp_path):
    import hashlib,json
    from e2e_matrix import split_sweep_journals
    shared=tmp_path/"raw";shared.mkdir()
    records=[dict(event="startup"),dict(event="sweep_control",key="a",action="prepare"),
        dict(event="decode_gate",key="gate-a",action="hold"),dict(event="step",step=1),
        dict(event="sweep_control",key="a",action="checkpoint"),
        dict(event="sweep_control",key="b",action="prepare"),dict(event="step",step=2),
        dict(event="sweep_control",key="b",action="checkpoint"),
        dict(event="shutdown",remaining_requests=0,all_kv_free=True)]
    raw="".join(json.dumps(r)+"\n" for r in records)
    (shared/"rank0.jsonl").write_text(raw)
    views=split_sweep_journals(shared,["a","b"],tmp_path/"views")
    for key,expected in (("a",records[1:5]),("b",records[5:8])):
        actual=list(map(json.loads,(views[key]/"rank0.jsonl").read_text().splitlines()))
        assert actual==expected+[records[-1]]
    evidence=json.loads((tmp_path/"views/provenance.json").read_text())["rank0.jsonl"]
    assert evidence["sha256"]==hashlib.sha256(raw.encode()).hexdigest()
    assert (shared/"rank0.jsonl").read_text()==raw
    (shared/"rank0.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records if r.get("key")!="b"))
    with pytest.raises(RuntimeError):split_sweep_journals(shared,["a","b"],tmp_path/"broken")


def test_fa3_dcp_uses_the_split_count_of_its_aot_schedule(monkeypatch):
    import torch
    from types import SimpleNamespace as NS
    import vllm.v1.attention.backends.flash_attn as fa
    calls=[]
    def kernel(**kwargs):
        calls.append(kwargs)
        q=kwargs["q"]
        return torch.zeros_like(q),torch.zeros(q.shape[1],q.shape[0])
    monkeypatch.setattr(fa,"flash_attn_varlen_func",kernel)
    monkeypatch.setattr(fa,"get_dcp_group",lambda:NS(all_gather=lambda q,dim:q.repeat(1,4,1)))
    monkeypatch.setattr(fa,"cp_lse_ag_out_rs",lambda out,lse,*a,**k:(out[:,:4],lse[:,:4]))
    monkeypatch.setattr(fa,"merge_attn_states",lambda out,*a:out.zero_())
    impl=fa.FlashAttentionImpl.__new__(fa.FlashAttentionImpl)
    impl.scale=128**-.5;impl.alibi_slopes=None;impl.sliding_window=(-1,-1)
    impl.logits_soft_cap=0.;impl.vllm_flash_attn_version=3
    q=torch.zeros(2,4,128);kv=torch.zeros(2,1,128)
    meta=NS(query_start_loc=torch.arange(3,dtype=torch.int32),max_query_len=1,
        block_table=torch.zeros(2,2,dtype=torch.int32),dcp_context_kv_lens=torch.full((2,),128),
        max_dcp_context_kv_len=128,scheduler_metadata=torch.zeros(20,dtype=torch.int32),
        max_num_splits=16,causal=True)
    impl._forward_with_dcp(q,kv,kv,kv,kv,torch.empty_like(q),meta)
    assert calls[0]["num_splits"]==meta.max_num_splits
    assert calls[0]["scheduler_metadata"] is meta.scheduler_metadata


def test_metadata_cache_never_borrows_another_cohort():
    import numpy as np
    import torch
    from types import SimpleNamespace as NS
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    class Builder:
        supports_update_block_table=True
        def __init__(self):self.calls=0
        def build(self,common_prefix_len,common_attn_metadata):
            self.calls+=1
            return common_attn_metadata
        def update_block_table(self,old,table,slots):
            from copy import copy
            result=copy(old);result.block_table_tensor=table;result.slot_mapping=slots
            return result
    builders=[Builder(),Builder()]
    group=NS(layer_names=["layer0"],get_metadata_builder=lambda i:builders[i])
    r=GPUModelRunner.__new__(GPUModelRunner)
    r.kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=FullAttentionSpec(
        block_size=16,num_kv_heads=1,head_size=128,dtype=torch.bfloat16))])
    r.attn_groups=[[group]]
    qloc=torch.arange(7,dtype=torch.int32)
    lengths=torch.arange(10,16,dtype=torch.int32)
    r.query_start_loc=NS(gpu=qloc,cpu=qloc)
    r.seq_lens=NS(gpu=lengths,cpu=lengths,np=lengths.numpy())
    r.input_batch=NS(block_table=[NS(get_device_tensor=lambda n:torch.zeros(n,2,dtype=torch.int32),
        slot_mapping=NS(gpu=torch.zeros(6,dtype=torch.int64)))],
        num_computed_tokens_cpu_tensor=torch.zeros(6,dtype=torch.int32))
    r.dcp_world_size=1;r.speculative_config=None;r.is_mm_prefix_lm=False
    r._get_encoder_seq_lens=lambda *a:(None,None)
    metadata,_=r._build_attention_metadata(num_tokens=6,num_reqs=6,max_query_len=1,
        ubatch_slices=[UBatchSlice(slice(0,2),slice(0,2)),UBatchSlice(slice(2,6),slice(2,6))])
    assert [b.calls for b in builders]==[1,1]
    assert [m["layer0"].num_reqs for m in metadata]==[2,4]
    torch.testing.assert_close(metadata[1]["layer0"].seq_lens,lengths[2:])
    assert metadata[1]["layer0"].query_start_loc.tolist()==[0,1,2,3,4]


@pytest.mark.parametrize("backend",["flashinfer","flashattn"])
@pytest.mark.parametrize("std",[0.,.02])
def test_synthetic_kv_fill_respects_backend_block_axis_and_strides(backend,std):
    import torch
    from types import SimpleNamespace
    from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
        DecodeBenchConnectorWorker,DecodeBenchConnectorMetadata)
    options=dict(fill_mean=.25,fill_std=std)
    cfg=SimpleNamespace(cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda k,d:options.get(k,d)))
    worker=DecodeBenchConnectorWorker(cfg)
    shape=(8,2,3,16,128) if backend=="flashinfer" else (2,8,3,16,128)
    cache=torch.zeros(shape,dtype=torch.bfloat16).permute(0,1,3,2,4)
    worker.register_kv_caches({"layer0":cache})
    axis=0 if backend=="flashinfer" else 1
    assert worker.kv_block_axes["layer0"]==axis
    worker.start_fill_kv(DecodeBenchConnectorMetadata({"r":(([1,5,999,-1],),32)}))
    assert torch.count_nonzero(cache.index_select(axis,torch.tensor([0,2,3,4,6,7])))==0
    values=cache.index_select(axis,torch.tensor([1,5]))
    if std:assert values.float().std()>.01
    else:torch.testing.assert_close(values,torch.full_like(values,.25),rtol=0,atol=0)
    assert worker.per_request_fill_calls==1


@pytest.mark.parametrize("method",FOUR_NODE_METHODS)
@pytest.mark.parametrize("batch",[16,32,64,96,128])
def test_full_budget_and_lifetime_capacity(method,batch):
    caps=([5,33,33,33] if method.startswith("tknp") else
          [6]*4 if method=="dp-tp8" else [106])
    trial=make_trial(method,batch,caps,32768,1024)
    validate_matrix(dict(trials=[trial]),32)
    assert sorted(g for r in trial["servers"] for g in r["gpus"])==list(range(32))
    for role in trial["servers"]:
        assert role["cuda_graph"]
        if role.get("owner_counts"):
            assert all(n<=c for n,c in zip(role["owner_counts"],caps))
        if method=="dcp":
            assert role["tp"]==32 and role["dcp"]==4
        if method=="tp8-pp4":
            assert role["tp"]==8 and role["pp"]==4


@pytest.mark.parametrize("method",["tknp-tp8","tknp-tp8-dbo"])
@pytest.mark.parametrize("requested",[0,2,4,16])
@pytest.mark.parametrize("batch",[16,32,64,96,128])
def test_root_ownership_tuning_preserves_shape_capacity(monkeypatch,method,requested,batch):
    caps=[5,33,33,33]
    role=make_trial(method,batch,caps,32768,1024,requested)["servers"][0]
    counts=role["owner_counts"]
    assert sum(counts)==role["graph_batch_size"]
    assert all(n<=c for n,c in zip(counts,caps))
    if requested==0 and role.get("enable_dbo"):
        assert counts[0]==0
    else:assert counts[0]>=(2 if role.get("enable_dbo") else 1)
    if role.get("enable_dbo"):
        for key,value in role["env"].items():
            monkeypatch.setenv(key,value)
        from vllm.forward_context import build_tknp_dbo_cohort_rank_order
        order=build_tknp_dbo_cohort_rank_order(counts)
        assert 0 not in order[len(order)//2:]
        assert role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"]==str(counts[0])


def test_query_grouping_is_explicit_and_narrow():
    from vllm.v1.attention.ops.grouped_decode import query_group_count
    assert query_group_count(16,1,is_hopper=True,override="0")==1
    assert query_group_count(16,1,is_hopper=True,override="2")==2
    assert query_group_count(16,1,is_hopper=False,override="2")==1
    assert query_group_count(16,1,is_hopper=True,batch_invariant=True,override="2")==1
    assert query_group_count(32,8,is_hopper=True,override="2")==1
    with pytest.raises(ValueError):
        query_group_count(16,1,is_hopper=True,override="4")


def test_admission_tuning_is_explicit_and_does_not_relabel_physical_capacity():
    caps=[5,33,33,33]
    role=make_trial("tknp-tp8-dbo",128,caps,32768,1024,2,100)["servers"][0]
    assert role["graph_batch_size"]==100
    assert role["owner_counts"]==[2,33,33,32]
    assert caps==[5,33,33,33]
    # A TKNP tuning flag must not constrain other baseline methods.
    dp=make_trial("dp-tp8",128,[6]*4,32768,1024,2,100)
    assert [r["max_num_seqs"] for r in dp["servers"]]==[6]*4


def test_grouped_cpu_metadata_does_not_mutate_original_or_allocate_live_kv():
    import torch
    from vllm.v1.attention.ops.grouped_decode import grouped_cpu_metadata
    indptr=torch.tensor([0,0,3,4],dtype=torch.int32)
    last=torch.tensor([0,2,16],dtype=torch.int32)
    seq=torch.tensor([0,34,16],dtype=torch.int32)
    expanded,ends,lengths=grouped_cpu_metadata(indptr,last,seq,2)
    assert expanded.tolist()==[0,1,2,5,8,9,10]
    assert ends.tolist()==[1,1,2,2,16,16]
    assert lengths.tolist()==[1,1,34,34,16,16]
    assert indptr.tolist()==[0,0,3,4] and seq.tolist()==[0,34,16]


def test_fixed_grid_layout_is_bounded_and_metadata_is_aligned():
    from vllm.v1.attention.ops.fixed_decode_plan import (
        choose_fixed_chunk_pages,fixed_decode_plan_layout)
    for batch,kvheads,qheads in [(4,1,8),(12,1,8),(34,1,8),(4,8,32),(128,1,4)]:
        chunk=choose_fixed_chunk_pages(batch,kvheads,2056,132)
        layout=fixed_decode_plan_layout(batch,qheads,128,2056,chunk)
        assert chunk>0 and chunk&(chunk-1)==0
        assert layout["padded"]>=batch
        assert len(layout["plan"])==10
        assert all(offset%16==0 for offset,_ in layout["fields"].values())
        assert layout["float_bytes"]<128*1024**2


def test_tail_replay_selection_preserves_sticky_owners_and_admission(monkeypatch):
    from types import SimpleNamespace
    from vllm.v1.worker.tknp_serving_slots import select_resident_replay_size
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    layouts={16:[4,4,4,4],32:[6,9,9,8],64:[6,20,19,19],104:[6,33,33,32]}
    assert select_resident_replay_size(layouts,104,[0]*6+[1]*6+[2]*6+[3]*6)==32
    assert select_resident_replay_size(layouts,104,[1]*20)==64
    # Total occupancy alone is insufficient: do not move KV off owner 1.
    assert select_resident_replay_size(layouts,104,[1]*21)==104
    with pytest.raises(ValueError):select_resident_replay_size(layouts,32,[1]*10)
    with pytest.raises(ValueError):select_resident_replay_size(layouts,104,[4])
    monkeypatch.setenv("VLLM_TKNP_GRAPH_SLOTS","104")
    monkeypatch.setenv("VLLM_TKNP_GRAPH_SLOT_LIMITS","[6,33,33,32]")
    monkeypatch.setenv("VLLM_TKNP_GRAPH_TAIL_DOWNSIZE","1")
    runner=GPUModelRunner.__new__(GPUModelRunner)
    runner._tknp_graph_splits=layouts
    runner.input_batch=SimpleNamespace(req_ids=["r"])
    schedule=SimpleNamespace(token_parallel_allocations=SimpleNamespace(req_to_tknp_rank={"r":1}))
    runner._select_tknp_step_graph(schedule)
    assert runner._tknp_serving_graph_size()==16
    import os
    assert os.environ["VLLM_TKNP_GRAPH_SLOTS"]=="104"
    assert os.environ["VLLM_TKNP_GRAPH_SLOT_LIMITS"]=="[6,33,33,32]"
    monkeypatch.setenv("VLLM_TKNP_GRAPH_TAIL_DOWNSIZE","0")
    runner._select_tknp_step_graph(schedule)
    assert runner._tknp_serving_graph_size()==104


@pytest.mark.parametrize("batch",[4,12,34])
@pytest.mark.parametrize("mode",["grouped","native","native-multikv","tensor-core"])
def test_grouped_runtime_wrapper_graph_and_changing_masked_rows(batch,mode):
    import math
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU primitive regression")
    if mode=="native-multikv" and batch!=4:
        pytest.skip("One multi-KV representative is sufficient")
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from vllm.v1.attention.ops.grouped_decode import QueryGroupedDecodeWrapper
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode
    torch.manual_seed(1019)
    length=32768 if batch==34 else 2048
    page,heads,dim=16,(16 if mode in ("grouped","tensor-core") else 32 if mode=="native-multikv" else 4),128
    kvheads=8 if mode=="native-multikv" else 1
    per=math.ceil(length/page)
    kv=torch.randn(batch*per,2,kvheads,page,dim,device="cuda",dtype=torch.bfloat16)
    indices=torch.empty(batch*per,device="cuda",dtype=torch.int32)
    order=torch.randperm(batch*per,device="cuda",dtype=torch.int32)
    # Match the strided Q view returned by the actual QKV projection.
    qkv=torch.randn(batch,heads+2*kvheads,dim,device="cuda",dtype=torch.bfloat16)
    query=qkv[:,:heads]
    out=torch.empty((batch,heads,dim),device="cuda",dtype=query.dtype)
    lse=torch.empty((batch,heads),device="cuda",dtype=torch.float32)
    if mode=="grouped":
        grouped=QueryGroupedDecodeWrapper(torch.empty(128*1024**2,device="cuda",dtype=torch.uint8),
            "HND",batch_size=batch,max_requests=batch,max_page_indices=indices.numel(),use_cuda_graph=True)
    else:
        from vllm.v1.attention.ops.fixed_decode_plan import choose_fixed_chunk_pages
        grouped=BatchDecodeWithPagedKVCacheWrapper(torch.empty(128*1024**2,device="cuda",dtype=torch.uint8),
            "HND",use_tensor_cores=mode=="tensor-core",use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.empty(batch+1,device="cuda",dtype=torch.int32),
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=torch.empty(batch,device="cuda",dtype=torch.int32))
        if mode!="tensor-core":
            grouped._vllm_fixed_cc_max_pages=per
            grouped._vllm_fixed_cc_chunk_pages=choose_fixed_chunk_pages(batch,kvheads,per,132)
    if mode=="grouped":
        public=QueryGroupedDecodeWrapper(torch.empty(128*1024**2,device="cuda",dtype=torch.uint8),
            "HND",batch_size=batch,max_requests=batch,max_page_indices=indices.numel(),use_cuda_graph=False)
    else:
        public=BatchDecodeWithPagedKVCacheWrapper(torch.empty(128*1024**2,device="cuda",dtype=torch.uint8),
            "HND",use_tensor_cores=mode=="tensor-core")
    graph=None
    for phase in range(3):
        # Capture shorter contexts, then grow 4x under the SAME graph. This
        # catches planner choices baked into a graph from short dummy inputs.
        current_length=length//4 if phase==0 else length
        lengths=[0 if (i+phase)%5==0 else max(1,current_length-(i*31+phase*7)%511)
                 for i in range(batch)]
        counts=[math.ceil(n/page) for n in lengths]
        ptr=[0]
        for n in counts:ptr.append(ptr[-1]+n)
        indptr=torch.tensor(ptr,dtype=torch.int32)
        last=torch.tensor([((n-1)%page+1) if n else 0 for n in lengths],dtype=torch.int32)
        seq=torch.tensor(lengths,dtype=torch.int32)
        payload=torch.cat([order[i*per:i*per+n] for i,n in enumerate(counts)])
        indices.zero_();indices[:len(payload)].copy_(payload)
        fast_plan_decode(grouped,indptr,indices,last,seq,heads,kvheads,dim,page,
            q_data_type=query.dtype,kv_data_type=kv.dtype,sm_scale=dim**-.5)
        qkv.normal_()
        if graph is None:
            for _ in range(3):grouped.run(query,kv,out=out,lse=lse,return_lse=True)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                grouped.run(query,kv,out=out,lse=lse,return_lse=True)
        graph.replay();torch.cuda.synchronize()
        active=torch.tensor([n>0 for n in lengths],device="cuda")
        # The tensor-core reference rounds attention probabilities to BF16;
        # near cancellation it can differ from FP32 by >5e-4. Check the
        # captured graph against a fresh launch AND an independent FP32
        # oracle, not one lower-precision backend against another.
        fresh,fresh_lse=grouped.run(query,kv,return_lse=True)
        torch.testing.assert_close(out[active],fresh[active],rtol=0,atol=0)
        torch.testing.assert_close(lse[active],fresh_lse[active],rtol=0,atol=0)
        fast_plan_decode(public,indptr,indices,last,seq,heads,kvheads,dim,page,
            q_data_type=query.dtype,kv_data_type=kv.dtype,sm_scale=dim**-.5)
        public_out,public_lse=public.run(query,kv,return_lse=True)
        torch.testing.assert_close(out[active],public_out[active],rtol=.02,atol=1e-3)
        torch.testing.assert_close(lse[active],public_lse[active],rtol=.005,atol=.005)
        from torch.nn.attention import SDPBackend,sdpa_kernel
        import torch.nn.functional as F
        rows=[i for i,n in enumerate(lengths) if n]
        if length>2048:rows=sorted({rows[0],rows[-1]})
        for i in rows:
            physical=order[i*per:i*per+counts[i]].long()
            keys=kv[physical,0].permute(1,0,2,3).reshape(kvheads,-1,dim)[:,:lengths[i]]
            values=kv[physical,1].permute(1,0,2,3).reshape(kvheads,-1,dim)[:,:lengths[i]]
            with sdpa_kernel(SDPBackend.MATH):
                reference=F.scaled_dot_product_attention(query[i].float()[None,:,None,:],
                    keys.float()[None],values.float()[None],enable_gqa=True)[0,:,0]
            # CUDA-core BF16 dot products themselves can differ from FP32
            # by ~1e-3 near cancellation. Require the same bound of the
            # unmodified public eager kernel, in addition to the bitwise
            # graph-vs-fresh check above.
            torch.testing.assert_close(out[i].float(),reference,rtol=.02,atol=1e-3)
            torch.testing.assert_close(public_out[i].float(),reference,rtol=.02,atol=1e-3)
            scores=torch.einsum("hd,hnd->hn",query[i].float(),
                keys.float().repeat_interleave(heads//kvheads,0))*dim**-.5
            # FlashInfer uses log2 LSE (the backend passes
            # is_lse_base_on_e=False to the context-parallel merge).
            torch.testing.assert_close(lse[i],scores.logsumexp(-1)/math.log(2),rtol=.005,atol=.005)
        assert torch.isfinite(out[active]).all()
        assert torch.count_nonzero(out[~active])==0
        if mode=="grouped":assert torch.isneginf(lse[~active]).all()
