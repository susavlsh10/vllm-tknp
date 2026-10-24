"""Matched eager/graph efficiency plans and fail-closed mode checks."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"
sys.path.insert(0, str(HERE))
from eos_efficiency import METHODS, make_trial
from e1_70b_suite import topology
from e2e_matrix import validate_live_segment, validate_matrix
from e2e_server import validate_sweep_control


@pytest.mark.parametrize("cache", ["0", "1"])
def test_immutable_prompt_array_reinsertion_preserves_live_output(monkeypatch, cache):
    import numpy as np
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
    monkeypatch.setenv("VLLM_CACHE_PROMPT_TOKEN_IDS", cache)
    prompt=list(range(32))
    req=CachedRequestState("r",prompt,[],SamplingParams(temperature=0),None,
                           ([],),31,[7])
    if cache == "1":
        assert req.prompt_token_ids_np.dtype==np.int32
        assert not req.prompt_token_ids_np.flags.writeable
    else:
        assert req.prompt_token_ids_np is None
    assert not CachedRequestState.__dataclass_fields__["prompt_token_ids_np"].compare
    array=req.prompt_token_ids_np
    batch=InputBatch(max_num_reqs=2,max_model_len=64,max_num_batched_tokens=2,
        device=torch.device("cpu"),pin_memory=False,vocab_size=128,
        block_sizes=[16],kernel_block_sizes=[16])
    for value in (9,11,13):
        index=batch.add_request(req)
        batch.refresh_metadata()
        np.testing.assert_array_equal(batch.token_ids_cpu[index,:req.num_tokens],
                                      prompt+req.output_token_ids)
        batch.remove_request(req.req_id)
        batch.condense()
        batch.refresh_metadata()
        req.output_token_ids.append(value)
        assert req.prompt_token_ids_np is array
    embeds=CachedRequestState("e",None,[],None,None,([],),0,[],
                             prompt_embeds=torch.zeros((3,4)))
    assert embeds.prompt_token_ids_np is None and embeds.num_prompt_tokens==3


@pytest.mark.parametrize("pp,async_mode,connector,flag,resumed,expect_history",[
    (True,False,None,"1",False,False),
    (True,False,False,"1",False,False),
    (True,False,True,"1",False,True),
    (True,True,False,"1",False,True),
    (True,False,False,"0",False,True),
    (True,False,False,"1",True,True),
    (False,False,None,"1",False,True),
])
def test_compact_pp_payload_preserves_tokens_blocks_and_recovery(
        monkeypatch,pp,async_mode,connector,flag,resumed,expect_history):
    from types import SimpleNamespace
    from vllm.v1.core.sched.scheduler import Scheduler
    monkeypatch.setenv("VLLM_PP_COMPACT_CACHED_TOKENS",flag)
    s=Scheduler.__new__(Scheduler)
    s.use_pp=pp;s.scheduler_config=SimpleNamespace(async_scheduling=async_mode)
    s.connector=(None if connector is None else
                 SimpleNamespace(requires_cached_request_token_ids=connector))
    s.prev_step_scheduled_req_ids=set()
    req=SimpleNamespace(request_id="r",all_token_ids=list(range(35)),
        num_computed_tokens=34,num_output_tokens=3,num_output_placeholders=0)
    blocks=SimpleNamespace(get_block_ids=lambda allow_none:([7,9,11],))
    result=s._make_cached_request_data([] if resumed else [req],
        [req] if resumed else [],{"r":1},{},{"r":blocks})
    assert result.req_ids==["r"] and result.num_computed_tokens==[34]
    assert result.num_output_tokens==[3] and result.new_block_ids==[([7,9,11],)]
    assert result.new_token_ids==([[34]] if pp else [])
    assert result.resumed_req_ids==({"r"} if resumed else set())
    assert result.all_token_ids==({"r":list(range(35))} if expect_history else {})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_full_single_node_budget_and_explicit_mode(method, mode):
    tp, pp, tk, dcp, replicas = topology(method, 8)
    caps = [12] + [17]*(tk-1) if tk>1 else [13]*replicas
    if mode=="eager" and method.endswith("-dbo"):
        with pytest.raises(ValueError, match="requires CUDA"):
            make_trial(method,mode,128,caps,32768,128)
        return
    trial = make_trial(method,mode,128,caps,32768,128)
    validate_matrix(dict(trials=[trial]),8)
    assert sorted(g for r in trial["servers"] for g in r["gpus"])==list(range(8))
    for r in trial["servers"]:
        assert r["cuda_graph"] == (mode=="graph")
        assert not r["disable_combo_kernels"]
        if mode=="eager":
            assert not r.get("graph_slots") and not r.get("owner_counts")
            assert "VLLM_TKNP_REQUIRE_CUDAGRAPH" not in r["env"]
        elif tk>1:
            assert r["graph_slots"] and sum(r["owner_counts"])==r["graph_batch_size"]
            assert all(q<=c for q,c in zip(r["owner_counts"],caps))


def test_eager_sweep_control_does_not_weaken_graph_layout_guard():
    opts=dict(tknp=8,pp=1,graph_maximum=128,preinitialized=True)
    for action in ("prepare","checkpoint"):
        validate_sweep_control(dict(key="b128",action=action),cuda_graph=False,**opts)
        with pytest.raises(ValueError,match="capture-layout"):
            validate_sweep_control(dict(key="b128",action=action),cuda_graph=True,**opts)
    with pytest.raises(ValueError,match="captured graph layout"):
        validate_sweep_control(dict(key="b128",action="prepare",graph_size=128),cuda_graph=False,**opts)


@pytest.mark.parametrize("graph", [False,True])
def test_live_gate_rejects_wrong_mode_and_missing_replay(tmp_path,graph):
    key="smoke"
    client=tmp_path/(key+"-client");server=tmp_path/"server0"
    client.mkdir();server.mkdir()
    (client/"summary.json").write_text(json.dumps(dict(completed=2,failed=0,output_tokens=8)))
    (server/"manifest.json").write_text(json.dumps(dict(worker_resources=[dict(rank=0)])))
    record=dict(rank=0,counts={"decode:FULL" if graph else "decode:NONE":4},per_request_kv_fill_calls=0)
    if graph:record["counts"]["graph:FULL_REPLAY"]=4
    path=server/(key+"-execution.json")
    path.write_text(json.dumps([record]))
    (server/"rank0.jsonl").write_text(json.dumps(dict(event="sweep_control",key=key,
        action="checkpoint",all_kv_free=True,remaining_requests=0))+"\n")
    trial=dict(servers=[dict(gpus=[0],cuda_graph=graph)])
    segment=dict(key=key,batch=2,smoke_output_tokens=4)
    validate_live_segment(trial,segment,tmp_path)
    record["counts"]["decode:NONE" if graph else "decode:FULL"]=1
    path.write_text(json.dumps([record]))
    with pytest.raises(RuntimeError,match="execution-mode"):
        validate_live_segment(trial,segment,tmp_path)


def test_immutable_index_cache_is_bounded_and_invalidates(monkeypatch):
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    runner=GPUModelRunner.__new__(GPUModelRunner)
    runner.device=torch.device("cpu")
    monkeypatch.setenv("VLLM_TKNP_CACHE_INDEX_TENSORS","1")
    a=runner._tknp_cached_index_tensor("slots",[0,2,4],torch.long)
    assert runner._tknp_cached_index_tensor("slots",(0,2,4),torch.long) is a
    b=runner._tknp_cached_index_tensor("slots",[4,2,0],torch.long)
    assert b is not a and b.tolist()==[4,2,0] and a.tolist()==[0,2,4]
    assert len(runner._tknp_index_tensor_cache)==1
    c=runner._tknp_cached_index_tensor("slots",[4,2,0],torch.int32)
    assert c.dtype==torch.int32 and c is not b
    for values in ([],[1],[3,4,8],[]):
        result=runner._tknp_cached_index_tensor("slots",values,torch.long)
        assert result.tolist()==values
    assert len(runner._tknp_index_tensor_cache)==1
    monkeypatch.setenv("VLLM_TKNP_CACHE_INDEX_TENSORS","0")
    assert runner._tknp_cached_index_tensor("slots",[1],torch.long) is not runner._tknp_cached_index_tensor("slots",[1],torch.long)


@pytest.mark.parametrize("flag,expected",[("0",False),("1",True)])
def test_flashinfer_decode_kernel_override_is_explicit(monkeypatch,flag,expected):
    from types import SimpleNamespace
    import vllm.v1.attention.backends.flashinfer as backend
    monkeypatch.setenv("VLLM_FLASHINFER_DECODE_USE_TENSOR_CORES",flag)
    monkeypatch.setattr(backend,"get_kv_cache_layout",lambda:"HND")
    monkeypatch.setattr(backend,"BatchDecodeWithPagedKVCacheWrapper",lambda *a,**kw:SimpleNamespace(**kw))
    builder=backend.FlashInferMetadataBuilder.__new__(backend.FlashInferMetadataBuilder)
    builder._decode_wrapper=None
    builder.num_qo_heads=32;builder.num_kv_heads=8;builder.dcp_world_size=1
    builder._get_workspace_buffer=lambda:None
    assert builder._get_decode_wrapper(8).use_tensor_cores is expected
    assert builder._get_decode_wrapper(8) is builder._decode_wrapper


def test_flashinfer_unset_split_uses_public_api_none():
    from types import SimpleNamespace
    import torch
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode
    calls=[]
    wrapper=SimpleNamespace(is_cuda_graph_enabled=False,plan=lambda *a,**kw:calls.append((a,kw)))
    x=torch.tensor([0,1],dtype=torch.int32)
    fast_plan_decode(wrapper,x,x,x[:1],x[:1],32,8,128,16)
    assert calls[0][0][-2] is None


@pytest.mark.parametrize("is_hopper,q,kv,invariant,expected",[
    (True,32,8,False,False),(True,4,1,False,False),
    (False,32,8,False,True),(True,64,8,False,True),
    (True,32,8,True,True),(True,8,8,False,True)])
def test_decode_auto_policy_is_narrow(is_hopper,q,kv,invariant,expected):
    from vllm.v1.attention.backends.flashinfer import decode_uses_tensor_cores
    assert decode_uses_tensor_cores(q,kv,is_hopper=is_hopper,
        batch_invariant=invariant,override="auto") is expected
    assert decode_uses_tensor_cores(q,kv,is_hopper=is_hopper,override="1")
    assert not decode_uses_tensor_cores(q,kv,is_hopper=is_hopper,override="0")
    with pytest.raises(ValueError,match="must be"):
        decode_uses_tensor_cores(q,kv,is_hopper=is_hopper,override="maybe")


@pytest.mark.parametrize("method",["tknp","tknp-dbo","tknp-tp2","tknp-tp2-dbo"])
@pytest.mark.parametrize("batch",[32,64,128,256])
def test_root_bandwidth_proxy_preserves_capacity_and_cohorts(method,batch):
    from tknp_bandwidth_layout import bandwidth_aware_counts
    degree=8 if method in {"tknp","tknp-dbo"} else 4
    caps=[13]+[17]*7 if degree==8 else [31,35,35,35]
    trial=make_trial(method,"graph",batch,caps,32768,128,True)
    r=trial["servers"][0];counts=r["owner_counts"]
    assert sum(counts)==r["graph_batch_size"]
    assert all(c<=cap for c,cap in zip(counts,caps))
    baseline=make_trial(method,"graph",batch,caps,32768,128)["servers"][0]
    assert r["graph_batch_size"]==baseline["graph_batch_size"]
    assert counts[0]<=baseline["owner_counts"][0]
    if method.endswith("dbo"):
        assert counts[0]<=r["graph_batch_size"]//2-2*(degree-1)
    assert bandwidth_aware_counts([13]+[17]*7,128,root_dense_passes=2,minimum_peer=4)[0]==9


def test_cpu_kv_control_reuses_cpu_buffers_and_keeps_growth_guard(monkeypatch):
    import torch
    from types import SimpleNamespace
    import vllm.v1.core.sched.scheduler as module
    monkeypatch.setenv("VLLM_TKNP_CPU_KV_CONTROL","1")
    monkeypatch.setenv("VLLM_TKNP_PLACEMENT_POLICY","cost")
    monkeypatch.setattr(module,"get_tknp_world_size",lambda:4)
    monkeypatch.setattr(module,"get_tknp_rank",lambda:0)
    monkeypatch.setattr(module,"get_tknp_group",lambda:SimpleNamespace(cpu_group="cpu"))
    identities=[]
    def gather(outputs,local,group):
        assert group=="cpu" and local.device.type=="cpu"
        identities.append((local.data_ptr(),[t.data_ptr() for t in outputs]))
        for i,out in enumerate(outputs):out.copy_(local+i)
    monkeypatch.setattr(torch.distributed,"all_gather",gather)
    scheduler=module.Scheduler.__new__(module.Scheduler)
    free=[10]
    scheduler.kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(get_num_free_blocks=lambda:free[0]),
        get_block_ids=lambda req:([1,2],))
    scheduler.cache_config=SimpleNamespace(block_size=16)
    scheduler.token_parallel_scheduler=SimpleNamespace(req_to_tknp_rank={"r":0})
    scheduler.requests={"r":SimpleNamespace(num_prompt_tokens=32,max_tokens=17)}
    # Full lifetime uses four blocks, two already allocated: reserve two more.
    assert scheduler._gather_tknp_free_blocks()==[8,9,10,11]
    free[0]=6
    assert scheduler._gather_tknp_free_blocks()==[4,5,6,7]
    assert identities[0]==identities[1]


@pytest.mark.parametrize("rank",[0,1,7])
def test_cohort_p2p_warmup_has_matching_disjoint_root_star_links(rank):
    from types import SimpleNamespace
    from vllm.v1.worker.gpu_ubatch_wrapper import _warmup_tknp_cohort_p2p
    events=[]
    comm=SimpleNamespace(rank=rank,world_size=8,device="cpu",
        group_start=lambda:events.append(("start",)),group_end=lambda:events.append(("end",)),
        send=lambda t,p,stream:events.append(("send",p,t.data_ptr())),
        recv=lambda t,p,stream:events.append(("recv",p,t.data_ptr())))
    stream=SimpleNamespace(synchronize=lambda:events.append(("sync",)))
    _warmup_tknp_cohort_p2p(comm,stream)
    assert events[0]==("start",) and events[-2:]==[("end",),("sync",)]
    peers=range(1,8) if rank==0 else [0]
    assert sorted(x[1] for x in events if x[0]=="send")==list(peers)
    assert sorted(x[1] for x in events if x[0]=="recv")==list(peers)
    assert not ({x[2] for x in events if x[0]=="send"}&{x[2] for x in events if x[0]=="recv"})


@pytest.mark.parametrize("custom,symm",[(True,True),(True,False),(False,True),(False,False)])
def test_private_tp_dispatch_preserves_native_priority(monkeypatch,custom,symm):
    from types import SimpleNamespace
    import vllm.v1.worker.ubatching as ub
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    monkeypatch.setattr(ub,"dbo_enabled",lambda:True)
    monkeypatch.setattr(ub,"dbo_current_ubatch_id",lambda:1)
    dc=CudaCommunicator.__new__(CudaCommunicator)
    dc.tknp_dbo_tp_comms=[None,SimpleNamespace(all_reduce=lambda x:"nccl")]
    dc.tknp_dbo_tp_ca_comms=[None,SimpleNamespace(disabled=False,
        should_custom_ar=lambda x:custom,custom_all_reduce=lambda x:"custom")]
    dc.tknp_dbo_tp_symm_comms=[None,SimpleNamespace(should_use_symm_mem=lambda x:symm,
        all_reduce=lambda x:"symm")]
    assert dc.all_reduce(object())==("custom" if custom else "symm" if symm else "nccl")


def test_graph_capture_registers_all_private_custom_buffers(monkeypatch):
    from contextlib import contextmanager,nullcontext
    from types import SimpleNamespace
    import torch
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    events=[]
    def comm(name):
        @contextmanager
        def capture():
            events.append("enter:"+name)
            try:yield
            finally:events.append("exit:"+name)
        return SimpleNamespace(capture=capture)
    dc=CudaCommunicator.__new__(CudaCommunicator)
    dc.ca_comm=comm("base");dc.tknp_dbo_tp_ca_comms=[comm("c0"),comm("c1")]
    group=GroupCoordinator.__new__(GroupCoordinator);group.device_communicator=dc
    stream=object();context=SimpleNamespace(stream=stream)
    monkeypatch.setattr(torch.cuda,"current_stream",lambda:stream)
    monkeypatch.setattr(torch.cuda,"stream",lambda s:nullcontext())
    with group.graph_capture(context):events.append("body")
    assert events==["enter:base","enter:c0","enter:c1","body","exit:c1","exit:c0","exit:base"]


def test_decode_gate_preserves_natural_queueing_and_rejects_unsafe_states():
    from e2e_server import validate_decode_gate
    valid=dict(synthetic=True,preinitialized=True,held=False,drained=True)
    validate_decode_gate("hold",**valid)
    validate_decode_gate("release",**dict(valid,held=True,drained=False))
    for action,override in (("hold",dict(drained=False)),("hold",dict(held=True)),
            ("release",dict(held=False)),("hold",dict(pending=True)),
            ("hold",dict(synthetic=False)),("release",dict(held=True,preinitialized=False))):
        with pytest.raises(ValueError):validate_decode_gate(action,**dict(valid,**override))
    with pytest.raises(ValueError):validate_decode_gate("unknown",**valid)
