from collections import Counter
from types import SimpleNamespace
import threading

import pytest
import torch

from vllm.forward_context import (
    build_tknp_dbo_cohort_rank_order,
    compute_dbo_tknp_split,
)


def test_static_shape_compilation_is_published_atomically():
    from vllm.compilation.piecewise_backend import PiecewiseBackend, RangeEntry
    from vllm.config.compilation import Range

    entered, release = threading.Event(), threading.Event()
    calls = []
    runnable = lambda *args: None

    def compile_shape(*args, **kwargs):
        calls.append(kwargs["compile_range"])
        entered.set()
        assert release.wait(5)
        return runnable

    backend = PiecewiseBackend.__new__(PiecewiseBackend)
    shape = Range(start=32, end=32)
    entry = RangeEntry(shape)
    backend.to_be_compiled_ranges = {shape}
    backend.graph = None
    backend.compilation_config = None
    backend.piecewise_compile_index = 0
    backend.total_piecewise_compiles = 1
    backend.vllm_backend = SimpleNamespace(
        compiler_manager=SimpleNamespace(compile=compile_shape),
        inductor_config={},
    )
    backend.check_for_ending_compilation = lambda: None
    failures = []

    def worker():
        try:
            backend._maybe_compile_for_range_entry(entry, [])
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    threads[0].start()
    assert entered.wait(5)
    threads[1].start()
    assert not entry.compiled and entry.runnable is None
    release.set()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not failures
    assert calls == [shape]
    assert entry.compiled and entry.runnable is runnable
    assert not backend.to_be_compiled_ranges


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("world,batch", [(2, 32), (4, 128), (8, 256)])
def test_request_ownership_and_cohort_capacity(monkeypatch, zero, world, batch):
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_ZERO_ROOT_C1", str(int(zero)))
    root_count = min(batch // world, batch // 2 - (world - 1))
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", str(root_count))
    counts = compute_dbo_tknp_split(batch, 4, [batch * 4] * world)
    order = build_tknp_dbo_cohort_rank_order(counts)
    assert [Counter(order)[rank] for rank in range(world)] == counts
    assert counts[0] == root_count
    c0, c1 = Counter(order[:batch // 2]), Counter(order[batch // 2:])
    assert c1[0] == (0 if zero else 1)
    assert c0[0] + c1[0] == root_count
    assert all(c0[rank] > 0 and c1[rank] > 0 for rank in range(1, world))


def test_root_capacity_is_not_silently_exceeded(monkeypatch):
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "24")
    with pytest.raises(RuntimeError, match="root rank"):
        compute_dbo_tknp_split(128, 4, [23 * 4, 128 * 4, 128 * 4, 128 * 4])


def test_projection_only_root_has_no_request_ownership(monkeypatch):
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "0")
    counts = compute_dbo_tknp_split(128, 4, [0, 256, 256, 256])
    order = build_tknp_dbo_cohort_rank_order(counts)
    assert counts[0] == 0 and 0 not in order
    assert len(order) == 128
    assert all(Counter(cohort)[rank] > 0
               for cohort in (order[:64], order[64:]) for rank in (1, 2, 3))


def test_zero_work_attention_never_reads_kv_metadata(monkeypatch):
    import vllm.attention.layer as layer

    monkeypatch.setattr(layer, "get_forward_context", lambda: SimpleNamespace(
        tknp_metadata=SimpleNamespace(tknp_enabled=True, local_num_tokens=0)))
    monkeypatch.setattr(layer, "get_attention_context", lambda *_: pytest.fail("KV accessed for padding"))
    query = torch.ones(1, 2, 4)
    output = torch.full_like(query, float("nan"))
    layer.unified_attention_with_output(query, query, query, output, "unused")
    assert torch.equal(output, torch.zeros_like(output))


def test_projection_only_unequal_cohorts_fail_closed(monkeypatch):
    from vllm.forward_context import tknp_cohort_split_point
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "0")
    monkeypatch.setenv("VLLM_TKNP_DBO_C0_FRACTION", "0.625")
    with pytest.raises(ValueError, match="equal cohorts"):
        tknp_cohort_split_point(256)


def test_unequal_cohorts_share_the_scheduler_split_point(monkeypatch):
    from vllm.forward_context import tknp_cohort_split_point
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_ZERO_ROOT_C1", "1")
    monkeypatch.setenv("VLLM_TKNP_DBO_C0_FRACTION", "0.625")
    counts = [48, 70, 69, 69]
    order = build_tknp_dbo_cohort_rank_order(counts)
    split = tknp_cohort_split_point(sum(counts))
    assert split == 160
    assert Counter(order[:split]) == {0: 48, 1: 38, 2: 37, 3: 37}
    assert Counter(order[split:]) == {1: 32, 2: 32, 3: 32}


def test_flashinfer_graph_plan_delegates_abi_and_stages_metadata(monkeypatch):
    import vllm.v1.attention.backends.flashinfer as backend
    calls = []
    monkeypatch.setattr(backend, "fast_decode_plan", lambda wrapper, **kw: calls.append(kw))
    wrapper = SimpleNamespace(
        is_cuda_graph_enabled=True, vllm_first_call=False, _fixed_batch_size=2,
        _paged_kv_indices_buf=torch.empty(4, dtype=torch.int32),
        _paged_kv_indptr_buf=torch.zeros(3, dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.zeros(2, dtype=torch.int32),
    )
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    last_page = torch.tensor([16, 15], dtype=torch.int32)
    backend.fast_plan_decode(wrapper, indptr, torch.arange(4), last_page,
                             torch.tensor([32, 31]), 16, 1, 128, 16)
    assert torch.equal(wrapper._paged_kv_indptr_buf, indptr)
    assert torch.equal(wrapper._paged_kv_last_page_len_buf, last_page)
    assert len(calls) == 1 and calls[0]["num_kv_heads"] == 1
    assert calls[0]["indptr"] is indptr


@pytest.mark.parametrize("root", [0, 40])
def test_tknp2_profile_uses_real_cohort_ownership(monkeypatch, root):
    import numpy as np
    import vllm.v1.worker.gpu_model_runner as runner
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", str(root))
    monkeypatch.setenv("VLLM_TKNP_DBO_ZERO_ROOT_C1", "1")
    monkeypatch.setenv("VLLM_TKNP_DBO_LAYOUT", "asymmetric")
    monkeypatch.setenv("VLLM_TKNP_DBO_C0_FRACTION", "0.5")
    monkeypatch.setattr(runner, "get_tknp_world_size", lambda: 2)
    fake = SimpleNamespace(_tknp_graph_splits={},
                           parallel_config=SimpleNamespace(enable_dbo=True))
    tokens, counts, order = runner.GPUModelRunner._dummy_tknp_layout(
        fake, 1024, 256, np.full(256, 4, dtype=np.int32))
    assert counts == [root, 256 - root]
    assert tokens == [root * 4, (256 - root) * 4]
    assert order[:128].count(1) > 0 and order[128:].count(1) > 0


def test_tp_allreduce_selects_only_the_active_cohort(monkeypatch):
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    import vllm.v1.worker.ubatching as ubatching

    calls = []
    fake = SimpleNamespace(tknp_dbo_tp_comms=[
        SimpleNamespace(all_reduce=lambda x: calls.append(0) or x + 1),
        SimpleNamespace(all_reduce=lambda x: calls.append(1) or x + 2),
    ])
    monkeypatch.setattr(ubatching, "dbo_enabled", lambda: True)
    monkeypatch.setattr(ubatching, "dbo_current_ubatch_id", lambda: 1)
    assert CudaCommunicator.all_reduce(fake, torch.tensor(3)).item() == 5
    assert calls == [1]


def test_cohort_plan_is_part_of_compile_cache_key(monkeypatch):
    from vllm.config import ParallelConfig
    config = ParallelConfig(enable_token_parallel=True, token_parallel_size=2,
                            enable_dbo=True)
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "0")
    zero = config.compute_hash()
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "64")
    assert config.compute_hash() != zero
    plain = ParallelConfig(enable_token_parallel=True, token_parallel_size=2)
    normal = plain.compute_hash()
    monkeypatch.setenv("VLLM_TKNP_DBO_ROOT_REQUESTS", "96")
    assert plain.compute_hash() == normal


def test_tp_allreduce_preserves_private_symmetric_backend(monkeypatch):
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    import vllm.v1.worker.ubatching as ubatching

    calls = []
    def nccl(x):
        calls.append("nccl")
        return x + 10
    def symm(x):
        calls.append("symm1")
        return x + 2
    fake = SimpleNamespace(
        tknp_dbo_tp_comms=[SimpleNamespace(all_reduce=nccl)] * 2,
        tknp_dbo_tp_symm_comms=[
            SimpleNamespace(should_use_symm_mem=lambda x: False),
            SimpleNamespace(should_use_symm_mem=lambda x: x.numel() == 1,
                            all_reduce=symm),
        ],
    )
    monkeypatch.setattr(ubatching, "dbo_enabled", lambda: True)
    monkeypatch.setattr(ubatching, "dbo_current_ubatch_id", lambda: 1)
    assert CudaCommunicator.all_reduce(fake, torch.tensor(3)).item() == 5
    assert calls == ["symm1"]
    assert torch.equal(CudaCommunicator.all_reduce(fake, torch.tensor([3, 4])),
                       torch.tensor([13, 14]))
    assert calls == ["symm1", "nccl"]
