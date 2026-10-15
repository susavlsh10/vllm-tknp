"""GPU regression for native FlashInfer single-KV-head graph planning."""
import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from vllm.v1.attention.backends.flashinfer import fast_plan_decode


def main():
    torch.manual_seed(1234)
    torch.cuda.set_device(0)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    indices = torch.arange(4, dtype=torch.int32, device="cuda")
    last_page = torch.tensor([16, 15], dtype=torch.int32)
    indptr_gpu = indptr.to("cuda")
    last_gpu = last_page.to("cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", use_cuda_graph=True, use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr_gpu, paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last_gpu,
    )
    query = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4, 2, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(num_qo_heads=16, num_kv_heads=1, head_dim=128, page_size=16,
                  q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    wrapper.plan(indptr, indices, last_page, **kwargs)
    expected = wrapper.run(query, kv).clone()
    fast_plan_decode(wrapper, indptr, indices, last_page,
                     torch.tensor([32, 31], dtype=torch.int32), **kwargs)
    # The first fast-plan call initializes the wrapper; the next uses its
    # cached native planning module, which exposed the stale 19/20-arg ABI.
    fast_plan_decode(wrapper, indptr, indices, last_page,
                     torch.tensor([32, 31], dtype=torch.int32), **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(query, kv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = wrapper.run(query, kv)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    # Re-plan a different KV length into the same graph input buffers, and
    # verify that replay sees fresh metadata rather than stale capture values.
    last_page[1] = 7
    wrapper.plan(indptr, indices, last_page, **kwargs)
    expected = wrapper.run(query, kv).clone()
    fast_plan_decode(wrapper, indptr, indices, last_page,
                     torch.tensor([32, 23], dtype=torch.int32), **kwargs)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    print("FLASHINFER_FAST_PLAN_GRAPH_PASS kv_heads=1 replay_metadata_updates=True")


if __name__ == "__main__":
    main()
