"""CPU-side fail-closed contracts for drained multi-layout TKNP sweeps."""
from types import SimpleNamespace

import pytest

from vllm.v1.worker.tknp_serving_slots import (
    cached_graph_inventory, empty_root_position_rows, fit_capture_split, normalize_capture_layouts,
    select_capture_layout, validate_multi_layout_dbo_shapes)


def test_multiple_capture_sizes_and_independent_capacity_fitting():
    layouts = normalize_capture_layouts({"8": [2, 6], "16": [4, 12], "32": [8, 24]}, 2)
    assert list(layouts) == [8, 16, 32]
    assert [fit_capture_split(counts, size, [601, 3301], 100)
            for size, counts in layouts.items()] == [[2, 6], [4, 12], [6, 26]]


@pytest.mark.parametrize("layouts", [{}, [], {"7": [2, 5]}, {"8": [2, 4]},
    {"8": [2, -1]}, {"8": [8, 0]}, {"8": [True, 7]}, {"08": [2, 6]}])
def test_reject_invalid_layouts(layouts):
    with pytest.raises((ValueError, TypeError)):
        normalize_capture_layouts(layouts, 2)


def test_drained_switches_reuse_exact_layouts_without_mutating_table():
    layouts = {8: [2, 6], 16: [4, 12], 32: [8, 24]}
    for size in [8, 32, 16, 32, 8]:
        selected = select_capture_layout(layouts, size)
        assert selected == layouts[size] and selected is not layouts[size]
    for conditions in (dict(remaining_requests=1), dict(batch_queue_depth=1), dict(all_kv_free=False)):
        with pytest.raises(ValueError):
            select_capture_layout(layouts, 16, **conditions)
    for size in (4, "8", 64, True):
        with pytest.raises(ValueError):
            select_capture_layout(layouts, size)


def test_graph_inventory_handles_native_and_dbo_wrappers_and_cycles():
    graph0, graph1 = object(), object()
    native = SimpleNamespace(concrete_cudagraph_entries={"shape8": SimpleNamespace(cudagraph=graph0)})
    wrapper = SimpleNamespace(cudagraph_wrapper=native, cudagraphs={16: SimpleNamespace(cudagraph=graph1)})
    wrapper.runnable = wrapper
    inventory = cached_graph_inventory(wrapper)
    assert set(inventory.values()) == {id(graph0), id(graph1)}
    assert inventory == cached_graph_inventory(wrapper)


def test_zero_work_root_cohort_has_matching_compiler_padding_only():
    assert empty_root_position_rows([0, 0]) == 1
    assert empty_root_position_rows([8, 0]) == 8
    assert empty_root_position_rows([4, 0]) == 4
    validate_multi_layout_dbo_shapes({8: [(2, 0), (2, 4)], 16: [(4, 0), (4, 8)]})
    with pytest.raises(ValueError):
        validate_multi_layout_dbo_shapes({8: [(2, 0), (1, 5)]})
    with pytest.raises(ValueError):
        validate_multi_layout_dbo_shapes({8: [(0, 0), (4, 4)], 16: [(4, 0), (4, 8)]})


def test_sweep_control_requires_explicit_captured_layout():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"))
    from e2e_server import validate_sweep_control
    base = dict(tknp=2, pp=1, graph_maximum=32, preinitialized=True,
                layouts={8: [2, 6], 32: [8, 24]})
    validate_sweep_control(dict(key="b8-r0", action="prepare", graph_size=8), **base)
    validate_sweep_control(dict(key="b8-r0", action="checkpoint"), **base)
    for body in (dict(key="b16", action="prepare", graph_size=16),
                 dict(key="b8", action="checkpoint", graph_size=8)):
        with pytest.raises(ValueError):
            validate_sweep_control(body, **base)


def test_four_node_plan_is_supported_without_allocating_nodes(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/offline_inference/TKNP"))
    from e2e_matrix import validate_matrix
    from eos_tknp_graph_readiness import configuration
    from tknp_graph_sweep import make_segments
    from fp8_decode_sweep import trial_for
    config = configuration(32768, 1024, tp=8, tknp=4, dbo=True, random_kv=True)
    validate_matrix(config, 32)
    segments = make_segments(config, [16, 32, 64, 96, 16], tmp_path)
    assert [segment["graph_size"] for segment in segments] == [16, 32, 64, 64, 16]
    assert len(config["trials"][0]["servers"][0]["gpus"]) == 32
    for batch in [4, 8, 16, 32, 48, 128]:
        trial = trial_for("tknp-tp8", 32768, 1024, batch, [6, 33, 33, 33], gpu_count=32)
        validate_matrix(dict(trials=[trial]), 32)
        role = trial["servers"][0]
        assert role["tp"] == 8 and role["tknp"] == 4
        assert sum(role["owner_counts"]) == role["graph_batch_size"]
        assert role["graph_batch_size"] <= 104
