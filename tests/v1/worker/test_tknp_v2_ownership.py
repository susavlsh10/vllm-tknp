"""Pure-Python ownership planning tests; no GPUs or vLLM import required."""
import importlib.util
from pathlib import Path

_path=Path(__file__).parents[3]/'vllm/distributed/tknp_layer_ownership.py'
_spec=importlib.util.spec_from_file_location('tknp_v2_plan',_path)
plan=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)


def test_equal_layers_with_endpoint_cost():
    owners=plan.partition_layers([10]*32,60,2)
    assert owners.count(0)==13 and owners.count(1)==19
    assert list(owners)==sorted(owners)


def test_llama8b_four_owners():
    owners=plan.llama_owners(32,4096,14336,32,8,128,128256,False,2,4)
    assert len(owners)==32 and set(owners)==set(range(4))
    assert owners.count(0)<owners.count(1)
    assert list(owners)==sorted(owners)
    assert [owners.count(i) for i in range(4)]==[5,9,9,9]


def test_single_owner():
    assert plan.partition_layers([3,4],100,1)==(0,0)


def test_recipe_exempt_layers_change_storage():
    skip=('model.layers.0.self_attn', 'model.layers.1.mlp')
    costs=plan.llama_layer_costs(3,4096,14336,32,8,128,1,skip)
    assert costs[1]>costs[0]>costs[2]
    assert plan.llama_layer_costs(3,4096,14336,32,8,128,2,skip)==plan.llama_layer_costs(3,4096,14336,32,8,128,2)


def test_minimax_matches_bruteforce():
    import itertools
    costs=(3,8,2,5,11,1)
    for degree in range(1,5):
        owners=plan.partition_layers(costs,7,degree)
        actual=max(sum(c for c,o in zip(costs,owners) if o==r)+(7 if r==0 else 0)
                   for r in range(degree))
        optimum=min(max(sum(costs[a:b])+(7 if r==0 else 0)
                        for r,(a,b) in enumerate(zip((0,)+cuts,cuts+(len(costs),))))
                    for cuts in itertools.combinations(range(1,len(costs)),degree-1))
        assert actual==optimum


def test_capacity_safe_equal_graph_layouts():
    import runpy
    sweep=runpy.run_path(str(Path(__file__).parents[3]/'examples/offline_inference/TKNP/tknp_v2_405b_sweep.py'))
    maximum,sizes=sweep['equal_layouts']([4,8,16,32,64,96,128],[27,28,28,27])
    assert maximum==108 and sizes==[4,8,16,32,64,96,108]
    assert all(size//4<=27 and size%4==0 for size in sizes)
