from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"examples/offline_inference/TKNP"))
from profile_decode_matrix import profile_plan


def test_profile_preserves_engine_and_layout_identity(tmp_path):
    import copy
    source=dict(trials=[dict(servers=[dict(tp=8,tknp=4,max_num_seqs=108,
        graph_batch_size=108,graph_layouts={32:[6,9,9,8],96:[6,30,30,30],108:[6,34,34,34]})])],
        client_sweep=[dict(key="b96-r0",batch=96,graph_size=96,workload="w",output="old")],
        runtime=dict(repo="frozen"))
    before=copy.deepcopy(source)
    cfg,point=profile_plan(source,"b96-r0",tmp_path)
    assert source==before
    role=cfg["trials"][0]["servers"][0]
    expected=dict(before["trials"][0]["servers"][0],profile_after_step=32,profile_steps=8)
    assert role==expected
    assert cfg["runtime"]==before["runtime"]
    assert point["graph_size"]==96 and point["batch"]==96
    assert point["output"]==str(tmp_path/"point")
