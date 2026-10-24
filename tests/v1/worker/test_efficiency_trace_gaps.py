from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"examples/offline_inference/TKNP"))
from eos_efficiency_trace import inter_kernel_gaps,is_dense_kernel


def test_cublas_nvjet_bf16_projections_are_counted_as_dense():
    assert is_dense_kernel("nvjet_tst_64x96_64x10_4x1_v_bz_splitK_TNT")
    assert not is_dense_kernel("ncclDevKernel_SendRecv")


def test_attention_gaps_are_not_summed_across_overlapping_intervals():
    assert inter_kernel_gaps([])=={"count":0}
    rows=[dict(ts=0,dur=10),dict(ts=2,dur=3),dict(ts=20,dur=4),dict(ts=130,dur=1)]
    result=inter_kernel_gaps(rows)
    assert result["count"]==3
    assert result["p50_us"]==10 and result["max_us"]==106
    assert result["above_100us"]==1
