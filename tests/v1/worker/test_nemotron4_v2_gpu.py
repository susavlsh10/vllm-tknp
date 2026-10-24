"""Small CUDA checks before launching distributed Nemotron compatibility runs."""
import pytest
import torch


def test_full_logit_comparison_checks_replicated_tp_copies(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    script=Path(__file__).resolve().parents[3]/'examples/offline_inference/TKNP/tknp_v2_compare.py'
    spec=importlib.util.spec_from_file_location('nemotron4_logit_comparison',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest=hashlib.sha256(b'server-request').hexdigest()
    for name in ['reference','candidate']:
        stage=tmp_path/name
        (stage/'b1/client').mkdir(parents=True)
        (stage/'b1/client/requests.jsonl').write_text(json.dumps(dict(
            request_id='request-0',server_request_id='server-request',token_ids=[2]))+'\n')
        (stage/'b1/server0').mkdir()
        (stage/'b1/server0/manifest.json').write_text(json.dumps(dict(
            worker_resources=[dict(rank=rank,tknp_v2=None) for rank in [0,1]])))
        for rank in [0,1]:
            path=stage/'engine/server0'/f'full-logits-rank{rank}'
            path.mkdir(parents=True)
            torch.save(dict(step=0,request_id='server-request',logits=torch.tensor([0.,1.,3.])),
                       path/(digest+'-0.pt'))
    result=module.compare(tmp_path/'reference',tmp_path/'candidate',1)
    assert result['passed'] and result['replicated_logit_copies_checked']==2
    stage=tmp_path/'candidate'
    dummy=stage/'engine/server0/full-logits-rank2'
    dummy.mkdir()
    torch.save(dict(step=0,request_id='server-request',logits=torch.zeros(3)),dummy/(digest+'-0.pt'))
    (stage/'b1/server0/manifest.json').write_text(json.dumps(dict(worker_resources=[
        dict(rank=rank,tknp_v2=dict(tknp_rank=0 if rank<2 else 1,control_rank=0))
        for rank in [0,1,2]])))
    result=module.compare(tmp_path/'reference',stage,1)
    assert result['passed'] and result['non_output_rank_placeholders_excluded']==1
    torch.save(dict(step=0,request_id='server-request',logits=torch.tensor([0.,1.,4.])),
               tmp_path/'candidate/engine/server0/full-logits-rank1'/(digest+'-0.pt'))
    with pytest.raises(AssertionError,match='Replicated TP logits disagree'):
        module.compare(tmp_path/'reference',tmp_path/'candidate',1)


def test_local_nemotron_tokenizer_has_correct_vocabulary_and_special_ids():
    import os
    from transformers import AutoTokenizer
    path=os.environ.get('NEMOTRON4_TOKENIZER')
    if not path:pytest.skip('Tokenizer path supplied by interactive gate')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    assert tokenizer.vocab_size==256000
    assert tokenizer.bos_token_id==2 and tokenizer.eos_token_id==3


def test_effective_128k_model_limit_without_loading_weights():
    import os
    from pathlib import Path
    from vllm.config import ModelConfig
    path=os.environ.get('NEMOTRON4_MODELS')
    if not path:pytest.skip('Model metadata supplied by interactive gate')
    config=ModelConfig(model=str(Path(path)/'tiny_fp8'),
        tokenizer=os.environ['NEMOTRON4_TOKENIZER'],dtype='bfloat16',max_model_len=131072)
    assert config.max_model_len==131072
    assert config.hf_config.head_dim==192
    assert config.hf_config.rope_parameters['partial_rotary_factor']==.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('head_dim',[128,192,256])
def test_dcp_correction_supports_non_power_of_two_heads(head_dim):
    from vllm.attention.ops.common import CPTritonContext,correct_attn_out
    raw=torch.randn(3,5,head_dim*2,device='cuda',dtype=torch.bfloat16)[...,::2]
    lses=torch.randn(4,5,3,device='cuda',dtype=torch.float32).permute(0,2,1)
    lses[:,0,0]=-float('inf')
    lses[0,1,0]=float('inf')
    clean=torch.where(torch.isnan(lses)|torch.isposinf(lses),-float('inf'),lses)
    expected_lse=torch.logsumexp(clean,dim=0)
    factor=torch.nan_to_num(torch.exp(clean[1]-expected_lse),nan=0.)
    expected=raw.float()*factor[...,None]
    ctx=CPTritonContext()
    for _ in range(2):
        actual,actual_lse=correct_attn_out(raw.clone(),lses,1,ctx)
        torch.testing.assert_close(actual.float(),expected,rtol=.01,atol=.01)
        torch.testing.assert_close(actual_lse,expected_lse,rtol=1e-5,atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_layernorm1p_preserves_residual_and_zero_centered_scale():
    from vllm.model_executor.models.nemotron import NemotronLayerNorm1P
    norm=NemotronLayerNorm1P(192,eps=1e-5,device='cuda',dtype=torch.bfloat16)
    x=torch.randn(4,192,device='cuda',dtype=torch.bfloat16)
    residual=torch.randn_like(x)
    combined=x+residual
    out,updated=norm(x,residual)
    raw=getattr(norm,'module',norm)
    expected=torch.nn.functional.layer_norm(combined,(192,),raw.weight+1,raw.bias,raw.eps)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    torch.testing.assert_close(updated,combined,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_partial_rotary_192_supports_extended_positions():
    from vllm.model_executor.layers.rotary_embedding import get_rope
    rope=get_rope(192,4096,rope_parameters=dict(rope_type='linear',factor=32.0,
                   rope_theta=10000.0,partial_rotary_factor=.5),dtype=torch.bfloat16).cuda()
    positions=torch.tensor([4095,32767,65535,131071],device='cuda')
    q=torch.randn(4,3*192,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(4,192,device='cuda',dtype=torch.bfloat16)
    a,b=rope(positions,q.clone(),k.clone())
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    torch.testing.assert_close(a.view(4,3,192)[...,96:],q.view(4,3,192)[...,96:],rtol=0,atol=0)
    torch.testing.assert_close(b[...,96:],k[...,96:],rtol=0,atol=0)
