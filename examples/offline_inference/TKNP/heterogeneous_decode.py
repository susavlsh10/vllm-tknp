"""Controlled mixed-context workloads and per-length cached-decode metrics."""
from collections import Counter
import json
from pathlib import Path
import random


DISTRIBUTIONS = {
    'uniform64k': [65536],
    'balanced32_64_96k': [32768,65536,98304],
    'uniform48k': [49152],
    'skewed32_96k': [32768,32768,32768,98304],
}


def make_mixed_workload(distribution, batch, output_tokens, seed=20261021, order='shuffled'):
    from e2e_workload import make_workload
    pattern=DISTRIBUTIONS[distribution]
    if batch<=0 or batch%len(pattern) or output_tokens<=0:
        raise ValueError('Exact controlled mixtures require batch divisible by pattern length')
    lengths=(pattern*(batch//len(pattern)))
    rows=make_workload(batch,lengths,[output_tokens],0,seed)
    if order=='shuffled':random.Random(seed+1).shuffle(rows)
    elif order=='long-first':rows.sort(key=lambda r:len(r['prompt_token_ids']),reverse=True)
    else:raise ValueError('Unsupported order')
    return rows


def workload_description(rows):
    hist=Counter(len(r['prompt_token_ids']) for r in rows)
    return dict(requests=len(rows),context_histogram=dict(sorted(hist.items())),
                total_prompt_tokens=sum(k*v for k,v in hist.items()),
                mean_prompt_tokens=sum(k*v for k,v in hist.items())/len(rows),
                total_output_tokens=sum(r['max_tokens'] for r in rows),
                arrival_mode='all queued behind decode release gate')


def summarize_mixture(point, workload):
    from e2e_workload import percentile
    point=Path(point)
    requests=[json.loads(l) for l in (point/'client/requests.jsonl').read_text().splitlines()]
    client=json.loads((point/'client/summary.json').read_text())
    expected={r['request_id']:r for r in workload}
    assert len(requests)==len(workload)
    assert len({r['request_id'] for r in requests})==len(workload)
    by_id={r['server_request_id']:r for r in requests}
    groups={};queues={};owner_imbalance=[];peak_context_work=[]
    for r in requests:
        req=expected[r['request_id']]
        assert r['output_tokens']==req['max_tokens'] and r['prompt_tokens']==len(req['prompt_token_ids'])
        groups.setdefault(r['prompt_tokens'],[]).append(r)
    for path in point.glob('server*/rank*.jsonl'):
        release=None;first={}
        for line in path.read_text().splitlines():
            e=json.loads(line)
            if e['event']=='decode_gate' and e['action']=='release':release=e['time']
            if e['event']=='first_compute' and e['request_id'] in by_id:
                first.setdefault(e['request_id'],e['time'])
            if path.stem=='rank0' and e['event']=='step' and set(e.get('scheduled_request_ids',[]))&by_id.keys():
                work=e.get('owner_context_tokens',[])
                if len(work)>1 and sum(work)>0:
                    owner_imbalance.append(max(work)/(sum(work)/len(work)))
                    peak_context_work.append(work)
        if release is not None:
            for key,t in first.items():queues.setdefault(key,max(0,t-release))
    preparation=client['admission_gated_decode']['preparation_seconds']
    per_length={}
    for length,rs in sorted(groups.items()):
        ttft=[r['arrival_s']+r['dispatch_lateness_s']+r['token_times_s'][0]-preparation for r in rs]
        delays=[queues[r['server_request_id']] for r in rs if r['server_request_id'] in queues]
        gaps=[g for r in rs for g in r['intertoken_s']]
        def quantiles(v):return {f'p{int(q*100)}':percentile(v,q) for q in (.5,.95,.99)}
        per_length[str(length)]=dict(requests=len(rs),output_tokens=sum(r['output_tokens'] for r in rs),
            ttft_from_release_s=quantiles(ttft),scheduler_queue_s=quantiles(delays),
            client_intertoken_s=quantiles(gaps),
            mean_tpot_s=quantiles([r['tpot_s'] for r in rs if r['tpot_s'] is not None]))
    return dict(workload=workload_description(workload),per_context=per_length,
        scheduler_queue_observed_requests=len(queues),
        owner_context_imbalance_max_over_mean=dict(p50=percentile(owner_imbalance,.5),p95=percentile(owner_imbalance,.95)),
        owner_context_work_peak=max(peak_context_work,key=sum) if peak_context_work else None,
        caveat='Owner context tokens are an attention-work proxy, not measured SM occupancy or KV bytes.')
