#!/usr/bin/env python3
"""Collect preserved single-node efficiency evidence without running GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    matrix=read(root/"optimized/summary.json",{})
    micro=read(root/"microbatch_warm/summary.json",{})
    root_layout=read(root/"root_bandwidth/summary.json",{})
    root_micro=read(root/"root_microbatch/summary.json",{})
    native_tp=read(root/"native_tp_root/summary.json",{})
    gated=read(root/"gated_standard/summary.json",{})
    gated_pp=read(root/"gated_pp_cached/summary.json",{})
    gated_micro=read(root/"gated_micro/summary.json",{})
    cpu_pairs=read(root/"cpu_kv_pairs/pairs.json",{})
    pairs=read(root/"matched_pairs/pairs.json",{})
    numeric=read(root/"attention_numerics.json",{})
    rows=matrix.get("trials",[])+micro.get("trials",[])
    valid=all(r["performance_valid"] and r["all_ranks_verified"] and not r["measured_preemptions"]
        and r["output_tokens"]==r["batch"]*128 for r in rows)
    keys={(r['method'],r['mode'],r['batch']) for r in rows}
    expected={(method,mode,batch) for method in ("tp","pp-pipelined","dp-lifetime","tknp","tknp-tp2",
        "tp2-pp4","dp-tp2","tknp-dbo","tknp-tp2-dbo") for mode in ("eager","graph")
        if not (method.endswith("-dbo") and mode=="eager") for batch in (32,64,128,256)}
    traces={}
    for tag,parent in (("before",root),("after",root/"optimized")):
        for mode in ("eager","graph"):
            directory=parent/("profile-tknp-"+mode)
            files=sorted(directory.glob("server0/traces/*.pt.trace.json*"))
            journal=directory/"server0/rank0.jsonl"
            events=([json.loads(line) for line in journal.read_text().splitlines()]
                    if journal.exists() else [])
            steps=[e for e in events if e.get("event")=="step" and e.get("scheduled_tokens",0)]
            captured=steps[16:24]
            traces[tag+"-"+mode]=dict(files=[str(f) for f in files],
                analysis=str(directory/"trace_analysis.json"),count=len(files),
                full_batch_verified=len(captured)==8 and all(len(e["scheduled_request_ids"])==128 for e in captured))
    matched=[]
    for i in range(3):
        a=[r for r in pairs.get("trials",[]) if r["pair"]==i and r["variant"]=="legacy"]
        b=[r for r in pairs.get("trials",[]) if r["pair"]==i and r["variant"]=="fixed"]
        if len(a)==len(b)==1:
            assert a[0]["workload_sha256"]==b[0]["workload_sha256"]
            matched.append(dict(pair=i,legacy_tps=a[0]["output_tps"],fixed_tps=b[0]["output_tps"],
                burst_speedup=b[0]["output_tps"]/a[0]["output_tps"],
                legacy_step_ms=a[0]["engine_step_p50_ms"],fixed_step_ms=b[0]["engine_step_p50_ms"],
                engine_latency_speedup=a[0]["engine_step_p50_ms"]/b[0]["engine_step_p50_ms"]))
    root_comparisons=[]
    for method in ("tknp","tknp-tp2"):
        for batch in (32,64,128,256):
            baseline=next((r for r in rows if r["method"]==method and r["mode"]=="graph" and r["batch"]==batch),None)
            variants=[r for r in root_layout.get("trials",[]) if r["method"]==method and r["batch"]==batch]
            if baseline and variants:
                root_comparisons.append(dict(method=method,batch=batch,repeats=len(variants),
                    baseline_engine_tps=baseline["steady_engine_output_tps"],
                    root_aware_engine_tps=statistics.median(r["steady_engine_output_tps"] for r in variants),
                    baseline_burst_tps=baseline["output_tps"],
                    root_aware_burst_tps=statistics.median(r["output_tps"] for r in variants),
                    scope="Exploratory ownership control, not alternating matched pairs"))
    result=dict(passed=valid and keys==expected and len(rows)==len(expected) and pairs.get("passed",False)
                and numeric.get("passed",False) and all(t["count"]==8 and t["full_batch_verified"] for t in traces.values()),
        expected_matrix_points=len(expected),validated_matrix_points=len(rows),
        matrix=rows,matched_pairs=matched,numerical_checks=numeric,traces=traces,
        capacities=dict(matrix.get("capacities",{}),**micro.get("capacities",{})),dcp=matrix.get("dcp"),
        root_layout_comparisons=root_comparisons,root_layout_trials=root_layout.get("trials",[]),
        cpu_counter_pairs=cpu_pairs,controls=read(root/"controls.json",{}),
        source_snapshot=str(root/"source_pp_compact_v2"),
        historical_main_source_snapshot=str(root/"source_fixed"),
        source_versions=dict(original_64_point_matrix=str(root/'source_fixed'),
            original_microbatch_controls=str(root/'source_controls_v2'),
            native_tp_controls=str(root/'source_native_tp_v2'),
            gated_non_pp_matrix=str(root/'source_decode_gate'),
            repaired_pp_microbatch_and_final_profiles=str(root/'source_pp_compact_v2')),
        control_source_snapshot=str(root/"source_controls_v2"),
        scope="One EOS node/eight H100s; Llama 3.1 8B dummy BF16 weights and BF16 random physical KV; ISL32768/OSL128; no prefill; performance excludes profiler runs. Burst throughput includes frontend admission and queueing; steady engine rates exclude frontend and are not GPU-kernel-only rates.")
    if matched:result['median_paired_burst_speedup']=statistics.median(r['burst_speedup'] for r in matched)
    result['median_paired_engine_latency_speedup']=(
        statistics.median(r['engine_latency_speedup'] for r in matched) if matched else None)
    result['root_microbatch_trials']=root_micro.get('trials',[])
    result['native_tp_trials']=native_tp.get('trials',[])
    result['native_tp_controls']=read(root/'native_tp_controls.json',{})
    result['selected_microbatch']=read(root/'selected_microbatch.json',{})
    result['prompt_copy_microbenchmark']=read(root/'prompt_copy_micro.json',{})
    result['final_controls']=read(root/'final_controls.json',{})
    result['kernel_geometry_checks']=read(root/'kernel_cliffs_retry1.json',{})
    result['pp_profile_concurrency']=read(root/'profile_pp_pipeline/concurrency.json',{})
    result['pp_cached_profile_status']=read(root/'pp_cached_profile_status.json',{})
    result['pp_cached_profile_concurrency']=read(root/'profile_pp_cached/concurrency.json',{})
    result['selected_microbatch_profile']=read(
        root/'profile_micro_selected/profile-tknp-dbo-graph/trace_analysis.json',{})
    final_profile=root/'profile_final_tknp/profile-tknp-dbo-graph'
    final_files=sorted(final_profile.glob('server0/traces/*.pt.trace.json*'))
    final_journal=final_profile/'server0/rank0.jsonl'
    final_steps=([json.loads(line) for line in final_journal.read_text().splitlines()]
                 if final_journal.exists() else [])
    final_steps=[e for e in final_steps if e.get('event')=='step' and e.get('scheduled_tokens',0)]
    final_capture=final_steps[16:24]
    result['final_token_parallel_profile']=dict(files=[str(p) for p in final_files],
        count=len(final_files),full_batch_verified=len(final_capture)==8 and
        all(len(e['scheduled_request_ids'])==128 for e in final_capture),
        scope='Final source, graph two-cohort TKNP8, B128/ISL32768; performance-invalid profiler diagnostic',
        analysis=read(final_profile/'trace_analysis.json',{}))
    # Keep historical rows and select repaired PP only where a matched row exists.
    repaired={(r['method'],r['mode'],r['batch']):r for r in gated_pp.get('trials',[])}
    fair=[repaired.get((r['method'],r['mode'],r['batch']),r) for r in gated.get('trials',[])]
    result['gated_standard_original']=gated.get('trials',[])
    result['gated_pp_cached']=gated_pp.get('trials',[])
    result['gated_primary_comparison']=fair
    result['gated_microbatch']=gated_micro.get('trials',[])
    result['prompt_cache_pairs']={flag:read(root/('prompt_cache_pair'+flag)/'summary.json',{})
                                  for flag in ('0','1')}
    fair_expected={(m,mode,b) for m in ('tp','pp-pipelined','dp-lifetime','tknp','tknp-tp2','tp2-pp4','dp-tp2')
                   for mode in ('eager','graph') for b in (128,256)}
    result['fair_decode_comparison_passed']=(
        {(r['method'],r['mode'],r['batch']) for r in fair}==fair_expected
        and len(fair)==28 and len(gated_micro.get('trials',[]))==8
        and all(r.get('admission_gated') and r['performance_valid'] and r['all_ranks_verified']
                and r['output_tokens']==r['batch']*128 and not r['measured_preemptions']
                for r in fair+gated_micro.get('trials',[])))
    for batch in (128,256):
        hashes={r['workload_sha256'] for r in fair+gated_micro.get('trials',[])
                if r['batch']==batch}
        if len(hashes)>1:raise RuntimeError('Gated comparison workload differs across methods')
    result['main_matrix_passed']=result['passed']
    result['passed']=(result['passed'] and result['fair_decode_comparison_passed']
                      and result['final_controls'].get('passed',False)
                      and result['pp_cached_profile_status'].get('passed',False)
                      and result['final_token_parallel_profile']['count']==8
                      and result['final_token_parallel_profile']['full_batch_verified'])
    result['report_generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['accepted_default_fixes']=[
        'Hopper GQA4 FlashInfer CUDA-core decode selection; other head ratios unchanged',
        'Public FlashInfer API split sentinel normalization',
        'Bounded immutable TKNP index tensor cache',
        'Initialize cohort NCCL P2P links before KV sizing and graph capture',
        'Immutable CPU prompt arrays avoid repeated long-prompt conversion in rotating PP request sets',
        'Synchronous PP avoids unused full-prompt IPC on rotations; retains full history for async recovery, preemption and non-opted-in connectors']
    result['not_promoted']=[
        'CPU KV-counter collective: no speed gain; exact-token A/B inconclusive due different admission/physical-KV placement; opt-in remains OFF',
        'Fixed CUDA-core chunk planner: diagnostic-only private ABI; inconsistent gain, not used in runtime',
        'Private custom TP all-reduce: component correctness passed but no gain; opt-in remains OFF',
        'Root-aware ownership is a proxy, opt-in; use microbatch only where measured beneficial']
    result['measurement_notes']=[
        'Primary gated decode TPS starts after all requests are CPU queued; includes natural KV waves, decode, streaming and completion',
        'Older burst TPS includes HTTP body/admission preparation and is not pure decode throughput',
        'Steady engine TPS is a peak-resident-wave service rate, not whole-burst completion or GPU-only time',
        'All weights/KV synthetic BF16; no real-model quality, prefill, production-arrival or multi-node scaling claim',
        'Nontrivial DCP invalid for exact 8B model on eight GPUs in this fork; DCP1 is TP8']
    result['microbatch_comparisons']=[]
    for method in ('tknp','tknp-tp2'):
        for batch in (128,256):
            base=next((r for r in fair if r['method']==method and r['mode']=='graph'
                       and r['batch']==batch),None)
            variants=[r for r in gated_micro.get('trials',[])
                      if r['method']==method+'-dbo' and r['batch']==batch]
            if base and variants:
                tps=statistics.median(r['output_tps'] for r in variants)
                engine=statistics.median(r['steady_engine_output_tps'] for r in variants)
                result['microbatch_comparisons'].append(dict(method=method,batch=batch,
                    repeats=len(variants),standard_decode_tps=base['output_tps'],
                    micro_decode_tps=tps,decode_speedup=tps/base['output_tps'],
                    standard_engine_tps=base['steady_engine_output_tps'],micro_engine_tps=engine,
                    engine_speedup=engine/base['steady_engine_output_tps'],
                    scope='Repeated microbatch controls versus one matched standard point; not alternating pairs'))
    result['pp_metadata_control']={}
    for flag in ('0','1'):
        trials=result['prompt_cache_pairs'][flag].get('trials',[])
        if trials:
            result['pp_metadata_control'][flag]=dict(repeats=len(trials),
                median_decode_tps=statistics.median(r['output_tps'] for r in trials),
                median_engine_tps=statistics.median(r['steady_engine_output_tps'] for r in trials),
                scope='Combined immutable prompt-array cache and compact synchronous PP IPC; two repeats per setting')
    for batch in (32,64,128,256):
        hashes={r["workload_sha256"] for r in rows if r["batch"]==batch}
        if len(hashes)>1:raise RuntimeError("Matrix workload differs across methods")
    compact=[]
    for method in ('tp','pp-pipelined','dp-lifetime','tknp','tknp-tp2','tp2-pp4','dp-tp2'):
        row=dict(method=method)
        for mode in ('eager','graph'):
            for batch in (128,256):
                value=next((r['output_tps'] for r in fair if r['method']==method
                            and r['mode']==mode and r['batch']==batch),None)
                row[mode+'_b'+str(batch)+'_tps']=value
        compact.append(row)
    for method in ('tknp-dbo','tknp-tp2-dbo'):
        row=dict(method=method,eager_supported=False)
        for batch in (128,256):
            values=[r['output_tps'] for r in gated_micro.get('trials',[])
                    if r['method']==method and r['batch']==batch]
            row['graph_b'+str(batch)+'_tps']=statistics.median(values) if values else None
        compact.append(row)
    result=dict(passed=result.pop('passed'),
        primary_scope='Authoritative completion-TPS comparison: single-node/eight H100s, Llama 3.1 8B Instruct, BF16 dummy weights/BF16 random physical KV, ISL32768/OSL128. All CPU requests prepared before release; natural KV waves and decode serving included; prefill, model loading, KV fill and profiling excluded. Micro rows are medians of two repeats.',
        primary_comparison_tps=compact,
        historical_data_note='The matrix field is the earlier 64-point diagnostic grid: its burst rates include HTTP admission and its PP rows precede the metadata repair. Do not substitute it for primary_comparison_tps or gated_primary_comparison.',
        **result)
    (root/"efficiency_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print('VALIDATED_MATRIX',len(rows),'/',len(expected),'PASSED',result['passed'])
    if matched:print('PAIRED_FIX_SPEEDUPS',[round(r['burst_speedup'],3) for r in matched])
    print('GATED_DECODE_COMPLETE',result['fair_decode_comparison_passed'],len(fair),
          'MICRO',len(gated_micro.get('trials',[])))
    for r in sorted(fair,key=lambda r:(r['method'],r['mode'],r['batch'])):
        print('GATED',r['method'],r['mode'],r['batch'],round(r['output_tps'],1))
    for method in ("tp","pp-pipelined","dp-lifetime","tknp","tknp-tp2","tp2-pp4","dp-tp2","tknp-dbo","tknp-tp2-dbo"):
        for mode in ("eager","graph"):
            data=sorted((r for r in rows if r['method']==method and r['mode']==mode),key=lambda r:r['batch'])
            if data:print(method,mode,[(r['batch'],round(r['output_tps'],1),round(r['steady_engine_output_tps'],1),round(r['engine_step_p50_ms'],2)) for r in data])


if __name__=="__main__":main()
