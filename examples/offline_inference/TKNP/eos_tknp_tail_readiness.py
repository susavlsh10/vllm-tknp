#!/usr/bin/env python3
"""Single-node 8B sticky-owner tail-graph correctness, inside an existing job.

Constant synthetic KV deliberately makes request outputs independent of physical
block placement. This is correctness/graph evidence, NOT throughput data.
Both policies keep the same admission layouts and compare every output token.
"""
import argparse
import json
import os
from pathlib import Path

from e2e_matrix import run_trial,validate_matrix
from eos_efficiency import configuration as runtime_configuration,write_json
from eos_tknp_graph_readiness import configuration
from tknp_graph_sweep import make_segments,audit_sweep
from e2e_workload import make_workload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--attention-backend",choices=["FLASHINFER","FLASH_ATTN"],default="FLASHINFER")
    p.add_argument("--dbo-threshold",type=int,default=2)
    p.add_argument("--dbo-only",action="store_true")
    p.add_argument("--root-empty",action="store_true",
                   help="Validate existing projection-only root path with zero owned KV requests")
    p.add_argument("--tokens",type=int,default=33)
    p.add_argument("--diagnostic-logits",action="store_true",
                   help="Persist full logits (no KV samples), retaining strict token failure")
    p.add_argument("--heterogeneous",action="store_true",
                   help="Use mixed 256/384/512-token contexts in the exact-output pairing")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for dbo in ([True] if args.dbo_only else (False,True)):
        golden=None
        for downsize in (False,True):
            destination=args.output/f"dbo{int(dbo)}-tail{int(downsize)}"
            destination.mkdir()
            cfg=configuration(512,args.tokens,2,4,dbo,False)
            cfg["preinitialize_constant_kv"]=True
            role=cfg["trials"][0]["servers"][0]
            if args.diagnostic_logits:
                role.update(record_full_logits=True,skip_diagnostic_kv=True)
            if args.heterogeneous:
                # Control owner counts so this check exercises BOTH replay
                # types; load-balancer quality is a separate experiment.
                role["placement"]="count"
            role["graph_layouts"]={16:[2,5,5,4],32:[2,10,10,10],64:[2,21,21,20]}
            if args.root_empty:
                if not dbo:raise ValueError("Projection-only compiler padding requires the DBO wrapper")
                role["graph_layouts"]={16:[0,6,5,5],32:[0,11,11,10],64:[0,22,21,21]}
                role["env"]["VLLM_TKNP_DBO_ROOT_REQUESTS"]="0"
            role["env"].update(VLLM_TKNP_GRAPH_TAIL_DOWNSIZE="1" if downsize else "0",
                                VLLM_TKNP_DBO_WARMUP_P2P="1",
                                VLLM_TKNP_DBO_DECODE_THRESHOLD=str(args.dbo_threshold))
            runtime=runtime_configuration(args.job,args.node,destination,512,args.tokens)
            runtime["runtime"]["env"]["VLLM_ATTENTION_BACKEND"]=args.attention_backend
            os.environ["VLLM_ATTENTION_BACKEND"]=args.attention_backend
            cfg.update(slurm_job_id=args.job,slurm_nodes=[args.node],gpus_per_node=8,
                runtime=runtime["runtime"],nixl_port_base=21200,rdzv_port_base=26500,
                # Align decode starts so HTTP arrival jitter cannot change
                # the last-token graph shape in an exact-output pairing.
                admission_gated_decode=True)
            validate_matrix(cfg,8)
            segments=make_segments(cfg,[16,32,96,32],destination)
            if args.heterogeneous:
                for segment in segments:
                    Path(segment["workload"]).write_text("".join(json.dumps(row)+"\n" for row in
                        make_workload(segment["batch"],[256,384,512],[args.tokens],0,20261016)))
            cfg["client_sweep"]=segments
            write_json(destination/"matrix.json",cfg)
            print("TAIL_READINESS_START",dbo,downsize,flush=True)
            run_trial(cfg["trials"][0],cfg,Path(segments[0]["workload"]),destination/"engine")
            summary=audit_sweep(cfg,segments)
            outputs=[]
            evidence=[]
            for segment in segments:
                point=Path(segment["output"])
                outputs.append({row["request_id"]:row["token_ids"] for row in
                    map(json.loads,(point/"client/requests.jsonl").read_text().splitlines())})
                records=json.loads((point/"server0/execution.json").read_text())
                if segment["batch"]==96:
                    for record in records:
                        sizes={int(k.split(":")[1]):v for k,v in record["counts"].items()
                               if k.startswith("tknp_graph_size:") and v}
                        if 64 not in sizes or (downsize and 32 not in sizes):
                            raise RuntimeError("Tail did not exercise both large/small actual graph replays")
                        if not downsize and set(sizes)!={64}:
                            raise RuntimeError("Disabled tail policy changed its replay layout")
                        if dbo and downsize and 32<args.dbo_threshold<=64:
                            if not all(record["counts"].get(key,0)>0 for key in
                                    ("graph:NATIVE_FULL_REPLAY","graph:UBATCH_FULL_REPLAY")):
                                raise RuntimeError("Adaptive tail did not replay both native and cohort graphs")
                        evidence.append(dict(rank=record["rank"],actual_replay_sizes=sizes))
            if golden is None:golden=outputs
            elif outputs!=golden:raise RuntimeError("Tail policy changed request-ordered generated tokens")
            summary.update(dbo=dbo,tail_downsize=downsize,dbo_threshold=args.dbo_threshold,
                heterogeneous_contexts=args.heterogeneous,
                projection_only_root=args.root_empty,
                placement=role["placement"],
                tail_replay_evidence=evidence,paired_tokens_checked=downsize,
                performance_valid=False)
            write_json(destination/"summary.json",summary)
            summaries.append(summary)
            print("TAIL_READINESS_PASS",dbo,downsize,flush=True)
    write_json(args.output/"summary.json",dict(passed=True,controls=summaries,
        scope="8B BF16 constant-KV correctness only, paired sequential single-node engines; no trained-model accuracy or throughput claim"))


if __name__=="__main__":main()
