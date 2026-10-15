# Token Parallel paper experiment plan

Updated October 15, 2026, America/Chicago. This is the portable configuration
policy for subsequent collections. Historical jobs/configurations are retained
for reproducibility; their presence does not make them approved new sweeps.

## Models and precision

Primary models are Llama 3.1 70B and 405B. Use TP8 for both before adding a
second parallel dimension. Mistral Medium 3.5 128B is conditional on existing
fork/container support; defer it rather than port a new architecture. Use
Llama 3.1 8B only for short correctness, graph, transport and stability checks,
not headline performance results.

Use BF16 weights and BF16 KV, without quantization. The current CLI expresses
BF16 KV as `--kv-cache-dtype auto` with model dtype `bfloat16`; verify actual
storage in worker manifests. Dummy weights are permitted for the authorized
systems-proxy tests but are not trained-model validation. Historical FP8 wins
are separate evidence, not primary BF16 results.

## Narrowed topology matrix

`N_D` is the number of decode GPUs, not the combined prefill/decode allocation.
DCP divides work inside TP and does not multiply the GPU count.

| Decode GPUs | Token parallel | Data parallel | Pipeline parallel | DCP or reference |
|---|---|---|---|---|
| 8 | TP8, TKNP1 reference | Same TP8 engine, DP1 | Same TP8 engine, PP1 | TP8 reference; nontrivial DCP unsupported for eight KV heads |
| 16, main comparison | TP8 × TKNP2 | TP8 × DP2 | TP8 × PP2 | TP16 + DCP2 |
| 32, sparse extension | TP8 × TKNP4 | TP8 × DP4 | No automatic extension | TP32 + DCP4 |

Do not sweep TP1/TP2/TP4 roots or replicas, TP1×TKNP16, TP1×DP16, PP16,
TP4×PP4, replicated DCP, or other PP variants. The requested PP baseline is
TP8×PP2. A future 32-GPU PP point needs a separate same-budget decision;
never compare a 16-GPU PP point to a 32-GPU method as equal-resource evidence.
Sub-TP8 small-model smokes are development exceptions only. This policy is a
bounded deployment study, not proof that TP8 is globally optimal.

For either Llama model with eight KV heads, this fork needs TP greater than
eight for nontrivial DCP. Use the full decode allocation as TP: TP16+DCP2 or
TP32+DCP4. Do not launch TP8+DCP2 or claim DCP consumes TP×DCP GPUs.

## Main throughput and queueing collection

Use 32,768, 65,536 and 98,304 input tokens. Reserve space for output tokens
within the model's supported total context. Start with controlled exact-length
prompts. RULER-derived prompts are optional; synthetic repeated prompts are
not a quality benchmark or unmodified real traffic.

Measure actual full-lifetime KV capacity for every fixed topology. At each
context choose the same offered batches for all methods:

1. A below-DP-capacity batch.
2. A batch above DP capacity and below TKNP capacity, if that interval exists.
3. A batch above TKNP capacity, where TKNP must also queue.

Allow each method to admit in waves and run the whole offered workload to
completion. Never give an out-of-resident-capacity method zero throughput or
compare TKNP's full workload against only DP's first wave. DP runs real TP8
replicas, with the same router/workload. Preserve native admission and label
full-lifetime reservation as a distinct policy rather than forcing a weaker
policy on the baseline. PP must use actual native pipeline concurrency.

Freeze one exploratory run per point initially. Repeat decisive wins and
negative controls as matched pairs; graph determinism does not eliminate
system-level variance. The initial 16-GPU matrix has four methods × three
contexts × three regimes = 36 points/model, or 72 for 70B+405B, before
repeats/readiness failures. Do not automatically multiply the whole matrix by
microbatch variants or a new hardware family.

## Timing and prefill decode setup

Prioritize decode-stage latency, per-request TPOT/ITL and completed-output
throughput. Also record TTFT, total makespan, prefill queue/compute,
handoff/admission-to-first-token, p50/p95/p99, preemptions, recomputation,
running/waiting requests, goodput, physical KV capacity and all-rank graph
execution. State whether a percentile is over tokens or per-request means.
Queue snapshots alone do not establish a KV-memory cause.

Use a common, adequately provisioned prefill pool for actual end-to-end
comparisons. Count both prefill and decode GPUs, including unused allocated
GPUs. A TP8 producer plus D16 needs 24 active GPUs: six four-GPU LYRIS nodes.
The last four-node approval cannot accommodate that layout. D32+P8 needs 40
GPUs/ten LYRIS nodes. Request a new resource envelope before allocating.

Keep three timing families distinct:

- Actual prefill → NIXL KV handoff → generation, including queueing and drain.
- Real-prefill decode-ready timing: prefill in waves, generate one token to
  establish resident KV, then measure continuing decode for those same live
  requests. Validate ownership, positions, occupancy and stage boundaries;
  this is not an unrelated second client request or free cache reconstruction.
- Dummy-KV decode controls: include initialization in client burst makespan,
  or explicitly isolate an already-resident common window. Never label these
  actual prefill or pure steady-state decode without that isolation.

Compare the latter two before choosing the formal decode setup. Real
NIXL-to-DCP interleaved KV mapping is not implemented, so DCP currently belongs
to the matched decode-only family. Do not compare its synthetic setup against
other methods' real P/D makespan as a speedup. No connector rewrite is implied.

## Microbatch ablation and serving

Pilot the fixed two-cohort implementation at one TP8×TKNP configuration with
BF16 KV. Freeze the faster validated off/on policy for the main matrix; do not
assume the earlier FP8 gains carry over. Then use one model/topology and a
small batch sweep to compare standard, matched-owner unsplit and two-cohort
execution. Capture short all-rank traces separately from timing. Record root
dense readiness, each cohort's output publication, exposed NCCL waits and
attention gaps; narrower gaps alone are not a throughput win.

After the main results, use one model/configuration for uniform versus
controlled mixed 32K/64K/96K contexts, varying arrivals and bursts. Compare
count versus context-work placement, keep ownership sticky, and record
per-owner KV/work plus per-length latency. A TKNP-only serving demonstration
establishes functionality, not superiority over DP; add one matched TP8×DP
anchor if making a comparative serving claim. Root scaling is a sparse
one-model follow-up. Skip a separate sensitivity grid and all 1T work.

## Reproducibility and collection gates

Pin the source commit plus container digest, Python/torch/FlashInfer/NCCL
versions, hardware/topology, placement, compile flags, KV policy and workload
hash. Require all assigned ranks to replay graphs, no silent eager fallback,
complete expected outputs, finite numerical diagnostics and full drain.
Short real-weight 8B checks are the numerical gate; dummy 70B token agreement
is only transport/determinism evidence.

The October 15 121-point collection remains historical exploratory evidence:
dummy BF16 weights, BF16 KV, DBO off; real P/D on smaller decode budgets and
synthetic KV at D16. Preserve its faster lower-TP DP baselines and all failures,
but do not expand those configurations in the new campaign or claim a blanket
TKNP win. Raw results are stored outside Git under
`ubatch_results/70b_matrix_20261014/`.

See [EOS_REPRODUCTION.md](EOS_REPRODUCTION.md) for H100 environment bring-up,
graph/microbatch launch recipes, evidence gates and retrieval. Writing this
plan/runbook and pushing source does not authorize new EOS or LYRIS jobs.
