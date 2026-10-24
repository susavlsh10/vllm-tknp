# Context-balanced Token Parallel placement

## Design

TKNP keeps each request's KV on a fixed attention owner. The existing cost
policy processes waiting requests in FIFO order and greedily assigns each to
the feasible owner with the smallest accumulated `prompt_tokens + max_tokens`.
With heterogeneous contexts, late long requests can leave unequal owner loads.

The proposed policy takes a bounded FIFO prefix (24 requests in the prototype),
sorts it longest-context-first for **ownership planning only**, and greedily
assigns owners using the same cost, lifetime KV reservations, and graph-lane
limits. If the entire prefix cannot be assigned, it retries a shorter prefix.
Requests are still admitted in FIFO order; no later request bypasses an
unassigned older request. Ownership is sticky: there is no KV migration.
Planning costs at most O(W^2 log W + W^2 R) for window W and R owners; the
implementation stops at the first fully feasible prefix.

The policy changes CPU-side placement, not captured GPU operations, tensor
addresses, or ownership-layout shapes. It therefore reuses the existing FULL
CUDA graphs. It does not introduce microbatching or change capacity.

Enable with `VLLM_TKNP_PLACEMENT_POLICY=cost` and
`VLLM_TKNP_BALANCED_WINDOW=24`; zero disables the new policy. Keep
`VLLM_TKNP_ROOT_BIAS_TOKENS=0`: an exploratory 65536-token root penalty did not
provide a consistent additional gain. The default remains unchanged.

## Single-node evidence

EOS job 6311940, 2026-10-21: one node, eight H100s, Llama 3.1 8B dummy BF16
weights, preinitialized physical BF16 random KV, TP2 x TKNP4, FULL graphs.
All variants used the same requests and fixed graph layout [6,6,6,6].
Numbers below are median admission-gated completion throughput (tokens/s),
three persistent measurements per point, OSL128.

| Context distribution | Batch | FIFO cost | Balanced | Change |
|---|---:|---:|---:|---:|
| Uniform 64K | 24 | 1304.89 | 1322.23 | +1.3% |
| Equal 32K/64K/96K | 24 | 1168.98 | 1337.92 | +14.5% |
| 75%32K / 25%96K | 24 | 1229.07 | 1396.77 | +13.6% |
| Uniform 64K | 48 | 1302.77 | 1344.73 | +3.2% |
| Equal 32K/64K/96K | 48 | 1253.34 | 1332.92 | +6.3% |
| 75%32K / 25%96K | 48 | 1519.65 | 1551.92 | +2.1% |

For equal-mixture B24, median client ITL fell from 19.7 to 17.2 ms;
the median owner context-work max/mean proxy fell from 1.25 to 1.00.
A separate unreplicated OSL1024 equal-mixture B24 control measured
1102.48 vs 1337.15 tokens/s (+21.3%); do not present this as a repeated result.
All 57 points across FIFO, balanced, and root-aware variants passed exact
output-count, all-eight-rank FULL replay, zero-preemption, drained-KV, and
stable graph/model/KV-address checks. Three pure placement regression tests
also passed. This is not trained-model accuracy validation.

## Interpretation and limits

The evidence supports reducing heterogeneous attention stragglers, not a
universal throughput gain. Context sums are a work proxy, not SM occupancy.
B48 queued because of the explicit 24-lane graph ceiling, not physical KV
exhaustion. All requests arrived behind a release gate: these are cached-decode
burst tests, not online-arrival or real-prefill results. Variant order was not
interleaved, so small differences may reflect run-order effects.

Sources: `vllm/v1/core/sched/tknp_placement.py` (`place_balanced_window`),
`vllm/v1/core/sched/scheduler.py` (opt-in selection), and
`examples/offline_inference/TKNP/eos_heterogeneous_scheduler.py` (experiment).
Full summaries are under `ubatch_results/hetero_scheduler_20261021_job6311940_`
with suffixes `fifo`, `window`, and `root-aware`.

## Four-node follow-up

Compare native and balanced TP8 x TKNP4, TP8 x DP4, TP32 x DCP4,
TP8 x PP4 (scheduling-only), and TP32 using the 405B FP8-recipe dummy model,
BF16 random KV, OSL1024, batches24/48/96. Pair the equal32/64/96K mixture
with uniform64K and the 75%32K/25%96K mixture with uniform48K.
Use identical TKNP layouts for both policies, no root penalty or DBO, natural
waves, all-rank FULL replay checks, per-length queue/ITL, and gated completion
throughput. Probe capacity at96K+1024 for conservative longest-context-safe
graphs; report these graph limits explicitly and do not claim maximal mixed
KV capacity. One measurement per large-model point is exploratory evidence.
