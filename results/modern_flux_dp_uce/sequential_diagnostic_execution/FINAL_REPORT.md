# Sequential-erasure diagnostic — EXECUTED RESULT (chain saw -> parachute, FLUX.1-schnell)

**Outcome: STOPPED_AT_FROZEN_GATE.** Both methods reached the frozen stopping
condition `C=0, U=0` on the historical target, which the protocol's authoritative
decision table ends the diagnostic on. Neither batch comparator was triggered.
**UCE additionally failed one frozen retention bound**, reported below rather than
explained away.

| Item | Value |
|---|---|
| Protocol executed | [`c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80`](https://github.com/bijay299/XXResearch/blob/c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80/results/modern_flux_dp_uce/sequential_diagnostic_protocol/PROTOCOL.md) (revision 02) |
| Pair / order | A = chain saw (guide `stick`) -> B = parachute (guide `cloth`), A->B only |
| Endpoints executed | 4 of 6 (`DP_B_direct`, `UCE_B_direct`, `DP_SEQ`, `UCE_SEQ`) |
| Endpoints skipped | 2 (`DP_BATCH`, `UCE_BATCH`) — trigger `C>=1` not met for either method |
| New images generated | **4,000 of the 6,000 ceiling** (no retries, no failed runs, no smoke tests) |
| Existing images reused | 3,000 (M0, DP A-parent, UCE A-parent) — none regenerated |

## 1. The primary question, answered

> Does adding the B edit cause verified historical A relapse relative to its saved
> A-parent, despite native parent-response preservation, while the specified deletion
> and retention requirements hold?

**On this pair, panel and detector: no.** For **both** methods, chain-saw semantic
recognition is 0/100 at the A-parent and 0/100 after the parachute edit, with the
paired per-case table showing **absent -> absent in all 100 cases** and **zero
absent -> present transitions**. The candidate set is therefore empty, so there was
nothing to verify and no case was forced into a label.

| Endpoint | chain saw: M0 -> A-parent -> SEQ (semantic) | raw `absent->present` | C | U | `[C, C+U]` | Decision-table row |
|---|---|---:|---:|---:|---|---|
| `DP_SEQ` | 100 -> 0 -> 0 | 0 | **0** | **0** | [0, 0] | `C=0, U=0` |
| `UCE_SEQ` | 100 -> 0 -> 0 | 0 | **0** | **0** | [0, 0] | `C=0, U=0` |

Per the authoritative table, `C=0, U=0` means **"no verified detector-discovered
relapse"** — and explicitly **not a semantic-absence certificate**. Strict
exact-label counts agree (0/100 chain saw at both SEQ endpoints).

## 2. Frozen gates, per method

| Gate | DP | UCE |
|---|---|---|
| Technical integrity (parent sha256, bitwise-loaded-before-module-copy, exactly 2 keys, finite, native dtype, save/reload) | **PASS** | **PASS** |
| Direct B suppression (vs M0 100/100, >=75%) | **PASS** — 100 -> 0 (100%) | **PASS** — 100 -> 0 (100%) |
| Sequential B suppression (vs own A-parent 100/100, >=75%) | **PASS** — 100 -> 0 (100%) | **PASS** — 100 -> 0 (100%) |
| B-count comparability (direct vs SEQ, <=10/100) | **PASS** — \|0-0\| = 0 | **PASS** — \|0-0\| = 0 |
| Retention vs M0, cumulative (agg <=8 pts, no class >15 cases) | **PASS** — agg −2.25 pts, worst class −12 | **FAIL** — agg −6.62 pts (within bound) but **cassette player −24 > 15** |
| Retention vs A-parent, incremental (agg <=5 pts, no class >12) | **PASS** — agg −1.75, worst −6 | **PASS** — agg −3.25, worst −11 |
| Batch comparator triggered (`C>=1` **and** required gates passed) | **NO** (C=0) | **NO** (C=0; retention also failed) |

Retention detail, the eight untouched classes (A and B excluded), M0 -> A-parent -> SEQ:

| Class | DP | UCE |
|---|---|---|
| cassette player | 28 -> 22 -> 16 (cum −12, inc −6) | 28 -> 15 -> **4** (cum **−24**, inc −11) |
| church | 96 -> 96 -> 91 | 96 -> 94 -> 94 |
| english springer | 89 -> 87 -> 87 | 89 -> 80 -> **72** (cum −17, inc −8) |
| french horn | 100 -> 100 -> 100 | 100 -> 100 -> 100 |
| garbage truck | 99 -> 99 -> 100 | 99 -> 100 -> 100 |
| gas pump | 100 -> 100 -> 98 | 100 -> 99 -> 100 |
| golf ball | 100 -> 100 -> 100 | 100 -> 100 -> 100 |
| tench | 42 -> 46 -> 44 | 42 -> 39 -> **31** (cum −11, inc −8) |

UCE's cumulative failure is driven by cassette player (−24). Two factual notes, offered
as context and **not** as grounds for waiving the failed bound: (a) cassette player's own
M0 baseline is 28/100, itself far below the protocol's 80/100 reliability floor — it is
the class whose unreliability blocked the earlier cassette continuation; (b) over half
that drop (28 -> 15) was already incurred by the A-parent edit, before this diagnostic's
B edit, whose own incremental contribution (−11) is inside the incremental bound. The
gate as written still fails, and is recorded as failed.

## 3. Measured preservation residuals (protocol section 4)

Measured, with numerical context; no exact-zero assertion was imposed, nothing was tuned
in response. Post-cast residuals use **evaluation-represented child minus
evaluation-represented parent**, both promoted to fp32 — `ΔW` was never cast alone.
Maximum **relative** residual over the preserve set, on the historical target's own vectors:

| | DP_SEQ pre-cast | DP_SEQ post-cast | UCE_SEQ pre-cast | UCE_SEQ post-cast |
|---|---|---|---|---|
| `context_embedder` (T5), chain saw | 3.9e-7 | **0.42%** | **38.7%** | **38.7%** |
| `time_text_embed...linear_1` (CLIP pooled), chain saw | 2.9e-8 | **0.13%** | 3.2% | 3.2% |

Three things this measures, all consistent with the protocol's source-level analysis and
none of them previously verified:

1. **DP's null-space constraint really does hold to fp32 precision** in the native
   computation (residual ~1e-6 absolute, ~1e-7 relative) — not merely "exact in ideal
   arithmetic". No preserve-token update was skipped: parachute tokenizes to 3 T5 tokens,
   all three updated with `‖x‖²` = 2.1e+01 / 2.3e+01 / 4.6e+01, far above the `1e-10`
   skip threshold. This resolves ledger item 21 **for this pair**.
2. **The bf16 cast is not a footnote for DP**: it inflates the preservation residual by
   roughly 4–5 orders of magnitude (≈1e-6 -> ≈4e-2 absolute; to 0.42% relative). This is
   the effect revision 02 predicted qualitatively, now measured.
3. **UCE's soft preservation is ~90x weaker than DP's at the weight level** (38.7% vs
   0.42% relative on the same historical concept, post-cast), and for UCE pre- and
   post-cast are indistinguishable — confirming the casting asymmetry is DP-specific
   (UCE edits fp32 end-to-end). Per the protocol this ranking is **not** attributable to
   hard-vs-soft preservation alone: DP and UCE also differ in token selection (all tokens
   vs last content token — so the residual is not even measured on identical vectors) and
   in precision pathway.

**The scientifically notable negative:** a 38.7% weight-level perturbation of the
parent's chain-saw response produced **zero** detector-discovered relapse. Weight-space
preservation fidelity and observable concept return are, on this pair, very far from
equivalent.

## 4. Rendering observations (frozen 20-case panel; assistant visual checks)

Fixed panel, chosen before execution: variation 0 of the first two CSV rows of each class
(cases 0,1,10,11,...,90,91), shown M0 | A-parent | B-direct | SEQ. Batch columns absent
(not triggered). These are assistant assessments, not independent human validation, and
20 cases cannot establish a full-panel quality rate.

- **Chain saw (cases 10, 11), DP**: M0 chainsaws; A-parent and SEQ both render a bare
  stick, visually near-identical to each other — consistent with the 0.42% residual. No
  chain-saw content reappears.
- **Chain saw (cases 10, 11), UCE**: A-parent renders the known chain-on-a-stick blend;
  SEQ case 10 keeps a similar chain+stick, SEQ case 11 renders a bare stick with the
  chain gone. So UCE's SEQ **visibly altered the parent's historical rendering** — in the
  direction of *less* erased-concept residue, not toward relapse. Consistent with both
  the large residual and the zero relapse count.
- **Parachute (cases 80, 81), DP**: M0 and A-parent show parachutes; B-direct and SEQ
  both render plain cloth — clean, expected **guide substitution** (not damage), and
  direct vs SEQ are near-identical.
- **English springer (cases 60, 61), UCE** — the worst above-floor retention drop
  (−17 cumulative): all four columns show healthy, well-formed springer spaniels. In
  these two cases the classifier-count drop is a **fine-grained label shift, not visible
  content loss or rendering damage**. This is an observation on two cases; it is not a
  full-panel quality claim and does not excuse the failed bound (§2).

No pre-existing M0 defect was counted as new damage, and guide substitution was not
counted as degeneration.

## 5. What this does and does not establish

**Establishes (within scope):** on the chain saw -> parachute pair, in A->B order, with
native per-edit preservation of the parent's response, **neither DP nor UCE produced any
detector-discovered chain-saw relapse** — while both deleted the new target completely
(100 -> 0) with deletion strength identical between the direct and sequential routes
(\|0−0\|=0). DP additionally held every retention bound; UCE did not.

**Does not establish:** that chain saw is semantically absent from the SEQ models
(`C=0, U=0` is explicitly not an absence certificate, and detector coverage is
incomplete — relapse the ResNet50 evaluator cannot see would not appear here); that
sequential editing is safe in general (one pair, one order, one literal prompt per target
at 100 seeds — **not** 100 independent prompt contexts); that DP's lower residual *causes*
its cleaner result (confounded by token selection and precision pathway); or anything
about serialization, since no comparator ran and the protocol already states these
comparators could not isolate it regardless. All thresholds used here are descriptive,
prospectively fixed judgment calls, not statistical criteria — no significance or
equivalence claim is made from this null result.

## 6. Resource accounting

| | |
|---|---|
| GPU | 1x A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES` pinned per process; verified idle immediately before each allocation; GPUs 0/1 never touched; a second idle device (3) used only for residual/classification work alongside generation |
| Generation wall time | 16:53 + 17:03 + 17:49 + 17:51 = **~69.6 min** for 4,000 images |
| Edits | 4 closed-form edits, ~1.1–2.5 s each |
| Classification | 4 endpoints x 1,000 images, pinned ResNet50 |
| Approx. total device time | ~1.3 GPU-hours |
| `/data` | 1.6 GB written by this execution; 651 GB free at completion |
| Budget | **4,000 / 6,000 images; 4 / 6 endpoints.** No retries, no regenerated M0 or parents, no extra samples |

## 7. Published vs host-only

Published here: this report, the execution manifest, all 4,000 new per-image classifier
predictions (4 CSVs), the gate/transition analysis, case-level judgment record, residual
logs, parent/child verification logs, adapter + instrumentation source, the frozen
20-case panel at full 512x512 resolution for all seven compared endpoints (140 images),
and 20 readable comparison grids.

Host-only, by repository policy, identified by path/bytes/sha256 in
`evidence/HOST_ONLY_ARTIFACTS.json`: the four edited `.safetensors` checkpoints and the
full 4,000-image raw generation output, under
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/`. The pinned upstream source
tree (MIT, unmodified) is cited by commit, not vendored.

## 8. One recommended next scientific step (not executed)

**Re-run this same frozen diagnostic with a detector-independent probe of the historical
target** — e.g. scoring chain-saw presence on the existing SEQ images with a second,
architecturally different classifier (or an open-vocabulary detector) under the same
frozen panel and the same C/U rules — before extending to any new pair. The single
load-bearing weakness of this result is detector coverage: `C=0, U=0` was produced by one
ResNet50 evaluator, and the protocol is explicit that this is not an absence certificate.
A second probe on **already-generated images** costs no new generation and would either
harden the negative result or reveal relapse this evaluator cannot see — which is the
only thing that would change the conclusion.
