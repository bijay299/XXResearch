# Sequential-erasure diagnostic — EXECUTED RESULT (chain saw -> parachute, FLUX.1-schnell)

**Outcome: STOPPED_AT_FROZEN_GATE. Diagnostic CLOSED (§8).** Both methods reached the
frozen stopping condition `C=0, U=0` on the historical target, which the protocol's
authoritative decision table ends the diagnostic on. Neither batch comparator was
triggered. **UCE additionally failed the cumulative retention gate — on two classes
independently (§2)** — so UCE does **not** satisfy the full gate conjunction; DP does.

> **This report carries a post-publication reporting correction.** The executed numbers,
> frozen gates, empty candidate sets and stopping decision are **unchanged**; several
> *interpretations* published at execution commit `ffc133f` were wrong and are corrected
> here — most importantly the residual metric's meaning (§3), the scope of UCE's
> retention failure (§2) and of the springer visual explanation (§4), and two provenance
> statements (§6, §7). See [`CORRECTIONS.md`](CORRECTIONS.md) for the issue-by-issue
> record. No experiment, model, encoder, detector, tensor computation, image review or
> GPU work was run to produce the correction.

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
| Retention vs M0, cumulative (agg <=8 pts, no class >15 cases) | **PASS** — agg −2.25 pts, worst class −12 | **FAIL** — agg −6.625 pts (inside the aggregate bound) but **two independent per-class violations: cassette player −24 and english springer −17, both > 15** |
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

**UCE's cumulative retention gate fails on two classes independently**, not one
(corrected — see `CORRECTIONS.md` C1):

- **cassette player**: 28 -> 15 -> 4, cumulative **−24**, incremental −11.
- **english springer**: 89 -> 80 -> 72, cumulative **−17**, incremental −8.

Either violation alone fails the 15-case per-class cumulative bound. The cumulative
aggregate (−6.625 pts) is inside the 8-point aggregate bound, and the incremental bounds
pass (aggregate −3.25, worst class −11), but the cumulative gate is failed on the
per-class criterion regardless.

Cassette player's own M0 baseline is 28/100 — below the protocol's 80/100 reliability
floor, the class whose unreliability blocked the earlier cassette continuation — and over
half its drop (28 -> 15) predates this diagnostic's B edit. That is **context, not grounds
to waive the gate**, and it does not extend to english springer, which has a solid M0
baseline of 89/100 and fails the same bound on its own. The gate is recorded as failed.

## 3. Measured preservation residuals (protocol section 4)

Measured, with numerical context; no exact-zero assertion was imposed, nothing was tuned
in response. Post-cast residuals use **evaluation-represented child minus
evaluation-represented parent**, both promoted to fp32 — `ΔW` was never cast alone.

**What the archived logger actually computes** (corrected — see `CORRECTIONS.md` C2):

```
q(c) = || (W_child − W_parent) c ||₂  /  || c ||₂
```

This is an **input-normalized isolated-vector response residual**. Its denominator is the
**input vector's** norm, not the parent's output norm `‖W_parent c‖`. **`q` is therefore
not a percentage change in the parent's response, and not a weight-matrix perturbation
fraction.** Expressing it as "38.7%" / "0.42%" of anything was wrong and is withdrawn. A
parent-response fraction would need a different denominator; it was **not** computed, and
is not computed now. Values below are the archived `relative_residual` (= `q`, decimal)
and `residual_l2` (absolute) fields on the historical target's own probe vectors:

| probe vector | DP_SEQ pre-cast | DP_SEQ post-cast | UCE_SEQ pre-cast | UCE_SEQ post-cast |
|---|---|---|---|---|
| `context_embedder` (T5), chain saw | q = 0.0000004 (abs 2.83e-06) | q = 0.0042419 (abs 0.03064) | q = 0.3865076 (abs 2.8839) | q = 0.3869613 (abs 2.8873) |
| `time_text_embed…linear_1` (CLIP pooled), chain saw | q ≈ 0.0000000 (abs 8.51e-07) | q = 0.0013254 (abs 0.03863) | q = 0.0316740 (abs 0.92307) | q = 0.0319098 (abs 0.92995) |

DP's T5 probes are two tokens (`t5_tok0`, `t5_tok1`); the row shows `t5_tok0`
(`t5_tok1`: pre-cast q = 0.0000003, post-cast q = 0.0042297). UCE's T5 probe is its
single native last content token.

**Important probe-input caveat (corrected — `CORRECTIONS.md` C3):** the archived logger
hardcodes **bf16 text encoders for both methods** (`residuals.py`, `encode_concepts`,
`torch_dtype=torch.bfloat16`), whereas native UCE encodes in fp32. Promoting
bf16-produced vectors to fp32 does not reconstruct fp32-encoder outputs. **UCE's values
here are therefore probes on bf16 encodings, not verified measurements of its native
fp32 preservation vectors.** The native-input reading is withdrawn. The logged
measurements stand as archived; no encoder was rerun.

What these numbers do and do not support:

1. **DP's captured pre-cast residuals are very small on these probe vectors** (abs ~1e-6,
   q ~1e-7) and its post-cast residuals are larger (abs ~3–4e-2, q ~1.3e-3 to 4.2e-3) —
   a narrow numerical observation about DP's edit-time/intermediate casting **on these
   isolated-string vectors only**. It is not a demonstration of exact preservation for
   contextual evaluation prompts, nor for the generator as a whole.
2. **No preserve-token update was skipped**: parachute tokenizes to 3 T5 tokens, all three
   updated with `‖x‖²` = 2.1e+01 / 2.3e+01 / 4.6e+01, far above the `1e-10` skip
   threshold. This resolves ledger item 21 **for this pair**.
3. **UCE's pre- and post-cast values differ** (T5 q 0.3865076 -> 0.3869613; CLIP q
   0.0316740 -> 0.0319098) — they are close but **not identical**, and the earlier
   "indistinguishable" phrasing is withdrawn. Both methods are evaluated in bf16; DP's
   asymmetry concerns its **native edit-time/intermediate** casting, not an absence of
   evaluation rounding for UCE.
4. **Withdrawn**: "38.7% weight-level perturbation", "0.42% parent-response change", and
   "UCE's preservation is ~90x weaker than DP's". A ratio between selected `q` values
   does not establish a general or causal preservation-strength ranking — token selection
   (all tokens vs last content token), precision pathway and probe inputs all differ
   between the two methods, so the values are not measured on comparable quantities.

**The supported statement, narrowly:** differing measured residuals coexist with **zero
detector-discovered relapse** on this panel. No broader equivalence, ranking, or causal
conclusion about preservation strength follows from it.

## 4. Rendering observations (frozen 20-case panel; assistant visual checks)

Fixed panel, chosen before execution: variation 0 of the first two CSV rows of each class
(cases 0,1,10,11,...,90,91), shown M0 | A-parent | B-direct | SEQ. Batch columns absent
(not triggered). These are assistant assessments, not independent human validation, and
20 cases cannot establish a full-panel quality rate.

- **Chain saw (cases 10, 11), DP**: M0 chainsaws; A-parent and SEQ both render a bare
  stick, visually near-identical to each other. No chain-saw content reappears.
- **Chain saw (cases 10, 11), UCE**: A-parent renders the known chain-on-a-stick blend;
  SEQ case 10 keeps a similar chain+stick, SEQ case 11 renders a bare stick with the
  chain gone. So UCE's SEQ **visibly altered the parent's historical rendering** — in the
  direction of *less* erased-concept residue, not toward relapse. Consistent with both
  the large residual and the zero relapse count.
- **Parachute (cases 80, 81), DP**: M0 and A-parent show parachutes; B-direct and SEQ
  both render plain cloth — clean, expected **guide substitution** (not damage), and
  direct vs SEQ are near-identical.
- **English springer (cases 60, 61), UCE** — one of the two classes failing the
  cumulative bound (−17): the two displayed images depict **coherent, well-formed dogs**
  in every column. That local observation is all that is retained here. Their recorded
  top-1 labels are M0 `English springer` / `English springer`; UCE A-parent
  `English setter` / `English springer`; UCE SEQ `English setter` / `English springer` —
  i.e. **the only disagreement visible in this pair of cases already exists at the
  A-parent, before this diagnostic's B edit.** These two cases therefore explain neither
  the 8 additional cases lost after B nor the 17 cumulative losses.
  **Withdrawn (`CORRECTIONS.md` C4):** the earlier claim that the overall springer
  decline is "a fine-grained label shift, not visible content loss" — two panel cases
  cannot support that extrapolation. No additional images were inspected to replace it,
  and the cause of the remaining losses is **not established**.

No pre-existing M0 defect was counted as new damage, and guide substitution was not
counted as degeneration.

## 5. What this does and does not establish

**Establishes (within scope):** on the chain saw -> parachute pair, in A->B order, with
native per-edit preservation of the parent's response, **neither DP nor UCE produced any
detector-discovered chain-saw relapse** — while both deleted the new target completely
(100 -> 0) with deletion strength identical between the direct and sequential routes
(\|0−0\|=0). **DP satisfies the full frozen gate conjunction** (technical integrity,
direct and sequential suppression, comparability, both retention bounds). **UCE does
not**: it shows the same detector-defined zero relapse but **fails the cumulative
retention gate**, so it must not be described as satisfying the gate conjunction.
This result does **not** demonstrate an unmet method gap.

**Does not establish:** that chain saw is semantically absent from the SEQ models
(`C=0, U=0` is explicitly not an absence certificate, and detector coverage is
incomplete — relapse the ResNet50 evaluator cannot see would not appear here); that
sequential editing is safe in general (one pair, one order, one literal prompt per target
at 100 seeds — **not** 100 independent prompt contexts); that DP's lower measured `q`
*causes* its cleaner result, or that it ranks the two methods' preservation strength at
all (§3: `q` is input-normalized, the probe vectors and encoder precision differ between
methods, and UCE's probes used bf16 encodings rather than its native fp32 ones); or
anything about serialization, since no comparator ran and the protocol already states
these comparators could not isolate it regardless. All thresholds used here are
descriptive, prospectively fixed judgment calls, not statistical criteria — no
significance or equivalence claim is made from this null result.

## 6. Resource accounting

| | |
|---|---|
| GPU | **Two** A100-SXM4-40GB devices were used over the run, not one throughout (corrected — `CORRECTIONS.md` C5): GPU 2 was the initial allocation and ran every edit and all four generations; **GPU 3 was additionally used, concurrently with GPU 2, for residual logging and for the `DP_SEQ` classification pass**. `CUDA_VISIBLE_DEVICES` was pinned per process; each device was verified idle immediately before its first allocation; GPUs 0/1 were never used |
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

Host-only, by repository policy, inventoried in `evidence/HOST_ONLY_ARTIFACTS.json` under
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/` — with **two different
levels of identification** (corrected — `CORRECTIONS.md` C6):

- **Hash-identified**: the four edited `.safetensors` checkpoints, and the DP pre-cast
  fp32 capture (`logs/DP_SEQ.precast_fp32.pt`) — each with host path, byte count and
  sha256.
- **Not hash-indexed**: the 4,000-image raw generation output, inventoried **only by
  directory path, image count and total byte count**. Individual images are *not*
  individually hashed; the earlier claim that all raw output is identified by sha256 was
  incorrect. The 140 published panel images *are* individually hashed, in
  `evidence/panel_hashes.csv`. No new hashing campaign was run.

The pinned upstream source tree (MIT, unmodified) is cited by commit, not vendored.

## 8. Closeout decision

**This diagnostic is CLOSED.** The earlier "recommended next step" (a second,
architecturally different detector re-scoring the existing SEQ images) is **withdrawn as
a recommendation and is not authorized** — along with any new pair, reverse order,
parameter sweep, ablation, or new method.

The frozen result stands as published: the classifier counts, the frozen gates, the empty
candidate sets (`C=0, U=0` for both methods) and the no-comparator stopping decision are
**unchanged by this correction pass**. Detector coverage remains the acknowledged
load-bearing limitation of the negative result, and `C=0, U=0` remains explicitly not a
certificate of semantic absence — but that limitation is now recorded as a **stated scope
boundary of a closed result**, not as a queued follow-up.

Any future measurement on this material requires a **separately declared protocol** and
must be published alongside this frozen result rather than overwriting it.
