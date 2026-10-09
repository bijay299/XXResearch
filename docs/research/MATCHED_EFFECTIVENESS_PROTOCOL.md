# MATCHED_EFFECTIVENESS_PROTOCOL — proposed diagnostic, for review

**Status: a proposal. Nothing here has been run.** No GPU work was launched in
preparing it. The GPU-hour figure is a **ceiling put forward for review, not
permission to run**. The PI decides; the main research chat reviews.

**Prepared by** AUDIT-01, from the audited snapshot
[`a9de625`](https://github.com/bijay299/XXResearch/tree/a9de625).
Evidence: [`results/audit_v1/AUDIT_REPORT.md`](../../results/audit_v1/AUDIT_REPORT.md).

---

## 1. The question, and why the pilot cannot answer it

The pilot compared L2-SP 25000 against no regularisation at a **fixed 1000
steps**, so the two arms deleted different amounts. L2-SP left +42.5 pp of dog
suppression where the unregularised arm achieved +82.5 pp (both seeds), and
retained ~35–40 pp more bird. Those two facts cannot be combined into a
preservation claim, because the arm that preserved more also deleted less.

> **The question.** At a *matched* level of new-request deletion, does L2-SP
> retain more than simply making a smaller unregularised update?
>
> If an unregularised run stopped early at the same deletion level retains just
> as much, then L2-SP's apparent preservation benefit is a consequence of
> reduced update magnitude and not of the regulariser. If L2-SP still retains
> more, the benefit is attributable to *where* it constrains the update rather
> than *how much*.

This is a **diagnostic**, not a proposed method. Its purpose is to decide
whether our current interpretation survives a fair comparison.

## 2. What upstream already does — cited before any distinction is claimed

**No novelty is claimed for this diagnostic.** CUIG already contains
early-stopping controls, and appendices D and E of the accepted paper report
them. In the pinned upstream tree
(`9932ac3271a122f6d38d19e0b8c8908fe5237ff7`) the implementation is:

- `Regularizers/Simultaneous/simultaneous.py` → `sample_and_evaluate_ua`,
  `check_early_stopping`
- wired into `UnlearningMethods/ConAbl/train_conabl.py:309–364`
- exposed as `--eval_interval`, `--patience` (default 2000),
  `--stop_threshold` (default 99.0) in `ConAbl/src/args.py:10–12`

**What that code does.** Every `eval_interval` optimizer steps it samples and
measures Unlearning Accuracy on the concept being removed, then stops when
either `ua >= stop_threshold` (default 99%, i.e. the concept is essentially
gone) or UA has failed to improve for `patience` steps. It is a **convergence
and compute-saving** rule, driven by one objective: maximise deletion of the
current request. Each arm stops at *its own* maximum achieved deletion.

**The difference here is one of purpose, not mechanism.** This protocol stops
each arm at a *prescribed level common to both arms*, so retention is compared
between arms of equal deletion strength. Upstream's rule equalises *convergence*;
this one equalises *effect size*. That is a measurement-protocol choice — the
kind of control an evaluation needs, not a contribution.

Also not novel, and not claimed: regularisation for concept editing, semantic
preservation metrics, and sequential-request evaluation itself. **No new
regularizer and no learned controller is built in this work.**

## 3. Design

Fixed throughout: base SD-1.5, native ConAbl object editing, kv-xattn (32
tensors, 19,169,280 params), effective LR 8e-6, AdamW, fp32, constant schedule
with 500 warmup, upstream mappings `horse+cat`, `horse+dog`. Anchor caches
reused byte-identically. Parents are the **existing seed-specific MA
checkpoints** — seed 17 MA for seed 17's children, seed 29 MA for seed 29's.

### Conditions, dog branch (`horse+dog` from MA)

| arm | source | deletion control |
|---|---|---|
| **A1** `MAB_L2` | **exists** (`8ee7d56f62b0` s17, `c4f3ecdcb945` s29) | L2-SP 25000, 1000 steps |
| **A2** `MAB_es@k` | **new trajectory** | unregularised, stopped at step *k* chosen to match A1's deletion level |
| **A3** `MAB_full` | **exists** (`63c9a8232cfd` s17, `d9f28f0fc6b8` s29) | unregularised, 1000 steps — the unmatched reference the pilot used |
| **A0** `MA` | **exists** | the parent, for both estimands |
| **M0** | **exists** | untouched backbone |

A2 is the only arm requiring new compute. A single unregularised trajectory per
seed yields every candidate *k*, because dumps are saved along the way.

### Why the sandwich branch is excluded from the confirmatory comparison

`MAC_L2` achieves **+7.5 pp [+0.0, +17.5]** (s17) and **+7.5 pp [+0.0, +15.0]**
(s29) of sandwich suppression — at or below what 10 prompt clusters can resolve.
There is no meaningful deletion level to match *to*. Matching an unregularised
arm down to +7.5 pp would compare two arms that both essentially failed to
delete, and any retention similarity between them would be an artifact of both
doing nothing.

**It is therefore declared infeasible at this coefficient and reported as
infeasible — not scored, and not presented as "L2-SP preserves better here".**
This costs no GPU time: the conclusion follows from existing evidence.

Bringing that branch in would first require a coefficient search to find a
sandwich L2-SP value landing inside the meaningful region. That is a separate,
separately-priced request (§9), and it must run on development prompts only.

### Checkpoint schedule for A2

Dump the unregularised trajectory every **100 steps** (10 dumps, storage only,
~730 MiB per trajectory). Then **bisect** on the development set for the
matched *k* rather than evaluating all ten: ~4 evaluations instead of 10.

Bisection assumes suppression is monotone in steps. That assumption is
**checked** at every evaluated point; if a non-monotonicity appears, the
fallback is a full 10-point scan on the development set (+6 dev evaluations,
≈0.13 GPU-h per trajectory). Prune unmatched dumps after selection.

## 4. Prompt sets — three disjoint sets, with distinct roles

The pilot's 280-prompt set **has informed the hypotheses in this protocol** and
is exploratory from here on. It may be used for sanity checks and for continuity
with the published tables, but **no configuration may be selected on it and no
confirmatory claim may be made from it**.

| set | size | role | may be used for |
|---|---|---|---|
| **DEV** (new) | 10 prompts × (new target, cat, bird) | choosing *k*; monotonicity check | all selection |
| **FINAL** (new, frozen before any arm is chosen) | 7 categories × 10 prompts × 4 gen seeds = 280 | the confirmatory comparison | one evaluation, reported as-is |
| **PILOT** (existing) | 280 | historical continuity | descriptive comparison only; never selection, never a confirmatory claim |

DEV composition, sized so the matching quantity is the best-resolved:

| category | prompts × gen seeds | n | resolution |
|---|---|---|---|
| new target (dog) | 10 × 4 | 40 | 2.5 pp |
| cat (historical) | 10 × 2 | 20 | 5.0 pp |
| bird (retained) | 10 × 2 | 20 | 5.0 pp |
| **per checkpoint** | | **80** | |

**Frozen before any arm is chosen**, with a recorded sha256 and a disjointness
check against the 999 training strings *and* against DEV and PILOT, exactly as
the pilot manifest was built. The FINAL set is generated and hashed **before**
the development stage begins, so selection cannot reach it.

## 5. Meaningful deletion region, matching tolerance, infeasibility

### Meaningful deletion region

An arm qualifies for a matched comparison only if, on DEV, against its own
seed's MA parent, at t=0.5:

> **new-target suppression ≥ 40 pp** *and* **new-target residual ≤ 60%**

Rationale: below 40 pp an arm has not meaningfully removed the concept, so its
retention is not informative about a preservation/effectiveness trade. `MAB_L2`
sits at +42.5 pp — just inside, which is why the dog branch is feasible and the
sandwich branch (+7.5 pp) is not. The threshold is **declared here, before the
data are collected**, and is not to be moved afterwards to admit an arm.

### Matching tolerance

Match A2 to A1's DEV suppression within **±5.0 pp**. With 40 new-target DEV
images the attainable resolution is 2.5 pp, so ±5.0 pp is two resolution steps —
tight enough to be a real match, loose enough to be reachable on a 100-step
grid. Choose the *k* minimising |suppression(A2@k) − suppression(A1)|; break
ties toward the **smaller** *k* (the weaker update), which is conservative
against the hypothesis that L2-SP is unnecessary.

### Infeasibility rule — declared in advance

Abandon the matched comparison for a branch, and report infeasibility, if any of:

1. **A1 is outside the meaningful region** (sandwich branch: already true).
2. **No dumped *k* matches within ±5.0 pp**, i.e. suppression jumps across the
   band between adjacent dumps. Report the bracketing pair and the gap; do not
   widen the tolerance to manufacture a match.
3. **Suppression is non-monotone in *k*** in a way that makes "the matched
   point" ambiguous (two separated *k* both inside the band with materially
   different retention). Report both.
4. **A2@k fails the meaningful region** even where it matches A1.

> Declaring infeasibility is a **result**, not a failure. "These two methods
> cannot be compared fairly at this coefficient" is the finding, and it is more
> useful than a comparison run anyway. What must never happen is equating two
> arms at near-zero suppression and reporting preservation as solved.

### Reporting imperfect matching without reselecting

The achieved match is measured on DEV and **fixed there**. On FINAL, the
realised suppression difference between A1 and A2 will not be exactly zero.
Report it:

- State `suppression(A1) − suppression(A2)` on FINAL as a **residual
  mismatch**, with its paired interval, beside every retention contrast.
- If |residual mismatch| on FINAL exceeds ±10 pp, the retention comparison is
  reported as **confounded by residual mismatch** and the primary endpoint is
  declared unresolved.
- **Under no circumstances re-pick *k* using FINAL.** If the match transfers
  poorly, that is reported as a finding about transfer, and any re-selection
  requires a new frozen FINAL set.

## 6. Endpoints

**Primary.** Retained-category detection rate at matched deletion, as the direct
paired contrast `D_retained(A1) − D_retained(A2)` at t=0.5, **per category,
never averaged** — horse, bird, chair, bicycle separately. Bird carries the
pilot's largest effect and is the pre-declared focus; the other three are
reported with it, not pooled into it.

**Co-primary (the matching check).** `D_dog(A1) − D_dog(A2)` on FINAL — the
residual mismatch that licenses or voids the primary.

**Secondary.**
1. Historical cat residual, direct: `D_cat(A1) − D_cat(A2)`.
2. Both parent-referenced estimands, `vs` the **same seed's** MA, reported
   alongside the direct contrasts so reference-dependence stays visible.
3. Literal vs paraphrase for every endpoint, reported separately — residual
   presence and parent-referenced change never merged.
4. Thresholds 0.3 / 0.5 / 0.7 for everything; 0.5 primary.
5. A3 (`MAB_full`) as the unmatched reference, to quantify how much the pilot's
   conclusion changes under matching.
6. Parameter movement (relative L2 vs MA) per arm — descriptive association
   only, **never** offered as a mechanism.

**Contrast construction.** All contrasts paired at identical
`(prompt_id, gen_seed)`; paired cluster bootstrap resampling **prompts** with
their generation seeds held together; 10 000 draws; 95% percentile intervals;
**training seeds analysed separately and never pooled**. Two seeds supports a
direction check only.

**Interval discipline, restated because it governs the stop rule.** An interval
including zero is **not** evidence of no effect. A difference in significance
labels between arms or seeds is **not** a significant difference. No multiplicity
correction is applied, so each interval is descriptive, not a test.

## 7. Visual validation — required, not optional

A detector result is a proxy. Before any conclusion from this diagnostic is
reported:

1. Extend the AUDIT-01 blinded packet with A2 and the new FINAL images, same
   construction: a stratified-random set **R** plus a separately labelled
   enriched diagnostic set **D**, key stored outside the packet, presence /
   ambiguity / quality / context in separate columns.
2. **A human must complete the sheet.** AI inspection is not human review and
   may not be described as validation.
3. Estimate detector error **only from set R**, with its sampling weights, and
   report it per prompt family. Set D characterises failure modes and is biased
   upward by construction; never pooled, never quoted as overall accuracy.
4. If detector error differs materially between the arms being compared, the
   primary endpoint is reported as **proxy-limited**.

## 8. Run matrix and resource accounting

Measured on this host, from the 10 pilot training runs and 11 pilot checkpoint
evaluations — not assumed:

| quantity | measured |
|---|---|
| per optimizer step | **0.8303 s** (n=10 runs) |
| per generated image | **1.0083 s** (n=12 reports) |
| per detected image | **0.0718 s** (n=10 reports) |

Stated assumptions: **25 s** process setup per invocation (pipeline build +
checkpoint load; an assumption, since the pilot's `seconds_per_image` already
amortises setup over 280 images), and **25%** contingency for one failed-stage
re-run plus bisection overshoot.

### CORE plan — dog branch, 2 training seeds

| stage | runs | images | GPU-h |
|---|---|---|---|
| A2 training (new trajectories, 1000 steps, dumps every 100) | 2 | — | 0.461 |
| Development selection (4 bisection evals × 2 seeds + 6 reference evals, 80 img each) | 14 | 1120 | 0.336 |
| Final confirmatory (M0 + 2 seeds × {MA, A2@k, A1}) = 7 ckpts × 280 | 7 | 1960 | 0.588 |
| Process setup (23 invocations × 25 s) | — | — | 0.160 |
| **subtotal** | | **3080** | **1.545** |
| **with 25% contingency** | | | **1.93** |

**≈1.93 GPU-hours, within the 4 GPU-hour ceiling.**

Wall time is **separate** and is not GPU-hours: ≈0.85 h on two idle GPUs,
≈0.6 h on four. The four A100s are shared without a scheduler, so availability
is not guaranteed and wall time cannot be promised. Only independently verified
idle GPUs would be used.

**Storage.** Peak 1.72 GiB — 0.73 GiB of intermediate dumps (20 × 73 MiB) plus
0.29 GiB of images, falling to ≈0.44 GiB once unmatched dumps are pruned.
`/data` has 1.4 TB free. No weights or image archives are committed.

**Tuning budget, comparable by construction.** A1 receives **zero** tuning here:
its coefficient 25000 is taken as given, exactly as the pilot ran it. A2
receives one selection over 10 pre-saved dumps on DEV — a 10-point grid on a
single scalar, ~4 of which are evaluated. Neither arm is tuned on FINAL. If the
PI prefers strict parity, the alternative is to spend a matching 10-point L2-SP
coefficient grid on A1, which is the §9 extension — the asymmetry is recorded
here rather than hidden, and the current asymmetry **favours A2**, i.e. it is
conservative against L2-SP.

### Alternatives, priced

| plan | GPU-h | what it buys / costs |
|---|---|---|
| **FALLBACK** — dog branch, 1 seed | **1.02** | one direction check; no cross-seed agreement possible |
| **CORE** (recommended) | **1.93** | dog branch, both seeds, cross-seed direction check |
| **EXTENDED** — both branches confirmatory | 3.29 | **not a drop-in**: presupposes a sandwich coefficient already inside the meaningful region. The coefficient search that would find one is **not** costed here |

Full arithmetic: [`results/audit_v1/protocol_cost_model.json`](../../results/audit_v1/protocol_cost_model.json),
re-derivable with `python3 scripts/seq/protocol_cost_model.py`.

### A free reproducibility check

Training reproducibility has **never** been tested on this host (see
`ARTIFACT_INVENTORY.md` §6). The A2 trajectory re-runs an unregularised dog
request at 1000 steps — the same configuration as the existing `MAB`. Compare
the re-run's final `delta.bin` sha256 against `63c9a8232cfd` (s17) /
`d9f28f0fc6b8` (s29). This costs nothing and is recorded as a **secondary
observation, not an assumption**: a mismatch does not invalidate the diagnostic
(each arm is evaluated as itself), but it would tell us bit-exact training
reproduction does not hold here, which matters for everything downstream.

## 9. Not in this request

Listed so the boundary is explicit, not as a plan.

- Sandwich L2-SP coefficient search (~10 training runs ≈ 2.3 GPU-h plus DEV
  evaluation) — required before the sandwich branch can be compared fairly.
- Matching L2-SP coefficient grid on A1 for strict tuning parity.
- More training seeds for population variance.
- Separating anchor sharing from semantic distance — the standing confound.
- Longer request streams.
- Human annotation labour (no GPU).
- Any new regularizer or learned controller. **Out of scope.**

## 10. Stop rule, and what would change our interpretation

**Stop immediately and report, without proceeding, if:**

- the infeasibility rule (§5) fires for the dog branch;
- A2's trajectory fails to reach the meaningful region by step 1000;
- the FINAL set is found to overlap DEV, PILOT or the training strings;
- any arm's artifacts fail the completeness guards in `run_seed.sh`;
- the FINAL residual mismatch exceeds ±10 pp — report confounding, do not
  re-select.

**Pre-declared readings of the primary endpoint** (per-category retention at
matched deletion, t=0.5, both seeds):

| outcome | reading |
|---|---|
| A1 − A2 retention is **small and includes zero in both seeds** on every retained category | **Weakens our interpretation.** L2-SP's apparent preservation benefit is consistent with being an effect of reduced update magnitude, reproducible by stopping an unregularised run early. The regulariser would then be an expensive way to take a smaller step, and "L2-SP protects retention" should be withdrawn as a method claim. |
| A1 − A2 retention **excludes zero favouring A1 in both seeds**, with FINAL mismatch within ±10 pp | **Supports our interpretation.** Where the update is constrained matters beyond how much, and a mechanism question becomes well-posed: which tensors A1 spares that a uniformly smaller step does not. |
| **Directions disagree between the two seeds** | **Unresolved.** Report as unresolved and cost more training seeds. Do **not** report the agreeing seed. |
| A2 **cannot be matched** within tolerance | **Rejects the comparison, not the hypothesis.** Report the bracketing dumps and the gap. |
| A1 − A2 retention favours **A2** | Reduced update magnitude preserves *better* than L2-SP at equal deletion — a result against L2-SP, reportable as such. |

**What would reject the broader framing.** If matched unregularised early
stopping reproduces L2-SP's retention on every retained category in both seeds,
then the pilot's "L2-SP is a tradeoff" framing collapses into the much duller
"smaller updates change less", and sequential-suppression interference is not
the limitation this line of work has been looking for. That outcome should be
reported as clearly as a positive one.

---

### Pre-commitments

Fixed before any data are collected, and not to be moved afterwards: the
meaningful deletion region (≥40 pp **and** ≤60% residual); the ±5.0 pp matching
tolerance and the tie-break toward smaller *k*; the ±10 pp FINAL mismatch limit;
the primary endpoint as per-category retention, never averaged; the FINAL set
frozen and hashed before the development stage; selection on DEV only; and
every interval caveat in §6.

**This document authorises nothing.** It is submitted for central review and
the PI's decision.
