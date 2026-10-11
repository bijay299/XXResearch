# Next screen — an identifiable probe of interference in sequential erasure

**CPU-only proposal, revision 3. This authorises nothing, and no GPU experiment
is authorised by it or by the Amendment 01 closeout.** For the PI's decision. If
approved, compute must run on GPUs **independently verified unoccupied by another
user** ([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)); the
hour ceiling stays retired and caps nothing.

> **What revision 3 fixes in revision 2**
>
> 1. **The declared operating point was arithmetically unreachable.** 20% residue
>    with a ≥ 60 pp own-parent drop requires a parent rate ≥ 80%. Measured on the
>    frozen-test probe, the parent generates `dog` at **68.75 / 75.00%** and
>    `sandwich` at **67.50 / 70.00%** — so the regime revision 2 declared cannot
>    be reached for either of its own request-2 targets (§2.1). Replaced by a
>    **relative** target, derived from measured rates rather than asserted.
> 2. **The four cells could not isolate anchor sharing.** "Shared" always meant
>    `horse` and "disjoint" always meant `flower`, because request 1 used `horse`.
>    Sharing was confounded with anchor identity and with the target–anchor
>    relation. Both anchors are now **counterbalanced** (§3).
> 3. **Forcing an anchor on LACU disables its central mechanism.** Controlled-
>    anchor runs are now a separately labelled **ablation family**, and **native
>    LACU stays in the method comparison** (§4).
> 4. **"Excludes +10" was not a predicate.** `[−2, +3]` excludes +10 and is not a
>    material effect. Four mutually exclusive, exhaustive interval predicates
>    replace it, dominance now requires a **direct** contrast, and "already
>    solved" now requires **utility** as well as low relapse (§5).
> 5. **Arm acceptance is now deterministic and predeclared**, prior collateral
>    suppression is distinguished from current erasure failure and from later
>    relapse, and a comparative conclusion requires **≥ 2 accepted arms** (§6).
> 6. **The certificate is an unproved direction**, with its representation,
>    assumptions and the missing proof stated (§7). Literature-absence claims are
>    scoped to the six works actually checked.

---

## 1. Prior work — scoped to what was checked

**Six works were read for this document.** No exhaustive survey was done, and
every absence claim below is scoped to these six.

| work | what it establishes |
|---|---|
| **CEASE** ([2610.01989](https://arxiv.org/html/2610.01989)) | **§3.2 derives** accumulated noise into an **anchor** channel (repeated activation of the shared replacement direction), an **off-anchor** channel (`αᵢᵀαⱼ` overlap) and a **cross** channel, vanishing under `βᵢᵀc∗ = 0` and `αᵢᵀαⱼ = 0`. **§3.3**: **AIC** appends the shared replacement token to the invariance matrix; **HOC** projects updates onto the orthogonal complement of the SVD of past updates. 100 sequential edits, SD 1.4/1.5/2.1. **A.5** tests **five random orders**, variation small against between-method differences |
| **LACU** ([2512.02657](https://arxiv.org/abs/2512.02657)) | **Locality-aware target selection**: per forget prompt, the mapping prompt the model itself treats as most similar by score-prediction distance. **Locality-aware replay**: the nearest retain concepts replayed as a local functional regulariser. Stable over **10 sequential steps** where prior work "collapses after 3–5" |
| **Sculpting Memory** ([2504.09039](https://arxiv.org/html/2504.09039v1)) | dynamic gradient mask, concept-aware loss, and **distillation explicitly targeting persistence of earlier erasures** |
| **M-ErasureBench** ([2512.22877](https://arxiv.org/pdf/2512.22877)) | erasure across prompts, learned embeddings, inverted latents — varies the **probe modality** |
| **Side Effects of Erasing Concepts** ([2508.15124](https://arxiv.org/pdf/2508.15124)) | collateral effects of single erasures |
| **When Are Concepts Erased?** ([2505.17013](https://arxiv.org/pdf/2505.17013)) | mechanism/timing of erasure within the model |

**Already addressed, and not claimed as gaps:** shared-anchor interference
(CEASE AIC), historical-update interference (CEASE HOC), order sensitivity
(CEASE A.5), persistence of earlier erasures as an objective (Sculpting Memory),
and locality-aware replacement selection (LACU).

### 1.1 The narrow question

CEASE's decomposition is in the algebra of the **edit operator** —
`βᵢᵀc∗`, `αᵢᵀαⱼ`. The screen's question is one level out, in the **structure of
the concept sequence**:

> At a **matched** end state on the newest target, is **relapse of an earlier
> target** separably associated with **(a) whether successive requests share a
> replacement anchor** and **(b) how semantically close successive targets are** —
> and does an existing remedy's benefit depend on the combination?

**Among the six works checked**, none varies target proximity, none separates it
from anchor sharing, and none reports a stagewise measurement of whether a
previously erased concept becomes generatable again against *its own*
post-erasure state. CEASE varies **order**, a different factor; in
celebrity-to-celebrity and style-to-style streams proximity is roughly constant
and the anchor is shared throughout, so `αᵢᵀαⱼ` and proximity cannot be told
apart from published results. Constraint ablations show a remedy helps on the
evaluated stream; they do not identify which sequence property made it hard.

**What the screen can deliver:** whether (a) and (b) are separably associated
with relapse **on the concepts it screens**, at a matched end state. **What it
cannot:** a mechanism, generality over concepts (§3.3), or any claim about
publishability.

---

## 2. The operating point, re-derived from measured rates

### 2.1 Revision 2's gate was unreachable — the arithmetic, from existing rows

Residue `r` with an own-parent drop `≥ D` requires parent rate `p ≥ r + D`. For
revision 2's `r = 20%`, `D = 60 pp`: `p ≥ 80%`. For *any* residue in a 15–25%
tolerance band: `p ≥ 75%`. Measured on the frozen-test probe at t = 0.5, with
`MA` as the request-2 parent (from the committed `analysis_test.json`, no new
compute):

| category | s17 `p` | s29 `p` | `p ≥ 80` (reach 20% at ≥60 pp)? | `p ≥ 75` (reach *any* band point)? | max residue at ≥60 pp |
|---|---|---|---|---|---|
| horse | 93.75 | 95.00 | **yes** | yes | 33.75 |
| bicycle | 82.50 | 83.75 | **yes** | yes | 22.50 |
| bird | 78.75 | 78.75 | no | yes | 18.75 |
| **dog** | **68.75** | **75.00** | **no** | **no (s17)** | 8.75 |
| **sandwich** | **67.50** | **70.00** | **no** | **no** | 7.50 |
| chair | 62.50 | 63.75 | no | no | 2.50 |
| cat *(erased at request 1)* | 30.00 | 23.75 | no | no | none |

**Both of revision 2's request-2 targets fail.** This is not a tuning detail: the
declared regime was unsatisfiable on this base model and probe, and one more
round would have discovered it after spending GPU time. It is also a second
instance of the Amendment 01 failure mode — a level asserted rather than checked
against the set it would be measured on.

### 2.2 Declared operating point (relative), and the matched quantity

| condition | definition | declared value |
|---|---|---|
| **generation precondition** | parent rate for the newest target, on the probe being reported | **`p_k ≥ 65%`** — below this the parent does not reliably generate the target and no deletion level is meaningful |
| **deletion target** | relative suppression `1 − r_k/p_k` | **`≥ 0.75`** |
| **absolute cap** | residue `r_k` | **`≤ 20%`** |
| **matched quantity (iso-deletion)** | **absolute residue `r_k`**, across arms | **within ± 5 pp** |
| reported, never matched | `p_k`, the own-parent drop `p_k − r_k`, relative suppression | — |
| reported, never matched | suppression vs the common base `M0` | — |

**Why residue is the matched quantity.** At request *k* each method's parent is
**its own** request-(*k*−1) checkpoint, so a pp drop is measured from a different
starting point per method and is not comparable. Residue is the end state — what
the model now does — and is parent-free.

**Why the deletion target is relative.** A fixed absolute drop is unsatisfiable
wherever the parent rate is low (§2.1), and a fixed absolute residue is trivially
satisfiable wherever it is lower still. 75% relative suppression means the same
thing at `p = 95` and at `p = 70`. At `p = 65` it demands `r ≤ 16.25%`, a
≥ 48.75 pp absolute drop — meaningful, and well above the probe's representable
step (1.25 pp at 80 pairs, 0.18 pp at 560).

**Parent divergence, declared in advance.** When accepted arms' `p_k` spread
exceeds **10 pp**, matching residue and matching the drop cannot both hold. The
comparison is then labelled **parent-divergent**, both quantities are shown, and
**the phrase "iso-deletion" is not used for it.** There is no single quantity
that makes two different-parent arms equally deleted; this is a property of the
sequential setting, not of the rule.

### 2.3 Calibration on DEV, acceptance on ACCEPT, one evaluation of TEST

Four sets, disjoint and hashed before anything runs. **The pilot's frozen TEST
set is spent and is not reused, for calibration or anything else.**

1. **DEV — calibration and selection only.** Each method is driven along its own
   strength parameter until it first meets §2.2 on DEV. The dial is then **frozen
   and written down**.
2. **ACCEPT — a separate held-out set, evaluated before TEST.** The frozen dial is
   re-measured. **No retuning is permitted at this step**; §6 decides acceptance
   mechanically from what ACCEPT shows.
3. **TEST — frozen, evaluated once.** The precondition, the deletion target, the
   cap and the cross-arm match are **re-checked here** before any relapse or
   retention number is read. A failure on TEST is reported as a failure, exactly
   as Amendment 01 now does.
4. **RELAPSE probe** — the earlier targets' prompts, drawn with the same frozen
   grouping, evaluated at every stage.

Everything in §2.2, §3, §5 and §6 is **frozen in a hashed pre-registration file
before any new data are collected.**

---

## 3. Cells — counterbalanced so sharing is identifiable

### 3.1 Why revision 2's four cells could not work

With request 1 fixed as `cat` / anchor `horse`, "shared" was always `horse` and
"disjoint" was always `flower`. So the sharing contrast also changed **which**
anchor request 2 used, and changed the **target–anchor relation** (`dog`–`horse`,
two animals, versus `dog`–`flower`). Three things varied together; none was
identified.

### 3.2 The counterbalanced design

Request 1 always erases `cat`, but with **either** anchor. Request 2 varies
target and anchor independently:

| | A1 (request 1 anchor) | A2 (request 2 anchor) | sharing | A2 identity |
|---|---|---|---|---|
| 1 | horse | horse | **shared** | horse |
| 2 | horse | flower | disjoint | flower |
| 3 | flower | flower | **shared** | flower |
| 4 | flower | horse | disjoint | horse |

× request-2 target ∈ {**`dog`** (near `cat`), **`sandwich`** (far)} × 2 training
seeds = **16 sequences**. Request-1 checkpoints needed: 2 anchors × 2 seeds =
**4**, of which the `horse` pair is the **existing MA** and is reused.

**Sharing is now orthogonal to A2 identity by construction:** shared occurs once
at A2 = `horse` and once at A2 = `flower`, and so does disjoint. The sharing main
effect averages over anchor identity; **A2 identity gets its own main effect**,
estimated separately, which absorbs the target–anchor relation.

**Fallback if the extra request-1 arm is refused.** Cells 1 and 2 only
(A1 = `horse`), reusing MA. That is **four particular anchor-choice cases**, and
it is then stated that way: it **cannot isolate sharing from anchor identity**,
and no sharing claim may be drawn from it. The full design is the only version
that answers §1.1(a).

**Proximity is measured and frozen on CPU before any run**, three ways, all
reported: COCO supercategory identity (`cat`, `dog` → *animal*; `sandwich` →
*food*), CLIP text-embedding cosine, and COCO co-occurrence rate.

### 3.3 One near pair, one far pair — an exploratory test, retained from revision 2

**`cat`→`dog` is the only near pair and `cat`→`sandwich` the only far pair.**
Proximity is therefore **confounded with concept identity**: anything attributed
to proximity could be a property of `dog` versus `sandwich`, their anchor caches,
their base-model frequency, or detector behaviour on those categories.

**Training-seed replication does not help.** Two seeds replicate the *run*, not
the *concept*; they bound run-to-run variability and support a direction check
only. No number of seeds converts this into evidence about proximity in general.

**Independent concept evidence needed before any mechanism claim**, pre-registered
before collection: **≥ 5 near and ≥ 5 far pairs**, members drawn from **disjoint
COCO supercategories** across pairs so concept identity varies independently of
proximity; the proximity measure **validated** against an external judgement; the
same anchor counterbalancing within every pair; every request-2 target satisfying
the §2.2 precondition `p ≥ 65%`; and a consistent sign across pairs with
intervals satisfying §5.1. Until that exists, results are about **these
concepts**, and the word "mechanism" is unavailable.

---

## 4. What the cell intervention is, per method — and native LACU stays

Prescribing an anchor is a **configuration change** for some methods and a
**mechanism change** for others. Conflating them would quietly disable a
competitor, so two families are run and reported separately.

**Family CA — controlled anchor.** Every method runs with the cell's prescribed
A1/A2. This is the factorial design of §3.

| method | what prescribing the anchor does |
|---|---|
| ConAbl (± L2-SP), ESD | the replacement/anchor concept is already an explicit argument — **configuration** |
| UCE, RECE, SPEED | the replacement embedding is an explicit solver input — **configuration** |
| Sculpting Memory | replacement target is an input; its mask and earlier-target distillation are untouched — **configuration** |
| **CEASE** | AIC holds the **shared replacement token** `c∗` invariant. In the **shared** cells this is its native premise. In the **disjoint** cells it changes *which* token is held invariant, which is a change to the mechanism's premise — reported as **`CA-CEASE(disjoint)`, an ablation**, never as CEASE |
| **LACU** | selection **is** the mechanism. A prescribed anchor disables locality-aware target selection — reported as **`CA-LACU`, an ablation**, never as LACU, in **every** cell |

**Family NAT — native.** Every method runs as published, choosing its own
replacements; **native LACU and native CEASE are both here.** This is the actual
method comparison. The §3 cells **do not apply to it**: a native method may not
sit in any cell, and forcing it into one would be the error above. NAT is
compared on the §2.2 operating point and the §5 endpoints over the same request
sequences, with each method's realised replacement choices reported.

**Rules, declared now.** No method is excluded for lacking a monotone strength
dial: it runs at its own declared operating point and every comparison involving
it is labelled **"not iso-deletion-matched"** with the realised gap stated.
**CEASE and LACU are not droppable** to reach a baseline count or for
convenience; if either cannot be run on this host that is a **blocking finding**
about the screen, not grounds for a weaker field. A CPU feasibility audit per
method precedes everything: code, licence, dependency closure against
`env/requirements.lock.txt`, dependence on the **blocked** UnlearnCanvas
classifiers, and whether a dial exists.

---

## 5. Endpoints, predicates, and what "already solved" requires

### 5.1 Interval predicates — mutually exclusive and exhaustive

For a contrast `C` with 95% interval `[lo, hi]` and threshold `Δ`, exactly one
holds:

| predicate | condition | meaning |
|---|---|---|
| **MATERIAL+** | `lo > +Δ` | a material increase |
| **MATERIAL−** | `hi < −Δ` | a material decrease |
| **NULL(Δ)** | `−Δ ≤ lo` **and** `hi ≤ +Δ` | **practically null** at the margin |
| **INCONCLUSIVE** | otherwise | the interval straddles a boundary |

`[−2, +3]` is **NULL(10)** — not material. "Excludes +10" is never used as a
predicate again: it is satisfied by both NULL and MATERIAL−.

**Dominance requires a direct contrast.** Comparing two predicates is not a
comparison. To claim one factor dominates, the **difference of the factor
effects** is computed as its own contrast with its own interval, and that must be
**MATERIAL** in the stated direction. Likewise the anchor × proximity interaction
is `(shared−disjoint | near) − (shared−disjoint | far)`, estimated directly.

**Uncertainty.** The existing frozen conditional paired cluster bootstrap —
scene cluster as the unit, category as the stratum, 10 000 draws, recorded RNG,
95% percentile intervals — derived from the actual manifests and hashed before
any data exist. Per training seed, **never pooled**. Read from interval bounds,
never point estimates. No multiplicity correction, so every interval is
**descriptive**; an interval containing zero is not evidence of no effect.
Conditional on this base model, these concepts, these anchors, these prompts and
**this detector**.

### 5.2 Endpoints and thresholds

| endpoint | definition | threshold |
|---|---|---|
| **newest-target deletion** | §2.2 — precondition, relative target, absolute cap | the acceptance condition, not a result |
| **relapse** of each earlier target `t_i` | §6.3 | `Δ_relapse = 10 pp` |
| **preserved categories** — each of bird, chair, bicycle, plus any never-targeted category separately | change vs that method's **own** parent | `Δ_retain = 10 pp` |
| **non-target control** — a category no request in the sequence touches | change vs own parent | `Δ_retain = 10 pp` |
| **anchor integrity** — each anchor used in the sequence | change in the anchor's own generation rate vs own parent | `Δ_anchor = 10 pp` |

All reported **separately and never averaged**. `Δ = 10 pp` throughout is the
material size already declared for this project, well above the probe's
representable step, so it is a judgement about consequence rather than
resolution.

### 5.3 "Already solved" requires utility, not just low relapse

A method is **ALREADY-SOLVES** a cell only if **all four** hold, in **both**
training seeds:

1. **deletion succeeded** — the arm is ACCEPTED under §6.1;
2. **relapse NULL(10)** for **every** earlier target in the sequence;
3. **every preserved category and the non-target control NULL(10)** — low relapse
   bought by destroying the retained set is not a solution;
4. **anchor integrity NULL(10)** — a collapsed anchor is a utility failure, and
   is precisely what shared-anchor reuse would be expected to cause.

If ALREADY-SOLVES holds in all cells of the design for any method in family NAT,
**the correct decision is to report that and stop.** The contribution would then
be the measurement protocol alone, and whether that is publishable is not for
this document to assert.

### 5.4 Decision table — exhaustive over the predicates

Let `S` = the direct sharing effect on relapse (counterbalanced, §3.2), `P` = the
direct proximity effect, `I` = their interaction, each with its own interval.

| condition | reading | decision |
|---|---|---|
| any NAT method is **ALREADY-SOLVES** in every cell | the chosen cells are solved | **report and stop.** No method work |
| `S` **MATERIAL** (either sign), `P` **NULL(10)**, and `S − P` **MATERIAL** | sharing dominates **on these concepts** | design the concept-varied study (§3.3) around anchor allocation |
| `P` **MATERIAL**, `S` **NULL(10)**, and `P − S` **MATERIAL** | proximity dominates **on these concepts** | design it around representation overlap; anchor allocation is not the lever |
| both `S` and `P` **MATERIAL**, or `I` **MATERIAL** | both matter, or the effect is conditional | report the interaction; no single remedy is indicated |
| `S` and `P` both **NULL(10)** | neither factor material at this margin | sequence structure is not the lever **in these cells**; report and do not build |
| any of `S`, `P`, `S − P`, `P − S` **INCONCLUSIVE** where it is load-bearing | **inconclusive** | report widths and the prompt count needed; **do not read point estimates**, and do not substitute a non-direct comparison |
| seeds disagree in direction on relapse | **unresolved** | cost more training seeds; do not report the agreeing seed |
| **< 2 accepted arms** in a comparison (§6.2) | no comparison exists | report per-arm descriptives only |
| NAT ranking differs from each method's own default operating point | default-point comparisons are confounded **in these cells** | report as a measurement finding, no claim about other settings |

### 5.5 Measurement validity — binding

The detector is a **proxy**. UnlearnCanvas classifiers remain **unobtainable on
this host**: no UA / IRA / CRA. **No human annotation has been performed anywhere
in this project.** The 180-item paired packet must be labelled before any screen
result is a finding, and the screen must add its own paired packet on the terms
verified in
[`annotation_packet_paired180/BLINDING_AUDIT.md`](../../results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md)
— secret salt, secret row order, **re-encoded image bytes with pixels verified**,
and all three attacks run against the published records *and the delivered files*
before anyone receives it.

---

## 6. Deterministic acceptance, and three outcomes that are not the same

### 6.1 Arm acceptance — mechanical, from ACCEPT and TEST

For arm `M` at stage `k`, evaluated in order, first match wins:

| | condition on the set being reported | state |
|---|---|---|
| 1 | `p_k < 65%` | **PRE-SUPPRESSED** |
| 2 | `p_k ≥ 65%` and (`1 − r_k/p_k < 0.75` or `r_k > 20%`) | **ERASURE-FAILURE** |
| 3 | `p_k ≥ 65%`, target and cap met | **ACCEPTED** |

Applied on **ACCEPT** to decide whether the arm proceeds, and **again on TEST**
before any endpoint is read. No tuning intervenes; the dial is frozen at DEV.

### 6.2 Comparison acceptance

A comparison between arms `A` and `B` at stage `k` is **ACCEPTED** iff both are
ACCEPTED **and** `|r_k(A) − r_k(B)| ≤ 5 pp` on TEST. If their `p_k` spread
exceeds 10 pp it is additionally labelled **parent-divergent** (§2.2).
**A comparative conclusion requires ≥ 2 accepted arms.** With fewer, per-arm
descriptives are reported and no contrast is formed — and that is the reported
outcome, not a gap to be filled by relaxing the rule.

### 6.3 Prior collateral suppression ≠ current erasure failure ≠ later relapse

Three states that a single "the target isn't being generated" number would
conflate:

| | what it is | how it is detected | how it is reported |
|---|---|---|---|
| **prior collateral suppression** | an **earlier** request already pushed `t_k` down, so the regime is unreachable by arithmetic before this request starts | `p_k < 65%` **measured before request k** | **PRE-SUPPRESSED**, with `p_k`. Excluded from the matched comparison at this stage and from relapse at later stages. **A finding about the earlier request**, and exactly the cross-edit interference the screen is about |
| **current erasure failure** | the parent did generate `t_k`, but this arm could not reach the target | `p_k ≥ 65%`, target or cap missed | **ERASURE-FAILURE** at stage *k*, with achieved `r_k`. Excluded from relapse analysis for `t_k`, with the count reported — never silently dropped |
| **later relapse** | `t_i` **was** erased, then came back | defined only where stage *i* was ACCEPTED | `Relapse_i(k) = D(C_k, t_i) − r_i` in pp, against **that method's own** post-erasure checkpoint `C_i` and its achieved `r_i`. **Positive = came back** |

`Relapse` is **never** computed from a reference that failed acceptance. A high
`D(C_k, t_i)` after an ERASURE-FAILURE at stage *i* is persistence of a concept
that was never removed — a statement about stage *i*. An arm with excluded stages
cannot be compared on relapse against an arm without them; that comparison is
reported as **unavailable**.

### 6.4 Stopping rule

Stop and report, without proceeding:

- **CEASE or LACU cannot be run on this host** — report the blockage; do not
  substitute a weaker field;
- **no request-2 target satisfies `p_k ≥ 65%`** on DEV for a given sequence —
  report the measured rates (§2.1 is this check, done in advance);
- **no method meets §2.2 on DEV** — report the achieved ceiling and the dials;
- **the frozen dial fails on ACCEPT for every method** — the operating point does
  not transfer; report it (Amendment 01's failure, caught a stage earlier);
- **fewer than 2 arms ACCEPTED on TEST** in the load-bearing comparison;
- **the load-bearing direct contrasts are INCONCLUSIVE** at the planned prompt
  count;
- **seeds disagree in direction** on relapse;
- **frozen-set overlap** between DEV, ACCEPT, TEST, the relapse probe, the pilot
  set or the anchor-training strings;
- **any artifact fails `scripts/seq/validate_stage.py`**.

A stop is a **result**.

---

## 7. The candidate direction — unproved, with the missing proof named

**No method is proposed for approval.** Revision 1's anchor-allocation sketch was
withdrawn because LACU occupies it. What remains open among the six works checked
is the **object held invariant**:

> CEASE's AIC appends the **shared replacement token** `c∗` to the invariance
> matrix. The candidate appends instead **each earlier target's achieved
> post-erasure state**, so the solved constraint is *"`t_i` stays where its
> erasure left it"*.

**Stated as a direction, with what is unproved made explicit:**

- **Representation.** The invariance row would act on the cross-attention
  key/value projections of a text embedding for `t_i` — the same objects CEASE's
  invariance matrix constrains. The quantity it can bound is a **distance in
  value-projection space**, evaluated at a finite set of embeddings.
- **Assumption 1 — monotonicity.** That image-level detectability of `t_i` is
  monotone, or at least bounded, in that projection distance. **Unproved.** A
  value-projection invariance row — like a distillation penalty — constrains an
  intermediate representation, and **neither is automatically sufficient for a
  bound on image-level relapse.**
- **Assumption 2 — coverage.** That a finite set of constrained embeddings
  controls behaviour on the prompt distribution actually probed, including
  paraphrases. **Unproved**, and the pilot's literal/paraphrase split shows these
  can diverge.
- **Assumption 3 — composition.** That the constraint survives *k* sequential
  solves without the admissible set becoming empty or degenerate. **Unproved**;
  it may reduce to HOC in some limit, or be infeasible once several earlier
  targets are pinned.
- **What a future proof would need.** A bound from value-projection distance to
  the sampler's output distribution for `t_i`-bearing prompts — a Lipschitz-type
  argument through the full sampling chain — composed with a calibrated
  relationship between that distribution and the detector's rate. **Neither
  exists**, and a *measured* relapse bound is not a *proved* one. Absent that,
  the construction can only be offered as an empirically-checked constraint, and
  the word "certificate" should not be used for it.

**It is a direction to be tested against CEASE and LACU under a matched end
state, not a contribution**, and its value depends on §5.4 not returning the
ALREADY-SOLVES row.

---

## 8. Not in this proposal

- **Any GPU work.** The costing is CPU arithmetic.
- Any repair, re-selection or extension of Amendment 01, or any reuse of its
  spent frozen test set — including for calibration.
- Any determinism or repeatability study.
- Any claim of generality over concepts (§3.3), any mechanism claim, any proof
  claim (§7), any exhaustive-literature claim (§1), and any assertion about
  publishability.
- Human annotation labour, which must be **scheduled**, not assumed.

## 9. CPU deliverables, if this direction is approved

1. the hashed **pre-registration**: §2.2 operating point and references, §3.2
   cells and counterbalancing, §5 predicates and thresholds, §6 acceptance
   states, the four set identities, the bootstrap grouping, and the per-arm
   advance decision for an ACCEPT failure;
2. a **feasibility screen for every candidate request-2 target** on the lines of
   §2.1 — measured parent rate, the residue the relative target implies, and
   whether the precondition holds — run before any target is adopted;
3. frozen manifests for the 16 sequences with the three frozen proximity
   measurements and all digests;
4. the §4 method audit, CEASE and LACU first, with the CA/NAT family labels and
   the ablation labels fixed in the code that names the arms;
5. the relapse estimator and its frozen grouping, with a CPU regression that
   **fails the build** if a relapse number is computed from a PRE-SUPPRESSED or
   ERASURE-FAILURE stage, if a comparison is formed from fewer than two accepted
   arms, or if an interval is read where the §6 conditions were not met;
6. a measured cost model per sequence from this host's existing per-step and
   per-image measurements, wall time kept separate from device-hours, including
   the **extra request-1 arm** the counterbalancing needs;
7. the screen's paired human-audit packet design on the terms in §5.5.

---

### Sources

The six works in §1, read for this document; no exhaustive survey was performed.
