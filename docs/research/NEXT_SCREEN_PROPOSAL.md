# Next screen — an identifiable probe of interference in sequential erasure

**CPU-only proposal, revision 4. This authorises nothing; no GPU run, no
generation, no annotation and no method implementation follows from it.** For the
PI's decision. If approved, compute must run on GPUs **independently verified
unoccupied by another user**
([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)); the hour
ceiling stays retired and caps nothing. **Novelty and publishability are
unestablished.**

> **What revision 4 fixes in revision 3**
>
> 1. **First-stage reuse was asserted, not qualified.** Revision 3 promised to
>    reuse the `horse` MA pair, but MA's historical `cat` residue is
>    **30.00 / 23.75%**, above the new 20% cap — and §6 requires an **ACCEPTED**
>    earlier erasure before relapse is even defined. Reuse is now **conditional**
>    on qualification against the new frozen sets, with a stated outcome if it
>    fails (§3).
> 2. **Checkpoint ownership was wrong.** "Four request-1 parents" was a global
>    count. Each method owns its own chain, so it is **2 anchors × 2 seeds *per
>    method*** (§3.2).
> 3. **`p_k < 65%` was labelled as if it proved prior damage.** It is an
>    eligibility condition and a **chosen floor**. `M0 = 60%, parent = 60%` is a
>    low parent with zero prior damage; `95% → 70%` is material prior damage that
>    *passes* the floor. Prior damage now needs a **paired M0→parent contrast with
>    uncertainty** (§4).
> 4. **Stopping logic was unordered and stagewise-only.** Two 8 pp stagewise
>    losses pass every stagewise check and breach a 10 pp overall margin, so
>    **cumulative preservation against M0** is now required alongside. Native
>    success is defined over each method's **own** sequences, stages and seeds.
>    The decision rules are **ordered**, so `S` and `P` both NULL with `I`
>    MATERIAL no longer triggers incompatible rows (§6).
>
> Retained unchanged because they were right: the four interval predicates,
> direct contrasts for dominance, anchor counterbalancing, and the native-versus-
> controlled distinction.

---

## 1. Prior work — scoped to the six works actually read

No exhaustive survey was done; every absence claim is scoped to these six.

| work | what it establishes |
|---|---|
| **CEASE** ([2610.01989](https://arxiv.org/html/2610.01989)) | **§3.2** derives accumulated noise into an **anchor** channel (repeated activation of the shared replacement direction), an **off-anchor** channel (`αᵢᵀαⱼ` overlap) and a **cross** channel, vanishing under `βᵢᵀc∗ = 0` and `αᵢᵀαⱼ = 0`. **§3.3**: **AIC** appends the shared replacement token to the invariance matrix; **HOC** projects updates onto the orthogonal complement of the SVD of past updates. 100 sequential edits, SD 1.4/1.5/2.1. **A.5** tests **five random orders** |
| **LACU** ([2512.02657](https://arxiv.org/abs/2512.02657)) | **locality-aware target selection** — per forget prompt, the mapping prompt the model itself treats as most similar by score-prediction distance; **locality-aware replay** — nearest retain concepts as a local functional regulariser. Stable over **10 sequential steps** |
| **Sculpting Memory** ([2504.09039](https://arxiv.org/html/2504.09039v1)) | dynamic gradient mask, concept-aware loss, **distillation explicitly targeting persistence of earlier erasures** |
| **M-ErasureBench** ([2512.22877](https://arxiv.org/pdf/2512.22877)) | erasure across prompts, embeddings, inverted latents — varies the **probe modality** |
| **Side Effects of Erasing Concepts** ([2508.15124](https://arxiv.org/pdf/2508.15124)) | collateral effects of single erasures |
| **When Are Concepts Erased?** ([2505.17013](https://arxiv.org/pdf/2505.17013)) | mechanism and timing of erasure within the model |

**Already addressed, not claimed as gaps:** shared-anchor interference (AIC),
historical-update interference (HOC), order sensitivity (A.5), persistence of
earlier erasures as an objective (Sculpting Memory), locality-aware replacement
selection (LACU).

### 1.1 The narrow question

> At a **matched end state** on the newest target, is **relapse of an earlier
> target** separably associated with **(a) whether successive requests share a
> replacement anchor** and **(b) how semantically close successive targets are** —
> and does an existing remedy's benefit depend on the combination?

Among the six works read, none varies target proximity, none separates it from
anchor sharing, and none reports a stagewise measurement of relapse against each
earlier target's *own* post-erasure state. CEASE varies **order**, a different
factor; in celebrity- and style-only streams proximity is roughly constant and
the anchor is shared throughout, so `αᵢᵀαⱼ` and proximity are not separable from
published results. Constraint ablations show a remedy helps on the evaluated
stream; they do not identify which sequence property made it hard.

**The screen can** test whether (a) and (b) are separably associated with relapse
**on the concepts it screens**. **It cannot** establish a mechanism, generalise
over concepts (§5.3), or make anything publishable.

---

## 2. Operating point, re-derived from measured rates

### 2.1 Planning inputs, not results

Residue `r` with an own-parent drop `≥ D` needs parent rate `p ≥ r + D`.
Revision 2's `r = 20%`, `D = 60 pp` needs `p ≥ 80%`. Measured on the **spent**
frozen-test probe at t = 0.5 with `MA` as the request-2 parent:

| category | s17 `p` | s29 `p` | | category | s17 `p` | s29 `p` |
|---|---|---|---|---|---|---|
| horse | 93.75 | 95.00 | | **dog** | **68.75** | **75.00** |
| bicycle | 82.50 | 83.75 | | **sandwich** | **67.50** | **70.00** |
| bird | 78.75 | 78.75 | | chair | 62.50 | 63.75 |
| | | | | cat *(erased at request 1)* | 30.00 | 23.75 |

Both of revision 2's request-2 targets fail `p ≥ 80%`, and `dog` on seed 17 fails
`p ≥ 75%`, so no point in a 15–25% band was reachable either. **These are
exploratory planning inputs from a spent probe, not results and not predictions.**
The new evaluation uses **fresh frozen sets**, every rate is re-measured there,
and no threshold is set from these numbers beyond choosing a *form* for the
target that does not depend on an unknown absolute rate.

### 2.2 The declared operating point

| condition | definition | declared value |
|---|---|---|
| **recognisability floor** | parent rate for the newest target on the set being reported | **`p_k ≥ 65%`** — a **chosen floor**, see §4.1 |
| **deletion target** | relative suppression `1 − r_k/p_k` | **`≥ 0.75`** |
| **absolute cap** | residue `r_k` | **`≤ 20%`** |
| **matched quantity** | **absolute residue `r_k`** across arms | **within ± 5 pp** |
| reported, never matched | `p_k`, own-parent drop, relative suppression, suppression vs `M0` | — |

**Residue is the matched quantity** because at request *k* each method's parent is
**its own** request-(*k*−1) checkpoint, so a pp drop starts from a different place
per method. Residue is the end state and is parent-free.

**The target is relative** because a fixed absolute drop is unsatisfiable wherever
the parent rate is low and a fixed absolute residue is trivial wherever it is
lower still. At `p = 65` the target demands `r ≤ 16.25%`, a ≥ 48.75 pp absolute
drop — well above the probe's representable step (1.25 pp at 80 pairs, 0.18 pp at
560).

**Parent divergence.** When accepted arms' `p_k` spread exceeds **10 pp**, residue
matching and drop matching cannot both hold: the comparison is labelled
**parent-divergent**, both quantities shown, and **"iso-deletion" is not used for
it.** No single quantity makes two different-parent arms equally deleted.

### 2.3 Four frozen sets; the spent TEST set is not reused

**DEV** calibration and selection only · **ACCEPT** held-out qualification, **no
retuning** · **TEST** evaluated once, all conditions re-checked there ·
**RELAPSE probe** the earlier targets' prompts, evaluated at every stage. All
disjoint and hashed. **Amendment 01's frozen TEST set is spent and is not reused
for anything, calibration included.** Everything in §2.2, §5, §6 and §3.1 is
frozen in a hashed pre-registration before any new data are collected.

---

## 3. Request 1 — qualification, not reuse by assumption

### 3.1 Each method's request-1 checkpoint must qualify on the new sets

Request 1 always erases `cat`. Its parent is **`M0`**, the common pretrained
base, so `p_1 = D(M0, cat)` **measured on the new sets** — not assumed.

**Qualification procedure**, identical in form to every other stage:

1. **calibrate on DEV** — drive that method's strength parameter until it first
   meets §2.2 for `cat` on DEV; **freeze the dial**;
2. **qualify on ACCEPT** — re-measure with the frozen dial. The checkpoint is
   **QUALIFIED** iff `p_1 ≥ 65%`, relative suppression `≥ 0.75` and `r_1 ≤ 20%`;
3. **re-check on TEST** — the reported reference residue `r_1` is the TEST value,
   and it is the reference every later relapse number is measured against (§6.2).

**No request-2 stage runs for a method whose request-1 checkpoint is not
QUALIFIED**, because §6 defines relapse only against an ACCEPTED earlier erasure.

### 3.2 Reuse of the existing MA pair is conditional, and ownership is per method

**Ownership.** Each method owns its own chain: a sequence erased by method *M* at
request 1 and method *M* at request 2. A checkpoint produced by ConAbl is not a
valid request-1 parent for UCE, RECE, SPEED, CEASE, LACU or Sculpting Memory.
Revision 3's "4 request-1 parents" was a global count and was wrong. The correct
count is **2 anchors × 2 training seeds = 4 request-1 checkpoints *per method***,
and **2 (A1) × 2 (A2) × 2 (targets) × 2 (seeds) = 16 request-2 checkpoints per
method.** This is a materially larger commitment than revision 3 implied and is
an input to the decision, not a detail.

**The existing `horse` MA pair may serve only as the ConAbl arm's request-1
checkpoint, and only if it QUALIFIES on the new ACCEPT and TEST sets.**

**Its status is in doubt, and is not decided here.** Its historical `cat` residue
on the spent probe is **30.00% (s17) and 23.75% (s29)**, both above the 20%
absolute cap — so on *that* probe it would not have qualified. But that probe is
spent, a different prompt sample gives a different rate (the entire Amendment 01
finding), and `p_1 = D(M0, cat)` has never been measured on the new sets at all.
So:

- **no claim that reuse will fail**, and **no unconditional acceptance** either;
- **no training and no threshold relaxation now.** The cap is not moved to admit
  a checkpoint, and nothing is retrained to meet it.

**If qualification fails** — the two prespecified options, declared now:

| | |
|---|---|
| **(i) stop that arm's sequence** | report request-1 non-qualification with the measured `p_1` and `r_1`. This is the default and needs no further approval. A method with no qualified request 1 contributes no relapse data, and if that leaves fewer than two accepted arms the comparison does not exist (§6.4 rule 0) |
| **(ii) a revised first stage** | *described for later approval, not executed:* re-run request 1 for **every** method under the frozen protocol with the dial calibrated on DEV, removing reuse entirely and making the count uniformly 4 request-1 checkpoints per method. This is the clean design; it is also strictly more compute, and **it is not authorised by this document** |

---

## 4. Low recognisability is not evidence of prior damage

### 4.1 `65%` is a chosen floor

The only arithmetic constraint is `r_k ≤ p_k` — a residue cannot exceed the rate
the parent generates at. **`p_k ≥ 65%` is a judgement**, picked so that 75%
relative suppression corresponds to at least a 48.75 pp absolute drop. It is not
an impossibility threshold, and nothing below it is impossible — only
uninformative at this margin. The state is therefore named **`LOW-PARENT`**, a
neutral description of the measurement, **not** `PRE-SUPPRESSED`.

### 4.2 Prior damage needs its own contrast

A low parent rate has at least three causes that `p_k` cannot distinguish: the
base model never generated the concept reliably; an earlier edit suppressed it;
or the probe is hard for that category. **Counterexample:** `D(M0, t_k) = 60%`
and `p_k = 60%` is `LOW-PARENT` with **zero** prior damage. **Converse:**
`D(M0, t_k) = 95%` falling to `p_k = 70%` is a **25 pp** material loss that
**passes** the floor.

So a prior-damage claim uses a **paired `M0` → parent contrast with
uncertainty**, on the same frozen grouping and bootstrap:

> **`PriorΔ_k = D(M0, t_k) − D(C_{k−1}, t_k)`**, with a 95% interval, read with
> the §5.1 predicates at **`Δ_prior = 10 pp`**.
> **MATERIAL+ → prior collateral suppression of `t_k` by the earlier edits.**
> NULL(10) → no material prior loss. INCONCLUSIVE → say so.

**The floor and the prior-damage test are independent and both are reported.** An
arm can be `LOW-PARENT` with `PriorΔ` NULL, or eligible with `PriorΔ` MATERIAL+.
Only the contrast supports a statement about what an earlier request did.

---

## 5. Endpoints and predicates

### 5.1 Four predicates — exclusive and exhaustive (retained)

For a contrast with 95% interval `[lo, hi]` and threshold `Δ`, exactly one holds:

| predicate | condition |
|---|---|
| **MATERIAL+** | `lo > +Δ` |
| **MATERIAL−** | `hi < −Δ` |
| **NULL(Δ)** | `−Δ ≤ lo` **and** `hi ≤ +Δ` |
| **INCONCLUSIVE** | otherwise |

`[−2, +3]` is **NULL(10)**, not material. "Excludes +10" is never a predicate:
both NULL and MATERIAL− satisfy it.

**Dominance requires a direct contrast (retained).** Comparing two predicates is
not a comparison. To claim a factor dominates, the **difference of the factor
effects** is estimated as its own contrast with its own interval and must be
MATERIAL in the stated direction. The interaction
`I = (shared−disjoint | near) − (shared−disjoint | far)` is likewise estimated
directly.

**Uncertainty.** The frozen conditional paired cluster bootstrap — scene cluster
as unit, category as stratum, 10 000 draws, recorded RNG, 95% percentile
intervals — hashed before any data exist. Per training seed, **never pooled**.
Read from bounds, never point estimates. No multiplicity correction, so every
interval is **descriptive**. Conditional on this base model, these concepts,
these anchors, these prompts and **this detector**.

### 5.2 Stagewise **and** cumulative preservation

Stagewise checks alone are insufficient: **two 8 pp losses each pass a 10 pp
stagewise margin and together breach it.** Both are therefore required, at
`Δ = 10 pp`, for **every** retained category, the non-target control, and **every
anchor used in the sequence**, each reported separately and never averaged:

| | definition | reference |
|---|---|---|
| **stagewise** | `D(C_{k−1}, c) − D(C_k, c)` at each stage *k* | that method's own previous checkpoint |
| **cumulative** | `D(M0, c) − D(C_K, c)` at the final stage *K* | the common base `M0` |

| endpoint | threshold |
|---|---|
| newest-target deletion | the §2.2 acceptance condition, not a result |
| **relapse** of each earlier target (§6.2) | `Δ_relapse = 10 pp` |
| retained categories (bird, chair, bicycle, …), each separately | `Δ_retain = 10 pp`, **stagewise and cumulative** |
| non-target control | `Δ_retain = 10 pp`, **stagewise and cumulative** |
| anchor integrity, each anchor separately | `Δ_anchor = 10 pp`, **stagewise and cumulative** |
| prior collateral suppression (§4.2) | `Δ_prior = 10 pp` |

### 5.3 Cells — counterbalanced, and still exploratory over concepts

Request 1 erases `cat` with **either** anchor; request 2 varies target and anchor
independently:

| | A1 | A2 | sharing | A2 identity |
|---|---|---|---|---|
| 1 | horse | horse | **shared** | horse |
| 2 | horse | flower | disjoint | flower |
| 3 | flower | flower | **shared** | flower |
| 4 | flower | horse | disjoint | horse |

× request-2 target ∈ {`dog` (near), `sandwich` (far)} × 2 seeds = **16 sequences
per method**. Sharing is **orthogonal to A2 identity by construction** — shared
occurs once at each anchor, and so does disjoint — so the sharing main effect
averages over anchor identity, and **A2 identity has its own main effect**, which
absorbs the target–anchor relation.

**Fallback if the extra request-1 arm is refused:** cells 1–2 only. That is
**four particular anchor-choice cases** and **cannot isolate sharing from anchor
identity**; no sharing claim may be drawn from it.

**Still exploratory over concepts (retained).** `cat`→`dog` is the only near pair
and `cat`→`sandwich` the only far pair, so **proximity is confounded with concept
identity**. Two training seeds replicate the *run*, not the *concept*.
**Independent concept evidence needed before any mechanism claim**, pre-registered:
≥ 5 near and ≥ 5 far pairs from **disjoint COCO supercategories**, proximity
**validated** against an external judgement, the same counterbalancing within
every pair, every target satisfying `p ≥ 65%` on the new sets, and a consistent
sign with intervals satisfying §5.1. Proximity is measured and frozen on CPU
before any run (COCO supercategory, CLIP text cosine, co-occurrence rate).

### 5.4 Two method families (retained)

**Family CA — controlled anchor**, carrying the §5.3 factorial. **Family NAT —
native**, every method as published, choosing its own replacements; this is the
actual method comparison and **native LACU and native CEASE are both in it.**
The cells **do not apply to NAT**.

| method | what prescribing the anchor does |
|---|---|
| ConAbl (± L2-SP), ESD | explicit argument — **configuration** |
| UCE, RECE, SPEED | explicit solver input — **configuration** |
| Sculpting Memory | replacement is an input; mask and earlier-target distillation untouched — **configuration** |
| **CEASE** | AIC holds the **shared replacement token** invariant; the disjoint cells change which token that is, a change to the mechanism's premise → **`CA-CEASE(disjoint)`, an ablation**, never reported as CEASE |
| **LACU** | selection **is** the mechanism → **`CA-LACU`, an ablation in every cell**, never reported as LACU |

No method is excluded for lacking a monotone dial: it runs at its own declared
operating point and every comparison involving it is labelled **"not
iso-deletion-matched"** with the realised gap. **CEASE and LACU are not droppable
for convenience or a count**; if either cannot be run here that is a **blocking
finding**. A CPU feasibility audit precedes everything.

---

## 6. Acceptance, relapse, and the ordered decision rules

### 6.1 Arm acceptance — mechanical, first match wins

| | condition on the set being reported | state |
|---|---|---|
| 1 | `p_k < 65%` | **LOW-PARENT** (neutral; see §4) |
| 2 | `p_k ≥ 65%` and (`1 − r_k/p_k < 0.75` or `r_k > 20%`) | **ERASURE-FAILURE** |
| 3 | `p_k ≥ 65%`, target and cap met | **ACCEPTED** |

Applied on ACCEPT to decide whether the arm proceeds, and **again on TEST**
before any endpoint is read. The dial is frozen at DEV; no tuning intervenes.
`PriorΔ_k` (§4.2) is reported for every stage regardless of state.

### 6.2 Relapse, and the three states it must not be confused with

> **`Relapse_i(k) = D(C_k, t_i) − r_i`**, against **that method's own**
> post-erasure checkpoint `C_i` and its **TEST** reference residue `r_i`.
> **Positive = the erased concept came back.** Defined **only** where stage *i*
> was ACCEPTED.

| | what it is | reported as |
|---|---|---|
| **LOW-PARENT** | the parent rate is below the chosen floor — a statement about **recognisability**, not causation | the state plus `p_k`; a prior-damage claim needs `PriorΔ` (§4.2) |
| **prior collateral suppression** | an earlier edit materially reduced `t_k` | `PriorΔ_k` **MATERIAL+** — a finding about the **earlier** request |
| **ERASURE-FAILURE** | the parent did generate `t_k`, this arm missed the target | the state plus achieved `r_k`; excluded from relapse for `t_k`, count reported |
| **relapse** | `t_i` was erased, then came back | the contrast above |

`Relapse` is **never** computed from a reference that failed acceptance. An arm
with excluded stages cannot be compared on relapse against an arm without them;
that comparison is **unavailable**.

### 6.3 Comparison acceptance

A comparison between arms `A` and `B` is **ACCEPTED** iff both are ACCEPTED and
`|r_k(A) − r_k(B)| ≤ 5 pp` on TEST; it is additionally **parent-divergent** if
their `p_k` spread exceeds 10 pp. **A comparative conclusion requires ≥ 2
accepted arms.** With fewer, per-arm descriptives are reported, no contrast is
formed, and that is the reported outcome.

### 6.4 ALREADY-SOLVES — utility as well as deletion and relapse

For **family CA** a method ALREADY-SOLVES a **cell**, and for **family NAT** it
ALREADY-SOLVES **its own realised sequence**, only if **all five** hold across
**every stage** of that sequence and in **both** training seeds:

1. every stage **ACCEPTED** (§6.1) — deletion actually succeeded;
2. **relapse NULL(10)** for every earlier target at every later stage;
3. every retained category and the non-target control **NULL(10) stagewise *and*
   cumulatively** (§5.2);
4. **anchor integrity NULL(10) stagewise *and* cumulatively**, for every anchor
   the sequence used;
5. no stage **LOW-PARENT** and no `PriorΔ` **MATERIAL+** inside the sequence —
   otherwise the sequence never reached the regime it claims to solve.

For NAT this is evaluated over the **method's actual target sequences, stages
and seeds**, never over the CA cells, because §5.4 says the cells do not apply to
native runs.

### 6.5 Decision rules, in strict precedence order

Let `S` = the direct sharing effect on relapse, `P` = the direct proximity
effect, `I` = their interaction (§5.1). **Evaluate in order; the first rule that
fires is the decision.** This is what prevents incompatible readings — `S` and
`P` both NULL with `I` MATERIAL now resolves at rule 3, not at rule 6.

| # | condition | decision |
|---|---|---|
| **0** | any **qualification failure**: no QUALIFIED request 1 for a needed arm (§3.1), fewer than 2 accepted arms (§6.3), or the delivery gate REJECTED the audit packet | **the screen has no result.** Report the failure. Nothing below is evaluated |
| **1** | the two training **seeds disagree in direction** on relapse | **UNRESOLVED.** Cost more seeds; do not report the agreeing seed |
| **2** | a NAT method **ALREADY-SOLVES** its own sequences (§6.4) | **report and stop.** No method work; the contribution would be the protocol alone, and its publishability is not asserted |
| **3** | `I` **MATERIAL** | **the effect is conditional.** Report the interaction; no main-effect or dominance reading is available, whatever `S` and `P` show |
| **4** | `S` and `P` both **MATERIAL** | both factors matter on these concepts; no single remedy indicated |
| **5** | `S` **MATERIAL**, `P` **NULL(10)**, **and `S − P` MATERIAL** | sharing dominates **on these concepts**; design the concept-varied study around anchor allocation |
| **6** | `P` **MATERIAL**, `S` **NULL(10)**, **and `P − S` MATERIAL** | proximity dominates **on these concepts**; design it around representation overlap |
| **7** | `S` and `P` both **NULL(10)** and `I` **NULL(10)** | sequence structure is not the lever **in these cells**; report and do not build |
| **8** | anything else — any load-bearing contrast **INCONCLUSIVE** | **inconclusive.** Report widths and the prompt count needed; do not read point estimates, and do not substitute a non-direct comparison |

### 6.6 Stopping rule

Stop and report, without proceeding: **CEASE or LACU cannot be run here**; **no
method's request 1 QUALIFIES** (§3.1); no request-2 target satisfies `p_k ≥ 65%`
on DEV; no method meets §2.2 on DEV; the frozen dial fails on ACCEPT for every
method; fewer than 2 arms ACCEPTED on TEST; the load-bearing direct contrasts are
INCONCLUSIVE; seeds disagree in direction; frozen-set overlap among DEV / ACCEPT
/ TEST / relapse probe / pilot / anchor-training strings; any artifact fails
`scripts/seq/validate_stage.py`. **A stop is a result.**

### 6.7 Measurement validity — binding

The detector is a **proxy**. UnlearnCanvas classifiers remain **unobtainable on
this host**: no UA / IRA / CRA. **No human annotation has been performed anywhere
in this project.** The 180-item paired packet must be labelled before any screen
result is a finding, by an annotator who has seen **neither earlier packet nor
any source image** (13 of its 180 images overlap the 228-item packet). The
screen's own packet must pass the delivery gate on the terms in
[`annotation_packet_paired180/BLINDING_AUDIT.md`](../../results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md).

---

## 7. The candidate direction — unproved, with the missing proof named

**No method is proposed for approval.** Revision 1's anchor-allocation sketch is
withdrawn: LACU occupies it. What remains open among the six works read is the
**object held invariant**:

> CEASE's AIC appends the **shared replacement token** `c∗` to the invariance
> matrix. The candidate appends instead **each earlier target's achieved
> post-erasure state**, so the solved constraint is *"`t_i` stays where its
> erasure left it"*.

- **Representation.** The row would act on cross-attention key/value projections
  of a text embedding for `t_i` — the objects CEASE's invariance matrix already
  constrains. What it can bound is a **distance in value-projection space** at a
  finite set of embeddings.
- **Assumption 1 — monotonicity.** That image-level detectability of `t_i` is
  monotone, or at least bounded, in that distance. **Unproved.** A
  value-projection invariance row — like a distillation penalty — constrains an
  intermediate representation and **is not automatically sufficient for a bound
  on image-level relapse.**
- **Assumption 2 — coverage.** That finitely many constrained embeddings control
  behaviour on the probed prompt distribution, paraphrases included.
  **Unproved**, and the pilot's literal/paraphrase split shows these diverge.
- **Assumption 3 — composition.** That the constraint survives *k* sequential
  solves without the admissible set emptying or degenerating. **Unproved**; it may
  reduce to HOC in some limit.
- **What a proof would need.** A bound from value-projection distance to the
  sampler's output distribution for `t_i`-bearing prompts — a Lipschitz-type
  argument through the full sampling chain — composed with a calibrated
  relationship between that distribution and the detector's rate. **Neither
  exists.** A *measured* relapse bound is not a *proved* one, and **the word
  "certificate" should not be used for this construction.**

A direction to be tested against CEASE and LACU at a matched end state, **not a
contribution**; its value depends on §6.5 rule 2 not firing.

---

## 8. Not in this proposal

Any GPU work, generation, annotation answer or method implementation. Any repair
or reuse of Amendment 01, including its spent TEST set. Any determinism study.
Any claim of generality over concepts, mechanism, proof, exhaustive literature
coverage, novelty or publishability. Any relaxation of the §2.2 thresholds to
admit the existing MA pair. Human annotation labour, which must be **scheduled**.

## 9. CPU deliverables, if this direction is approved

1. the hashed **pre-registration** of §2.2, §3.1, §5, §6, the four set
   identities, the bootstrap grouping, and the per-arm advance decision for an
   ACCEPT failure;
2. the **request-1 qualification plan** of §3.1 per method, with the ConAbl reuse
   case marked conditional and option (ii) costed but unauthorised;
3. a **feasibility screen per candidate target** on the lines of §2.1, run on the
   new sets before any target is adopted;
4. frozen manifests for the 16 sequences per method, with the three frozen
   proximity measurements and all digests;
5. the §5.4 method audit, CEASE and LACU first, with CA/NAT and ablation labels
   fixed **in the code that names the arms**;
6. the relapse and `PriorΔ` estimators with their frozen grouping, and a CPU
   regression that **fails the build** if a relapse number is computed from a
   LOW-PARENT or ERASURE-FAILURE stage, if a comparison is formed from fewer than
   two accepted arms, if a cumulative check is skipped, or if the §6.5 rules are
   evaluated out of order;
7. a measured cost model **per method** — 4 request-1 plus 16 request-2
   checkpoints each — wall time separate from device-hours;
8. the screen's paired packet design on the §6.7 terms.
