# Next screen — an exploratory factorial probe of interference in sequential erasure

**CPU-only proposal, revision 2. This document authorises nothing, and no GPU
experiment is authorised by the Amendment 01 closeout or by this file.**
Submitted for the PI's decision. If approved, any compute must run on GPUs
**independently verified unoccupied by another user**
([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)); the hour
ceiling stays retired and caps nothing.

> **Revision 2 corrects revision 1**, which overstated the gap. Revision 1 said
> nobody had separated shared-anchor from historical-update interference. **CEASE
> addresses both, derives them, and tests order robustness** — see §1. Revision 1
> also excluded LACU and proposed an anchor-allocation method that **LACU's own
> mechanism overlaps**. Both are corrected here, the claimed gap is narrowed to
> what the literature actually leaves open, and the guarantees about
> publishability are withdrawn: a screen can establish a question is worth
> answering; it cannot promise that answering it is publishable.

---

## 1. Closest prior work, read for what it actually does

| work | what it establishes | checked against |
|---|---|---|
| **CEASE** — *Continual Concept Erasure by Suppressing Cross-Edit Interference* (arXiv [2610.01989](https://arxiv.org/html/2610.01989)) | **§3.2 derives** the accumulated-noise decomposition into an **anchor channel** (repeated activation of the shared replacement direction), an **off-anchor channel** (influence overlap `αᵢᵀαⱼ` between edits) and a **cross channel** coupling them, and shows the terms vanish under `βᵢᵀc∗ = 0` and `αᵢᵀαⱼ = 0`. **§3.3** supplies both remedies: **AIC** appends the shared replacement token to the invariance matrix (killing anchor + cross), **HOC** projects each update onto the orthogonal complement of the SVD of cumulative past updates (reducing off-anchor overlap). 100 sequential edits, SD 1.4/1.5/2.1, celebrities + styles + instances. **Appendix A.5** tests **five random permutations** of the 100 targets and reports mean ± std, with variation "small on every metric relative to between-method differences" | §3.2, §3.3, A.5 |
| **LACU** — *Locality-Aware Continual Unlearning* (arXiv [2512.02657](https://arxiv.org/abs/2512.02657)) | **Locality-Aware Target Selection** picks, per forget prompt, the context-preserving mapping prompt the model *itself* treats as most similar, by score-prediction distance. **Locality-Aware Replay** uses the same metric to find the retain concepts nearest the forget concept and replays them as a local functional regulariser. Reports stable unlearning over **10 sequential steps** where prior approaches "collapse after only 3–5" | abstract, method summary |
| **Sculpting Memory** (arXiv [2504.09039](https://arxiv.org/html/2504.09039v1)) | dynamic gradient mask + concept-aware loss + **distillation that explicitly targets persistence of earlier erasures** | method summary |
| **M-ErasureBench** (WACV 2026, arXiv [2512.22877](https://arxiv.org/pdf/2512.22877)) | benchmarks erasure across prompts, learned embeddings and inverted latents — varies the **probe modality** | abstract |

**Three claims from revision 1 are withdrawn.** Shared-anchor interference is
not unaddressed (CEASE AIC). Historical-update interference is not unaddressed
(CEASE HOC). Order sensitivity is not untested (A.5, five permutations).
Persistence of earlier erasures is not an unaddressed objective (Sculpting
Memory). **And the absence of a named relapse metric in a paper is not evidence
that the phenomenon is unstudied** — revision 1 treated it as such, which was
wrong.

### 1.1 The narrower question that is actually open

CEASE's decomposition is stated in the algebra of the **edit operator**: anchor
activation `βᵢᵀc∗` and update overlap `αᵢᵀαⱼ`. Those are properties of the
solver's updates. The screen's question is one level out, in the **structure of
the concept sequence**:

> Holding achieved deletion of the newest target **equal across arms**, is
> interference — and specifically **relapse of an earlier target** — separably
> associated with **(a) whether successive requests share a replacement anchor**
> and **(b) how semantically close successive targets are**, and does the
> benefit of an existing remedy depend on which combination you are in?

Why this is not already answered:

1. **No published experiment varies target proximity, or separates it from
   anchor sharing.** Confirmed for CEASE, which varies **order** (A.5) — a
   different factor. In the concept sets used by this literature the two
   covary: celebrity-to-celebrity and style-to-style streams hold proximity
   roughly constant and share an anchor throughout, so `αᵢᵀαⱼ` and target
   proximity cannot be told apart from the results.
2. **Constraint ablations are not an identification design.** Removing AIC or
   HOC shows a remedy helps on the evaluated stream. It does not show which
   sequence property made the stream hard, so it cannot say when the remedy is
   needed and when it is redundant.
3. **CEASE does not report whether previously erased concepts become
   generatable again.** Confirmed. Sculpting Memory *optimises* for persistence;
   neither reports a stagewise measurement of relapse against each earlier
   target's own post-erasure state. That is a measurement gap, not a claim that
   the phenomenon is unknown.

**What this screen can and cannot deliver.** It can establish whether (a) and
(b) are separably associated with relapse **on the specific concept pairs it
screens**, at matched deletion. It cannot establish a mechanism, cannot
generalise over concepts (§3), and does not make anything publishable on its
own. Those limits are stated here so they are not re-discovered after the
compute is spent.

---

## 2. Eligibility and iso-deletion are two different conditions

Amendment 01 failed because one condition was checked and the other was not.
Revision 1 then repeated a version of the error by treating "≥ 60 pp / ≤ 30%" as
iso-deletion. **It is not.** A regime gate says each arm deleted *enough*; it
says nothing about whether two arms deleted the *same*. Both are declared, both
are checked, and they are never collapsed.

### 2.1 What is matched, and against what reference

**The matching quantity is the achieved RESIDUE of the newest target, not the
suppression delta.** This is forced by the sequential setting: at request *k*,
each method's parent is **that method's own** request-(*k*−1) checkpoint, so a
"pp drop vs own parent" is measured from a different starting point for every
method and is not comparable across them. Residue — the absolute detection rate
on the newest target, on a fixed probe set — is parent-free and directly
comparable.

| condition | quantity | declared value | reference |
|---|---|---|---|
| **operating target** (iso-deletion) | residue on the newest target, `D_k(t_k)` | **20%**, matching tolerance **± 5 pp across arms** | a fixed probe set; no parent term |
| **regime eligibility** | suppression of the newest target | **≥ 60 pp** | **that method's own immediate parent** `C_{k−1}` |
| **regime eligibility** | residue on the newest target | **≤ 30%** | — |
| **context, reported never matched** | suppression vs the common base | — | **M0**, the shared pretrained model |

Eligibility uses the own-parent reference because it asks whether *this edit*
did meaningful work. Matching uses residue because it asks whether *the arms are
comparable*. An arm can pass one and fail the other, and both outcomes are
reported.

### 2.2 Calibration on development only, acceptance on a held-out set

Three sets, disjoint and hashed before anything runs. This is the direct
Amendment 01 repair: a level that holds on the selection set is not evidence it
holds anywhere else.

1. **DEV — calibration, selection only.** Each method is driven along its own
   strength parameter (ESD/ConAbl: steps and η; UCE/RECE/SPEED/CEASE: the
   solver's regularisation; LACU: its own) until its newest-target residue first
   reaches the 20% target on DEV. The dial value is then **frozen and written
   down**.
2. **ACCEPT — a separate held-out acceptance set, evaluated before TEST.** The
   frozen dial is re-measured here. An arm is **accepted** only if, on ACCEPT,
   it is inside the regime gates **and** within ± 5 pp of the operating target
   **and** within ± 5 pp of every other accepted arm. **No retuning is permitted
   at this step.** An arm that fails is reported as failing and either runs
   unmatched with that label attached (§4.4) or is withdrawn from the matched
   comparison — the PI's call, declared in advance per arm, not chosen after
   seeing ACCEPT.
3. **TEST — frozen, evaluated once**, no reselection, no reuse for calibration.
   Both gates and the cross-arm match are **re-checked on TEST** before any
   retention or relapse number is read, and a failure there is reported as a
   failure, exactly as Amendment 01 now does.

Everything in §2.1 and §2.2 — the 20% target, the ± 5 pp tolerance, the gates,
the references, the dial per method, the three set identities and the bootstrap
grouping — is **frozen in a hashed pre-registration file before any evaluation**.
Failed matching is a reported outcome. Test-driven retuning is prohibited, and
the frozen test set of the closed pilot is **spent** and is not reused.

### 2.3 What iso-deletion cannot fix

Matching residue on the newest target does **not** make the arms' histories
equal: each method arrived at `C_{k-1}` by its own path, and relapse at stage
*k* depends on that path. So an arm's relapse is always *conditional on its own
history*, and a cross-arm relapse difference is not attributable to the stage-*k*
edit alone. This is a limitation of the sequential setting, not of the matching
rule, and it is reported alongside every cross-arm comparison.

---

## 3. The cells, what each identifies, and what it does not

Request 1 is fixed for every cell: **erase `cat`**, replacement anchor `horse`.
Only request 2 varies. The two cells marked **new** change *only the anchor
assignment* relative to a cell the pilot already ran.

| cell | request 2 target | request 2 anchor | proximity to `cat` | anchor | status |
|---|---|---|---|---|---|
| **NS** | `dog` | `horse` (same as r1) | **near** | **shared** | the pilot's dog branch |
| **Nd** | `dog` | `flower` | **near** | **disjoint** | **new** |
| **FS** | `sandwich` | `horse` | **far** | **shared** | **new** |
| **Fd** | `sandwich` | `flower` | **far** | **disjoint** | the pilot's sandwich branch |

Proximity is **measured and frozen on CPU before any run**, three ways, all
reported: COCO supercategory identity (`cat`/`dog` → *animal*; `sandwich` →
*food*), CLIP text-embedding cosine between target names, and COCO
co-occurrence rate. No run starts until those values are hashed.

**What each contrast identifies:**

| contrast | holds fixed | identifies |
|---|---|---|
| **NS − Nd** | proximity = near | the **anchor-sharing** association, at near proximity |
| **FS − Fd** | proximity = far | the **anchor-sharing** association, at far proximity |
| **NS − FS** | anchor = shared | the **proximity** association, under a shared anchor |
| **Nd − Fd** | anchor = disjoint | the **proximity** association, under disjoint anchors |
| **(NS − Nd) − (FS − Fd)** | — | whether the anchor effect **depends on** proximity (interaction) |

### 3.1 This is an exploratory test on two concept pairs

**Said plainly: one near pair (`cat`→`dog`) and one far pair
(`cat`→`sandwich`).** "Proximity" is therefore **confounded with concept
identity**: anything attributed to proximity could be a property of `dog` versus
`sandwich`, of their anchor caches, of their base-model frequency, or of detector
behaviour on those categories. The 2×2 **identifies the factors within these two
pairs and nowhere else.**

**Training-seed replication does not fix this.** Two training seeds replicate the
*run*, not the *concept*; they bound run-to-run variability within a cell and
support a direction check only. They are not independent draws of "a near pair",
so no number of seeds converts this into evidence about proximity in general.
Revision 1 blurred this and the blur is withdrawn.

**What independent concept evidence would be needed before any mechanism
claim.** At minimum, pre-registered before collection: **≥ 5 near pairs and ≥ 5
far pairs**, with the members of each pair drawn from **disjoint COCO
supercategories** across pairs so concept identity varies independently of
proximity; the proximity measure **validated** against an external judgement
(human or a held-out model) rather than assumed; the same anchor-sharing
manipulation applied within every pair; and a consistent sign across pairs with
intervals that exclude the declared threshold. Until that exists, any statement
from this screen is about **these cases**, and the word "mechanism" is not
available. This screen is a decision about whether that larger study is worth
designing — nothing more.

---

## 4. Relapse, defined stagewise

### 4.1 The definition

For a method *M* and an earlier target `t_i` erased at request *i*, let `C_i^M`
be **M's own checkpoint immediately after request *i***, and
`r_i^M = D(C_i^M, t_i)` its achieved residue there. For any later stage *k* > *i*:

> **`Relapse_i^M(k) = D(C_k^M, t_i) − r_i^M`**, in percentage points.
> **Positive means the erased concept came back.**

The reference is each target's **own successful post-erasure checkpoint for that
method** — not M0, not the other method's checkpoint, not the stage-*k* parent.
That is what makes "came back" mean what it says.

### 4.2 Initial non-erasure is a different outcome and is kept separate

`Relapse` is defined **only** for stages where erasure succeeded in the first
place. At request *i*, if `C_i^M` fails the regime gates or the operating target
(§2.1), then `t_i` was **never erased** by *M*, and:

- the stage is recorded as an **erasure failure** with its achieved `r_i^M`;
- it is **excluded** from that method's relapse analysis, and the exclusion is
  reported with the count — never dropped silently;
- it is **not** reported as relapse, and `Relapse` is not computed from a
  reference that never met the target. A large `D(C_k, t_i)` after a failed
  erasure is persistence of a concept that was never removed, which is a
  statement about stage *i*, not stage *k*.

An arm with excluded stages cannot be compared on relapse against an arm without
them; that comparison is reported as unavailable.

### 4.3 Threshold, uncertainty, and the decision table

**Practical effect threshold: 10 pp of relapse**, declared now. Rationale, not
convention: it is the material size already declared for this project's
retention endpoint, and it sits well above the probe's representable step
(1.25 pp at 80 pairs, 0.18 pp at 560), so the threshold is a judgement about
consequence rather than about resolution.

**Uncertainty rule.** The existing frozen conditional paired cluster bootstrap —
scene cluster as the resampling unit, category as the stratum, 10 000 draws, 95%
percentile intervals, RNG recorded — derived from the actual manifests and hashed
before any data exist. Per training seed, **never pooled**. Decisions are read
from **interval bounds**, never from point estimates. No multiplicity correction
is applied, so every interval is **descriptive**, and an interval including zero
is not evidence of no effect. Intervals stay conditional on this base model,
these concepts, these anchors, these prompts and **this detector**.

| outcome | reading | decision |
|---|---|---|
| anchor contrasts (NS−Nd, FS−Fd) exclude **+10 pp** in the same direction, in both seeds | anchor sharing is associated with relapse **on these pairs** | a larger, concept-varied study (§3.1) is warranted; method work waits for it |
| proximity contrasts (NS−FS, Nd−Fd) exclude +10 pp, anchor contrasts do not | proximity dominates **on these pairs** | anchor allocation is not the lever here; design the concept-varied study around representation overlap |
| both sets exclude +10 pp | both factors matter on these pairs | report the interaction; neither single remedy is indicated |
| intervals **span** ±10 pp | **inconclusive** | report widths and the prompt count that would be needed; do **not** read point estimates |
| seeds disagree in direction | **unresolved** | cost more training seeds; do not report the agreeing seed |
| **a strong existing method already holds relapse within ±10 pp in all four cells, at matched deletion** | **the chosen cells are solved** | **report that and stop.** Do not build a method. The contribution would be the measurement protocol alone, and whether that is publishable is not for this document to assert |
| iso-deletion changes the method ranking relative to each method's own default operating point | published preservation comparisons are confounded **in these cells** | report it as a measurement finding, with no claim about other settings |

### 4.4 Baselines — and the closest competitor is not droppable

Of record: **ESD**, **UCE**, **RECE**, **SPEED**, **CEASE**, **LACU**, and
**Sculpting Memory**. ConAbl ± L2-SP is retained for continuity with our own
tables and is **not** a baseline of record.

**LACU is in, and must be.** Its locality-aware target selection and local replay
are precisely the directions §5 would otherwise propose, and running the screen
without it would compare against a weaker field than exists. **CEASE and LACU are
not droppable for convenience.** Specifically:

- A method **without a monotone strength dial** is **not excluded**. It is run at
  its own declared operating point, its achieved residue is reported, and every
  comparison involving it is labelled **"not iso-deletion-matched"** with the
  realised gap stated. An unmatched comparison against the closest competitor,
  clearly labelled, is more informative than its absence.
- Revision 1's stopping condition "fewer than three usable baselines with
  monotone dials → stop" is **withdrawn**: it would have licensed dropping CEASE
  or LACU to reach a count. The replacement is in §4.5.
- Reproduction difficulty is reported, not used as grounds for exclusion. If
  CEASE or LACU cannot be run on this host, that is a **blocking finding** about
  the screen, not a reason to run a weaker comparison.

A CPU feasibility audit per method, before anything runs: code availability and
licence, dependency closure against `env/requirements.lock.txt`, dependence on
the **blocked** UnlearnCanvas classifiers, and whether a monotone dial exists.

### 4.5 Stopping rule, declared in advance

Stop and report, without proceeding:

- **CEASE or LACU cannot be run on this host** — report the blockage; do not
  substitute a weaker field;
- **no method reaches the 20% operating target** on DEV — report the achieved
  ceiling and the dial values;
- **the frozen dial fails on ACCEPT for every method** — the operating point does
  not transfer; report it (this is Amendment 01's failure, caught one stage
  earlier);
- **the cross-arm match fails on TEST** — report, do not reselect, do not widen;
- **relapse intervals cannot separate the four cells** at the planned prompt
  count — report widths and the count that would be needed;
- **the two training seeds disagree in direction** on the primary relapse
  endpoint;
- **frozen-set overlap** with DEV, ACCEPT, the pilot set or the anchor-training
  strings;
- **any artifact fails `scripts/seq/validate_stage.py`**.

A stop is a **result**.

### 4.6 Measurement validity — binding, unchanged

The detector is a **proxy**. UnlearnCanvas classifiers remain **unobtainable on
this host**, so no UA / IRA / CRA. **No human annotation has been performed
anywhere in this project**; the 180-item paired packet must be labelled before
any screen result is called a finding, and the screen must add its own paired
packet on the terms in §6.

---

## 5. Candidate method directions — stated with their overlaps

**No method is proposed for approval here.** Revision 1's anchor-allocation
sketch is withdrawn as a novelty claim: LACU already selects mapping targets by
score-prediction distance and replays the local neighbourhood, so "choose the
anchor by representational distance" is **occupied territory** — and LACU chooses
the *nearest* safe target, so even the opposite allocation rule is a variation on
a published mechanism rather than a distinct one.

What is *not* occupied, as far as §1's reading goes, is the **object being held
invariant**:

> CEASE's AIC appends the **shared replacement token** `c∗` to the invariance
> matrix: the anchor must not move. A candidate direction appends, instead, **each
> earlier target's achieved post-erasure state** as its own invariance row, so the
> solved constraint is *"target `t_i` stays where its erasure left it"* —
> `Relapse_i(k) ≤ δ` as a constraint on the edit, with a checkable certificate,
> rather than as a distillation objective (Sculpting Memory) or as orthogonality
> to past update directions (CEASE HOC), both of which are sufficient-not-
> necessary proxies for it.

**Honest caveats on that direction, which the screen is meant to test, not
assume:** it reuses CEASE's machinery and may reduce to HOC in some limit; it
requires each earlier target's post-erasure state to be representable as an
invariance row, which is unverified; and its value depends entirely on §4.3
*not* returning the "already solved" row. **It is a direction, not a
contribution, and its technical distinctness is a claim to be tested against
CEASE and LACU under matched deletion — not asserted.**

If the screen returns "already solved in these cells", the correct conclusion is
that this line does not support a method paper, and that should be said rather
than worked around.

---

## 6. Not in this proposal

- **Any GPU work.** Nothing here is authorised; the costing is CPU arithmetic.
- Any repair, re-selection or extension of Amendment 01, which is closed, or any
  reuse of its spent frozen test set.
- Any determinism or repeatability study.
- Any claim of generality over concepts (§3.1), any mechanism claim, and any
  assertion about publishability.
- Human annotation labour, which must be **scheduled**, not assumed.

## 7. CPU deliverables, if this direction is approved

1. the hashed **pre-registration** of §2 — operating target, tolerance,
   references, per-method dial, the three set identities, the bootstrap
   grouping, and the per-arm advance decision for an ACCEPT failure;
2. frozen manifests for the four cells, with the three frozen proximity
   measurements and all digests;
3. the §4.4 feasibility audit, per method, with CEASE and LACU reported
   first and blockages reported as blockages;
4. the relapse estimator and its frozen grouping, with a CPU regression that
   **fails the build** if a relapse number is computed from an excluded stage
   (§4.2) or read where the match or gates were not met (the Amendment 01
   defect, now a test);
5. a measured cost model per cell from this host's existing per-step and
   per-image measurements, with wall time kept separate from device-hours;
6. the screen's paired human-audit packet design, on the terms verified in
   [`annotation_packet_paired180/BLINDING_AUDIT.md`](../../results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md)
   — secret salt, secret row order, and an attack run against the published
   records before the packet is handed to anyone.

---

### Sources

- CEASE — [arXiv 2610.01989](https://arxiv.org/html/2610.01989) (§3.2 noise decomposition; §3.3 AIC/HOC; A.5 five random orders)
- LACU — [arXiv 2512.02657](https://arxiv.org/abs/2512.02657) (locality-aware target selection; local replay; 10 sequential steps)
- Sculpting Memory — [arXiv 2504.09039](https://arxiv.org/html/2504.09039v1) (persistence of earlier erasures)
- M-ErasureBench — [arXiv 2512.22877](https://arxiv.org/pdf/2512.22877) (probe-modality benchmark)
- Side Effects of Erasing Concepts — [arXiv 2508.15124](https://arxiv.org/pdf/2508.15124)
- When Are Concepts Erased From Diffusion Models? — [arXiv 2505.17013](https://arxiv.org/pdf/2505.17013)
