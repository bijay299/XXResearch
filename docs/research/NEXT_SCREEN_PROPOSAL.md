# Next screen — what drives interference in sequential concept erasure

**CPU-only proposal. This document authorises nothing and no GPU experiment is
authorised by the Amendment 01 closeout.** Submitted for the PI's decision. If
approved, any compute must run on GPUs **independently verified unoccupied by
another user** ([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md));
the hour ceiling stays retired and caps nothing.

The point of this screen is to **decide whether a method is warranted and which
one** — not to repair or extend the Amendment 01 pilot, which is closed
([`AMENDMENT_01_CLOSEOUT.md`](../../results/audit_v1/AMENDMENT_01_CLOSEOUT.md)).

---

## 1. Closest prior work, checked specifically

Not a fresh survey — the four papers that bound the gap.

| work | what it does | what it leaves open |
|---|---|---|
| **CEASE**, *Continual Concept Erasure by Suppressing Cross-Edit Interference* (arXiv 2610.01989) | **the closest.** 100 sequential edits, SD 1.4/1.5/2.1, celebrities + styles + instances. Two training-free subspace constraints on a closed-form solver: anchor invariance, and orthogonality to the SVD of cumulative past updates. Baselines ESD, ConAbl, UCE, RECE, SPEED, CUIG-GradProj, Sculpting Memory | **does not equalise deletion strength** when comparing preservation — deeper-erasing baselines corrupt the retain set, so preservation comparisons are confounded; **relapse of previously erased targets is not a separate metric**; constrains *update directions*, never *behaviour on earlier targets*; sequences are long but their **structure is not varied** |
| **Sculpting Memory** (arXiv 2504.09039) | dynamic gradient mask + concept-aware loss + distillation **to keep previously unlearned concepts forgotten** | behaviour-level relapse control already exists — a bare "keep old targets suppressed" loss is **not** a novel contribution |
| **LACU**, *Locality-Aware Continual Unlearning* (arXiv 2512.02657) | grounds mapping-target selection and replay in score-prediction distance | selects the *nearest safe* target per request; does not ask whether interference is driven by that choice or by target proximity |
| **M-ErasureBench** (WACV 2026, arXiv 2512.22877) | benchmarks erasure across prompts, learned embeddings, inverted latents | varies the **probe modality**, not the **sequence structure** |

**Sequential-erasure interference is therefore not an open problem, and we should
stop describing it as one.** Three things about it still are.

## 2. The gap, stated narrowly

**(G1) Nobody has separated the two mechanisms that could cause interference.**
A later request can damage an earlier one because the two requests **share a
replacement anchor**, or because their **targets are semantically close** — two
different causes with two different fixes. In our own pilot they are *perfectly*
confounded: the dog branch shares `MA`'s `horse` anchor **and** is semantically
near `cat`; the sandwich branch uses `flower` **and** is far. The standing
confound in `MATCHED_EFFECTIVENESS_PROTOCOL.md` §12 is the same one, and CEASE's
anchor-invariance and historical-orthogonality constraints are *both* active at
once, so its ablations cannot separate them either.

**(G2) Preservation is compared at unequal deletion strength.** CEASE says so
about its own baselines. Our pilot's whole AUDIT-01 finding was this confound,
and Amendment 01 — the attempt to fix it — failed because the gates were not
verified on the set that was reported. The repair is cheap and nobody applies
it: drive every method to a **declared deletion level**, and **verify that level
on the set the retention numbers come from**.

**(G3) Relapse is not a declared endpoint.** "Did the earlier target come back?"
is reported, when at all, as degraded retain-set quality. It is a distinct
outcome from retention of never-targeted concepts and must not be averaged with
it.

**The gap is the conjunction**: *how* interference depends on sequence structure,
measured at matched deletion verified where it is reported, with relapse as its
own endpoint. None of the four works above answers that.

## 3. The screen

Each cell is a *sequence*: `M0 → request 1 → request 2 (→ 3 …)`, one erasure
method throughout.

### 3.1 Deliberately varied sequences

**Factor A — anchor sharing.** Request 2 reuses request 1's replacement anchor,
or uses a disjoint one.
**Factor B — target proximity.** Request 2's target is semantically near
request 1's target, or far. Proximity is **measured and frozen before any run**
(CLIP text-embedding cosine plus the two targets' COCO co-occurrence rate),
recorded with the manifests, not asserted.

|  | anchor **shared** | anchor **disjoint** |
|---|---|---|
| target **near** | the pilot's dog branch | **never run** |
| target **far** | **never run** | the pilot's sandwich branch |

The two empty cells are the screen. They are the only way G1 can be answered,
and they cost no more than the two cells we already have.

**Factor C — length**: 2 requests (the pilot's depth) and **5**. Long enough for
accumulation; short enough that every intermediate state can be evaluated.
**Factor D — order**: the same request *set* in two permutations, so "request 2
damaged request 1" is separated from "these two concepts interact".

Minimum informative design: **2×2 at length 2**, plus **one length-5 chain per
anchor condition**, plus **one order swap**, on **two training seeds**, analysed
separately and never pooled.

### 3.2 Strong existing baselines

Not our pilot's arms. **ESD**, **UCE**, **RECE**, **SPEED** (the closed-form
line), and **CEASE** as the continual-specific state of the art, with
**ConAbl ± L2-SP** retained only for continuity with our own tables. L2-SP is
**not** a baseline of record — it is the thing AUDIT-01 could not interpret.

CPU-checkable before anything runs: code availability and licence, dependency
closure against `env/requirements.lock.txt`, whether each method needs the
**blocked** UnlearnCanvas classifiers, and whether each exposes a **monotone
deletion dial** (§3.3). A method without a usable dial is reported as such, not
run at a vendor default and compared anyway.

### 3.3 Meaningful target deletion, verified where it is reported

This is Amendment 01's lesson, written into the design:

1. **Declare the level first.** The newest target must reach **≥ 60 pp
   suppression** against that sequence's own immediate parent, with **≤ 30%
   residue**. Amendment 01's ≥ 30 pp / ≤ 60% gates produced a regime so weak
   that the comparison was not worth making; 60/30 is chosen because the
   original pilot's unregularised arm reached +82.5 pp, so the level is known to
   be reachable on this stack.
2. **Dial, don't fix the budget.** Each method is driven along its own strength
   parameter (ESD/ConAbl: steps and η; UCE/RECE/SPEED/CEASE: the solver's
   regularisation) until it first meets the declared level on the **development**
   probe. Selection happens **only** there.
3. **Verify on the reported set.** Before any retention or relapse number is
   read, re-check **both gates, for every arm, on the frozen test set**, and the
   cross-method agreement within a declared tolerance. Where that fails, the
   comparison is **rejected and reported as rejected** — exactly the rule that
   closed Amendment 01, now applied before rather than after the write-up.
4. **One frozen test set, evaluated once**, with no reselection. Development,
   test and pilot prompt sets stay disjoint and hashed, as now.

### 3.4 Endpoints — reported separately, never averaged

| endpoint | measured as |
|---|---|
| **newest-target deletion** | suppression vs that sequence's immediate parent — the eligibility check, not a result |
| **previous-target relapse** | `D(target_1)` at step *k* **minus** `D(target_1)` immediately after request 1. **Positive = the erased concept came back.** The G3 endpoint, and the primary one |
| **retained concepts** | each never-targeted category separately (horse, bird, chair, bicycle) vs the sequence's own parent |
| **non-target control** | a category no request in the sequence touches |
| **anchor integrity** | can the replacement anchor still be generated at all — a failure mode anchor reuse should produce if G1's anchor branch is right |

Uncertainty: the existing **conditional paired cluster bootstrap**, scene cluster
as the unit, category as the stratum, 10,000 draws, 95% percentile intervals,
frozen from the manifests before any data exist. Intervals stay **conditional**
on this base model, these concepts, these anchors, these prompts and **this
detector**. They are not broad inference.

**Measurement validity, unchanged and binding.** The detector is a **proxy**;
UnlearnCanvas classifiers remain **unobtainable on this host**, so no UA / IRA /
CRA. **No human annotation has been performed anywhere in this project.** The
180-item paired packet must be labelled before any screen result is called a
finding, and the screen must add its own paired packet on the same terms.

### 3.5 What each comparison decides

| comparison | if it comes out this way | the decision |
|---|---|---|
| relapse: anchor **shared** vs **disjoint**, proximity held fixed | relapse materially larger when the anchor is shared | **G1 answered: anchor-driven.** Build the **anchor-allocation** method (§4) |
| relapse: target **near** vs **far**, anchor held fixed | relapse materially larger when targets are near, anchor sharing irrelevant | **G1 answered: proximity-driven.** Anchor allocation cannot help; the method must act on the shared representation, and this screen says which representation |
| both factors material, or they interact | neither fix suffices alone | **report the interaction**; a method must address both, and the design of §3.1 is what makes that statement possible |
| neither material at length 2 but relapse accumulates at length 5 | interference is a depth effect, not a pairwise one | **the unit of analysis is the stream**, not the edit pair — a different method shape |
| iso-deletion **changes the method ranking** vs the published fixed-budget ranking | published preservation comparisons are confounded in practice, not just in principle | **G2 is a publishable result on its own**, independent of any new method |
| CEASE already removes relapse across all four cells at matched deletion | the problem is solved | **say so and stop.** Do not build a method; the contribution would be the protocol only |

The last row matters: this screen is designed so that *"CEASE already handles
this"* is a reportable outcome, not a failure of the screen.

### 3.6 Stopping rule, declared in advance

Stop and report, without proceeding:

- **any arm misses the declared deletion level on the reported set** — the
  Amendment 01 rule, applied up front;
- **no method reaches the declared level at all** on this stack — report the
  achieved ceiling and the dial values; propose nothing further;
- **relapse intervals are too wide to separate the 2×2 cells** at the planned
  prompt count — report the width and what count would be needed; do **not**
  read point estimates;
- **the two training seeds disagree in direction** on the primary relapse
  endpoint — **unresolved**; cost more seeds, do not report the agreeing one;
- **frozen test overlap** with development, pilot or anchor-training strings;
- **any artifact fails `scripts/seq/validate_stage.py`**;
- the screen's **CPU feasibility audit** finds fewer than **three** usable
  baselines with monotone deletion dials — a three-method comparison is the
  minimum worth the compute.

A stop is a **result**: "interference in this setting cannot be attributed to
anchor sharing or proximity at this precision" is more useful than an
attribution produced anyway.

## 4. The candidate method — and why the screen comes first

**If anchor-driven (G1, anchor branch):** a **per-request anchor allocator** that
chooses or synthesises a replacement target provably separated from every prior
request's anchor subspace, with the deletion level driven to the calibrated
target rather than a fixed budget. Distinct from CEASE, which constrains the
*update* to be orthogonal to past update directions but still reuses a shared
replacement token; and from LACU, which picks the *nearest* safe target — the
opposite allocation rule. The separation would be a declared, measurable
property of the method, not a loss term that is hoped to act.

**If proximity-driven (G1, proximity branch):** anchor allocation is useless and
the method must separate overlapping target representations before erasing. That
is a different construction, and proposing it now without the screen would be
guessing.

**Either way, the protocol contribution stands**: iso-deletion verified on the
reported set, plus relapse as a declared endpoint, is a reusable evaluation
result (G2 + G3) that does not depend on which branch wins.

**We are not proposing to build a method yet.** The pilot has already produced
one method claim that did not survive its own audit, and one matched-design
diagnostic that did not reach its authorised regime. The screen is cheap, it is
decisive, and committing to a mechanism before it is what produced both.

## 5. Not in this proposal

- Any GPU work. **Nothing here is authorised**; the costing below is CPU
  arithmetic, and the next deliverable is CPU artifacts, not runs.
- Any repair, re-selection, re-tolerance or extension of Amendment 01. It is
  closed.
- Any determinism or repeatability study. Not requested, not proposed.
- New prompt-set generation before the frozen manifests are reviewed.
- Any claim about strongly-suppressed regimes beyond the declared 60/30 level.
- Human annotation labour, which must be **scheduled**, not assumed.

## 6. The CPU deliverables, if this direction is approved

Produced on CPU, reviewed, and only then costed for GPUs verified idle:

1. frozen sequence manifests for the 2×2, both length-5 chains and the order
   swap, with the proximity measurements and all digests;
2. the baseline feasibility audit of §3.2 — per method: licence, dependency
   closure, blocked-asset dependence, and whether a monotone deletion dial
   exists;
3. the calibration protocol as runnable code, with the on-reported-set gate
   check wired in from the start and a CPU regression that fails the build if a
   comparison is read where the gates were not met;
4. the frozen bootstrap grouping for the new endpoints, including relapse;
5. a measured cost model per cell, derived from the existing per-step and
   per-image measurements on this host, with wall time kept separate from
   device-hours;
6. the paired human-audit packet design for the screen, on the same terms as
   [`annotation_packet_paired180/`](../../results/audit_v1/annotation_packet_paired180/).

---

### Sources

- [Continual Concept Erasure in Diffusion Models by Suppressing Cross-Edit Interference (CEASE)](https://arxiv.org/html/2610.01989)
- [Locality-Aware Continual Unlearning for Diffusion Models (LACU)](https://arxiv.org/pdf/2512.02657)
- [Sculpting Memory: Multi-Concept Forgetting in Diffusion Models via Dynamic Mask and Concept-Aware Optimization](https://arxiv.org/html/2504.09039v1)
- [M-ErasureBench: A Comprehensive Multimodal Evaluation Benchmark for Concept Erasure in Diffusion Models](https://arxiv.org/pdf/2512.22877)
- [Side Effects of Erasing Concepts from Diffusion Models](https://arxiv.org/pdf/2508.15124)
- [When Are Concepts Erased From Diffusion Models?](https://arxiv.org/pdf/2505.17013)
