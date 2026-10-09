# MATCHED_EFFECTIVENESS_PROTOCOL — bounded diagnostic, for review

**Status: a proposal. Nothing here has been run.** No GPU work was launched in
preparing it. The GPU-hour figure and the 4-hour ceiling are **proposed for
review, not approved**. The PI decides; the main research chat reviews.

**Revision.** This replaces the first version (commit `f7f4ba8`, preserved at
[`results/audit_v1/preserved_originals/v1/`](../../results/audit_v1/preserved_originals/v1/MATCHED_EFFECTIVENESS_PROTOCOL.md)).
Changes: the run matrix is the bounded one specified for review (full fixed
scan, no bisection; no lower-L2 stage; no sandwich coefficient search), and
seven overstatements in the v1 text are corrected — see
[`CLAIM_CORRECTIONS_V2.md`](../../results/audit_v1/CLAIM_CORRECTIONS_V2.md).

Evidence base: [`results/audit_v1/AUDIT_REPORT.md`](../../results/audit_v1/AUDIT_REPORT.md).

---

## 1. The question

The pilot compared L2-SP 25000 against no regularisation at a **fixed 1000
steps**, so the two arms deleted different amounts of the newest target. L2-SP
achieved +42.5 pp of dog suppression where the unregularised arm achieved
+82.5 pp (both seeds), and retained ~35–40 pp more bird. Those two facts cannot
be combined into a preservation claim: the arm that preserved more also deleted
much less.

> **The question.** At a *matched* level of dog deletion, does L2-SP retain more
> bird than simply making a smaller unregularised update?

**This is a diagnostic of our own pilot explanation, not a method proposal.** It
asks whether the pilot's "L2-SP protects retention" reading survives equalising
deletion strength. It is explicitly a **partial-suppression** test: the matched
level will be around 40–50 pp of dog suppression with substantial dog residue,
not a strongly-suppressed regime, and nothing about strong suppression follows
from it.

## 2. What upstream already does — cited before any distinction is claimed

**No novelty is claimed.** The brief states that appendices D and E of the
accepted paper report early-stopping controls; that is taken as given here. The
paper was **not read** for this document, so what follows is strictly what the
pinned code (`9932ac3271a122f6d38d19e0b8c8908fe5237ff7`) shows, reported
separately from the appendix claim.

**The mechanism.** `Regularizers/Simultaneous/simultaneous.py:check_early_stopping`,
called from `UnlearningMethods/ConAbl/train_conabl.py:309–364`, gated on
`--eval_interval` (with `--patience`, default 2000, and `--stop_threshold`,
default 99.0, in `ConAbl/src/args.py:10–12`).

**It is threshold-or-patience stopping.** It stops when `ua >= stop_threshold`
**or** when `no_improvement_count >= patience`. It tracks `best_ua`, but the
weights saved at stop time are the **current** ones, not the best-scoring ones.
So it does **not** certify a per-arm maximum, and the v1 text calling it one was
wrong.

**The scored concept pool comes from the caller, not the helper.**
`train_conabl.py` passes `args.target_concepts` (the CLI arg is
`--anchor_target_concepts`) together with `--eval_classifier_dir`. The helper
itself says nothing about which concepts are scored, so it is **no evidence for
newest-target-only scoring**. In upstream's sequential object script the caller
passes a single `"${anchor_name}+${object_name}"` mapping; scoring uses an
UnlearnCanvas classifier, not a COCO detector.

**It is not wired into the sequential setting.** `--eval_interval` is passed in
exactly six `BashScripts/Simultaneous/**` scripts and in **no**
`BashScripts/Sequential/**` script. Our own pilot runs recorded
`eval_interval: null`.

**The distinction, stated narrowly.** Upstream's rule stops each arm by its own
convergence criterion. This protocol stops the unregularised arm at a
**prescribed level matched to the L2 arm's achieved level**, so retention is
compared between arms of comparable deletion strength. That is a
measurement-protocol choice — the kind of control an evaluation needs, not a
contribution. Also not novel and not claimed: regularisation for concept
editing, semantic-preservation metrics, and sequential-request evaluation.
**No new regularizer and no learned controller is built.**

## 3. Design

### Reused without new compute

- **MA17 and MA29** — the existing seed-specific parents.
- **MAB_L2 at L2=25000** for both seeds — the existing final L2 children
  (`8ee7d56f62b0`, `c4f3ecdcb945`).

### Explicitly excluded

- Any lower-L2 or alternative-coefficient stage.
- Any sandwich-branch coefficient search.
- A new M0 evaluation on the frozen test set.
- An unmatched full-step (MAB) reference arm on the frozen test set.

### New compute: two unregularised dog trajectories

| | |
|---|---|
| trajectories | 2 (one per training seed, parent = that seed's MA) |
| steps | 1000, unregularised, configuration otherwise matched to the pilot |
| dumps | steps 100, 200, … 1000 — **saved and scored, full fixed scan** |
| selection | **no bisection**; every dump is scored on the development set |

A full fixed scan is used rather than bisection because bisection assumes
suppression is monotone in steps, which is untested; scanning all ten costs
little more here and makes non-monotonicity visible instead of fatal.

**Training configuration matched to the pilot**: kv-xattn (32 tensors,
19,169,280 params), effective LR 8e-6, AdamW, fp32, constant schedule with 500
warmup, `horse+dog`, 1000 iterations, same anchor caches byte-identically. The
only intended differences from the existing `MAB` are the periodic dumps and
`l2sp_weight 0`. **Evaluation RNG is kept separate from training RNG**:
generation seeds are an evaluation-side quantity and are never reused as
training seeds (see §4).

### Why the sandwich branch is excluded

`MAC_L2` achieves **+7.5 pp** of sandwich suppression in both seeds
(`[+0.0, +17.5]` s17, `[+0.0, +15.0]` s29). **L2-SP suppression there is too
weak for this question**: a matched comparison needs a meaningful deletion level
to match *to*, and +7.5 pp does not supply one. Matching an unregularised arm
down to it would compare two arms that both barely deleted, and any retention
similarity would be an artifact of both doing nothing.

That is a statement about fitness for the question, **not** a claim that 7.5 pp
is unmeasurable. With 10 prompt clusters of 4 images the representable step is
2.5 pp, so 7.5 pp is a real three-step value; separately, its interval reaches
zero, so it is imprecise. (The v1 text called it a "resolution floor"; that was
wrong.)

## 4. Prompt sets — drafted, hashed, and frozen before any launch

Three sets with distinct roles. The pilot's set **has informed the hypotheses in
this document** and is exploratory from here on: it may be used for continuity
with published tables, but **no configuration may be selected on it and no
confirmatory claim may be made from it**.

| set | distinct prompt texts | gen seeds | pairs / checkpoint | role |
|---|---|---|---|---|
| **DEV** (drafted) | **20** (dog only, 10 literal + 10 paraphrase) | 511, 622, 733, 844 | **80** | selection only |
| **TEST** (drafted, frozen) | **140** (20 per category × 7) | 1301, 1402, 1503, 1604 | **560** | the single confirmatory evaluation |
| **PILOT** (existing) | 70 (10 per category × 7) | 101, 202, 303, 404 | 280 | descriptive continuity only |

**Counting, stated explicitly** because the pilot's reports conflated it: these
are *distinct prompt texts* × *generation seeds* = *prompt × generation-seed
pairs*. The pilot set is **70 prompt texts**, not 280; 280 is its pair count.

The development set is **dog only** by design. Cat and the retained categories
are endpoints of the confirmatory test; putting them in the development set
would create an opportunity to select on them.

**Drafted and hashed on CPU** —
[`results/audit_v1/draft_manifests/`](../../results/audit_v1/draft_manifests/):

| manifest | sha256 |
|---|---|
| `dev_manifest_DRAFT.json` | `ebfdd37270dea750…` |
| `test_manifest_DRAFT.json` | `170696eb5fc2832b…` |

**Disjointness verified** across all five pairings (dev/test, dev/pilot,
test/pilot, dev/anchor-training, test/anchor-training): **0 exact collisions,
0 near-duplicates at similarity ≥ 0.90**, against 400 upstream anchor training
strings. Generation-seed sets are pairwise disjoint across pilot/dev/test, and
no generation seed equals a training seed (17, 29).

**Shared surface structure is measured and reported, not denied.** Both sets are
template-built. Dev and test share **0** scene tails, but where any prompt texts
share a template the bootstrap's prompt clusters are not fully independent and
intervals are mildly optimistic. That limitation applies to the pilot set too;
see `disjointness_report.json`.

The test manifest is **frozen and hashed before the development stage begins**,
so selection cannot reach it.

## 5. Selection, gates, and the infeasibility rule

All on the **development set**, against that seed's **own MA**, at t=0.5.

**Reference gates.** A dump qualifies only with **≥ 30 pp dog suppression** and
**≤ 60% dog residue**. These gates define an explicitly **partial-suppression**
comparison — they are the regime the pilot actually explored, and they are what
makes this a test of the pilot's explanation. No claim about a
strongly-suppressed regime follows.

**Match rule.** For each seed, select the dump whose dog suppression is closest
to that seed's L2 endpoint, **earliest step breaking ties**; the mismatch must
be **≤ 5 pp**. Earliest-step tie-break is conservative against the hypothesis
that L2-SP is unnecessary, since it favours the smaller update.

**Infeasibility rule — declared in advance.** If **either seed** has no
qualifying dump within 5 pp, **the planned paired test stops**. Report the
bracketing steps, their suppression values and the gap. Do not widen the
tolerance, do not fall back to one seed, and do not proceed with an unmatched
comparison.

Also stop and report if suppression is **non-monotone in steps** in a way that
makes "the matched step" ambiguous — two separated dumps both inside the band
with materially different bird retention. Report both.

> Declaring infeasibility is a **result**. "These two arms cannot be compared
> fairly at this coefficient" is more useful than a comparison run anyway.

**Reporting imperfect matching on the test set without reselecting.** The match
is fixed on DEV. On TEST the realised dog suppression difference will not be
exactly zero. Re-check the same gates and the same match on TEST and report the
achieved values beside every retention contrast. **Nothing is reselected using
TEST.** If the match transfers poorly, that is reported as a finding about
transfer; any re-selection would require a new frozen test set.

## 6. Endpoints and uncertainty

**Primary.** The **paired bird contrast** `D_bird(L2) − D_bird(U*)` on TEST at
t=0.5, where `U*` is the selected matched unregularised dump.

**Reported alongside, never averaged into it:**

- dog residue and dog suppression for both arms (the matching check);
- cat change from MA, for both arms, referenced to that seed's own MA;
- **every retained category separately** — horse, bird, chair, bicycle;
- the sandwich category, reported separately as a non-target;
- literal vs paraphrase for each endpoint, with residual presence and
  parent-referenced change kept distinct;
- thresholds 0.3 / 0.5 / 0.7, with 0.5 primary.

**Uncertainty.** Conditional paired prompt-cluster bootstrap: resample **prompt
texts** carrying their four generation seeds together; one resampled list indexes
both arms; 10 000 draws; 95% percentile intervals. **Training seeds are analysed
separately and never pooled.** These intervals are conditional on this base
model, these concepts, these anchors, these prompts and this detector — they are
**not** broad inference, and two training runs support a **direction check
only**.

## 7. Decision rule — a prespecified material effect size

**Material bird advantage: 10 percentage points.** Declared now, not afterwards.

| outcome on TEST, both seeds | reading |
|---|---|
| **upper 95% bound < +10** in both seeds | A bird benefit of 10 pp or more is **ruled out, conditionally** on this design. The pilot's "L2-SP protects retention" reading does not survive matching at a materially useful size. |
| **lower 95% bound > +10** in both seeds | A **material** bird benefit carries forward: at matched dog deletion, L2-SP retains more bird by at least the declared size. |
| intervals **wholly within [−10, +10]** in both seeds | **Practical equivalence** at the declared margin — and only then. |
| anything else | **Inconclusive.** Report as inconclusive; do not read a point estimate as a result. |
| directions **disagree** between seeds | **Unresolved.** Cost more training seeds; do **not** report the agreeing seed. |
| no match within 5 pp (either seed) | **The comparison is rejected, not the hypothesis.** Report the bracketing dumps. |

**Three things no outcome here establishes.**

1. **Lack of significance is not equivalence.** An interval that merely includes
   zero does not show the arms retain equally. Only the "wholly within
   [−10, +10]" row supports an equivalence statement, and only at that margin.
2. **A matched benefit would not prove a geometric mechanism.** A bird advantage
   at matched dog deletion is consistent with L2-SP sparing particular
   parameters, but equally with differences in update direction, effective step
   schedule, or optimisation path. This design does not distinguish them, and
   "where the update is constrained matters" is **not** a conclusion available
   from it. Mechanism is a separate follow-up.
3. **It says nothing about the cat history.** The newest-target quantities are
   about dog. Cat change is reported as a secondary endpoint, referenced to each
   seed's own MA, and is not what the primary endpoint measures.

## 8. Resource accounting

Measured on this host from the completed pilot runs — not assumed:

| quantity | measured | n |
|---|---|---|
| per optimizer step | **0.8303 s** | 10 runs |
| per generated image | **1.0083 s** | 12 reports |
| per detected image | **0.0718 s** | 10 reports |

| stage | runs | images | GPU-h |
|---|---|---|---|
| Training: 2 trajectories × 1000 steps | 2 | — | **0.461** |
| Generation + detection, all stages | — | **5280** | **1.584** |
| └ development: 24 ckpts × 80 | 24 | 1920 | |
| └ frozen test: 6 ckpts × 560 | 6 | 3360 | |
| Process setup: 62 invocations × 25 s | — | — | **0.431** |
| **subtotal** | | **5280** | **2.476** |
| **with 25% contingency** | | | **3.095** |

Development checkpoints = 2 seeds × (MA + L2 endpoint + 10 dumps) = **24**.
Frozen-test checkpoints = 2 seeds × (MA, L2, selected-U) = **6**.
Invocations = 2 training + 30 generation + 30 detection = **62**.

> **Disclosure on the setup term.** The 0.431 GPU-h may be **partly
> double-counted**. The measured `s_per_generated_image` is elapsed ÷ images from
> the pilot reports, so it already amortises that run's pipeline construction and
> checkpoint load across its 280 images. Charging a further 25 s per invocation
> therefore counts some setup twice. It is kept as a deliberate **upper bound**
> because the new runs include many *small* evaluations (80 images) where
> per-invocation setup is a much larger share than in the pilot's 280-image runs.
> **Reconcile against the actual implementation before relying on the figure**:
> if the 24 development checkpoints are scored within one process, most of this
> term disappears and the subtotal falls toward **2.045 GPU-h**.

**Wall time is separate and is not GPU-hours**: ≈1.45 h with the two
trajectories in parallel and scoring split across two idle GPUs. The four A100s
are shared without a scheduler, availability is not guaranteed, and only
independently verified idle devices would be used.

**Storage.**

| item | size |
|---|---|
| 20 intermediate dumps × 76,686,190 bytes | **1.43 GiB** |
| 5280 images at ~98 KB | 0.49 GiB |
| **peak total** | **1.92 GiB** |
| after pruning unselected dumps | 0.63 GiB |

Excludes optimizer/RNG state (the current trainer saves none) and logs.
`/data` has 1.4 TB free.

**Tuning budget.** L2 receives **zero** tuning: 25000 is taken as given, as the
pilot ran it. The unregularised arm receives one selection over 10 pre-saved
dumps on DEV. Neither arm is tuned on TEST. The asymmetry is recorded rather
than hidden, and it **favours the unregularised arm**, i.e. it is conservative
against L2-SP.

Full arithmetic, re-derivable:
[`results/audit_v1/protocol_cost_model.json`](../../results/audit_v1/protocol_cost_model.json),
`python3 scripts/seq/protocol_cost_model.py`.

## 9. Reproducibility check — tensor-level, prepared on CPU

Training reproducibility has **never** been tested on this host. The new
unregularised trajectory re-runs the same configuration as the existing `MAB`,
so its final dump can be compared against `63c9a8232cfd` (s17) /
`d9f28f0fc6b8` (s29) at no extra GPU cost.

The comparison is **tensor-level**, not hash-level:
`scripts/seq/compare_checkpoints.py` reports, per tensor, max absolute
difference, mean absolute difference and relative Frobenius difference, on CPU.

**Three distinct claims, not to be conflated.**

1. **File hashes equal** — byte-identical serialisation. A mismatch can come
   from serialisation order or metadata alone and is **not** evidence of a
   training difference.
2. **Tensors numerically equal** — the weights match to a stated tolerance.
3. **Training equivalent** — a much stronger claim that neither of the above
   establishes on its own, and that behavioural comparison would be needed for.

The check is recorded as a **secondary observation**. A mismatch does not
invalidate the diagnostic — each arm is evaluated as itself — but it would tell
us bit-exact training reproduction does not hold here, which matters for
everything downstream.

## 10. Stop rule

Stop and report, without proceeding, if:

- the infeasibility rule (§5) fires for either seed;
- a trajectory fails the reference gates by step 1000;
- suppression is non-monotone in a way that makes the matched step ambiguous;
- the frozen test set is found to overlap DEV, PILOT or the training strings;
- any artifact fails `scripts/seq/validate_stage.py`;
- the TEST-set dog mismatch materially exceeds the DEV match — report
  confounding, do **not** reselect.

### What would weaken, support or reject the current interpretation

- **Weakens it.** Upper bound < +10 on bird in both seeds. L2-SP's apparent
  preservation benefit would not survive matching at a materially useful size,
  and "L2-SP protects retention" should be withdrawn as a method claim. Note
  this is *ruling out a 10 pp benefit*, not demonstrating equality.
- **Supports it.** Lower bound > +10 on bird in both seeds, with the dog match
  holding on TEST. A material benefit at matched deletion — with mechanism still
  open (§7).
- **Rejects the broader framing.** If matched unregularised early stopping
  reproduces L2-SP's bird retention within ±10 pp in both seeds, the pilot's
  "L2-SP is a tradeoff" framing reduces to "smaller updates change less", and
  sequential-suppression interference is not the limitation this line of work has
  been looking for. That outcome should be reported as clearly as a positive one.

## 11. Not in this request

- Sandwich-branch coefficient search, and any lower-L2 stage.
- A new M0 evaluation, and an unmatched full-step reference arm on TEST.
- More training seeds for population variance.
- Separating anchor sharing from semantic distance — the standing confound.
- Longer request streams; mechanism attribution; human annotation labour.
- Any new regularizer or learned controller. **Out of scope.**

---

### Pre-commitments

Fixed before any data are collected: the reference gates (≥30 pp suppression
**and** ≤60% residue); the ≤5 pp match tolerance with earliest-step tie-break;
the 10 pp material effect size and the [−10, +10] equivalence margin; the
primary endpoint as the paired bird contrast with every other category reported
separately and never averaged; both manifest hashes; selection on DEV only; one
evaluation of TEST with no reselection; and every uncertainty caveat in §6.

**This document authorises nothing.** Submitted for central review and the PI's
decision.
