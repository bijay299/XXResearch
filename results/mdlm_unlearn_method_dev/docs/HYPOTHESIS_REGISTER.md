# Hypothesis register

Coordinator-authored, written **before** wave-1 deliverables land, so that the formalization
and criticism agents attack stated positions rather than ratify whatever emerges. Each
hypothesis must receive a serious objection and a substantive response before it is
implemented. Status is updated as evidence arrives; nothing here is a result.

Grounding evidence, all from our own closed studies (immutable, see `../RECORD.md`):

- **E1** The exclusion-target / anchor-KL control reaches **0.049 disclosure, 1.000 retain,
  0 degenerate** on 128 synthetic facts at 300 steps. Single-shot forgetting is close to
  saturated; a method claiming "more forgetting" on this axis has almost no headroom.
- **E2** MDU (τ=1) and EMDU reach apparent **0.000 disclosure by emitting nothing** —
  empty on 15/16 and 16/16 forget probes. Their advantage on disclosure is degeneracy.
- **E3** An unlearned model at 0.049 disclosure returns to **0.605** on 43 held-out forget
  facts after **10 fine-tuning steps on 8 leaked facts** (never-memorized floor 0.023).
- **E4** Recovery is **directional, not positional**: random perturbation of up to **2×**
  the Frobenius norm of the whole unlearning edit leaves disclosure at 0.031, while the same
  perturbation leaves a *present* fact at 1.000.
- **E5** A published gradient-norm (flatness) defense reduced `‖∇L_mem‖` by **40%** at
  unchanged forget loss and bought **one probe in 43** — no robustness gain.
- **E6** Full-parameter 8B training does not fit 2×40 GB; everything is LoRA, declared.

---

## H1 — Fact-disentangled unlearning: the shared edit direction *is* the leak

**Limitation addressed.** E3+E4 together say something specific. Recovery needs *gradient
information about the forget set* and is indifferent to large undirected displacement. And
8 leaked facts restore **43 facts the adversary never saw**. The only way facts the
adversary never touched come back together is if their removal was carried by a **shared
low-dimensional direction**. A single rank-16 LoRA applied to all 64 facts at once is
exactly that: one shared subspace. Push back along it with any subset of facts and the
whole set returns.

**Distinguishing operation.** Constrain the unlearning edit so per-fact removal directions
are mutually **low-coherence / orthogonal**, instead of letting them collapse into one
shared subspace. Candidate realizations (to be narrowed by the architecture agent):
(a) a coherence penalty on per-fact update directions; (b) projection of each fact's update
orthogonal to the span of previously applied fact updates; (c) explicit per-fact subspace
allocation in a higher-rank adapter.

**Why it should improve the target trade-off.** Two distinct predicted effects, which must
be measured separately because they are different claims:
1. *Robustness* — recovering fact A's direction no longer recovers fact B.
2. *Preservation* — projecting the forget update orthogonal to the **retain** gradient
   subspace should reduce collateral damage on shared-answer facts (the "Bob lives in
   Paris" case), which is the primary trade-off axis.

**Observable prediction that separates it from a simpler explanation.** Per-fact, not
aggregate: under the RTT attack with 8 leaked facts, held-out recovery should fall
**specifically for facts whose removal directions are near-orthogonal to the leaked
facts'**, and the per-fact recovery should correlate with that measured cosine. A method
that merely forgets harder, or that is simply slower to train, predicts a *uniform* shift
with no cosine correlation. **I got exactly this kind of per-fact prediction wrong before**
(ARC, where in-pool vs absent differed by only 0.101 and the prediction was refuted), so
the cosine correlation — not the aggregate gap — is the test.

**Conditions under which it should fail.** If the 64 facts genuinely require overlapping
parameter directions (because they share relation templates and answer tokens), enforced
orthogonality will either be infeasible at rank 16 or will damage retain. There is a hard
capacity bound: at most `r` mutually orthogonal directions exist in a rank-`r` adapter, so
64 facts cannot have 64 orthogonal directions unless `r ≥ 64`. **This may be fatal and must
be checked analytically before any GPU time.**

**Mathematical object.** A constrained optimization procedure, not a scalar loss — if
realized by projection it is a custom update with no loss whose gradient it is. Must be
labelled as such; no claim of ordinary loss descent.

### H1 PREMISE CONFIRMED — and it refutes my own proposed test

Measured on the existing unlearning adapter (`spex/adapters/big_uncached/ckpt_300`, the
control that removed all 64 forget facts), CPU-only, `outputs/edit_spectrum.txt`:

| singular index | mean energy fraction | cumulative |
|---|---|---|
| 1 | **0.5821** | 0.5821 |
| 2 | 0.1310 | 0.7131 |
| 3 | 0.0731 | 0.7862 |
| 4-16 | <= 0.047 each | 1.0000 |

**Effective rank (participation ratio) = 3.13 of 16 available**; min 1.12, max 7.63 across
224 modules. 141/224 modules put >50% of their edit energy into a *single* direction.

So the removal of **64 facts** is carried by an edit of effective dimensionality **~3**.
That is the shared-direction structure H1 predicted, and it is a plausible mechanism for
E3 (8 leaked facts restoring 43 unseen ones).

**But the rank-capacity bound I wrote above is the wrong concern, and the experiment I
proposed from it is the wrong test.** 16 directions were *available* and the optimizer used
3.1. The collapse is not imposed by the adapter's capacity — it is **chosen** by the
optimization. Two consequences:

1. **"Raise the rank" is predicted NOT to help.** If the optimizer collapses to ~3
   directions when given 16, there is no reason it would not collapse when given 64. A rank
   sweep is therefore demoted from primary test to a *control*: if `r = 64` also yields
   effective rank ~3, collapse is confirmed as optimizer-driven and the capacity
   explanation is dead.
2. **The distinguishing operation must be an explicit anti-collapse constraint**, not extra
   capacity. H1(a)/(b)/(c) survive; a plain rank increase does not.

**Still unmeasured, and the next thing H1 needs (GPU):** whether *per-fact* forget gradients
at theta_u align with that top direction. Low effective rank of the final edit is consistent
with shared directions, but does not by itself prove the per-fact gradients are collinear —
the edit could be a low-rank *average* of diverse per-fact directions. Measuring the
pairwise cosine matrix of per-fact gradients is the direct test, and it also supplies the
per-fact quantity that H1's falsifiable prediction is stated against.

**Status: H1 REFINED (was briefly recorded as REFUTED; that was too strong).** The per-fact gradient-cosine measurement came back and the
mechanism is not there: forget-forget mean cos = **+0.067**, and the 64-fact forget gradient
set has effective rank **29.48 of 64** -- not low-dimensional. The rank-3.13 edit is exactly
the low-rank *average of diverse directions* that I named as fatal to H1. The alignment is
also not specific to forgetting (retain-retain = +0.055), so it is a property of the loss
and parameterization.

What replaced it is **measured, not assumed**: the coupling is carried by the **shared
answer token** (forget-forget same-answer cos = **+0.916** vs +0.053 different-answer), and
that predicted a per-fact behavioural split which was then confirmed -- every held-out fact
sharing an answer with a leaked fact recovered fully (**1.000**, 8/8) versus **0.514** for
non-sharing facts at k=10. See `FINDING_answer_channel.md`. This supersedes H1 and is the
grounding the method design now proceeds from.

**E4 is also withdrawn** pending re-measurement: the published noise probe perturbed only
the 128-module attention subspace holding 28.7% of the edit energy. See
`ERRATUM_relearn_a67d610.md`.

---

## H2 — Decode-trajectory-targeted unlearning (MDLM-native)

**Limitation addressed.** Every MDLM unlearning method we have implemented or reproduced
trains the per-position denoising loss at mask rates `t ~ U(0,1)`. But disclosure happens
through the **iterative unmasking trajectory**: generation starts from a fully-masked canvas
and the answer token is *committed* at a particular step, at a particular position, under
low-confidence remasking. Uniform `t` spends most of the training budget on states the
decoder never visits, and the states that actually decide disclosure get a vanishing share.

**Distinguishing operation.** Train on the **decoder's own visited states** — the actual
sequence of partially-unmasked canvases produced by the author's sampler — and concentrate
the forget objective at the commitment step/position, rather than on teacher-forced states
at uniformly sampled `t`.

**Why it should improve the trade-off.** It aligns the trained object with the measured
object. E1's control already achieves 0.049 with uniform `t`; the hypothesis is not "more
forgetting" but **less collateral per unit of forgetting**, because the edit is applied only
where it changes the decode outcome instead of across the whole mask-rate range.

**Observable prediction.** At matched disclosure, retain damage and degeneracy should be
strictly lower, and the edit's Frobenius norm should be smaller. A simpler explanation
("it just trains less") is excluded by matching disclosure and comparing edit norm.

**Conditions under which it should fail.** Sampling decoder trajectories is expensive
(`steps` forwards per example vs 1), so cost may be prohibitive; and the visited-state
distribution depends on the student, so it shifts during training — a moving target that
may not converge. If disclosure turns out to be decided diffusely rather than at an
identifiable commitment step, there is nothing to target.

**Prerequisite measurement, cheap and decisive.** Instrument the author's sampler on
memorized forget facts and record *at which step and position the answer token first
becomes the argmax and is kept*. If that distribution is concentrated, H2 is live; if it is
diffuse, H2 is dead before any training. **This is the first GPU experiment H2 earns.**

**Novelty position (to be checked against primary sources by the prior-art agent).** If no
published MDLM unlearning method trains on sampler-visited states, this is category (d), a
potentially novel combination with a substantive interaction. If some method already does,
H2 collapses to a reproduction.

**Status:** UNTESTED. Prerequisite: the commitment-step measurement.

---

## H3 — Removal versus replacement: forgetting as calibrated uncertainty

**Limitation addressed.** E2 shows the field's disclosure metric is satisfiable by emitting
nothing. Our own clean-forgetting metric fixes the empty-output case, but not the subtler
degenerate solution the formalization agent is being asked to formalize: **memorizing a
replacement answer**. A model that answers "Alice lives in Tbilisi" scores 0.000 disclosure
and is perfectly fluent, yet the association has been *overwritten*, not removed — and the
replacement is itself a false fact the model will now assert.

**Distinguishing operation.** Target not a shifted point mass but the **answer distribution
of a model that never knew the fact** — i.e. match the never-memorized reference's
*distribution shape* over plausible answers at the forget query, rather than merely moving
mass off the true answer.

**Why it should improve the trade-off.** It is the only one of the three that changes what
"success" means in a way we can measure: it makes the forget behaviour *indistinguishable
from ignorance* rather than *distinguishable as suppression*. We already hold the
never-memorized reference model, so the target is available.

**Observable prediction.** A replacement-memorizing method yields a low-entropy answer
distribution concentrated on a single wrong answer; H3 yields entropy and a top-k answer
set matching the never-memorized reference. Measurable directly, and it separates H3 from
every disclosure-only method.

**Conditions under which it should fail.** The never-memorized reference's answer
distribution at a forget query may be arbitrary and high-variance, making it a noisy
target; and matching it may be unachievable without also matching it on retain queries,
which would undo memorization of the retain facts.

**Status:** UNTESTED. Depends on the formalization agent's criteria for (d).

---

## Scheduling and the open question H1/H2/H3 cannot settle

All three target **different** axes: H1 robustness-and-preservation, H2 collateral-per-unit-
forgetting, H3 the definition of success. They are not mutually exclusive and must not be
bundled: the mandate forbids causal claims when recipes differ in several operations.

Prerequisite checks come first because each can kill its hypothesis for near-zero GPU cost:
H1's rank-capacity bound (analytical, no GPU), H2's commitment-step distribution (one short
GPU job), H3's reference-entropy measurement (one short GPU job).

---

## Novelty update, 2026-10-20 — see `NOVELTY_POSITION.md`

A work our first prior-art pass could not locate has been found and verified: **Zhang et
al., "Adversarial Reinforcement Learning for Robust Diffusion Large Language Model
Unlearning," ICML 2026** (OpenReview `91bMgklyeQ`; no arXiv version, which is why it was
missed). Abstract verified directly from the ICML site.

**H2 (decode-trajectory / mask-rate targeting) — novelty materially damaged.** They claim
*"the first comprehensive study of unlearning for DLMs"*, explicitly address robustness, and
use an adversarial RL framework. The following claims are now forbidden: "first
trajectory-aware MDLM unlearning", "first study of MDLM unlearning robustness", and anything
premised on no MDLM method training outside uniformly-sampled teacher-forced masks. H2 may
still be correct science; it can no longer be framed as opening an unoccupied region. Its
mask-rate/position half survives, with a live risk that their "credit assignment" component
is per-position attribution.

**What distinguishes our axis, and it is a real distinction.** Their adversary is the **input
context** — they defend against *elicitation*, an attacker who writes a prompt exploiting
prefix-and-suffix conditioning. Ours is the **weights** — relearning, an attacker who
fine-tunes on leaked examples. Different threat model, different defense, different
measurement. This must be stated explicitly in any claim, not relied on implicitly.

**H3 (ignorance-matching rather than suppression) is now the strongest position.** It is
independently the target our own formalization derived from the deconvolution; it appears
unoccupied in every primary source read; and it makes the falsifiable prediction that
separates it from the exclusion target — it should **not** amplify same-answer coupling
(predicted ~0.388, versus the exclusion target's measured 0.916).

**The method therefore proceeds from H3 + Finding 1**, with H2's position/mask-rate element
demoted to a secondary, separately-ablated component rather than a headline claim.

**Blocking action, flagged for a human.** The camera-ready PDF is behind OpenReview's
Cloudflare challenge; the coordinator confirmed 403 independently. Until it is read we do
not know their state distribution, whether "credit assignment" is per-position, or their
benchmark and metric. **No novelty claim should be frozen before someone with an OpenReview
login downloads it.**

---

## Correction, same day — Finding 1's causal claim retracted

The replacement for H1 (answer-token indexing as the causal leak channel) was **published
and then retracted within the hour**, after two agents objected independently and a decisive
experiment confirmed them. See `FINDING_answer_channel.md`, which now carries the corrected
version with the original preserved.

Decisive data: two baselines that are **not** answer-token-indexed, both driven to disclosure
0.000, show substantial same-answer gradient coupling anyway — EMDU **+0.773**, MDU tau=1
**+0.573** — and in both, forget-same-answer equals retain-same-answer (0.773/0.753 and
0.573/0.575). Suppression produces the coupling; the target's indexing is not required.

What survives: same-answer coupling is the dominant structure in per-fact gradient geometry
across every method measured, and its **selectivity** differs sharply — 17.2x (our exclusion
target) vs 3.9x (EMDU) vs 1.4x (MDU). MDU's near-uniform coupling with effective rank 4.5/64
indicates globally collapsed gradients, a mechanistic correlate of its measured output
collapse. Descriptive only.

**H1 revised to REFINED**: `||sum of 8 leaked unit gradients|| = 3.213`, with cos to leaked
facts 0.402 vs **0.119** to non-sharing held-out facts, so non-sharing facts receive ~**30%**
of the per-step first-order loss reduction a leaked fact receives. Effective rank 29.5/64 is
far from isotropic (64 iid unit vectors give 63.98). There is no single global direction, but
the gradient set is strongly anisotropic and cross-fact transfer is real.

**A standing lesson, recorded because it has now happened twice.** ARC's error was presenting
an aggregate gap as a per-fact confirmation. Finding 1 repeated it in a new form: an 8-vs-35
group difference presented as per-fact confirmation while its own continuous predictor had
r = 0.014, p = 0.936. **From here on, a per-fact prediction is confirmed only by a
within-stratum continuous correlation, computed before the group comparison is reported.**


---

# STANDING HYPOTHESIS QUEUE (added 2026-10-20 on explicit instruction)

**Operating rule from the user:** whenever a hypothesis or candidate is shown not to work,
pivot to the next one *immediately* and amend the plan — do not pause on a negative. This
queue exists so a death costs no planning time. Each entry carries its distinguishing
operation, its kill rule, and the **cheapest decisive experiment**, specified before any of
them is run.

Everything is ranked against the one measurement the method must beat: the bimodal failure
(9/15 grid points collapse to `''`; the 2 fluent arms disclose ~0.45 and substitute
~0.31–0.38; the desirable cell holds 4 of 960 rows).

## H6 — mask-rate bands select the failure mode — **SUPERSEDED, not tested**

Withdrawn before testing. Two of this project's own agents had already shown the mask-rate
version is wrong — the decoder's visited rates are a uniform ladder at our configuration, so
`t ~ U(0,1)` already matches the marginal — and I had accepted both deflations and then
written the experiment anyway. Replaced by H6'.

## H6' — the failure mode is set by the TARGET'S OWN MODE *(in test, Phase B)*

Anchor-family targets are built from a masked/null prompt, where the likeliest continuation
is end-of-text → collapse. Renormalisation-family targets are the reference's second choice
→ substitution. **Neither family is buggy; each student matches its target faithfully and the
target's mode is the failure.** Explains the one thing mask rate does not: that the grouping
tracks objective *family* and is invariant to tuning. Subsumes H7.

Decided by forward passes only, no training. Kill rule: EOS/pad mass not elevated over the
reference for the anchor family, or target mode failing to predict any arm's measured cell.
See `METHOD_PLAN.md` §1a–1b.

## H7 — the collapse attractor is EOS, and the anchor target literally asks for it

**Observation.** MDU and EMDU emit `''` on all 64 forget probes. An empty generation means
mass is on EOS at the answer positions. Both build their anchor from a **masked or null
prompt**, where EOS is the likeliest continuation — so the target is not pathological, it is
being obeyed. The collapse may be the objective doing exactly what it says.

**Distinguishing operation.** Bar EOS and pad from the target's support and renormalise. A
one-line change to a pinned, already-working objective.

**Why it could matter more than anything else in this queue.** It addresses the failure mode
that accounts for **9 of 15** measured grid points, and if right it converts three collapsed
baselines into fluent arms — which would *also* make them honest comparators for the first
time.

**Cheapest decisive experiment (no training, minutes).** Teacher-forced forward on the saved
forget states for each trained arm; report EOS mass at answer positions versus the memorized
reference. **Kill rule:** if MDU/EMDU's EOS mass at answer positions is not substantially
elevated over the reference, H7 is wrong and the collapse has another cause.

**Status:** UNTESTED. Runnable now against existing adapters.

## H8 — no method preserves; the forget and retain gradients are antiparallel

**Observation.** The retain **margin** falls 12.16 → 6.38 nats in *every* arm, including the
collapsed ones, while retain *accuracy* reads 1.000 at 14 of 15 grid points and two points
already hold a retain probe at margin ≤ 0. So every method does large collateral damage that
the standard metric hides.

**Hypothesis.** At positions where forget and retain items share content, the two objectives'
gradients are near-antiparallel, so the damage is structural rather than incidental — a
prediction already derived analytically (`∇L_forget` carries `+w_v*` where `∇L_retain`
carries `−w_v*`, giving cos ≈ −0.9).

**Distinguishing operation.** Restrict the forget edit to a subspace orthogonal to the retain
gradient span, computed **once** rather than per step (so it stays ordinary loss descent in a
fixed subspace, not a constrained procedure).

**Cheapest decisive experiment.** Per-item cosine between `∇L_forget` and `∇L_retain` on
**TOFU800** — not the synthetic set, whose surname/answer pairing makes any such measurement
uninterpretable. ~10 GPU-min. **Kill rule:** cosine near 0 means the margin collapse has
another cause and the projection buys nothing.

**Status:** UNTESTED. Blocked only on the TOFU backbone.

## H9 — intervene only at the decoder's commitment states *(= C2 standalone)*

Covered in `METHOD_PLAN.md` §4. Prerequisite: the commitment distribution must be
concentrated. **Kill rule:** a diffuse commitment distribution leaves nothing to target.

## H10 — forget by re-ordering commitment, not by suppressing probability

**The most MDLM-native idea available, and the one with no autoregressive analogue.**

**Observation.** LLaDA's sampler commits the **highest-confidence** masked position each
step. Disclosure requires the answer position to be committed while the surrounding context
still pins the memorized answer.

**Distinguishing operation.** Do not lower `p(answer)`. Lower the **confidence at the answer
position relative to its neighbours**, so that position is committed *last* — by which point
the surrounding text is model-written and no longer determines the memorized answer.

**Why it could succeed where every arm in the table fails.** It is the only candidate that
asks the model neither to assert something false (so no REPLACE) nor to suppress its output
distribution (so no DEGENERATE). It changes the decode *order*, which is exactly the degree
of freedom the bimodality says both existing families lack.

**Signature prediction that separates it from everything else.** Disclosure falls **while the
answer token's probability at the original teacher-forced state is unchanged.** No existing
method can produce that signature: both families work by moving that probability.

**Caveat, recorded up front.** Our formalization's Lemma 5 — which asserted that suppression
lowers confidence and therefore delays commitment — was found **wrong** by the mathematical
critic, with a counterexample in the multi-surface regime. What survives, and what H10 needs,
is the weaker claim that `θ` *controls* commit order; the specific inequality does not hold
and must be re-derived.

**Cheapest decisive experiment.** Instrument the author's sampler on memorized forget facts
and record the step and position at which the answer token first becomes argmax and is kept;
then test whether a confidence penalty at that position delays commitment without moving
`p(answer)`. ~30 GPU-min for the measurement. **Kill rule:** commit order not manipulable
independently of `p(answer)`, or late commitment still yielding the answer.

**Status:** UNTESTED. Strongest novelty position in the queue.

## Pivot order

`H6 → H7 → H10 → H8 → H9`. H7 is placed second only because it is nearly free and addresses
the majority failure mode; if H6 dies, H7's experiment can run the same hour. H10 is ranked
above H8 because it is the only entry that could plausibly reach the empty region rather than
improve one side of the trade-off.

**If all five die**, the fallback in `METHOD_PLAN.md` §6 stands: map the
collapse-versus-substitution frontier to establish whether the empty region is *unreachable*
rather than merely unreached. That is a weaker contribution but a real one, and it is the
terminal state the mandate specifies — best prototype preserved, exact obstacle named.

---

# QUEUE PRUNED — see `RESEARCH_PLAN.md`

Six live hypotheses was thrashing, not planning. The register is now subordinate to a
research plan written backwards from the paper's money table. Status of every entry:

| hypothesis | status | reason |
|---|---|---|
| H1 shared edit direction | **refined, closed** | no global direction; gradient set effective rank 29.5/64, but ~30% per-step transfer is real |
| H6 mask-rate bands | **superseded, never tested** | the visited mask rates are already a uniform ladder at our configuration |
| H6' target mode selects the failure mode | **in test** — this is risk R2 | forward passes only; also subsumes H7 |
| H7 EOS attractor | **subsumed by R2** | Phase B measures EOS mass in the anchor targets directly; no separate experiment needed |
| H8 antiparallel forget/retain gradients | **parked** | its evidence (per-probe retain margin ≤ 0) was ULP noise on a bf16 1/16-nat grid; the aggregate −47% collapse stands but the mechanism claim lost its basis |
| H9 commit-state-only intervention | **parked** | an efficiency refinement, not a failure-mode fix |
| H10 commitment re-ordering | **one experiment remaining** | its stated form was refuted by derivation; the salvaged form gets its 7-GPU-minute oracle test and nothing more |
| **H11 minimal interdiction** | **COMMITTED** | attacks the measured structural cause (uniform pressure across all answer tokens forces collapse) rather than proposing another target and hoping for a different attractor |

**Rule adopted:** a new hypothesis enters this register only if it feeds a row or a column of
the money table in `RESEARCH_PLAN.md` §2, and only if an existing one is closed to make room.

