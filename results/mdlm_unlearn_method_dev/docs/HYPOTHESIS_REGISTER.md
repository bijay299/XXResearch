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

**Status: H1 REFUTED.** The per-fact gradient-cosine measurement came back and the
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
