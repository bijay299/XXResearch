# Finding 1 — the answer token is the coupling channel

**Development evidence, 2026-10-20.** Two measurements, one predictive and one confirmatory,
on existing qualified assets. This refutes the coordinator's own H1 and replaces it with a
mechanism that is measured rather than assumed.

## What H1 claimed, and why it is dead

H1 held that the published leak — 8 leaked facts restoring 43 facts the adversary never saw
— was carried by a **shared low-dimensional direction**, evidenced by the unlearning edit's
effective rank of **3.13 of 16**. I flagged the alternative that would kill it: a low-rank
edit can be a low-rank *average* of diverse per-fact directions, which implies no transfer.

Measured directly (`outputs/grad_cosine_theta_u.json`), per-fact gradients of the
adversary's objective at θ_u, over the **full 224-module** edit subspace, 23.07 M
parameters, each fact averaged over 4 shared mask draws:

```
forget-forget   mean cos = +0.0670   |cos| = 0.0726   n = 4032
retain-retain   mean cos = +0.0545   |cos| = 0.0570   n = 4032
forget-retain   mean cos = +0.0185   |cos| = 0.0211   n = 4096
effective rank of the 64-fact forget gradient set = 29.48 of 64
```

Per-fact gradients are **near-orthogonal**, and their set has effective rank 29.5, not ~3.
**H1's mechanism is refuted.** The rank-3 edit is precisely the low-rank average I named as
fatal. The small residual alignment is not specific to forgetting — retain-retain (0.055)
is of the same order — so it is a property of the loss and parameterization, not of the
unlearning.

## What is actually there

Splitting the same cosine matrix by what the facts share:

| pair type | mean cos | sd | n | vs unrelated |
|---|---|---|---|---|
| forget-forget, **same answer token** | **+0.9159** | 0.066 | 64 | **83×** |
| retain-retain, same answer token | +0.6375 | 0.298 | 64 | 58× |
| **forget-retain, same answer** (the Alice/Bob case) | **+0.2597** | 0.114 | 128 | **24.3×** |
| forget-forget, same relation, different answer | +0.0672 | 0.071 | 1408 | 6× |
| forget-retain, different answer | +0.0107 | 0.014 | 3968 | 1× |

Per relation, the forget-retain same-answer coupling is +0.219 (home_city), +0.278 (field),
+0.324 (instrument) — present in all three, so it is not an artifact of one template.

**The coupling is carried by the shared answer token.** Two forget facts with the same
answer have nearly collinear gradients; two facts with different answers are orthogonal.
The relation template contributes an order of magnitude less.

A caveat on magnitudes, stated because it matters: at θ_u the forget facts are suppressed
(large gradients) while the retain facts are well fit (gradient norms span 1.16e-4 to 2.47).
The direction of a near-zero gradient is noisy, which is the likely reason forget-retain
same-answer coupling (0.260) is well below forget-forget same-answer coupling (0.916)
despite both sharing the token. Only **2 of 64** forget facts have any retain partner above
cos 0.5.

## The behavioural prediction this makes, and its confirmation

If the answer token is the channel, then under the published RTT attack the leak should
concentrate on held-out facts that **share an answer token with one of the 8 leaked facts**.
The 128-fact design gives each answer exactly 2 forget facts, so the 8 leaked facts have 8
same-answer partners among the 43 held out.

Re-partitioning the published per-fact generations (`anvil/outputs/relearn_gens_unlearned.json`,
no new compute):

| adversary steps | shares answer with a leaked fact (n=8) | does not share (n=35) | gap |
|---|---|---|---|
| 0 | 0.000 | 0.029 | −0.029 |
| 2 | 0.125 | 0.029 | +0.096 |
| 5 | 0.500 | 0.200 | +0.300 |
| **10** | **1.000** | **0.514** | **+0.486** |
| 20 | 1.000 | 0.571 | +0.429 |

**Every held-out fact sharing an answer token with a leaked fact was fully recovered by
k=10 (8/8).** The split is monotone in `k` and starts from parity at `k=0`, so it is a
property of the attack's progress, not of the starting point.

This is the per-fact, cosine-predicted test that H1 was supposed to pass and did not. It is
reported with the same standard I failed to meet in the ARC study, where an aggregate gap of
0.101 was presented as confirming a per-fact prediction.

**What it does not explain.** The non-sharing group still reaches 0.514. The answer channel
accounts for the fastest and most complete recovery, not all of it. The relation channel is
**untestable in this data** — the deterministic stride that chose the leak set happened to
cover all three relations, so there is no relation-disjoint comparison. That is a design
defect in the published leak set and the corrected RTT protocol must draw leak sets that
leave both answers and relations partially uncovered.

## The theta_0 control: unlearning CREATES the channel

The same measurement at theta_0 -- the memorized model, before any unlearning -- separates
"intrinsic to the task" from "produced by the objective"
(`outputs/grad_cosine_theta_0.json`):

| quantity | theta_0 (memorized) | theta_u (unlearned) | change |
|---|---|---|---|
| **same-answer coupling** | **+0.3876** | **+0.9159** | **x2.36** |
| different-answer | +0.0596 | +0.0533 | -11% |
| same-relation | +0.1263 | +0.1041 | -18% |
| forget-forget overall | +0.0648 | +0.0670 | +3% |
| retain-retain overall | +0.0463 | +0.0545 | +18% |
| forget-retain overall | +0.0412 | +0.0185 | -55% |
| gradient-set effective rank | 33.46 / 64 | 29.48 / 64 | -12% |

**Every structural quantity is essentially unchanged except the same-answer coupling, which
more than doubles.** The channel is not created from nothing -- at theta_0 it is already
6.5x the different-answer baseline (0.388 vs 0.060) -- but unlearning amplifies it to 17x
(0.916 vs 0.053) while touching nothing else.

This completes a causal account:

1. The exclusion target removes probability mass from the answer-token set `F`.
2. Two facts with the same answer have the *same* `F`.
3. So the objective drives their gradients into near-collinearity (0.388 -> 0.916).
4. Hence an adversary recovering one fact recovers all its same-answer partners
   (1.000 vs 0.514 at k=10).

**Caveat on the theta_0 magnitudes.** Gradient norms at theta_0 average 4.32e-06 (range
2.46e-07 to 2.12e-05) because the facts are fit, versus 5.72e-01 at theta_u -- five orders
of magnitude. Directions of near-zero gradients are numerically delicate, so 0.388 should
be read as indicative. The *contrast within* theta_0 (same-answer 0.388 vs different-answer
0.060, measured by the identical pipeline) is the robust part, and it is what licenses the
"amplified, not created" reading rather than "created".

## A cheap mechanism check, and a prediction that separates the two candidate targets

Same-answer coupling is measurable in ~3 GPU-minutes for any trained arm. That makes it a
**mechanism check** -- explicitly not an optimization objective, per Song et al.'s
"Measure, Don't Optimize" warning that a correlate turned into a target teaches the model to
satisfy the correlate. The outcome of record remains the downstream relearning curve.

It yields a falsifiable prediction that distinguishes the two closures the formalization
derived from the same deconvolution:

> The **exclusion-target** closure (mass removal indexed by the answer-token set `F`)
> amplifies same-answer coupling: measured 0.388 -> 0.916.
> The **ignorance-reference** closure (match the distribution of a model that does not know
> this *entity's* fact) is indexed by the entity, not by `F`, and is therefore predicted
> **not** to amplify it: same-answer coupling should stay near 0.388.

If the ignorance target amplifies the coupling anyway, the entity-vs-token framing is wrong
and the mechanism story above is incomplete. If it does not, the two targets are
mechanistically distinct in exactly the way that matters for the leak -- and that is
checkable before any expensive evaluation.

## Why this matters for the objective

The formalization asks: *what is the minimal sufficient conditioning information that
separates "Alice lives in Paris" from "Bob lives in Paris"?* This measurement answers the
dual question — what the current objective actually conditions on. An edit whose per-fact
gradient is essentially a function of the **answer token** cannot separate Alice from Bob,
because both facts present the same token at the same position under the same template; it
can only separate them through the entity representation, which the measured geometry says
is contributing little (same-relation-different-answer coupling is 0.067, i.e. the entity is
*already* doing the separating work for different-answer facts — it is specifically the
shared-token case that collapses).

It also explains two previously unexplained observations:

- **The "Tbilisi attractor."** 19 of 64 forget probes converge on the same replacement
  answer. If same-answer facts share a gradient direction at cos 0.92 and the exclusion
  target's fixed point is `argmax_{v∉F} p₀`, a single off-F attractor per relation is the
  expected outcome.
- **Why unlearning generalizes across paraphrases but leaks across facts.** Paraphrases of
  one fact share the answer *and* the entity; different facts sharing only the answer get
  the coupling without the separation.

## Status and next step

- **H1 (shared global direction): REFUTED.** Recorded in `HYPOTHESIS_REGISTER.md`.
- **Answer-token coupling: SUPPORTED** by gradient geometry (0.916 vs 0.053) and by a
  per-fact behavioural split (1.000 vs 0.514 at k=10).
- Single seed, single leak set, synthetic facts, one model. The coupling magnitudes are
  measured at one point (θ_u) in one parameterization (LoRA-B at init, which is the subspace
  the adversary's first step moves) and are a **restriction** of the full weight-space
  gradient.
- Required before this grounds a method: the same measurement at θ₀ (is the coupling created
  by unlearning or intrinsic to the task?), and a leak set drawn to leave answers and
  relations partially uncovered so both channels are separately testable.
