# Finding 41 — the cross-phase trainability contrast is now computable, and it mostly shows the statistic is not cross-phase comparable

**Date:** 2026-10-23 · **Cost:** zero GPU — computed from `trainability.frac_loss_removed` already
recorded on every `H65` and `H64` attack by the issue-2/7 repairs.
**Resolves:** the audit's observation that the T2I attack records carried no before/after loss, so
`FINDING33`'s normalised statistic could not be computed here and the cross-phase contrast was
**unverifiable**.

---

## 1. The numbers

Frozen fixed-noise, fixed-timestep probe on 8 of the attacker's own images, evaluated before and
after the attack — the same construction as the MDLM phase's statistic.

| phase / study | arms | attacks | `frac_loss_removed` | recovery |
|---|---|---|---|---|
| **MDLM** (`FINDING33`) | all | — | **0.936 – 0.989** | recovery *collapses* |
| **T2I `H65`** (official AdvUnlearn, recoverable) | 2 | 12 valid | **−0.016 – +0.102**, mean 0.057 | **up to H = 1.000** |
| **T2I `H64`** (frag, destroyed, unrecoverable) | 2 | 10 | **+0.014 – +0.069**, mean 0.050 | max H 0.333 |

Probe loss *before* the attack: `H65` mean 0.1415, `H64` mean 0.1819 — the damaged arm does fit its
own data worse, which is the trainability deficit `FINDING32` described.

## 2. ⚠ The headline is a retraction, not a contrast

Read naively, the table says the MDLM attacker removed **96 %** of available loss and recovered
nothing, while the T2I attacker removes **5 %** and recovers everything — an arresting inversion.

**That reading is wrong, and the reason matters.** The two denominators are not comparable in kind:

- a masked-token cross-entropy can be driven near zero, so "fraction of available loss removed" is
  close to "fraction of the learnable signal captured";
- a **diffusion MSE has a large irreducible floor** — predicting the sampled noise is a
  Bayes-limited problem, and most of the 0.14 is unremovable *in principle*. A 5 % reduction may
  represent most of the *reducible* loss, and nothing in this measurement separates the reducible
  part from the floor.

So the honest conclusion is the one `ERRATUM_r2` C14 already flagged as an assumption we had not
earned: **this statistic does not support cross-modality comparison**, and the apparent inversion is
an artifact of comparing two different denominators. The audit's point is resolved by showing the
comparison *cannot* be made, not by making it.

**What is withdrawn:** any use of "the T2I attacker removes far less loss than the MDLM attacker" as
evidence about either adversary's strength, including any softer phrasing of it. The cross-phase
claim is **unsupported in both directions**.

## 3. ✅ What the T2I-internal comparison does support

Within one modality the denominators *are* shared, and here the result is clean and negative:

> **`frac_loss_removed` does not distinguish an arm whose concept is fully recoverable from an arm
> whose concept cannot be recovered at all.** `H65` (reaches **H = 1.000**) spans −0.016 to +0.102;
> `H64` (max **H = 0.333**, mostly gate-failing) spans +0.014 to +0.069. **The ranges overlap
> substantially**, and `H64`'s mean (0.050) sits inside `H65`'s spread.

Two arms per group, so this is **not a population estimate** and arm-level confounds are not
excluded. But the two regimes it is being asked to separate differ in outcome by 0.333 versus 1.000
— about as far apart as this metric allows — and the statistic does not separate them.

**This is the matched control `ERRATUM_r2` C13 said was missing.** `FINDING32` reported late attack
loss rising monotonically with α and treated it as establishing a trainability deficit; C13
downgraded that to provisional because a higher loss is equally consistent with the arm simply being
more damaged. The probe now shows both: `H64`'s *absolute* probe loss is higher (0.182 vs 0.142 —
the arm is harder to fit), while the *fraction removed* is indistinguishable from the fully
recoverable arms. So:

- **trainability loss is real** (absolute probe loss rises with damage), **and**
- **it does not explain the robustness**, because the fraction of loss the attacker can remove is
  the same whether the concept comes back or not.

`FINDING32`'s observation survives as a measurement; its *inference* to "this explains the
robustness" does not.

## 4. The within-`H65` correlation is INCONCLUSIVE, and is reported as such

Spearman(`frac_loss_removed`, `H_after`) over the 12 valid `H65` attacks:

| statistic | value |
|---|---|
| ρ | **+0.531** |
| bootstrap CI95 (10 000 resamples) | **[−0.111, +0.911]** |
| excludes zero | **No** |

Under the rule pre-committed in `PREREG_H67` §5 and applied here for consistency, **no claim is
made** — not "trending positive", not "suggestive". A point estimate of +0.531 on n = 12 drawn from
2 arms is compatible with anything from a weak negative to a strong positive.

Noting it explicitly because +0.531 is exactly the kind of number this project has previously
over-read: `FINDING35` §0 records two Spearmans published with wrong values and one with a flipped
sign, and `T2I_REPORT_r2` §5 lists four separate false signals from n-too-small.

## 5. Scope

One model; two arm families; one attack family; a probe over 8 images at fixed noise and fixed
timesteps, which is a deliberately low-variance instrument and not a general measure of
trainability. `H64`'s rows come from an arm that **K2 disqualified** (`FINDING39`) — they are used
here only as the *unrecoverable* end of a contrast, never as a robustness measurement. The MDLM
figures are quoted from `FINDING33` as published and were not recomputed.
