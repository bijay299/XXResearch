# Erratum — interpretations that outran their controls, rewritten to what the data supports

**Raised by:** external audit of snapshot `0b435427`, §8–10. Accepted in substance after checking
each against the records.
**Affects:** `FINDING57`, `FINDING59`, `FINDING60`, `FINDING61`, `FINDING62`, `FINDING63`,
`FINDING64`, `T2I_REPORT_r4`.
**Affects no measurement.** Every number below is unchanged; what changes is what it is said to
show.

The audit's closing observation about this project is the one worth keeping at the top:

> *"new explanations and summary claims are added faster than they are verified. A preserved
> retraction is useful, but a current report that repeats the same inference in different words has
> not completed the correction."*

That is accurate, and `ERRATUM_r2` C17 → `T2I_REPORT_r4` §7 is a documented instance: I corrected
"suppresses expression rather than removing capability" and then reintroduced it. This file is
written to be the statement report r5 is generated against, not a separate document r5 can drift
from.

---

## 1. A saturated ceiling cannot rank predictors

**Said:** erasure depth, weight-space geometry and unroll depth "do not predict resistance"
(`FINDING60`, `FINDING62`, `FINDING63` §3a(a)).

**Problem:** every usable arm sits at H = 1.000. `FINDING60` itself notes this and then concludes
the predictor's transfer is falsified. Both cannot hold: equal saturated endpoints are consistent
with large differences in recovery **rate**, in **minimum sufficient budget**, and in robustness at
**lower** budgets — none of which we measured.

**Supported statement:** *the measured differences did not prevent reaching the ceiling under the
attacks run.* That is all. In particular, L2 spanning 1470× with identical endpoints
(`FINDING60`) shows the endpoint is insensitive to L2 **at 300 steps, lr 1e-5**; it does not show
L2 carries no information.

**What would settle it:** time-to-threshold, recovery-versus-budget curves, area under a
fixed-budget curve, and comparison at **matched initial utility**. These are different estimands
from final maximum recovery and we have none of them.

**Also corrected:** the LMC diagnostic samples **three** interior interpolation points and reports a
**CLIP retain-accuracy** curve. That is evidence about this sampled functional path, not a
characterisation of a denoising-loss basin. Any barrier claim needs the actual training objective
and a denser path. `FINDING60`'s "barrier-free (monotone)" label is narrowed accordingly.

## 2. Re-zeroing shows final-coordinate dispensability, nothing wider

**`FINDING57`'s measurement, unchanged and still the strongest intervention in the project:**

| intervention on the attacked model | target H | retain8 |
|---|---:|---:|
| reapply FRP church mask (4,415,427 entries) | 1.0000 | 0.9667 |
| random count-matched church mask, seed 1234 | 1.0000 | 0.9667 |
| reapply FRP parachute mask (7,434,307 entries) | 1.0000 | 0.9583 |

**Supported:** the originally pruned coordinates are not necessary **for the final evaluated model**
to produce the target **on this panel**.

**Not supported, and withdrawn:**
- that those coordinates were unnecessary **during the attack trajectory**;
- that no important computation **transiently** used them;
- that the **surviving feed-forward weights alone** carry the target — the rest of the network was
  present and trainable throughout, so nothing localises the capability there;
- that **every** 3% mask is harmless. One seed, one concept, matched on count per tensor but not on
  magnitude or other correlated properties;
- that the attack **relocated an encoding** rather than exploiting or rebuilding redundant
  computation;
- that the original concept information was **never removed**.

The clean-model random control (`FINDING57`: random mask erases nothing, 1.0000 vs 0.1333) does
establish that **mask selection matters** versus those random masks. It does not establish that
forget-critical **scoring** rather than magnitude or another correlated property is what does the
work.

**The intervention that addresses the remaining alternatives** — clamp the original mask to zero
**after every attack step** for the full 300, compare against final-only re-zeroing, with repeated
count-matched **and magnitude-matched** controls at ≥3 seeds — is experiment **E5**, ~9 GPU-h,
unblocked, not run.

## 3. Target-assisted recovery cannot separate reactivation from relearning

`ERRATUM_r2` C17 stated this correctly: when the attacker is handed target images, recovery does not
distinguish **residual-memory reactivation** from **learning the concept again from those images**.

`T2I_REPORT_r4` §7 reintroduced the stronger reading — "suppresses expression rather than removing
capability". **Withdrawn again**, and report r5 does not contain it in any phrasing.

The learned-token (TI) results are relevant because the **backbone is frozen**, which is a genuinely
different kind of evidence. But the newly trained embedding still receives target information, so TI
does not supply a never-trained oracle either. Nothing in this corpus does. A stronger attack cannot
supply it — only a tractable setup with a never-trained reference (experiment **E10**, unscoped), or
narrowing the paper to the **operational relearning threat**, which is the honest alternative.

## 4. BK-SDM does not isolate capacity

**Said** (`FINDING64`): halving the UNet changed nothing, so parameter abundance is not the
explanation.

**Problem:** both models may sit **above** a capacity threshold, in which case the comparison is
uninformative about capacity. Pruning also changes architecture, weights **and** starting erasure
depth simultaneously. A two-point observational contrast cannot isolate one of three varying
factors.

**Also corrected:** shallower erasure is not automatically "easier to defend". A higher starting
target score may simply mean **less recovery is needed** — the BK-SDM FRP arm starts at target
0.4667 and the SD-1.5 one at 0.1333, so their endpoints are not comparable evidence about
robustness.

**Supported statement:** FRP, GIFT, UTRAP, Meta variant A and UCE ports, on SD-1.4, SD-1.5 and
BK-SDM-small, **church only**, **one defence build per setting**, three primary-learning-rate
attacks each, were all reversed to H = 1.000 where the arm cleared the utility floor. That is a
narrower claim than "cross-model generalisation" and it is what `FINDING64` is reduced to.

## 5. An expanded budget does not erase a fixed-budget result — in either direction

`FINDING59`'s 300-step observations **remain measurements at 300 steps** after 900-step attacks
succeed. Bounded-budget robustness and unlimited-budget permanence are different questions and the
`FINDING35` §4 ratchet, as written, collapses them.

**Withdrawn:** that success at 900 rather than 300 steps establishes a **minimal 3× cost
multiplier**. Intermediate budgets and matched baseline curves are needed; we measured two points.

**Withdrawn:** the TI "monotone dose-response". That curve changes **seeds, learning rates and step
counts simultaneously**; it is not a controlled one-factor curve.

**Corrected scope of the headline:** the project's main "300 steps" statement is about **full-UNet**
attacks. It must not be extended to the four-family table, whose recorded successes include **TI at
6,000 steps** and **LoRA at 900**. The generated tables now split every arm by budget so the two
cannot be pooled again.

## 6. The validity gate: an ineffective attack is inconclusive

The gate (T must rise ≥ 0.20) is a sound guard against reading a flat H as certification. But the
converse reading must be stated too: **preventing the attacker from fitting T can itself be part of
a defence.** So "no valid attack" means **the evaluation is inconclusive**, not that the defence
fails and not that it succeeds.

Both outcomes are now reported: `TABLES_v1_generated.md` §1 carries **max H (valid)** and
**max H (ungated)** side by side for every arm × family × budget.

For the core UTRAP result this distinction does not rescue anything — multiple **effective** attacks
recover H to 1.000 — and that is why it remains the project's strongest negative result.

## 7. The WSR2 utility result is descriptive, and not attributable to the cited baseline

Three builds: retain8 **0.9500, 0.9750, 0.9667** (mean 0.9639, sample SD 0.0127); targets 0.0667,
0.2667, 0.2000.

**Supported:** a promising observed retention improvement relative to **this particular FRP
teacher's** 0.9417, at a **selected configuration**.

**Not supported:** a statistically reliable frontier improvement. An illustrative one-sample 95% t
interval over three points is ≈ **[0.9323, 0.9955]**, which **contains** the teacher value. And that
interval is itself optimistic: the teacher has one build, the configuration was selected, and all
arms share a small evaluation panel.

**Withdrawn:** "statistically indistinguishable erasure". Equivalence requires a prespecified margin
and an appropriate analysis; **failure to establish a difference is not equivalence**.

**Withdrawn separately:** any attribution of this arm to 'From Dormant to Deleted'. It implements
the opposite direction (`ERRATUM_wsr_direction.md`). The result is ours.

**Also corrected** (`FINDING61/62`): "every MSE term is bounded". MSE is bounded **below**, not
above. Lower-boundedness does not imply the optimiser converges, and the sentence was being used to
argue that it would.

## 8. H is a 15-image closed-set measurement

H = 1.000 means the classifier labelled **15/15** generated images as the target — 3 prompts × 5
images, fixed 12-label CLIP argmax, **no reject option**, few semantic neighbours, quantised to
1/15. H = 0 means 0/15.

**Neither establishes exact zero capability nor complete model restoration.** For scale only: an
exact two-sided 95% binomial interval for 15/15 has lower endpoint ≈ **0.782**; for 0/15 the upper
endpoint is ≈ **0.218**. That independent-binomial model is **inadequate** for this study — prompt
clustering, reused evaluation seeds and adaptive attack selection all violate it — and is quoted
only to show why four-decimal presentation is not four-decimal certainty.

Generating more images from the same three templates does not add prompt diversity. What would:
a larger preregistered prompt set, an independent target detector or blinded panel, hard semantic
negatives, per-image labels with confidence margins, fixed paired seeds, and uncertainty units that
respect prompt and defence-build clustering.

**All target numbers are additionally development measurements** — see `ERRATUM_split_leak.md`.

## 9. Protocol compliance: the frozen grid was not unchanged

`PROTOCOL_FROZEN_v1` requires exactly six attacks per arm/concept with no exceptions unless amended
before outcomes are seen. `FINDING64` stops at three; other studies use three-point subsets,
additional attacks, or new families and larger budgets.

**Stopping after a decisive successful attack is logically defensible** for finding a
counterexample — adding attacks cannot lower a maximum. **Claiming compliance with an identical
six-point comparative protocol is not.** The two are now separated in the generated tables:

1. the **fixed confirmatory grid**;
2. **prespecified secondary** attacks;
3. **adaptive exploratory escalation**.

That mathematical fact does not retroactively amend the protocol, and `PREREG_H85`'s three-point run
is recorded as a deviation in `PREREG_PRECEDENCE.md` rather than by editing the plan.

Two further corrections in the same area:
- `PREREG_H80`'s "both concepts before any claim" requirement is **not met** by the church-only
  result. Recorded as an unmet commitment.
- `PREREG_H84`'s seed rule **explicitly permits** a clearly labelled single build. Its single builds
  are therefore a **limitation**, not a protocol violation, and were briefly described as the
  latter. Corrected.

## 10. FID

Its absence does not invalidate the target-recovery counterexamples. It **does** limit any claim
that an attacked model is globally repaired or preserves general image quality — and the paired CLIP
panel is not a substitute.

`FID_PROTOCOL`'s refusal to report a 48-image estimate was right. Its 5,000-image proposal needs
exact preprocessing, resolution, reference subset and uncertainty stated; COCO plus Inception alone
does not make a number comparable to every published FID. And the cost drops substantially under
the audit's observation: one shared pre-attack state reused across six attacks is **35,000** images
(40,000 with a fresh base anchor), not 7 × 5,000. Experiment **E8**.
