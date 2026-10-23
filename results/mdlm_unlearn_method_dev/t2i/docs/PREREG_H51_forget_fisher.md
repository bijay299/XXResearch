# Pre-registration H51 — concept-targeted (forget-Fisher) damage, with a LOWERED prior recorded before results

**Date:** 2026-10-23 · **Status:** arms are **running and unread** as this is written.
**Why this document exists:** a primary-source novelty check returned while H51 was still training.
It is unfavourable in two specific ways, and recording that *before* the results is the only
version of this that is worth anything.

---

## 1. The mechanism

Inside the perturb-then-distil pipeline (UNDO, 2506.06278): instead of scaling each parameter's
perturbation by `(1 − normalised RETAIN Fisher)` — sparing the weights general capability needs —
scale it by the **normalised FORGET Fisher**, concentrating the same mean damage on the parameters
that most encode the erased concept. Motivation from `FINDING35`: every arm so far damaged all
gradient paths roughly uniformly, so utility and the concept's route died together.

## 2. ⚠ Novelty verdict: PARTIAL, and thinner than I assumed

| component | verdict |
|---|---|
| Fisher-weighted parameter **noise** for unlearning | **ANTICIPATED** — *Fisher Forgetting*, Golatkar et al., **1911.04933** (CVPR 2020) |
| damage magnitude scaled by **forget**-set Fisher | **ANTICIPATED** — SSD **2308.07707** (deterministic dampening, not noise) |
| forget-salient weights **randomly re-initialised**, then repaired | **ANTICIPATED** — Scissorhands **2401.06187** |
| forget-vs-retain Fisher **ratio** selecting weights to zero + repair | **ANTICIPATED** — DeepClean **2311.10448**, FisherMask **2310.05331** |
| random weight damage + repair evaluated for **tamper-resistance to relearning** | **ANTICIPATED** — *From Dormant to Deleted*, **2505.22310** |
| continuous, **mean-renormalised**, forget-Fisher-weighted *random* perturbation | NOT FOUND |
| forget-targeted damage whose repair is **distillation from the erased teacher** | NOT FOUND |

**A correction to our own earlier claim.** `FINDING24` (MDLM phase) presented retain-Fisher
protected damage as our modification to UNDO, with novelty recorded as "partially anticipated" by
EWC/SSD/WAGLE. The exact anticipation is now identified: **Fisher Forgetting (1911.04933) adds
Gaussian weight noise with covariance ∝ F(𝒟_r)^(−1/2)** — i.e. *more* noise where retain-Fisher is
*low* — and states its purpose verbatim as *"adding noise to destroy the weights that may have been
informative about 𝒟_f but not 𝒟_r."* That is our scheme's intent in a sharper functional form.
`(1 − normalised F_retain)` is a bounded reparameterisation of it, not a new idea, and should be
cited as such wherever we have described it as ours.

**So H51 is a noise-allocation ablation on an existing pipeline, using the allocation signal that
SSD/DeepClean/FisherMask already established as the right one for forget-targeted damage.** It is
reportable as a component ablation. It is not a new mechanism, and will not be described as one.

## 3. ⚠ The prior is LOWERED, on the designers' own evidence

UNDO's own ablation reports that **the noise *scale* β matters much less than the mixing
coefficient α** — *"Changes in mixing coefficient (α) produce a more pronounced effect on robustness
than the noise scale (β)"*. If the magnitude of the noise is second-order, there is prior reason to
doubt that its **distribution across parameters** is where the leverage lives.

Combined with `FINDING35`'s corrected statistics — Spearman(retain, recovery) = **+0.499**, i.e.
higher-utility arms are recovered *more* completely — **I expect H51 to fail**, and specifically to
fail in the way every previous arm has: either utility collapses, or utility survives and the arm
is fully reversed.

Recording that expectation now so that a negative result is not dressed up as confirmation of a
hypothesis I never held, and a positive one cannot be claimed as predicted.

## 4. Kill rules, including the one the audit forced

**K1.** Pre-declared **symmetric** attack budget: 6 attacks per arm — seeds {0, 1, 6} at lr 1e-5
plus {0, 1} at 5e-5 and {0} at 1e-4 — fixed before any arm is read, and **identical** for every
arm. The audit established that `max-over-valid-attacks` is monotone in attacks run and that this
project ran 9–10 attacks on hoped-for arms versus 2–6 on already-reversed ones, which made
`PREREG_H40` unfalsifiable in practice. No arm gets extra attacks after its number is seen.

**K2.** If utility (retain8) falls below **0.60**, the arm is not a candidate regardless of its
robustness number — the `FINDING35` band where resistance has only ever appeared alongside
unusable models.

**K3.** A positive result requires **both concepts** (church and parachute), since the composite's
apparent success on church evaporated on parachute (`H47` §7).

## 5. What a positive result would and would not be

Even a clean positive — retain ≥ 0.85 with max-over-valid H ≤ 0.40 under the fixed budget, on both
concepts — would be **a component ablation that beats the arms we tested**, not a SOTA claim. Four
existing defences for this threat model remain unrun: Meta-Unlearning (2410.12777),
IMMA (2311.18815), GIFT (2507.13598), and now *From Dormant to Deleted* (2505.22310), which is the
nearest published neighbour to this exact pipeline and the one a reviewer would demand first.

## 6. Verification debt, carried openly

The novelty check ran without web search (budget exhausted) via the arXiv and OpenAlex APIs, so
non-arXiv and very recent work is under-covered. Two specific items were flagged unverified and are
*not* treated as cleared: *"Weight initialization based on gradient similarity for versatile machine
unlearning"* (PoPETs 2026) — squarely in the Scissorhands family and unretrievable — and arXiv
2607.21353, *"Gradient Concentration, Not Weight Saliency, Explains Representation-Level Class
Unlearning"*, which bears directly on whether H51's premise (that a concept has a localisable
gradient route) holds at all.

---

## 7. ❌ VERDICT: H51 IS FALSE. Concept-targeted damage destroys utility *worse* than uniform damage.

The α=0.3 arm, under the pre-declared fixed budget:

| arm | church | **retain8** | attacks | gate |
|---|---|---|---|---|
| forget-Fisher α=0.3 | 0.000 | **0.0917** | 3/3 run at lr 1e-5 | **0/3 pass** (T rise −0.250, +0.000, −0.083) |
| *retain-Fisher α=0.3, same teacher, same α* | *0.000* | ***0.867*** | — | — |

**Kill rule K2 fires immediately:** retain 0.0917 is far below the 0.60 floor, so this is not a
candidate regardless of any robustness number — and no robustness number is even obtainable, since
0 of 3 attacks pass the validity gate with H pinned at 0.000. The model is too broken to attack.

**This confirms the expectation recorded in §3 before the arms were read** ("I expect H51 to fail,
and specifically to fail in the way every previous arm has: either utility collapses, or utility
survives and the arm is fully reversed"). It failed by the first route.

### Why it failed, and why that is informative

Concentrating the *same mean damage* on the parameters most important to the erased concept
destroyed the model **~9× worse** than spreading it away from retain-important parameters
(retain 0.0917 vs 0.867). The only way that happens is if **the parameters that matter most to the
concept are largely the same ones that matter most to general generation.** The concept is not
separably localised in weight space.

That is the weight-space counterpart of the measurement already in `T2I_BASELINES.md` §2: in
*embedding* space, the cross-attention positions the UNet actually reads are half-collinear across
concepts (cos 0.52–0.59) while the discriminative content tokens are not (cos 0.13–0.21). The same
non-separability now shows up in parameter space. Both findings say erasure cannot be made surgical
in this model, which is the mechanism `PREREG_H40` posits, arrived at from a second direction.

**It also retires the obvious remaining lever.** `FINDING35` §3 diagnosed that resistance appears
only in damaged arms at low utility, leaving "make the damage selective" as the one untried
response. Selectivity, implemented as forget-Fisher weighting, makes it strictly worse. UNDO's own
ablation anticipated this: the noise *distribution* is not where the leverage is.

α=0.5 is finishing for completeness of the ladder; it applies *more* damage and cannot clear K2.
