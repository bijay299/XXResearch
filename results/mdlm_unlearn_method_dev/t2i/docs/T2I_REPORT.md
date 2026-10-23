# Text-to-image concept erasure — phase report

> ⚠ **SUPERSEDED by `T2I_REPORT_r2.md` (2026-10-23).** This document is preserved as published
> and is NOT the current state of the phase. Corrections are itemised in
> `ERRATUM_r2_claim_corrections.md` (C1 max-over-attacks phrasing, C2, C4 the 2.14 ratio, C5,
> C10 in-sample utility) and incorporated in r2; neither is applied here.


**Date:** 2026-10-22 · **Model:** Stable Diffusion v1.4, pinned `133a221b8aa7292a167afc5127cb63fb5005638b`
**Mandate:** an unlearning algorithm that beats the strongest existing methods on a clearly defined
forgetting–utility trade-off, with real-model results.
**Status of the last open candidate:** `PREREG_H47` is running and unread. Everything below is
settled.

---

## 1. The honest verdict so far

**The mandate's bar is not met in this phase either.** Every concept-erasure method tested —
including the published robust SOTA — is completely undone by a cheap fine-tuning adversary with
weight access, and the only intervention that resists costs more utility than it is worth.

**⚠ And that headline is largely ANTICIPATED, per a primary-source check.** Meta-Unlearning
(2410.12777) already reports T2I erasure reversed by malicious fine-tuning at 50–300 steps on a
prompt-disjoint split, and names AdvUnlearn in exactly that claim; `Erased or Dormant?`
(2505.16174) reports it for UCE and ESD. Our measurement is a **confirmation** — with one modest
gap filled, since Meta-Unlearning asserts the AdvUnlearn case in prose while measuring RECE. It is
not a discovery, and `FINDING34` now carries that correction at the top.

**A competitive-baseline gap that bounds everything below:** Meta-Unlearning, IMMA (2311.18815) and
GIFT (2507.13598) are three existing defences for precisely this threat model, and **none was run
here**. No contribution claim in this phase is credible without them, and this phase does not have
them.

| arm | church (erased) | retain8 | **max-over-valid-gate H after attack** | valid/total |
|---|---|---|---|---|
| base SD v1.4 | 1.000 | 1.000 | — | — |
| **AdvUnlearn** (official, robust SOTA) | 0.400 | **0.992** | **1.000** | 6/6 |
| ESD (our re-impl) | 0.467 | 0.992 | **1.000** | 3/3 |
| UCE (our re-impl, tuned) | 0.067 | 0.850 | **1.000** | 3/3 ⚠ |
| UCE at matched utility | 0.000 | 0.350 | **1.000** | 4/4 |
| UNDO α=0.5 (damage+distil) | 0.000 | 0.317 | **0.467** | 3/10 |
| UNDO α=0.7 | 0.000 | 0.000 | no valid attack | 0/3 |

**Attacker cost: 300 gradient steps on 16 images, generated from the public base model.** No
scraped data, no privileged access beyond the weights.

## 2. What is solid

**(i) Every usable erasure is fully reversible.** `FINDING34`: the official AdvUnlearn church
encoder — retain 0.992, a genuinely good operating point — is restored to the base model's 1.000 by
**6 of 6** gated attacks. UCE and ESD likewise, **6 of 6** (corrected from "8 of 8": two attacks
previously pooled into the UCE-tuned row were run against a *different* build — the checkpoint path
`adapters/uce_church/unet.pt` was overwritten between a retain-0.350 and a retain-0.850 build).
⚠ And "not one arm with usable utility survived" is **too strong**: see `FINDING35` §1-2, where one
arm (retain 0.867) reached H 0.533 on a valid attack and could not be rebuilt.

**(ii) Prompt-space adversarial training does not transfer to weight space.** AdvUnlearn is
hardened over a neighbourhood of *text embeddings* and is undone by 300 gradient steps on the
*UNet* — the weaker of the two adversaries we ran; its text encoder never had to be touched. This
is a scope statement, not a refutation of their paper, which claims prompt-space robustness only.
The T2I literature should state that scope explicitly.

**(iii) Damage+distil does add real robustness — at matched utility.** `H45`, seed-replicated: at
retain ≈ 0.33 UCE is reversed to 1.000 by 4/4 seeds while damage+distil holds at **0.467**, a 2.1×
reduction at equal utility cost. It is the only component measured to contribute robustness.

**(iv) But its utility cost is not a budget artifact.** `H41`: retain stalls at **0.32–0.34** across
a 6× larger distillation pool and 4× more steps (0.092 → 0.317 → 0.342). This **refuted our own
repair-capacity explanation** from `FINDING33` §3 within reachable budgets.

**(v) A structural asymmetry, pre-registered before its tests.** `PREREG_H40`: re-associating a text
direction with visual features that are still present is cheaper than removing those features
without destroying utility. Its declared refutation condition (retain ≥ 0.85 at post-attack ≤ 0.40)
is unmet by every arm; the robust SOTA fails it at retain 0.992 / H 1.000.

## 3. Measurement results about the methods we and others use

**The cross-attention edit surface cannot cleanly separate concepts.** Content tokens discriminate
between concepts (cos 0.13–0.21) but the trailing positions that dominate cross-attention are
**half-collinear** (cos 0.52–0.59). A linear edit there necessarily damages retain concepts — a
measured reason erasure trades off against utility, not an argued one.

**Damage's collateral destruction is as reversible as its intended erasure.** `FINDING32` §4a:
damage at α=0.3 knocked an untargeted concept (tench) from 1.000 to 0.133, and the attacker
restored it to 1.000 in 300 steps. Nothing the damage destroyed stays destroyed.

**Trainability loss is real but does not explain the robustness.** `FINDING32`: late attack loss
rises monotonically with α (0.3393 → 0.4864). In the MDLM phase the matching control came out the
other way (`FINDING33`: the attacker removes 93.6–98.9% of its loss on every arm while recovery
collapses), so robustness there is fact-specific. In T2I the two are harder to separate because
the arms that resist are also badly degraded.

## 4. Our own errors, recorded

| error | how it was caught |
|---|---|
| UCE re-impl #1 did not erase (5.7% weight change, church 1.000) | checked erasure before trusting the arm |
| adversarial erasure re-impl, 3 attempts, never erased (1.000 / 0.933 / 0.800) | verified against working ESD at the same config; resolved with official weights |
| adversarial inner loop was not adversarial (budget negligible beside ‖e_c‖) | printed ‖δ‖/‖e_c‖ and saw 0.030 |
| a concept-specificity result that would have contradicted `FINDING32` | tested **before** publishing: survived nothing — `H42` (adversary strength) weakened it, `H43` (seed) killed it |
| per-arm robustness numbers rested on single-seed gates | `H43` showed gate outcomes flip with seed; all headline arms re-run seed-replicated |

**The validity gate has three documented failure modes**, all found by using it: a ceiling artifact
on at-ceiling arms; a false negative when T prompts are intrinsically harder than H prompts; and
seed instability. Every T2I number is bounded by these, and the 12-way closed-set CLIP metric is
coarse and top-saturated (`T2I_BASELINES.md` §1).

## 5. Limitations

- **One concept** (church) carries the robustness results; one model; one attack family
  (relearning). Prompt-space attacks (UnlearnDiffAtk, Ring-A-Bell, concept inversion) were **not**
  run — the literature already reports those.
- **UCE and ESD are our re-implementations**, labelled throughout; our UCE preserves utility worse
  than published UCE. Only AdvUnlearn is an official checkpoint.
- The NSFW track — the field's main safety application — was **deliberately excluded**; benign
  concepts only.
- No exact retrain oracle exists at SD scale. This was the MDLM phase's single most valuable asset
  and this phase lacks it, so "erased" is always relative to the base model, never to a model that
  never saw the concept.

## 6. What would change the verdict

`PREREG_H47` is the one live candidate: AdvUnlearn's text-encoder erasure (retain 0.992) composed
with damage+distil UNet robustification (the only component measured to add robustness), acting on
disjoint parameters. Its bar was declared in advance — retain ≥ 0.85 with max-over-valid H ≤ 0.50
would meet the mandate's bar for the first time in either phase and refute `PREREG_H40` by its own
condition. A novelty check runs alongside, and an ANTICIPATED verdict will be reported as a
reproduction regardless of performance.
