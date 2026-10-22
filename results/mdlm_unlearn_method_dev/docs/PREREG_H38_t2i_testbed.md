# Pre-registration H38 — text-to-image concept erasure: testbed, baselines, and the method candidate

**Date:** 2026-10-22 · **Status:** pre-registered before any run · **Domain:** Stable Diffusion
**Mandate:** unchanged — beat the strongest relevant existing methods on a clearly defined
forgetting–utility trade-off, with real-model results.

---

## 1. Why this domain, and what the MDLM phase hands it

The MDLM phase's central result is an **evaluation** result: behavioural metrics and static
knowledge probes both fail to predict whether an attacker can recover what was "unlearned"; only a
validated attack measures it (`REPORT.md` §10.1). T2I concept erasure is the setting where that
result has the most leverage:

- the standard reported metrics there are **static** — erased-concept classifier accuracy, CLIP
  score, FID — the same classes measured and found uninformative
- the attack literature already shows erasure is reversible (UnlearnDiffAtk 2404.19382, Ring-A-Bell
  2310.10012, P4D 2309.06135, concept inversion 2308.11508), so the gap is **known to exist** here
- but the protocol discipline this phase developed — **validity gate**, **matched attacker budget**,
  **max-over-attacks**, and the **trainability control** (`FINDING30` §4) — is not standard practice
  in that literature
- and the one method family that resisted relearning in the MDLM phase, **damage+distil**, appears
  **untested** for T2I concept erasure

## 2. The method candidate, stated before any run

> **H38.** Perturb-then-distil robustification (UNDO-style) transfers to T2I concept erasure:
> taking an erased UNet, damaging its weights with retain-Fisher-protected noise, and distilling
> the **erased** model's own outputs into the damaged student yields erasure that survives a
> relearning attack at a matched budget, where closed-form and fine-tuning erasure do not.

Why it is not obvious: UNDO's result is for LLM text classifiers/generators, where the suppressed
teacher's outputs are a token distribution. In a diffusion model the teacher's output is a
**noise prediction field**, distillation is over (timestep, latent, prompt) triples, and the
erasure methods being robustified are often **closed-form edits to cross-attention** (UCE, RECE)
rather than fine-tunes — so there is no gradient history for damage to disrupt. Whether damage
plus distillation does anything in that setting is an open empirical question, and the mechanism
established in the MDLM phase (`damage removes the attacker's route, not the knowledge`) makes a
specific prediction that can fail.

**Legitimacy, fixed now:** the erased teacher is produced by an ordinary published method on data
the practitioner has. No oracle, no ground-truth concept images in the loss. Targets are always
teacher outputs.

## 3. Testbed

- **Model:** Stable Diffusion **v1.4** (`CompVis/stable-diffusion-v1-4`), pinned revision recorded
  at download. v1.4 is the version ESD/UCE/RECE/UnlearnDiffAtk benchmark on, so baselines are
  comparable to published numbers.
- **Concepts (forget):** object classes — **church, parachute, garbage truck** (the UnlearnDiffAtk /
  Imagenette benchmark set) and one artist style, **Van Gogh**.
- **Concepts (retain):** the remaining Imagenette classes, plus COCO prompts for general utility.
- **Deliberately excluded: the NSFW / nudity track.** It is the canonical safety application in
  this literature, and erasing it is the real-world motivation — but generating that content is
  not needed to compare methods, so the testbed uses benign concepts only. This is a scope choice,
  recorded as a limitation, not a claim that the methods transfer to it.

**No exact oracle exists at this scale** — SD cannot be retrained from scratch on this budget. This
is the single biggest asset the MDLM phase had and the T2I phase lacks. Two mitigations, both
pre-declared:

1. **Floor proxy:** a concept the base model demonstrably cannot render serves as "never learned".
2. **Tier 2 (if the method shows promise):** a small text-conditional diffusion model trained from
   scratch with a concept genuinely absent, reproducing the two-tier structure that worked in the
   MDLM phase (real model for reality, small model for ground truth). Not built unless Tier 1
   justifies it.

## 4. Baselines — the bar to beat

| method | kind | why included |
|---|---|---|
| **ESD** (2303.07345) | fine-tune, negative guidance | the standard baseline |
| **UCE** (2308.14761) | closed-form cross-attention edit | strong, near-free to run |
| **RECE** (2407.12383) | closed-form + analytic adversarial embedding | current closed-form SOTA for robustness |
| **AdvUnlearn** (2405.15234) *or* **Receler** (2311.17717) | adversarially-trained erasure | the robust SOTA; **the real bar.** Included if it fits the budget; if not, its omission is stated in the headline, not a footnote |

Re-implementations are ours and will be labelled as such; published numbers are not claimed as
reproductions without matching their protocol.

## 5. Attacks — three families, with the gate

1. **Relearning attack** (transfers directly from this project): fine-tune the erased model on a
   small set of images of the concept, held-out-prompt evaluation, **author/prompt-disjoint T/H
   split**, under `FINDING20`'s **validity gate** — the attack must demonstrably relearn the
   material it was given before a flat curve on held-out prompts counts as robustness.
2. **Discrete prompt attack** (UnlearnDiffAtk-style): optimise a prompt/token sequence to resurface
   the concept. No auxiliary model needed.
3. **Concept inversion** (2308.11508): learn a new token embedding for the erased concept.

Reported as **max-over-attacks at a matched budget**, never best-case. Plus the **trainability
control** from `FINDING30` §4 — attack the arm on a *retained* concept to establish whether it is
harder to fine-tune in general — so a sluggish optimisation landscape cannot read as robustness.

## 6. Metrics — all three axes, every arm

- **forget:** erased-concept accuracy under a zero-shot CLIP classifier, on held-out prompts
- **utility:** retained-class accuracy, CLIP score and FID on COCO prompts
- **robustness:** max-over-attacks recovery at matched budget, gated

No arm is reported on fewer than all three axes. This rule exists because omitting the behavioural
axis caused two retractions in the MDLM phase.

## 7. Separating predictions, declared now

| outcome | reading |
|---|---|
| damage+distil holds erasure under all three attacks where ESD/UCE/RECE do not, at equal utility | **H38 true** — and the first method-level contribution across both phases |
| holds under relearning but **not** under prompt attack / concept inversion | partial: robustness is attack-family-specific, which is itself a reportable result given §1 |
| utility (FID/CLIP) collapses at the damage level needed | the trade-off is intrinsic in T2I as it was in MDLM; H38 false |
| no separation from baselines | damage+distil does not transfer out of the LLM setting; H38 false. A clean negative about the only family that survived the MDLM phase, and worth publishing as such |

## 8. Kill rules

**K1.** If the relearning attack cannot be made to pass the validity gate on the **baselines**
(i.e. the attack does not work at all in this setting), stop and fix the attack before measuring any
treatment. A defence measured against a broken attack is the failure mode this project has hit
three times.

**K2.** If damage+distil at the utility-matched damage level gives < 1.5× robustness improvement
over the best baseline, close the line rather than tuning α indefinitely.

**K3.** A novelty check runs **before** any claim of contribution, and an ANTICIPATED verdict is
reported as a reproduction regardless of performance — as was honoured for `H36`.

## 9. Order of work

1. Separate venv (the LLaDA venv was broken once by a dependency upgrade and must not be touched),
   SD v1.4 download, generation + CLIP-classifier eval harness, **base-model baseline numbers**
2. Baselines: UCE and RECE first (closed-form, cheapest), then ESD
3. The relearning attack **and its validity gate**, verified to work on the baselines (K1)
4. Only then: damage+distil, α swept against utility
5. Prompt attack and concept inversion on whatever survives
