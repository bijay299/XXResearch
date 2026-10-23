# FID protocol for post-attack image quality — written, declared PENDING, not yet run

**Date:** 2026-10-23 · **Raised by:** audit issue 7 ("distinguish target regeneration, retained
capability, general image quality") · **Status:** `general_image_quality.fid = null`,
`status = "PENDING"` in every `atk_*.json` record.

This document exists so that the absence of an FID number is a **declared gap with a price tag**,
not an omission. It states what would be required, why the cheap version would be worthless, and
what is reported in its place.

---

## 1. Why no FID number is reported

Every utility measurement in this phase generates **48 images** per side (6 concepts × 2 prompts ×
4 images). FID is a Fréchet distance between two 2048-dimensional Gaussians fitted to Inception-v3
pool3 features. At n = 48 the sample covariance is **rank-deficient** (48 < 2048), so the estimate
is dominated by its own bias: the standard references put the usable floor at a few thousand images,
and report FID falling monotonically with sample size at fixed model quality. A number computed at
n = 48 would be a function mostly of n, and comparing two such numbers across arms would be
comparing noise.

Reporting it anyway would be the exact failure this project has already recorded four times: a
statistic too small to carry the claim placed on it (`FINDING35` §1 on H's 1/15 quantisation, and
the four n-too-small false signals in `T2I_REPORT.md` §4).

## 2. What is reported instead, and its exact scope

`post_attack_utility.prompt_adherence` — mean CLIP similarity between each generated image and
**its own prompt string** (not the class label), over the same 48 images, paired on identical
generator seeds before and after the attack.

| property | prompt adherence | FID |
|---|---|---|
| per-image, so n = 48 is 48 observations | ✔ | ✘ (one number from a covariance fit) |
| sensitive to loss of prompt conditioning | ✔ | partly |
| sensitive to texture/artefact degradation | **✘ weakly** | ✔ |
| sensitive to distributional collapse (mode drop) | **✘** | ✔ |
| needs a reference image distribution | ✘ | ✔ |

**So the gap is real and specific:** adherence can miss an attack that keeps images on-prompt while
degrading their realism or collapsing their diversity. Nothing in this phase currently measures
that. The regression test `issue7_probe_is_sensitive` establishes only that adherence responds to
gross UNet damage (σ = 0.01 noise moved it 0.2869 → 0.1656); it does not establish sensitivity to
subtle degradation.

## 3. The protocol that would close it

1. **Reference set:** 5 000 MS-COCO 2014 validation captions, fixed by index, with the 5 000
   corresponding real images as the reference distribution. COCO is the convention in this
   literature (ESD, UCE, RECE, AdvUnlearn all report COCO FID), so numbers would be comparable to
   published ones rather than only internally.
2. **Generation:** 5 000 images per model state, 25 steps, guidance 7.5, seeds fixed by caption
   index so that before/after and arm/arm comparisons are paired on initial latents.
3. **Features:** Inception-v3 pool3, the standard `pytorch-fid` weights, not a CLIP substitute —
   substituting the feature extractor makes the number non-comparable to the literature.
4. **Report** FID(base, real) as the anchor, then FID(arm, real) and FID(attacked arm, real), with
   the base anchor stated alongside every number so that reviewers can see the floor this model
   family sits at (SD v1.4 does not reach 0).
5. **Paired companion:** CLIPScore on the same 5 000 captions, which is what §2's adherence
   becomes at defensible n.

## 4. Cost, measured not guessed

Generation on this machine runs at **≈ 1.0 s/image** at 25 steps, 512×512, fp16, batch 4–5 on one
A100 (from the H65 utility probes: 48 images per side inside each attack).

| item | images | GPU-hours (1 A100) |
|---|---|---|
| one model state | 5 000 | **≈ 1.4 h** |
| one arm, before + after attack | 10 000 | **≈ 2.8 h** |
| the six-point frozen grid for one arm | 65 000 | **≈ 18 h** |
| base-model anchor (once, reusable) | 5 000 | ≈ 1.4 h |
| COCO val2014 download (real images) | — | ≈ 6 GB storage |

Plus Inception feature extraction, which is negligible beside generation.

**At the two-GPU limit in force, the frozen grid for a single arm is ~9 wall-clock hours.** That is
affordable for *one* decisive comparison and not affordable per-attack across 234 runs. The honest
plan is therefore selective, not comprehensive.

## 5. The minimum that would be defensible, and the trigger

Do **not** run this for every attack. Run it once, for the single comparison a claim would rest on:

> **Trigger:** if any arm reaches the mandate's bar (retain ≥ 0.85 with max-over-valid H ≤ 0.40
> under the frozen grid), FID becomes mandatory for that arm — base anchor, arm, and attacked arm,
> at n = 5 000 — before the result is published as a contribution.

As of this date **no arm has reached that bar**, so the trigger has not fired, and FID is recorded
as PENDING rather than run. If an arm does reach it, the ~4.2 GPU-hours for that one three-way
comparison are budgeted and authorised in advance by this document.

A second, cheaper trigger: if `prompt_adherence` ever *disagrees* with `clip_acc` on whether an
attack preserved utility (one flat, the other collapsed), that disagreement cannot be adjudicated
without FID, and the comparison above must be run for the arm in question.

## 6. Storage note

The COCO val2014 download (~6 GB) fits the authorised 60 GiB budget (21.0 GiB used). It has **not**
been downloaded, because an unused 6 GB download is not free on a filesystem that is 99 % full. It
is fetched when the trigger fires, not before.
