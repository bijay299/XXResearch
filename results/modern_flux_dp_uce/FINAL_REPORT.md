# Modern-model reproduction: FLUX.1-schnell DP + UCE, cassette player / chain saw

First reproduction of the released Double Proxy (DP) and UCE adaptations for
`black-forest-labs/FLUX.1-schnell`, from `czhang024/CVPR_DouleProjection`
(pinned commit `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`,
https://arxiv.org/abs/2604.10032), plus the conditional sequential extension
defined in the protocol that authorized this run. **The sequential extension
did not run**: neither method satisfied the pre-registered eligibility gate.
This report covers exactly what did run — the native single-edit
reproduction — honestly, including that negative gate result.

Full provenance (source/model hashes, environment, hardware, exact native
settings) is frozen in [`RUN_MANIFEST.md`](RUN_MANIFEST.md), written before
any edit or generation. Per-image classifier predictions for all 5,000
generated images are in `evidence/{M0,DP_A,DP_B,UCE_A,UCE_B}_per_image.csv`.

## Scope actually executed

| Endpoint | What it is | Images |
|---|---|---|
| M0 | Pristine FLUX.1-schnell | 1,000 |
| DP_A | Double Proxy, cassette player -> box | 1,000 |
| DP_B | Double Proxy, chain saw -> stick | 1,000 |
| UCE_A | UCE, cassette player -> box | 1,000 |
| UCE_B | UCE, chain saw -> stick | 1,000 |

**5,000 images total**, out of the 9,000-image ceiling fixed by the protocol
(M0 + 4 singles + up to 4 sequential endpoints). No sequential endpoint ran
(see Gate result below), so the remaining budget was not used. No SD1.x/RECE
generation was touched, repeated, or counted against this scope.

Both methods' single edits matched exactly the two expected weight keys
(`context_embedder.weight`, `time_text_embed.text_embedder.linear_1.weight`),
confirmed finite, with DP saved in its native bf16 and UCE in its native
fp32 (verified directly from the saved `.safetensors`, not assumed); every
generation run logged "Applied 2 edited parameter tensors" with no skipped
keys. Technical integrity holds for all four single edits.

## Native reproduction: semantic recognition

Classifier: pinned `Flux/compare_accuracy.py`'s own `ErasureAccuracyEvaluator`
(ResNet50, `ResNet50_Weights.DEFAULT` = `IMAGENET1K_V2`), imported unmodified
and run once per endpoint (not reimplemented) to avoid re-classifying the
shared 1,000-image M0 panel four times. Inner-join row counts were asserted
equal to 1,000 for every endpoint before any analysis ran — no silent
shrinkage.

| Endpoint | Target | Semantic (incl. tape player / radio) | Strict (exact label) |
|---|---|---|---|
| M0 | cassette player | **28/100** | 6/100 |
| M0 | chain saw | 100/100 | 100/100 |
| DP_A | cassette player | 0/100 | 0/100 |
| UCE_A | cassette player | 1/100 | 1/100 |
| DP_B | chain saw | 0/100 | 0/100 |
| UCE_B | chain saw | 0/100 | 0/100 |

## Eligibility gate for the sequential extension — result: NEITHER method qualifies

The protocol's gates (development continuation gates, not paper-claimed
thresholds) are conjunctive per method: *both* of a method's single edits
must pass, where passing a single requires (1) M0 recognizing that target in
>=80/100 of its own cases, AND (2) that method's single edit reducing semantic
recognition by >=75% relative to M0's own count.

| Target | M0 semantic | Gate 1 (M0>=80) |
|---|---|---|
| A (cassette player) | 28/100 | **FAIL** |
| B (chain saw) | 100/100 | PASS |

| Single | M0 -> edited | Relative reduction | Gate 2 (>=75%) | Gate 1 (from above) | Combined |
|---|---|---|---|---|---|
| DP-A | 28 -> 0 | 100.0% | PASS | FAIL | **FAIL** |
| UCE-A | 28 -> 1 | 96.4% | PASS | FAIL | **FAIL** |
| DP-B | 100 -> 0 | 100.0% | PASS | PASS | PASS |
| UCE-B | 100 -> 0 | 100.0% | PASS | PASS | PASS |

Both methods' B-edits are clean, complete suppressions from an excellent
baseline. Both methods' A-edits also show very large *relative* reductions
(96–100%) — the edits themselves are not weak. But target A's own M0 baseline
(28/100 semantic, 6/100 strict) never clears the 80/100 reliability floor, so
per the protocol's own rule ("a method qualifies only if both singles pass")
**neither DP nor UCE qualifies for the sequential extension** — not because
either method's erasure is weak, but because the A-side baseline isn't solid
enough to treat a large relative drop as a validated erasure signal. Full
gate arithmetic, including the strict-label numbers (reported separately,
never used for gating): `evidence/gates_and_retention.txt`.

Per protocol ("If none qualifies, finish and publish the native reproduction
with the exact limitation"), **the sequential A->B / B->A extension was not
attempted.** No target substitution, no re-tuning of either method, and no
new target search were performed in response to this result.

## Retention (the other 9 classes, per single edit, vs. M0)

| Single | Aggregate mean semantic, M0 | Aggregate mean semantic, edited | Largest single-class drop |
|---|---|---|---|
| DP_A | 91.8 | 91.4 | church −2, english springer −1 |
| DP_B | 83.8 | 83.3 | cassette player −6 (itself unreliable at M0, see above) |
| UCE_A | 91.8 | 89.4 | english springer −11, tench −6 |
| UCE_B | 83.8 | 80.8 | cassette player −13, english springer −9 |

DP's retention cost is smaller than UCE's at both targets in this round; UCE
shows a consistently larger (though still modest, <=13-point) retained-class
drop, most visible on english springer. Full per-class tables:
`evidence/gates_and_retention.txt`.

## Qualitative rendering panel (20 fixed cases per endpoint, var00 of the first two CSV rows per class — not a statistical sample)

Four of these cases were inspected directly (images published in `images/`):

- **M0, cassette player case 0**: a legible boombox/stereo with a cassette
  tape loaded — a human would call this "a cassette player." Yet across the
  full 100-case panel ResNet50 only semantically recognizes this class 28% of
  the time. The gap looks more like a classifier/ImageNet-definition mismatch
  against FLUX-schnell's 4-step distilled rendering style than a case of FLUX
  failing to depict the concept.
- **DP_A, case 0**: cleanly redirected to the guide concept — an open
  cardboard box, matching "box" directly.
- **UCE_A, cases 0 and 1**: both still show a cassette tape (not a box),
  despite the classifier no longer labeling either one "cassette player." In
  these two spot-checked cases, UCE's classifier-based suppression did not
  correspond to a visible shift toward the stated guide concept.
- **M0, chain saw case 10**: a realistic chainsaw, correctly recognized.
- **DP_B, case 10**: cleanly redirected to a single wooden stick/twig.
- **UCE_B, case 10**: a metal chain draped across a wooden stick — a blended
  result carrying visible residue of the erased concept ("chain") alongside
  guide-like content ("stick"), rather than a clean guide-only image.

This is 4 inspected images out of 2,000 single-edit generations, offered as a
qualitative observation and a candidate question, not a validated or
panel-wide finding. It is not a classifier-recognition failure being
mislabeled as "editing damage," and it is not a human-review substitute — no
human review was run or required for this task.

## Interpretation vs. unresolved hypotheses

**Observed, not interpreted:** the exact recognition counts and gate
arithmetic above; the 2/2 matched-key technical integrity on every edit; the
4 spot-checked rendering-panel images.

**Interpretation (not yet established):** cassette player's low M0
recognizability is plausibly a classifier/ImageNet-definition issue rather
than a FLUX generation failure, based on one visually legible counter-example
(M0 case 0). The apparent DP-vs-UCE difference in guide-concept redirection
(clean box/stick for DP vs. residual cassette-tape/chain content for UCE in
the 4 inspected cases) is a candidate distinction between the two methods'
mechanisms (direct projection vs. embedding-regularized closed-form edit),
not a demonstrated general property.

**Unresolved:** (1) why native FLUX.1-schnell's "Image of cassette player"
prompt is poorly recognized by ResNet50/ImageNet-1k even with tape
player/radio counted as equivalent; (2) whether UCE's apparent non-redirection
to the guide concept, versus DP's clean redirection, holds beyond the 4
spot-checked cases or beyond this concept pair.

No small-sample FID was added, no statistical-equivalence claim is made from
any null result, and target-prompt CLIP loss was not used as a rendering-
failure proxy, per the protocol's explicit constraints.

## Essential prior art (not run here)

BLOCK (https://proceedings.mlr.press/v306/kumar26c.html) already reports
sequential-removal results on modern models; its inspected release
(`sprintml/block-concept-removal` at `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`)
has missing imported helpers and no located pretrained-transcoder download in
the inspected tree/README, making it a candidate for a future SD3.5
reproduction attempt — not attempted or claimed as novel here.

## Recommended next action (exactly one)

**Diagnose the cassette-player / M0 baseline-recognition gap directly** —
e.g., compare ResNet50's/ImageNet-1k's "cassette player" training exemplars
and decision boundary against a larger native FLUX.1-schnell
"Image of cassette player" sample, independent of either erasure method —
before attempting any further FLUX DP/UCE sequential work on this concept
pair. The sequential-extension gate failure here traces entirely to that
baseline, not to either erasure method's own performance; and the DP-vs-UCE
guide-redirection difference noticed in the 4 spot-checked images is a
second, separate question best investigated on a target whose own M0
baseline is already solid (e.g., chain saw, which cleared both gates for
both methods).

## What is here vs. host-only

Published: this report, `RUN_MANIFEST.md`, all per-image classifier
predictions (5 CSVs, 1,000 rows each), the recognition/gate/retention
computation outputs, the exact scripts used for case-manifest construction,
evaluation, and gate computation, the four native edit logs, the pip-frozen
dependency list, the downloaded-weight hash table, and the 100
fixed-rendering-panel images (20 cases x 5 endpoints) at full 512x512
resolution.

Host-only (`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/`, not published):
the full 5,000-image generation output, the four edited `.safetensors`
checkpoints (2 tensors each), the FLUX.1-schnell model snapshot, and the
isolated venv. The pinned upstream source tree
(`czhang024/CVPR_DouleProjection` at `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`,
MIT-licensed per its own `LICENSE`) is not vendored here — it was not
modified, so citing its commit is sufficient; see `RUN_MANIFEST.md` for its
file-level hashes.
