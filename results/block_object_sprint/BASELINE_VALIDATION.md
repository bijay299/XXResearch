# Baseline-validation record

Committed before any prototype code or tuning, as required. Budget state at the time of
this record: **3.20 of 48 GPU-hours**, **403 of 1,600 images**, **3.09 of 120 elapsed hours**.

## 1. Independent scorer controls — PASSED, but with an unresolved generated-domain failure

**Real-image controls (frozen rule, run once, not tuned): PASS.**
120 COCO val2014 photographs, labels from COCO's own human instance annotations.

| object | sensitivity | specificity | related-animal FP | verdict |
|---|---|---|---|---|
| Bears | 20/20 | 20/20 | 0/10 | PASS |
| Cats | 20/20 | 20/20 | 0/10 | PASS |
| Horses | 20/20 | 20/20 | 0/10 | PASS |

Threshold was >= 18/20 each, uncertain counting as error. There were no uncertain or
unparseable answers at all. Raw answers: `artifacts/scorer_validation/answers.jsonl`.

**Generated-domain audit: FAILS, systematically and in one direction.**
All 40 native-BLOCK (N/Bears) target images were audited by the agent, plus an
arm-blinded 20-tile sheet mixing M0/T0/N.

- On the arm-blinded sheet the agent agreed with the scorer on **15/20** tiles. All
  M0 and T0 bear calls and all retained-object calls were confirmed.
- The **5 disagreements were exactly the 5 images the scorer called "residual bear" at N.**
- Auditing all 40 N target images, the agent finds **0 bears**; the scorer reports **5**.
- Failure mode: stylised cartoon quadrupeds (dogs, foxes, cats) and crowded
  Byzantine / Van Gogh scenes read as "bear" to the judge.
- The native forced-choice evaluator shows the same weakness independently, labelling
  2/40 N images "Bears" (both Watercolor cartoon animals).

The COCO controls could not detect this: real photographs do not contain that
stylisation. This is precisely the limitation the protocol anticipated — small real-image
controls do not establish synthetic-domain validity.

**Direction:** the error is *conservative with respect to erasure*. The scorer
over-reports residual target, therefore **under-reports suppression**. No reported
effectiveness number is inflated by it.

**Consequence (recorded here, before any prototype work):** residual-target measurement on
edited arms is **not qualified at fine resolution**. The scorer's demonstrated error on
this arm is 5/40 (12.5 points), which is larger than the +/-5-point (2-image) resolution
the frozen prototype screening criteria require. Per the protocol, an unresolved material
generated-domain scoring failure blocks a prototype.

Numbers are reported from the frozen instrument as measured; the agent audit is reported
separately and is never substituted into the scorer's denominators.

## 2. Correct modules, native update, finite computations — PASS

- Replaced exactly the two gates the paper's Appendix D.1 and the release's code specify:
  `transformer.context_embedder` (`Linear(4096, 3072)`) and
  `transformer.time_text_embed.text_embedder` (`PixArtAlphaTextProjection(768 -> 3072 -> 3072)`).
  Shapes asserted against the transcoders at install time.
- Activation pairs verified **bit-identical** to the live original modules recomputed on a
  random sample (max abs diff 0.000e+00 for both) — the transcoders are trained to imitate
  the real modules, not something else. 321,920 context token-pairs; 330,000 pooled pairs,
  matching E.1's stated 330,000 exactly.
- Native erasure applied via the authors' own `apply_block_erasure` (their expressions,
  moved into a callable function). For Bears it writes **57 unique rows of 65,536** in the
  context transcoder and **32 of 12,288** in the pooled transcoder.
- All edited tensors finite; `max |delta|` 0.170 (context), 0.150 (pooled).
- **Unrelated tensors bit-identical after erasure**: encoder weight, encoder bias and
  decoder bias of both transcoders — all `True`. Only `W_dec` rows are touched.

Measured native hazard, reported not "fixed": the update writes 64 context rows of which
only 57 are unique, so **7 writes are order-ambiguous** (the tokens `▁Bear` and `s` share
7 active latents and PyTorch leaves the scatter winner unspecified). De-duplicating would
change the native update, so it was left intact and measured.

## 3. Storage separation, save/reload agreement, paired noise — PASS

- **Save/reload**: two independent loads of each checkpoint produce **bit-identical**
  outputs on identical inputs (stated tolerance: exact equality — same bytes, same dtype,
  same kernel; max abs diff 0.0). Numerical agreement within tolerance is not claimed to be
  anything more than that.
- **Reference separation**: erasure is applied to in-memory modules reloaded from a
  pristine checkpoint *per target*, so arms cannot mutate one another. Verified that the
  on-disk checkpoint is unchanged after running all three targets' erasures in one process.
- **Paired noise**: every case's initial latent comes from
  `torch.Generator(device="cpu").manual_seed(case.init_seed)`, rebuilt per image. Verified
  bit-identical for repeated construction, and identical across arms by construction
  (CPU generator, so device- and order-independent). Recorded in every `arm_report.json`.

## 4. M0 and T0 target presence — PASS (threshold 32/40)

| object | M0 definitely present | T0 definitely present |
|---|---|---|
| Bears | 40/40 | 40/40 |
| Cats | 40/40 | 40/40 |
| Horses | 40/40 | 40/40 |

Zero uncertain judgments. Agent audit confirmed every sampled M0/T0 bear call.

## 5. M0 retention — PASS (threshold 16/20 per stratum)

| object | related (Dogs, Rabbits) | unrelated (Architectures, Flowers) |
|---|---|---|
| Bears / Cats / Horses | 20/20 | 20/20 |

(The retained panels are shared across objects by the frozen design, so these are one
measurement, recorded as deduplicated reuse.)

## 6. T0 substitution cost — PASS (threshold <= 10 points)

| measure | M0 | T0 | change |
|---|---|---|---|
| related retention | 100% | 95% | **-5.0 pts** |
| unrelated retention | 100% | 100% | **0.0 pts** |
| severe rendering failure (80 cases) | 0% | 0% | **0.0 pts** |

By the native forced-choice metric: IRA related 100% -> 100%, IRA unrelated 100% -> 90%,
CRA 30% -> 40%, UA 2.5% -> 10%.

Installing the native transcoders is therefore a usable substrate: it costs one related
retained image (Dogs/Bricks) and no rendering degradation.

## Verdict

Gates 2, 3, 4, 5 and 6 **pass**. Gate 1's independent real-image controls **pass**, but the
generated-domain audit exposes an **unresolved, systematic, one-directional scoring failure
on edited arms**, which under the protocol blocks prototype development.

The platform is validated for measuring the native baseline, and the native baseline below
was faithfully executed and measured. It is **not** qualified to resolve the few-image
differences the frozen prototype screen depends on.

## Native BLOCK baseline on the development object (Bears)

Frozen 80-case panel. Binary presence procedure (adaptation) and native forced-choice
evaluator reported separately, never mixed.

| measure | M0 | T0 | N (native BLOCK) |
|---|---|---|---|
| target present (binary, /40) | 40 | 40 | **5** |
| relative suppression vs M0 (conservative) | — | 0.0% | **87.5%** |
| related retention (binary) | 100% | 95% | **95%** |
| unrelated retention (binary) | 100% | 100% | **90%** |
| severe rendering failure (/80) | 0 | 0 | **0** |
| native UA | 2.5% | 10.0% | **95.0%** |
| native IRA related | 100% | 100% | **100%** |
| native IRA unrelated | 100% | 90% | **75%** |
| native CRA (style) | 30% | 40% | **40%** |

Paired per-case transitions on the 40 target cases: `M0=yes, T0=yes, N=no` for 35 cases and
`M0=yes, T0=yes, N=yes` for 5 — with zero uncertain labels, and the agent audit judging all
5 of those residuals to be scorer false positives.

Retention failures, paired:
- related: exactly one (Dogs / Bricks), and it already fails at **T0** — caused by
  transcoder substitution, not by erasure.
- unrelated: two (Architectures / Pop_Art and Architectures / Watercolor), both introduced
  at N — caused by erasure.
