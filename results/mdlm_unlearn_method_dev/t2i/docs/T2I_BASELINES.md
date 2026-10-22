# T2I testbed and baselines — build log and honest state

**Date:** 2026-10-22 · **Model:** Stable Diffusion v1.4, pinned `133a221b8aa7292a167afc5127cb63fb5005638b`
**Pre-registration:** `PREREG_H38_t2i_testbed.md`

---

## 1. The testbed has a clean ceiling

Base SD v1.4, 15 images per concept (3 prompts × 5 seeds), zero-shot CLIP ViT-L/14 classification
over a fixed 12-way label set:

**Every one of the 12 concepts scores CLIP accuracy 1.000.** Forget targets (church, parachute,
garbage truck, Van Gogh) and retain neighbours alike. So erasure has unambiguous ground to move
against and there is no base-rate confound.

**Limit of this metric, stated up front:** a 12-way closed-set decision at 15 images per concept is
**coarse**, saturated at the top, and cannot detect improvement — only loss. The continuous
`clip_align` (CLIP cosine to the concept's own text) is recorded alongside for resolution, and FID
on COCO prompts is still to be added for the utility axis.

## 2. UCE re-implementation — two corrections, both recorded

**Correction 1: the first version did not erase at all.** It took the concept direction as the
single hidden state at the prompt's final content token. Result: 5.7% mean weight change and
church **unchanged at CLIP accuracy 1.000**.

The cause is structural, and worth recording because it is a property of SD rather than a typo:
SD cross-attention attends over **all 77 token positions**, and CLIP's text encoder is **causal**,
so the concept is propagated into every trailing position. The trailing "padding" states are what
the UNet actually reads. Editing one position leaves the concept fully reachable. Corrected to the
full range from the content token to position 76, matching UCE's official implementation — weight
change rose to 25–30% and erasure worked.

**A measured consequence, which is a general fact about this edit surface:**

| prompt pair | cos at the content token | mean cos at trailing positions |
|---|---|---|
| church vs tench | +0.1450 | **+0.5874** |
| church vs gas pump | +0.2127 | **+0.5670** |
| church vs monet | +0.1326 | **+0.5229** |

> The positions that dominate cross-attention carry **~4× more cross-concept overlap** than the
> discriminative content tokens. So a linear edit on those positions **cannot** separate the erase
> concept from retain concepts cleanly. This is a structural reason concept erasure trades off
> against utility in SD, measured rather than argued — the same shape of result as the MDLM phase's
> "two walls".

**Correction 2: the first working operating point destroyed utility** — church 1.000 → 0.000 but
retain 1.000 → **0.350**. The erase term (76 positions) carried the same weight as all 750 preserve
positions combined. A crippled baseline would make any treatment look good for the wrong reason, so
the balance was swept:

| preserve_scale | lam | church | retain mean |
|---|---|---|---|
| 0.3 | 0.1 | 0.000 | 0.311 |
| 0.3 | 1.0 | 0.111 | 0.422 |
| 1.0 | 0.1 | 0.000 | 0.511 |
| 1.0 | 1.0 | 0.000 | 0.422 |
| **3.0** | **0.1** | **0.000** | **0.778** ← selected |
| 3.0 | 1.0 | 0.000 | 0.667 |
| 10.0 | 0.1 | 0.000 | **0.044** ← collapses |

**The ps=10.0 collapse is non-monotone and unexplained.** A large preserve term should drive
`W_new → W_old`, not destroy the model; the likely cause is ill-conditioning of the matrix inverse.
It is recorded as a caveat on this re-implementation rather than ignored, and it is a reason not to
read the sweep as a smooth frontier.

**Selected baseline** (`preserve_scale=3.0, lam=0.1`), full 12-concept evaluation:

| | base SD v1.4 | UCE (ours, tuned) |
|---|---|---|
| **church** (erase target) | 1.000 | **0.067** |
| retain mean (8 concepts) | 1.000 | **0.850** |

Residual collateral damage concentrates on **styles** (monet 0.467, van_gogh 0.600) and two objects
(cassette player 0.733, golf ball 0.733).

**This is weaker preservation than published UCE reports.** It is our re-implementation at our
tuning, as `PREREG_H38` §4 requires it to be labelled; published numbers are not claimed, and the
gap is stated here rather than left for a reader to discover.

## 3. ESD re-implementation

Negatively-guided fine-tuning against the frozen original, no concept images. ESD-u (everything
except cross-attention, the paper's object setting): 815.6M trainable parameters, ~2.1 s/step.
Running at 500 steps. Results to follow.

## 4. State and next

| item | status |
|---|---|
| testbed + CLIP eval harness | done, ceiling 1.000 on all 12 concepts |
| UCE baseline | done and tuned, with two corrections recorded |
| ESD baseline | running |
| RECE, AdvUnlearn/Receler | not started |
| **relearning attack + validity gate** | **not started — kill rule K1 gates everything on it** |
| damage+distil treatment (the H38 hypothesis) | not started, correctly blocked behind the attack |

`PREREG_H38` K1 is explicit: if the relearning attack cannot be made to pass the validity gate on
the **baselines**, stop and fix the attack before measuring any treatment. Nothing about the
treatment will be measured until the attack is shown to work on something.
