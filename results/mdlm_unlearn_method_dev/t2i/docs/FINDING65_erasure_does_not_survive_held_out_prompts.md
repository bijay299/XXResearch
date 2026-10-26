# FINDING 65 — On prompts never used for anything, erasure is **~3× shallower** than every reported figure. FRP is 54% undone, UCE 60% undone. And the cause is **not** the prompt leak: the leaked arm degraded least.

> # ⚠ DO NOT QUOTE THE RATIOS. Audit defect D-022/D-022a, 2026-10-26.
>
> **Every FINAL number in this document was scored against the wrong label set, and the error runs
> in the direction that produces this finding.**
>
> `gen_eval.py` defines `LABELS` (:48) — the English text CLIP should see — and `LAB_KEY` (:50) —
> the underscored record keys. **Nine of the twelve differ.** `gen_eval_final.py:84` used
> `list(ge.CONCEPTS)`, whose keys equal `LAB_KEY`, so CLIP was prompted with *"a photo of a
> van_gogh"* and *"a photo of a english_springer"*.
>
> **Why that inflates this document's numbers.** `church`, `parachute` and `tench` are spelled
> **identically** in both lists. Every experiment here erases **church** — so the *target* label's
> text was correct while all nine competing distractors were degraded. Weaker distractors pull
> images toward church, so **`target_hits` is biased upward**. Measured across the 180 FINAL images,
> **37 still landed on a degraded label**, at median decision margins of **0.0027–0.0065** — far
> below the perturbation a label rewrite causes.
>
> **And the ratios compare two different scorers.** The DEV figures come from `gen_eval.py`, which
> uses `LABELS` **correctly**; the FINAL figures come from the defective path. FRP's "3.5×" is
> DEV 0.1333 against FINAL 7/15 = 0.4667, with only the FINAL side inflated. The same applies to
> UCE's "3×" and to §1's "54% undone" / "60% undone".
>
> **Status.** The 180 images were not saved, so correcting this requires regenerating them at the
> same seeds (deterministic, so the *same* images — a pure re-score, not a second use of the FINAL
> split) and re-scoring from `LABELS`. That run is queued as gate condition **G12**. Until it
> completes, treat every ratio and percentage in this document as **unmeasured**. The qualitative
> question — whether erasure is shallower on held-out prompts — remains open and may well survive;
> the magnitudes do not currently stand. `FINDING66` inherits the same defect.

**Date:** 2026-10-26 · **Experiment:** E6, the first and only use of the FINAL prompt set
**Pre-committed:** arm list fixed in `scripts/gen_eval_final.py` before any FINAL number was seen;
FINAL frozen 2026-10-25T23:30:00-04:00 and verified disjoint from DEF_TRAIN / ATK_TRAIN / DEV by
computation (`outputs/split_registry_v2.json`)
**Evidence:** `outputs/final_*.json` (per-image predictions retained),
`outputs/gen_E6cal_BASE_sd14_on_DEV.json`

---

## 1. The result

Target-concept generation on **FINAL** — three paraphrases per concept that were never used for
training, attacking, selection or any prior evaluation — against the **DEV** figure every report
has quoted:

| arm | base model | DEV target | **FINAL target** | base on FINAL | **FINAL as % of its own base** |
|---|---|---:|---:|---:|---:|
| `utrap_church_l0.15k8` | SD-1.4 | 0.0000 | **0.0667** (1/15) | 0.8667 | **7.7%** |
| `frp_church_r3_l05` | SD-1.4 | 0.1333 | **0.4667** (7/15) | 0.8667 | **53.8%** |
| `utrap_sd15_church_l0.15k8` | SD-1.5 | 0.0000 | **0.1333** (2/15) | 1.0000 | **13.3%** |
| `uce_sd15_church_lam1000` | SD-1.5 | 0.2000 | **0.6000** (9/15) | 1.0000 | **60.0%** |

**FRP's reported 0.1333 understates its residual target generation by 3.5×. UCE's 0.2000 understates
it by 3×.**

## 2. The calibration that makes this readable, and it is decisive

The obvious alternative explanation is that FINAL prompts are simply harder. They are not, or barely:

| model | DEV | FINAL |
|---|---:|---:|
| **base SD-1.4** (unmodified) | **1.0000** (15/15) | **0.8667** (13/15) |
| **base SD-1.5** (unmodified) | 1.0000 | **1.0000** (15/15) |

So prompt difficulty costs the base model **13.3% on SD-1.4 and 0.0% on SD-1.5**. It cannot account
for FRP moving +0.333 or UCE moving +0.400. On SD-1.5 in particular there is **no difficulty
penalty at all**, and UCE still loses 0.400.

## 3. The cause is NOT the split leak — and that is the interesting part

`ERRATUM_split_leak.md` established that `"a photo of a church"` — evaluation prompt H[0] — is also
the defence data-generation prompt for **GIFT, IMMA, WSR and UTRAP**. The natural prediction is that
the leaked arms should degrade most on held-out prompts.

**The opposite happened.** The two leaked arms (`utrap_*`) degrade **least** (7.7%, 13.3%). The two
arms that are **not** in the leak list (`frp_*` uses its own pools; `uce_*` is closed-form over
`erase_uce.FORGET_PROMPTS`) degrade **most** (53.8%, 60.0%).

So the leak is not the mechanism. What this measures instead is **prompt-specific erasure**: these
methods suppress the phrasings they were built and tuned around, and generalise poorly to
paraphrases. For FRP and UCE that is a property of the method as ported. For the selection process
it is worse than the leak would have been, because λ, β and K were all chosen while watching DEV H
(`ERRATUM_split_leak.md`), so the **operating points themselves** were fitted to three prompts.

A plausible secondary reading, not separable with this data: UTRAP's full-UNet 859.5M-parameter
scope may simply produce broader suppression than FRP's 3% mask or UCE's closed-form cross-attention
edit. Scope and prompt-generality are confounded here.

## 4. What this does to the project's claims

- **Every target-suppression number in the corpus is optimistic**, by a factor that is ~1.1× for
  UTRAP-class arms and ~3–3.5× for FRP- and UCE-class arms. The generated tables continue to report
  DEV values, now explicitly labelled DEV, with this finding as the correction.
- **It does not rescue any arm.** The direction is uniformly *worse* erasure, so no arm moves toward
  the mandate's bar. `uce_sd15_church_lam1000` at FINAL 0.6000 is 60% of base **before any attack**.
- **It does not change any robustness result.** All H measurements are DEV-prompt measurements of
  *recovery*, and the attacks reached the ceiling there. A FINAL-prompt recovery measurement has not
  been run (see §6).
- **UTRAP's standing improves slightly and narrowly.** At FINAL 0.0667 / 0.1333 it is the only
  tested method whose suppression substantially survives held-out phrasing. That is a real, modest
  point in its favour — and it remains reversed to H = 1.000 on 24/24 valid attacks, so it changes
  nothing about the mandate.

## 5. Limits, stated plainly

- **n = 15 per arm** (3 FINAL prompts × 5 images), quantised to 1/15, closed-set 12-label CLIP
  argmax with no reject option. The same measurement weaknesses as H (`ERRATUM_interpretations_r5.md`
  §8) apply unchanged.
- **One concept** (church), **four arms**, **one defence build each**, **one evaluation seed block**
  (common across arms by construction, via `split_registry.seed_for("final_eval", ·)`).
- **Three FINAL prompts is not prompt diversity.** It is three held-out paraphrases, which is enough
  to show the DEV figures are optimistic and not enough to estimate how optimistic in general.
- **FINAL is now partially spent.** Four defence arms and two base anchors have been scored on it.
  Re-scoring these arms after any further tuning would be illegitimate, and adding arms is a second,
  contaminated use — recorded here so the constraint is enforceable rather than remembered.

## 6. What this makes worth running next

**Recovery on FINAL prompts.** Every H number is DEV-prompt recovery. `attack_relearn.py` scores H
from the hardcoded `SPLIT[c]["H"]` pool, so measuring post-attack generation on FINAL needs a
separate scorer rather than a change to the historical evaluator. Cheap (JSONs only), and it would
say whether the ceiling is a DEV artifact — the one remaining way the robustness result could be
softer than reported.

Also newly motivated: **a paraphrase-robustness axis in the erasure objective itself.** Three of
four tested methods suppress their training phrasings and little else, which is a concrete,
measurable weakness no arm in this project was designed against.
