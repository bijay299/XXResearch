# Finding 21 — the H18 backbone qualifies, and we finally have an EXACT retrain oracle

**Date:** 2026-10-22 · **Pre-registration:** `PREREG_H18_small_mdlm_testbed.md`
**Scripts:** `build_small_corpus.py`, `train_small_mdlm.py`, `eval_small.py`, `rank_probe_small.py`
**Status:** K1, K2 and K3 all pass. The testbed is usable for unlearning arms.

---

## 1. The gates

| gate | requirement | measured | verdict |
|---|---|---|---|
| **K1** `theta_ref` recall on forget facts | ≥ 0.60 | **0.6975** | ✓ |
| **K2** `theta_oracle` recall on forget facts | ≤ 0.15 | **0.1100** (chance 0.1135) | ✓ |
| **K3** `theta_ref` DEGENERATE | ≤ 0.10 | **0.0050** | ✓ |

**K2 is the result that matters.** The oracle scores **−0.0035 relative to chance** on the forget
facts: it has literally no knowledge of those authors, because they never entered its pretraining
or its memorisation fine-tune. This is the exact retrain reference the project has never had —
every 8B evaluation was anchored to an *approximated* `pristine`, and `FINDING7` closed oracle
matching partly because our oracle was itself questionable.

## 2. The backbone

| component | value |
|---|---|
| model | 124.0M bidirectional pre-LN transformer, GPT-2-small shape, no causal mask |
| objective (pretraining) | LLaDA/MDLM: `t ~ U(0,1]`, `p_mask = (1-ε)t + ε`, CE on masked positions, `1/t` weighted |
| corpus | 200,375,617 FineWeb-Edu tokens + 200 synthetic authors (100 forget / 100 retain) |
| shared base | general-only, 30,000 steps, 67 min, probe CE 4.738 |
| memorisation | value-span SFT phase, 6,000 steps, 8.3 min per model |
| answer pools | shared across splits by construction, verified: `forget_only` empty for all 4 attributes |
| author identity | 200 unique surnames, 200/200 identifiable by surname alone, 0 shared across splits |

Two-phase is deliberate and mirrors the 8B setup (`theta0_tofu` = base + memorisation SFT): one
shared general-only base, then `base → theta_ref` on all facts and `base → theta_oracle` on
retain facts only. The forget facts never enter the oracle at any stage, which is what makes it
exact.

## 3. The unlearning objective, now with ground truth

| measure | `theta_ref` | `theta_oracle` | meaning |
|---|---|---|---|
| forget recall (ranking top-1) | **0.6975** | **0.1100** | the gap a method must close |
| retain recall (ranking top-1) | 0.6725 | 0.8875 | what a method must preserve |
| forget profile DISCLOSE | 0.2625 | 0.0325 | six-cell, infill protocol |
| forget profile REPLACE | 0.0625 | 0.2625 | |
| forget profile IGNORANT | 0.6700 | 0.6775 | |
| forget profile DEGENERATE | 0.0050 | 0.0000 | |

> **The objective is now exactly specified:** take `theta_ref` from 0.6975 → 0.1100 on forget
> facts while holding retain at 0.6725, and move the forget six-cell profile toward the oracle's
> measured profile rather than toward an assumed one.

### A confound, stated rather than buried

The oracle's retain recall (0.8875) is **higher** than ref's (0.6725). The oracle had to memorise
only 100 authors (2000 value-span examples) against ref's 200 (4000) at identical compute, so its
task was strictly easier. That gap is a property of the construction, not of unlearning.
**Retain-side comparisons must therefore use `theta_ref` as the baseline, never the oracle.** The
oracle is a valid reference for the *forget* side only, which is what it is for.

## 4. Two metrics, both reported, because they disagree

| protocol | what it measures | ref forget | oracle forget |
|---|---|---|---|
| **ranking top-1** over the closed answer pool | knowledge | 0.6975 | 0.1100 |
| **infill** six-cell DISCLOSE | behaviour | 0.2625 | 0.0325 |

Ranking is the primary *knowledge* measure and is what K1/K2 are assessed on. The six-cell infill
profile is the *behavioural* measure and is what TV→oracle is computed on. They differ because
multi-token values are hard to emit under the pinned sampler: by attribute, year (1 token) reaches
0.58 while genre (~4 tokens) reaches 0.01. The sampler commits the highest-confidence slot first
and can place a value's second token before its first — the same commit-ordering problem this
project measured in `FINDING11`/`FINDING12`. Reporting both avoids choosing the flattering one.

## 5. Four instrument bugs found on the way, and what they cost

Memorisation looked impossible for seven configurations. Most of the apparent failure was
measurement, not the model.

| bug | symptom | effect on the measurement |
|---|---|---|
| probe prompts drawn from the pool | frozen_KL not comparable across arms | a withdrawn claim, and a wrongly-justified job kill |
| trailing-run masking at the sequence end | generation 0.06 on a model that had memorised | looked like total failure |
| unpadded probe sequences | ranking 0.4050 | understated |
| **BPE leading space** — training tokenised the value bare after a prefix ending in a space (`'Ghent'` = `[38, 6925]`), the probe tokenised `' ' + value` (`[402, 6925]`) | ranking 0.4675 | **understated by 0.26** |

Ranking as each was fixed: **0.4050 → 0.4675 → 0.6650**. The model had learned the facts all
along; the probe was scoring a continuation it was never taught. Probes now use the **exact token
ids stored at corpus-build time**, so probe and training share one construction path and this
class of error cannot recur. The same BPE leading-space bug previously cost a kill rule on the 8B
side — it is the second time in this project.

## 6. What was genuinely a design finding, not an instrument bug

**Isolation is the active ingredient for memorisation under the MDLM objective.** Measured across
configurations at matched compute:

| supervision | ranking top-1 |
|---|---|
| random masking over document windows | 0.1550 |
| value spans packed into 512-token sequences, all masked | 0.1200 |
| value spans packed, one masked per sequence | 0.1225 |
| value spans packed, value/fluency losses weighted separately | 0.1450 |
| **isolated L=32 examples, one fact each, alternating with general batches** | **0.6650** |

Loss weighting gave value tokens ~200× more per-token gradient and produced almost nothing.
Packing the identical supervision into sequences where ~31 other same-type values remain
**visible** collapsed it. The model takes the cheap route — emit a plausible nearby value —
whenever that route exists, and binds author→value only when the fact is the sole context.

That is the same shape as `FINDING18`: when a cheaper path to a plausible answer is available, an
MDLM uses it instead of the specific knowledge.

## 7. Next

1. Re-run MDU, EMDU, AFC and RMU on this backbone, scored against the **exact** oracle.
2. UNDO inside its validated regime (α ∈ {0.1, 0.3, 0.5, 0.7}, β ∈ {0.05, 0.1}) with the α=0
   control.
3. RTT under `FINDING20`'s validity gate plus the adversarial lr sweep, now affordable.
4. The joint prediction of `FINDING18` + `FINDING19`, which is the reason for the whole testbed.

**No 8B result is altered.** The two walls were measured there and stand on that backbone.
