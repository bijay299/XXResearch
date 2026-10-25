# Finding 64 — the negative result is **not SD-1.4-specific**. A different checkpoint and a **half-size pruned architecture** both reverse to H = 1.000. And that **narrows our own mechanism claim.**

**Date:** 2026-10-25 · **Pre-registration:** `PREREG_H85_cross_model.md`
**Closes:** `T2I_REPORT_r4` §8's largest declared gap — thirteen reversals, all on one checkpoint.
**Evidence:** `outputs/gen_frp_{sd15,bksdm}_church_r3_l05.json`, `outputs/atk_H85_*.json`

---

## 1. The result

The **identical** pipeline — FRP at ρ=3%, β=0.05, λ=0.05, then `PROTOCOL_FROZEN_v1` — on two models
other than the one every previous result used:

| model | relation to SD-1.4 | UNet size | as-is target / retain8 | valid | **max-over-valid H** |
|---|---|---|---|---|---|
| SD-1.4 (all prior work) | — | 1.6 GB | 0.1333 / 0.9417 | 17 | **1.000** |
| **SD-1.5** | same architecture, different weights | 1.6 GB | **0.1333 / 0.9333** | **3/3** | **1.000** |
| **BK-SDM-small** | **structurally pruned UNet** | **0.9 GB** | **0.4667 / 0.9250** | **3/3** | **1.000** |

T rose +0.792…+0.875 (SD-1.5) and +0.625…+0.667 (BK-SDM), so all six attacks are valid and the H
values mean what they say. Both pre-registered predictions 1 and 2 are **confirmed**.

**SD-1.5 reproduced SD-1.4's FRP operating point to four decimals on the target** — 0.1333 both, with
retain 0.9333 vs 0.9417. The same prune fraction on a differently-trained checkpoint lands in the
same place and fails the same way.

## 2. ⚠ This forces a correction to our own mechanism claim

`T2I_REPORT_r4` §4 and `FINDING58`/`FINDING59` explained the failures partly by a **parameter-ratio**
argument: the defence constrains a small subspace (ψ = 5.11% of the UNet; FRP's scope = 17.3%) while
the adversary has **859.5M** parameters, so the adversary is not meaningfully constrained.

**BK-SDM-small halves the UNet and the outcome does not move.** The ratio argument predicts the
attack should be *relatively* more constrained on a smaller model — fewer surplus parameters to
relocate the concept into. It isn't: H = 1.000 on 3/3, from a *shallower* starting erasure (0.4667)
that should have been easier to defend.

So the ratio framing is **incomplete** and is narrowed here rather than left standing:

> Subspace-restricted defence is *sufficient* to explain the failures, but **parameter abundance is
> not the operative variable** — halving it changes nothing. `FINDING57`'s redundancy result is the
> better explanation: the concept is re-encodable in whatever weights survive, and a half-size model
> evidently has enough of them.

This is a correction to our own reasoning, not to any measurement; the numbers in `FINDING58`/`59`
stand.

## 3. What this does and does not establish

**Establishes:** across **three** checkpoints and **two** architectures, FRP-style erasure is
reversed to the base model's ceiling by 300 steps on 16 self-generated images. The phase's conclusion
is no longer "true of SD-1.4" — it is **true across checkpoints and across a halved architecture.**

**Does not establish** it for SD-2.x or SDXL: `stabilityai/*` returns **401** on this host (gated, no
token), so those were unavailable and the test used what was reachable rather than stalling. A
latent-space change of that size is the obvious remaining generalisation question.

**Does not re-test the other twelve defences cross-model.** Only FRP — our best published port and
the deepest published-method arm — was run on the new models. The other defences' cross-model
behaviour is **inferred, not measured.**

**One concept (church), one seed per build, lr 1e-5 only** on the new models: three attacks each
rather than the full six-point grid, because the three lr 1e-5 points already reached the ceiling and
`max-over-valid` cannot be lowered by adding more.

## 4. Provenance, and three bugs of mine fixed before any number existed

`BASE_MODEL`/`BASE_REV`/`BASE_VARIANT` overrides were added to `erase_frp.py`, `gen_eval.py` and
`attack_relearn.py`. **Verified before the run: with the variables unset they reproduce pinned SD-1.4
exactly**, so none of the 609 prior attack measurements changes. Revisions are pinned per model
(SD-1.5 `451f4fe16113…`, BK-SDM `572238db7ed3`).

Three defects, all mine, all caught before producing a result:

1. `local TAG=$1 L="${TAG}…"` — bash expands every argument before assigning, so `${TAG}` was unset
   and `set -u` aborted the script **silently**. Split into separate statements, with the reason in a
   comment.
2. The override fell back to `MODEL_PIN.txt`, feeding **SD-1.4's commit hash to SD-1.5** → Hub 404 on
   metadata. `BASE_REV` is now mandatory for a non-default model.
3. **`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`/`HUGGINGFACE_HUB_CACHE` are set in this host's environment
   to another project's cache** (`cuig_pilot`, 386 GB) and **override `HF_HOME`** — so model
   downloads were silently inflating a different project's storage. All four are now pinned to this
   phase's own cache.

⚠ `logs/crossmodel.log` contains two `BUILD FAILED` lines written by the **pre-fix** attempt, before
the successful builds in the same file. They are **stale artifacts of bug 2, not kill-rule results**,
and are annotated as such.
