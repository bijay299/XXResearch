# Finding 34 — the official robust SOTA is fully reversed by a weight-access fine-tuning adversary, 6/6 attacks

**Date:** 2026-10-22 · **Model:** Stable Diffusion v1.4 (pinned `133a221b`)
**Defence:** official `OPTML-Group/AdvUnlearn`, `church_unlearned`, pinned `bca3b279` (arXiv
2405.15234)
**Protocol:** `attack_relearn.py`, prompt-disjoint T/H, `FINDING20` validity gate, seed-replicated
**Status:** the decisive test `PREREG_H40` asked for, run against a **working** robust baseline
after three of my own re-implementations failed to erase at all.

> ### ⚠ CORRECTION — the headline is a CONFIRMATION, not a discovery
>
> A primary-source novelty check run after this finding was written returns **ANTICIPATED** for its
> central claim. **Meta-Unlearning (arXiv 2410.12777)** already establishes that T2I erasure is
> reversed by malicious fine-tuning, using essentially this protocol — fine-tuning on generated
> harmful-image sets for **50/100/200/300 steps**, evaluated on a **prompt-disjoint** held-out set
> — and states verbatim that *"models unlearned by these methods can still be maliciously finetuned
> to relearn unlearned concepts, as observed in our experiments"*, naming **AdvUnlearn** explicitly
> in that sentence. `Erased or Dormant?` (2505.16174) reports the same for UCE and ESD.
>
> **What survives as ours is narrow and must be stated that way:** Meta-Unlearning asserts the
> AdvUnlearn case in prose but its tables measure ESD/SDD/UCE/**RECE**, not AdvUnlearn. So §2 below
> is a **clean AdvUnlearn-specific datapoint filling a gap in a published claim** — modest,
> gap-filling confirmation, *not* a new threat-model finding. §3's framing of the prompt-space vs
> weight-space distinction is likewise their thesis, not ours.
>
> This correction is recorded because the finding as first written reads as a discovery. It is not.

---

## 1. The defence, verified before attacking it

My three `erase_adv.py` attempts reached church 1.000 / 0.933 / 0.800 — none of them a competitive
erasure, so attacking them would have proved nothing. The official released weights replace them:

| arm | church | retain8 | church CLIP align |
|---|---|---|---|
| base SD v1.4 | 1.000 | 1.000 | 0.2406 |
| **AdvUnlearn (official)** | **0.400** | **0.992** | **0.1388** |
| ESD (our re-impl) | 0.467 | 0.992 | 0.1631 |
| UCE (our re-impl, tuned) | 0.067 | 0.850 | 0.1453 |

AdvUnlearn is the strongest operating point available here and preserves utility far better than
our UCE.

⚠ **Two claims withdrawn from this paragraph after audit.** It previously read "it erases more than
our ESD" and "the 42% drop in CLIP alignment confirms the erasure is real". Neither holds: 0.400 vs
ESD's 0.467 is a **one-image difference** on a 15-image metric and cannot support a ranking; and
`clip_align` does not track erasure depth at all — an arm that generates nothing recognisable
(α=0.7, all 12 concepts at 0.000) shows a *smaller* align drop (−17%) than AdvUnlearn (−42%), while
UCE at church 0.067 shows the same −40% as AdvUnlearn at church 0.400. The alignment drop is a
second coarse number, not a confirmation. AdvUnlearn erases the **text encoder**, so its release is a `CLIPTextModel`; it was loaded
directly into our pinned pipeline, with no legacy-repo clone and no environment change.

*(church 0.400 is above the ≤0.20 figure a stricter protocol would want. That is the official
checkpoint's own operating point under our coarse 12-way closed-set metric, whose saturation and
coarseness are recorded in `T2I_BASELINES.md` §1. Reported as measured.)*

## 2. The measurement

Two adversaries, because the defence lives in the text encoder and attacking only the UNet would be
needlessly weak against it:

| adversary | seed | T before → after | gate | **H before → after** |
|---|---|---|---|---|
| UNet only | 0 | 0.333 → 1.000 | PASS | 0.400 → **1.000** |
| UNet only | 1 | 0.083 → 1.000 | PASS | 0.200 → **1.000** |
| UNet only | 3 | 0.167 → 1.000 | PASS | 0.133 → **1.000** |
| UNet only | 4 | 0.208 → 1.000 | PASS | 0.200 → **1.000** |
| UNet + text encoder | 0 | 0.333 → 1.000 | PASS | 0.400 → **1.000** |
| UNet + text encoder | 1 | 0.083 → 1.000 | PASS | 0.200 → **1.000** |

> **Six of six attacks pass the validity gate and six of six restore the concept to the base
> model's own 1.000 on held-out prompts.** The attacker's cost is 300 steps on 16 images it
> generated from the public base model.

Two details worth keeping:

- **The weaker adversary suffices.** UNet-only tuning fully reverses a text-encoder defence; the
  text encoder never had to be touched. Whatever AdvUnlearn removed from the text representation,
  the UNet retained enough to be re-pointed at it.
- **No gate ambiguity.** Unlike several of our treated arms, every attack here passes decisively
  (T rises +0.667 to +0.917), so none of the gate's three known failure modes applies.

## 3. Fair framing — this is a threat-model result, not a refutation of AdvUnlearn

**AdvUnlearn claims robustness against adversarial *prompt* attacks** (UnlearnDiffAtk, Ring-A-Bell
and similar), where an attacker searches text space against fixed weights. This finding uses a
**strictly stronger** adversary: full weight access and gradient fine-tuning, which their paper
does not claim to defend against.

So the honest statement is **not** "AdvUnlearn fails". It is:

> **Adversarial training in prompt space does not transfer to robustness in weight space.** A
> defence hardened over a neighbourhood of text embeddings is undone by 300 gradient steps on the
> UNet.

That gap is precisely what `PREREG_H40` describes, and it is a scope statement the T2I erasure
literature should make explicitly: results reported as "robust erasure" are robust *within prompt
space*, and the weight-access case is a different and much harder problem.

## 4. Verdict on PREREG_H40

`PREREG_H40` claimed that in a T2I model retaining general generative capability, erasure cannot be
made robust to a fine-tuning adversary, because re-associating a text direction with intact visual
features is easier than removing those features without destroying utility.

Its §3 refutation condition was: *an arm with retain ≥ 0.85 whose max-over-valid-gate post-attack
accuracy stays ≤ 0.40*. AdvUnlearn has retain **0.992** — comfortably above the bar — and
post-attack accuracy **1.000**, the worst possible value. **H40 is not refuted; it is supported by
the strongest available defence.**

| arm | retain8 | max-over-valid H |
|---|---|---|
| AdvUnlearn (official robust SOTA) | **0.992** | **1.000** |
| ESD | 0.992 | 1.000 |
| UCE tuned | 0.850 | 1.000 |
| UCE at matched utility | 0.350 | 1.000 |
| UNDO α=0.5 (damage+distil) | 0.317 | **0.467** |

**Every arm with usable utility is fully reversed. The only arm that resists has unusable
utility.** That is H40's asymmetry stated as a table.

## 5. What remains open

`H40` is supported, not proved, and one live candidate could still refute it: `PREREG_H47`'s
**composite** defence — AdvUnlearn's text-encoder erasure (retain 0.992) plus damage+distil UNet
robustification (the only component measured to add robustness, `H45`: 2.1× at matched utility).
The two act on disjoint parts of the model, and `H38`'s utility collapse was diagnosed as inherited
from a mediocre teacher, which AdvUnlearn is not. Those arms are running and unread.

## 6. Scope

- One concept (church), one model, one attack family (relearning), 6 attacks.
- The attacker uses base-model generations as stand-in concept images; a real attacker with
  scraped photographs would be at least as strong.
- AdvUnlearn released one checkpoint per concept; no hyperparameter sweep of their method was
  performed, and none is claimed. This is their published operating point, not a tuned-down version.
- Our re-implementations of UCE and ESD are labelled as such throughout and are not claimed to
  reproduce published numbers.
