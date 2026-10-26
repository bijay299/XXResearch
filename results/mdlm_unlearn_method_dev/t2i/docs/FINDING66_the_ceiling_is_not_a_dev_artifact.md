# FINDING 66 — The H = 1.000 ceiling is **not** a DEV artifact. On prompts never used for anything, the attack restores the target to **15/15** — and on SD-1.4 that is **more reliable than the unmodified base model**, which scores 13/15.

**Date:** 2026-10-26 · **Experiment:** E6b
**Adversary:** `attack_relearn.py` **unmodified** — bit-identical to all 642 historical attack
records. Only `--keep-checkpoint` was added, so the attacked model could be scored on FINAL and then
deleted.
**Evidence:** `outputs/final_ATK_*.json`, `outputs/atk_E6B_*.json`, `logs/e6b.log`

---

## 1. The result

`FINDING65` established that pre-attack suppression is ~3× shallower on held-out phrasings than every
reported figure. That raised the converse question, which was the last remaining way the robustness
result could have been softer than reported: **is the H = 1.000 ceiling itself an artifact of the
three DEV prompts the project has always scored recovery on?**

It is not.

| arm | FINAL **pre**-attack | DEV H after | **FINAL post-attack** | its base on FINAL |
|---|---:|---:|---:|---:|
| `utrap_church_l0.15k8` (SD-1.4) | 0.0667 (1/15) | 1.000 | **1.0000 (15/15)** | **0.8667 (13/15)** |
| `utrap_sd15_church_l0.15k8` (SD-1.5) | 0.1333 (2/15) | 1.000 | **1.0000 (15/15)** | 1.0000 (15/15) |

Both attacks gate-valid with **T 0.000 → 1.000 (rise +1.000)**. DEV H and FINAL target agree
exactly, at the ceiling, on both checkpoints and both architectures.

## 2. The sharper version: the attacked model beats the original on held-out phrasings

On SD-1.4 the **unmodified base model** scores **13/15** on the FINAL prompts, while the
**attacked** UTRAP checkpoint scores **15/15**. So 300 gradient steps on 16 self-generated images
did not merely undo the erasure — on paraphrases neither the defence nor the attacker was tuned on,
it left the model generating the target **more reliably than the model that was never erased at
all**.

Two hedges, both load-bearing: the gap is **2/15**, one quantisation step beyond noise at this n;
and the base model's 13/15 is itself a single 15-image measurement. So this is a suggestive
instance, not an established effect. It is consistent in direction with `FINDING46` ("the attack
hardens the concept"), which is why it is recorded rather than dismissed.

The paired utility moved the same way: **0.8333 → 0.9792** on the SD-1.4 arm (40/48 → 47/48).

## 3. What this closes, and what it does not

**Closes:** the DEV-artifact hypothesis for recovery. Every H = 1.000 in this project was measured on
prompts that were, for four defences, also defence-training prompts and were in all cases a
development set. A reader could reasonably have asked whether the ceiling was an artifact of that
reuse. On the two arms with the strongest held-out suppression, it is not — the ceiling is reached on
prompts used for nothing.

**Does not close:**

- **Two arms, one seed, one lr, one concept.** This is a confirmatory probe, not a grid. `frp_*` and
  `uce_*` — the two arms whose pre-attack FINAL suppression was *weakest* (0.4667, 0.6000) — have
  not been measured post-attack on FINAL.
- **It says nothing about budget.** The attack ran 300 steps. Recovery-versus-budget curves on FINAL
  are unmeasured, so this cannot distinguish "reversible" from "reversible cheaply"
  (`ERRATUM_interpretations_r5.md` §1, §5).
- **n = 15, quantised to 1/15**, closed-set 12-label CLIP argmax, no reject option. All of
  `ERRATUM_interpretations_r5.md` §8 applies unchanged. 15/15 does not mean the model was restored
  in any general sense; it means the classifier labelled 15 of 15 images as the target.
- **FINAL is now further spent.** Four defence arms, two post-attack arms and two base anchors have
  been scored on it. Any further use is confirmatory only; it can no longer serve as a clean
  held-out set for a *new* operating point.

## 4. Effect on the record

`FINDING65` lowered every reported erasure depth. This finding confirms that the robustness half is
**unchanged** by the same scrutiny. Net effect on the project's standing: **the negative result is
stronger, not weaker, than the audited snapshot claimed** — erasure was shallower than reported and
recovery is just as complete on genuinely held-out prompts.

The mandate remains **unmet** and this moves nothing toward it: on FINAL prompts the
mandate-qualifying arm goes from 1/15 to 15/15 under its primary attack.

## 5. Addendum (E6c, 2026-10-26) — extended to the two weakest-suppression arms, and one sub-ceiling point appears

`FINDING66` §3 listed as not-closed that `frp_*` and `uce_*` — the arms whose **pre**-attack FINAL
suppression was weakest — had not been measured post-attack. E6c measures them. It is a
**deliberate extension added after seeing E6b's results**, labelled as such in
`run_e6c_final_recovery.sh`, not pre-committed; leaving them out would have been selective, since
they are where E6b's conclusion is least surprising.

| arm | FINAL pre | DEV H after | **FINAL post** | gate |
|---|---:|---:|---:|---|
| `utrap_church_l0.15k8` (SD-1.4) | 0.0667 | 1.000 | **1.0000** (15/15) | T 0.000→1.000 |
| `utrap_sd15_church_l0.15k8` (SD-1.5) | 0.1333 | 1.000 | **1.0000** (15/15) | T 0.000→1.000 |
| `frp_church_r3_l05` (SD-1.4) | 0.4667 | 1.000 | **1.0000** (15/15) | T 0.208→1.000 |
| `uce_sd15_church_lam1000` (SD-1.5) | 0.6000 | 1.000 | **0.9333** (14/15) | T 0.417→1.000 |

**Three of four reach the ceiling on FINAL; the fourth is one quantisation step below it.** §1's
conclusion holds across all four arms and both architectures.

**The one divergence is worth naming rather than rounding away.** `uce_sd15_church_lam1000` is the
only arm where FINAL post-attack (**0.9333**) is *below* its DEV H (**1.000**). The gap is 1/15 —
exactly one quantisation step — so it is not yet distinguishable from noise at n=15. It is also the
only sub-ceiling post-attack number anywhere in the FINAL measurements, and the `FINDING35` §4
ratchet applies: a sub-ceiling value is **provisional**, and running more seeds can only raise the
max. Two further seeds on this arm are running for exactly that reason.

Utility moved in both directions here, unlike E6b: FRP **0.9792 → 0.9167** (−0.0625) while UCE went
**0.9167 → 0.9583** (+0.0417). So the "recovery without collateral damage" pattern is not universal
even among arms that all reach the target ceiling.
