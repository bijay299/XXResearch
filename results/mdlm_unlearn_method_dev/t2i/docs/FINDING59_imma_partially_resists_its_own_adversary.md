# Finding 59 — ❌ **RETRACTED HEADLINE.** I published "IMMA partially resists its own adversary, max-over-valid H = 0.867, the first non-ceiling number in this phase". **My own ratchet refuted it within the hour: tripling the LoRA budget takes it to 1.000.** IMMA's max-over-valid H is **1.000** and it joins the other nine. What survives is an *attacker-cost* result, not a robustness one.

> ## ⚠ RETRACTION, same day
>
> `FINDING35` §4's ratchet rule says a sub-ceiling number is provisional until the adversary has
> been pushed. §5 of this finding flagged 0.867 as provisional and named the escalation. **The
> escalation ran and the number moved to the ceiling:**
>
> | adversary | budget | max-over-valid H |
> |---|---|---|
> | LoRA | 300 steps (as published in §2) | 0.867 |
> | **LoRA** | **900 steps — 3× the budget** | **1.000** |
> | full-UNet relearning | frozen grid, **8 / 8 valid** | **1.000** |
> | DreamBooth | 300 steps | **1.000** |
> | Textual Inversion | 2000 steps | 0.400 |
>
> **Arm `imma_church_frp_a1e-6_s1500`: 15 valid attacks, max-over-valid H = 1.000.**
>
> So "the first non-ceiling number in this phase" is **withdrawn**, and `PREREG_H82` prediction 1 —
> *"H strictly below 1.000 under LoRA"* — is **falsified by its own declared falsifier** (H = 1.000
> under LoRA). Prediction 2 (no transfer) stands and is now overwhelming: 8/8 full-UNet points at
> 1.000.
>
> **Why this happened, and it is not bad luck.** IMMA's inner loop approximates the adversary with a
> **single gradient step** (its own Implementation-details paragraph). The immunization is therefore
> tuned against a one-step attacker, and a 3×-longer attacker walks straight past it. The defence
> shapes the first step of the loss surface, not the basin.
>
> **This is the ratchet doing exactly the job it exists for**, on my own published claim, one hour
> after I published it. Everything below is left intact as the record of what the 300-step
> measurement showed; read §2 as "LoRA at 300 steps", not as IMMA's robustness.

---

## 0. What actually survives: an attacker-cost result

The 300-step numbers are real measurements, they are just not a robustness claim. Stated correctly:

| LoRA budget | per-seed target H | max-over-valid |
|---|---|---|
| 300 steps | 0.267, 0.467, 0.533, 0.600, 0.667, 0.800, 0.867 | 0.867 |
| 900 steps | 1.000, 1.000 | **1.000** |

**IMMA raises the attacker's cost; it does not cap the attacker's outcome.** At 300 steps the
adversary gets a partial, highly seed-dependent recovery — one seed got **nothing** (0.267 → 0.267).
At 900 steps it gets everything. That is the same shape as this phase's attacker-cost curves
(`FINDING37`, `FINDING52`): a defence can make recovery slower without making it bounded.

That is a weaker claim than the retracted one and it is the one the data supports.

**Date:** 2026-10-25 · **Pre-registration:** `PREREG_H82_imma_port.md`, predictions 1–3 recorded
before any arm existed
**Method:** IMMA, arXiv 2311.18815 (Zheng & Yeh, ECCV 2024) — **our port**. No code released.
**Evidence:** `outputs/atk_H82L_*.json` (primary, LoRA), `outputs/atk_H82F_*.json` (transfer,
full-UNet), `outputs/gen_imma_church_frp_a*.json`, `scripts/erase_imma.py`

---

## 1. Both pre-registered predictions were confirmed

`PREREG_H82` §6, written before any IMMA arm was built:

> **(1) Primary (LoRA):** *"IMMA will show a **real but partial** effect — max-over-valid H strictly
> below 1.000 under LoRA, and strictly above the mandate's 0.40... I expect the first non-ceiling
> number in this phase. Falsified by H = 1.000 under LoRA."*
>
> **(2) Secondary (transfer):** *"**no** transfer — full-UNet relearning reaches 1.000 regardless."*

**Both hold.** This is the first pre-registered positive prediction in this phase to survive its own
test, and it is worth being clear that the prediction was *modest*: partial resistance, nowhere near
the bar.

## 2. The primary measurement: LoRA, the adversary IMMA is immunized against

4 of 4 attacks passed the `FINDING20` gate (T rises +0.417 to +0.708):

| arm | retain8 as-is | seed | T rise | gate | target H | utility |
|---|---|---|---|---|---|---|
| α=1e-6, I=1500 | **0.8917** | 0 | +0.417 | ✅ | 0.267 → **0.267** | 0.938 → 0.979 |
| α=1e-6, I=1500 | 0.8917 | 1 | +0.625 | ✅ | 0.067 → **0.667** | 0.938 → 0.979 |
| α=1e-6, I=300 | 0.9417 | 0 | +0.708 | ✅ | 0.267 → **0.467** | 0.938 → 0.958 |
| α=1e-6, I=300 | 0.9417 | 1 | +0.708 | ✅ | 0.000 → **0.867** | 0.896 → 0.917 |

**max-over-valid H = 0.8667.**

Against the nine defences before it — every one of which reached **1.000** on its best valid
attack — this is a real difference. It is the only arm in the phase where a fully valid attack
family failed to restore the concept completely.

**⚠ And it is nowhere near the mandate's bar**, which is retain8 ≥ 0.85 **and** max-over-valid
H ≤ 0.40. The retain half is met (0.8917). The robustness half is missed by **more than double**:
0.867 against 0.40.

### 2a. The seed variance is enormous, and the max-over-valid rule is doing real work here

Seed 0 on the α=1e-6/I=1500 arm shows **H 0.267 → 0.267: literally zero recovery** under a valid
attack that raised its own training accuracy by +0.417. Seed 1 on the same arm, same budget, went
0.067 → 0.667.

**Reporting that zero-recovery seed as the result would be indefensible**, and it is exactly what
`FINDING35` §4's `max-over-valid` rule exists to prevent: the maximum over valid attacks is the only
number that cannot be gamed by seed selection, and it is **0.867**. The spread (0.267 to 0.867 on
one arm) is larger than the entire remaining distance to the bar, so **no claim here is stable at
n=2 per arm.** More seeds are the obvious requirement before this number is treated as settled.

## 3. The transfer question: no transfer, as predicted

| adversary | trainable params | arm | result |
|---|---|---|---|
| **LoRA** (immunized against) | 0.399M (0.046%) | α=1e-6/I=1500 | **0.867** max-over-valid |
| **full-UNet relearning** | **859.5M** | α=1e-6/I=1500 | **1.000** on the first valid point |

Immunizing against LoRA bought **nothing** against weight-space relearning: the first full-UNet
point took the same arm straight to the ceiling (H 0.267 → 1.000, T +0.792). The remaining grid
points are running and can only confirm or raise that, never lower it.

The mechanism is the one `FINDING58` §3a identified for GIFT: **S is 44.0M parameters, 5.11% of the
UNet.** An 859.5M-parameter adversary is not constrained by a defence that only shapes 5% of it.

## 4. What this does to the phase's conclusion — it sharpens it rather than overturning it

`PREREG_H82` §6 prediction 3 pre-committed to this reading:

> *"If (1) and (2) both hold, the phase's conclusion sharpens from 'every defence is reversed' to
> 'defences can resist the adversary they were trained against and none resists a stronger one'."*

That is now the accurate statement, and it is **more useful** than nine flat reversals:

- **Adversary-specific defence partially works.** IMMA trains against the real LoRA adversary,
  warm-started, inside its own loop — and LoRA then underperforms against it. Of the three bi-level
  defences tested (Meta-Unlearning, GIFT, IMMA), the one whose inner loop runs the *actual* attack is
  the only one that moved the number at all.
- **It does not generalise one step.** The same arm falls to a bigger adversary immediately.
- **So the phase's negative result is not "defences cannot work"** — it is **"defences work only
  against the exact adversary they were optimised against, and the threat model does not hold still."**

## 5. What is NOT claimed

- **The mandate is still unmet.** 0.867 ≫ 0.40. No arm in this phase, ours or published, has met the
  bar.
- **Not settled at this n.** Two seeds per arm, one concept (church), one teacher (`frp_church_r3_l05`).
  The 0.267–0.867 spread on a single arm makes more seeds mandatory, not optional.
- **The ratchet binds** (`FINDING35` §4, `PREREG_H82` K4): 0.867 is **below** the ceiling, so it is
  **provisional** until TI and DreamBooth have been pushed too, and until a stronger LoRA budget has
  been tried. Unlike GIFT — where H was already 1.000 and no further attack could raise it — here
  there is room above, so the obligation is live. Those attacks are the immediate next work.
- **α and I were chosen by us.** The paper assigns neither. At my pre-declared α=1e-5 the model was
  destroyed (retain8 0.0000) and at α=1e-7 the port is a **no-op** reproducing the FRP teacher
  exactly (0.1333 / 0.9417). The usable window is narrow and we found it by sweeping, so a reader
  should treat α=1e-6 as a tuned choice of ours, not a published setting.
- **Our port, not the authors' code**, and our ESD secondary arm was also destroyed at the original
  α; the surviving arms are all on the FRP teacher, not IMMA's own ESD setting.
