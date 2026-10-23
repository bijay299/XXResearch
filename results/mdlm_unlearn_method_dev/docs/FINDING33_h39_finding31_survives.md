# Finding 33 — FINDING31 survives the trainability control: in the MDLM setting the robustness is fact-specific

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM · **Pre-registered:** `FINDING30` §4, condition
declared in `PREREG_H40` §3
**Status:** the control that `FINDING32` said was the project's most important open item, run on the
MDLM side. It **clears** my strongest result, and it does so against a pre-declared condition
rather than a retrofitted one.

---

## 1. The question, and the answer

`FINDING32` showed that in T2I, damage+distil's apparent robustness **is** general trainability
loss. `FINDING31`'s band separation — the project's strongest claim — showed the same surface
signature. If the same explanation held here, `FINDING31` would be an artifact.

The measure is ceiling-free: a **frozen fixed-mask probe** on the attacker's own training rows,
evaluated before and after the attack. Frozen because a raw per-step attack loss here is dominated
by the random mask-rate draw — the trap that already forced one retraction in this project.

Reported as the **fraction of available loss the attacker removes**, because the arms start from
different loss levels and a raw "fit drop" would therefore be misleading:

| arm | α | probe before | probe after | **% of loss removed** | max gated recovery | gate |
|---|---|---|---|---|---|---|
| suppression, MDU λ=0.5 | — | 2.1062 | 0.0270 | **98.72%** | +0.5608 | 2/2 |
| damage+distil | 0.0 | 1.8957 | 0.0204 | **98.92%** | +0.4352 | 2/2 |
| damage+distil | 0.3 | 1.6856 | 0.0274 | **98.37%** | +0.2541 | 2/2 |
| damage+distil | 0.5 | 1.5042 | 0.0372 | **97.53%** | −0.0106 | 2/2 |
| damage+distil | 0.6 | 1.3377 | 0.0855 | **93.61%** | **+0.0000** | 2/2 |

> **The attacker fits every arm to near-zero loss on its own data — 93.6% to 98.9% of the available
> loss removed — and every one of the 10 attacks passes the validity gate. Yet recovery on
> held-out facts collapses from +0.5608 to +0.0000 across the same ladder.**

A 5-percentage-point difference in fit capacity cannot explain a recovery collapse from 0.56 to
0.00. **The robustness is fact-specific.** `FINDING31` stands.

`PREREG_H40` §3 declared this exact condition in advance: *"if the MDLM `H39` control shows
damage+distil arms fit the attacker's data as fast as suppression arms → the T2I confound does not
generalise; `FINDING31` is clean."* It does, so it is.

## 2. What is not being overclaimed

- **There IS a mild trainability effect**, and it is monotone: 98.72% → 93.61% as α goes 0 → 0.6.
  The direction matches `FINDING32` exactly. The claim is that its *magnitude* is far too small to
  account for the robustness, not that it is zero.
- **α=0.5 and α=0.6 reach recovery ≈ 0** (−0.0106 and 0.0000) at 3,000 repair steps. That is a
  stronger result than the 12,000-step arms in `FINDING31` showed, which is consistent but was not
  predicted and rests on 2 attacks per arm.
- 5 arms, 2 attack learning rates each, seed 0, one attack family, one budget (`t_frac` 0.5). The
  `H37` replication is the broader evidence; this is a targeted control.
- The probe measures fit on the attacker's **objective**. A different adversary could face a
  different landscape.

## 3. Why MDLM and T2I came out opposite — the operative variable

The two phases disagree, and the disagreement has a concrete, measurable cause: **repair capacity
relative to the damage inflicted.**

| | MDLM phase | T2I phase |
|---|---|---|
| model | 124M | **859M** UNet |
| repair budget | 3,000–12,000 steps | 1,500–6,000 steps |
| distillation pool | ~1,400 prompts | **48–144 images** |
| retain after repair at the working α | **0.63–0.71** | **0.09–0.32** |
| attacker can fit the arm? | **yes, 93.6–98.9%** | no — 2/3 to 3/3 attacks fail the gate |
| robustness | fact-specific (this finding) | confounded with brokenness (`FINDING32`) |

> **Damage+distil appears to work when repair can actually restore the model, and to degenerate
> into "break the model" when it cannot.** In the MDLM phase repair restored utility to 0.63–0.71
> and the arms stayed fully trainable; in T2I the same procedure left retain at 0.09–0.32 and the
> arms became hard to fit. The mechanism did not change — the repair budget relative to model size
> and pool diversity did.

That is a testable prediction rather than a post-hoc story: **a T2I arm given enough repair to
restore utility should also become trainable again and show fact-specific robustness.** The 4×
budget run moved in that direction on every axis at once — retain 0.092 → 0.317 and late attack
loss 0.4397 → 0.4176 — but did not get far enough to test the prediction, and `PREREG_H38` K2
forbids escalating budget indefinitely to chase it.

## 4. What changes

| document | change |
|---|---|
| `FINDING31` | **qualification lifted.** Its separation may be read as fact-specific robustness on this backbone, not merely "resists this attack". |
| `FINDING32` §5 | the MDLM half of its warning is **resolved**; its T2I findings are unaffected |
| `PREREG_H40` | **scoped to T2I**, as its own §3 specified. Its mechanism is wrong about the LLM case. |
| `FINDING23` | the α=0 control giving ~0.12 (not zero) is reconfirmed here: α=0 recovers 0.4352 vs suppression's 0.5608 |

## 5. Note on the record

The frozen probe printed its values to the run logs correctly, but the field did not reach the
result JSONs — my string-insertion anchor for the JSON did not match. The numbers above are parsed
from the logs, which are the primary record and are published alongside. The JSON writer is fixed
for subsequent runs. Flagging it because a measurement that exists only in a log is easier to lose
than one in a structured result file.
