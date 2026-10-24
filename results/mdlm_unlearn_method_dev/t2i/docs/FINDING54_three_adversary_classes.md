# Finding 54 — three adversary classes spanning 859.5 M → 768 trainable parameters, three defence families: erasure is reversed in every combination. 768 scalars take the robust SOTA to 1.000.

**Date:** 2026-10-24 · **Adversaries:** full-UNet fine-tuning, LoRA, Textual Inversion
**Arms:** official AdvUnlearn (text-encoder erasure), FRP port (UNet pruning), damage+distil
(UNet noise + distillation) · **Protocol:** `PROTOCOL_FROZEN_v1` T/H split, `FINDING20` gate
**Evidence:** `outputs/atk_H71*.json`, `outputs/atk_H72*.json`

---

## 1. The comparison

| adversary | trainable params | % of UNet | AdvUnlearn | damage+distil | FRP |
|---|---|---|---|---|---|
| full-UNet fine-tuning | 859 520 000 | 100 % | **1.000** | **1.000** | **1.000** |
| LoRA rank 16 | 399 000 | 0.046 % | 0.867 / **1.000** † | **1.000** | **1.000** |
| **Textual Inversion** | **768** | **0.000089 %** | **1.000** | **0.933** | **0.933** ‡ |

Max-over-valid H, church unless marked. † church / parachute. ‡ church; parachute reads 0.667 with
3 of 8 valid and is being pushed further (§4).

> **Across three adversary classes spanning six orders of magnitude in trainable parameters, and
> three structurally different defences, erasure is substantially reversed in every combination.**

**The smallest adversary is the most effective against the strongest defence.** Textual Inversion
trains **one 768-dimensional token embedding**, changes **no weight at all**, and takes the official
robust SOTA to **1.000 on both concepts** (5 of 5 valid attacks). AdvUnlearn is adversarially trained
in *prompt space* and claims robustness there; a learned token is a prompt-space attack.

**Caveat on comparability, stated because it is easy to miss:** TI produces a *new token*, so its H
is measured on prompts containing that token rather than the concept word. The T/H structure and the
validity gate carry over, but **a TI H is not interchangeable with a weight-space H.** It answers a
different question — can the erased concept be re-addressed by a new token without touching a
weight — and the answer for text-encoder erasure is yes, completely.

## 2. ⚠ Two explanations of mine that this work refuted

**(a) "The ranking inverts with the adversary."** At 2 valid TI attacks it looked like FRP and
AdvUnlearn swapped places between LoRA and TI. The LoRA direction turned out to be a two-image
difference on church only (0.867 vs 1.000) and nothing on parachute — too small to carry a ranking.
Withdrawn.

**(b) "The erasure SITE predicts vulnerability to embedding-space attacks."** The prediction was
specific and testable: damage+distil erases in the UNet, so it should resist TI as FRP appeared to.
**It does not — 0.933, 3 of 3 valid.** The site explanation is refuted by its own test.

**(c) And FRP's apparent partial resistance was an attack-strength artifact.** It read 0.667 on both
concepts across 3 valid attacks each — the only partial resistance measured anywhere in this phase.
A stronger TI (3000 steps, lr 1e-2) took FRP-church to **0.933 with the gate passing**.

All three were caught by controls queued before publication, so nothing wrong entered the record.
But the pattern is the same every time and it is worth naming: **`max-over-valid` rises with attack
strength, so every sub-ceiling number is provisional until the adversary has been pushed.** This is
`FINDING35` §4's ratchet, and I have now walked into it six times in one week on my own
observations. The discipline that catches it is the only reason the record is clean.

## 3. What this adds beyond the earlier negatives

The phase's previous results used a **full-UNet** adversary — a larger attack surface than the
protection literature targets. IMMA (2311.18815) immunises against LoRA, Textual Inversion and
DreamBooth; Glaze and MIST poison data against the same family. Those results were therefore
repeatedly about a bigger adversary than those papers claim to stop (`ERRATUM_r2` C15,
`FINDING43` §2).

**That gap is now closed from the attack side.** Two of IMMA's three target methods have been run
directly against our erased arms, and both defeat erasure. This is a **stronger** negative than the
full-fine-tuning version, and it was obtained for a fraction of what porting IMMA would have cost —
a bi-level optimisation whose threat model is narrower than our adversary and which would most
likely have produced a third threat-model-mismatch result.

**It does not meet the mandate.** No arm reaches retain ≥ 0.85 with max-over-valid H ≤ 0.40 under any
of the three adversaries.

## 4. Scope and what is still open

- **DreamBooth**, IMMA's third target, is not run.
- **FRP-parachute** is the one remaining sub-ceiling figure (0.667, 3 of 8 valid). Stronger TI
  configs are **running**; given (c) above the honest expectation is that it breaks too, and it will
  be reported either way.
- Two concepts; one erased-arm instance per defence family; LoRA restricted to attention projections;
  TI to a single placeholder token.
- `Meta-Unlearning` (2410.12777), `GIFT` (2507.13598) and `From Dormant to Deleted` (2505.22310)
  remain unrun, none with usable released code. That is still the binding constraint on any
  contribution claim.
