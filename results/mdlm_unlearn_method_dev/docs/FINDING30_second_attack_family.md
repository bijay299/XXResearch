# Finding 30 — the advantage replicates under a second attack family, but the gate exposes a trainability confound

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM
**Protocol:** RTT, author-disjoint T/H, `t_frac 0.8`, `FINDING20` validity gate, full lr grid
{1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, recovery as max-over-**valid** attacks
**Status:** a positive replication and a new, unresolved threat to the central claim. Both reported.

---

## 1. Why a second family was needed

Every robustness number in this project — all 125 arms of `FINDING28`, all 37 gated runs of
`FINDING25` — comes from **one** attack: SFT on the answer span. If the protection is specific to
that optimisation surface, the central claim is an artifact of the attack, not a property of the
method. `FINDING25` §4 lists "one attack family" as a limitation; this tests it.

**The second family** (`train_small_rtt.py --attack-mode diffusion`): the attacker optimises the
**MDLM pretraining objective over whole passages** at a random mask rate, instead of supervised
cross-entropy on the answer span. Different loss, different masked positions, different surface —
the same T/H split, budget and gate.

## 2. The measurement

| arm | lr | T_rise (own training set) | gate | recovery on H |
|---|---|---|---|---|
| MDU λ=0.5 | 1e-5 | 0.284 | ✓ | +0.5917 |
| MDU λ=0.5 | **3e-5** | 0.369 | ✓ | **+0.6509** |
| MDU λ=0.5 | 1e-4 | 0.428 | ✓ | +0.5030 |
| MDU λ=0.5 | 3e-4 | 0.400 | ✓ | −0.1479 |
| MDU λ=0.5 | 1e-3 | 0.244 | ✓ | −0.2071 |
| MDU λ=0.5 | 3e-3 | 0.019 | ✗ | *(ungated)* |
| **selected point** | 1e-5 | 0.106 | ✗ | *(ungated)* |
| **selected point** | 3e-5 | 0.169 | ✗ | *(ungated)* |
| **selected point** | **1e-4** | 0.234 | ✓ | **+0.0862** |
| **selected point** | 3e-4 | 0.191 | ✗ | *(ungated)* |
| **selected point** | 1e-3 | 0.178 | ✗ | *(ungated)* |
| **selected point** | 3e-3 | 0.044 | ✗ | *(ungated)* |

> **The advantage replicates across attack families.** Max-over-valid recovery: **+0.6509** for the
> MDU baseline vs **+0.0862** for the selected point — a **7.6×** separation, against 3.2× for
> span-SFT at the same budget. The protection is not specific to the span-SFT surface.

## 3. The confound this exposed, which is more important than the replication

**The baseline admits 5 of 6 attacks; the selected point admits 1 of 6.** Not because its recovery
was low — because the attacker could not relearn **its own training set** on it. T_rise peaks at
**0.234** on the treated arm against **0.428** on the baseline.

That is consistent with two very different explanations:

| explanation | implication |
|---|---|
| **(a) fact-specific** — the forget facts are genuinely harder to resurface in the damaged model | robustness, as claimed |
| **(b) general trainability loss** — the damaged model is harder to fine-tune on *anything*, so every relearning attack is handicapped on it | the "robustness" is partly an artifact of a degraded optimisation landscape, and a sufficiently patient or better-tuned attacker closes it |

**I cannot currently distinguish these, and the project's headline robustness claim depends on
which is true.** This is the same failure shape as the three adversary-strength weakenings already
on record this session (α=0.5 "fully robust" → +0.057 under lr search; 18× → 3.2× at a matched
budget): a defence that looks strong because the adversary was weak.

`FINDING20`'s gate was designed to catch exactly this and it did — it rejected 5 of 6 attacks on
the treated arm rather than letting a flat H curve read as success. But the gate only **flags** the
problem; it does not resolve which explanation holds, and it leaves the treated side of the
headline resting on **n=1 valid attack**.

## 4. The deciding experiment, specified but not yet run

Train the attacker on **retain** facts instead of forget facts, same budget and lr grid, and
compare how fast each arm learns them:

- if the treated arm learns retain facts **as fast as** the baseline → the slowdown is
  **fact-specific**, explanation (a), and the robustness claim stands
- if the treated arm is **uniformly slower** → explanation (b), and every robustness number in this
  project is inflated by an unknown amount

This is a **trainability control**: a per-arm measurement of general fine-tuning capacity, used to
normalise attack strength. It is the natural extension of the validity gate (see
`NOVELTY_attack_validity_gate.md`) — the gate checks that the attack works on the material it was
given; the control checks that the *arm* is attackable at all. It is queued behind `H37`.

## 5. Scope

- One backbone, one corpus, two attack families, one arm pair for this comparison
  (selected point vs MDU λ=0.5), seed 0 only.
- Both families are *relearning* attacks. Neither is a probing, latent-space or
  prompt-based extraction attack, so "two families" is not "attack-agnostic".
- The negative recoveries at high lr (down to −0.21) are the attack destroying the model rather
  than recovering anything; they pass the gate on T_rise but are not meaningful recoveries. Taking
  the **max** over valid attacks handles this correctly, and no claim rests on the negative values.
- `t_frac 0.8` only. The span-SFT comparison number (3.2×) is at the same budget, so the
  cross-family comparison is budget-matched.
