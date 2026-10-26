> ### ⚠ CORRECTION NOTICE (audit of `0b435427`, remediation r5)
> **Two corrections.** The FRP paper evaluates **retain, forget, and forget+retain** attacks (Appendix B.2) and varies optimiser and horizon (Appendix C.3), so §2's retain-only characterisation and the "direct target gradients" explanation are **withdrawn** (`T2I_BASELINES_v2.md` §5). Line 106's "no usable released code" over-generalises: it is correct for FRAG/FRP but false for IMMA, Meta-Unlearning and From Dormant to Deleted (`T2I_BASELINES_v2.md`).
>
> Original text below is **unaltered**. Register: `docs/CORRECTION_REGISTER_r5.md`.

---

# Finding 43 — FRP, a published method built for relearning robustness, is fully reversed. The prediction made before the attacks ran was correct.

**Date:** 2026-10-23 · **Pre-registration:** `PREREG_H66_frp_port.md`, prediction in §5a
**Method:** Forget-Retain Pruning, arXiv 2608.25429 — **our port**, labelled as such throughout.
**Protocol:** `PROTOCOL_FROZEN_v1`, the identical six-point grid `H65` measured the official
AdvUnlearn baselines under. Budget-matched by construction.

---

## 1. The result

| arm | ρ | target (erased) | retain8 | gate-passing | **max-over-valid H** | ungated |
|---|---|---|---|---|---|---|
| `frp_church_r3_l05` | **3 %** | **0.1333** | **0.9417** | **6 / 6** | **1.000** | 1.000 |
| `frp_parachute_r5_l05` | 5 % | 0.2000 | 0.7417 | 5 / 6 | **1.000** | 1.000 |
| *AdvUnlearn church (official, same grid)* | — | 0.400 | 0.992 | 6/6 | *1.000* | — |

Per-attack, church (the arm that matters — parachute's retain was already below the bar):

| seed | lr | T rise | target H | utility after |
|---|---|---|---|---|
| 0 | 1e-5 | +0.792 | 0.133 → **1.000** | 0.9167 |
| 1 | 1e-5 | +0.750 | 0.133 → **1.000** | 0.9167 |
| 6 | 1e-5 | +0.792 | 0.200 → **1.000** | 0.9792 |

**FRP at retain 0.9417 is restored to the base model's ceiling by 300 gradient steps on 16
self-generated images, with 92–98 % of retained capability intact.** Not one of the twelve attacks
left it below 1.000 on its best configuration.

**The first of the five competing published defences to be run at all behaves exactly like
everything else in this phase.** That makes the phase's negative result *stronger*: a method
explicitly designed for relearning robustness fails against this adversary too.

## 2. The prediction, and why it being right matters

`PREREG_H66` §5a was written after the ρ ladder was measured and **before a single FRP attack
existed**. It predicted full reversal, with a stated mechanism:

> *"FRP zeroes weights and nothing holds them at zero. It is unstructured pruning with no mask
> re-applied during any subsequent training... the mask is an initial condition, not a constraint."*

It also stated what would falsify it — H ≤ 0.40 at retain 0.9417 — and committed to reporting that
as the mandate's bar met by a published method. That did not happen. **H = 1.000, six attacks of
six.**

The threat-model difference is the whole explanation:

| | FRP's paper | this phase |
|---|---|---|
| adversary data | **retain** split, one epoch | **16 images of the target concept**, 300 steps |
| gradient toward the erased concept | incidental | **direct and explicit** |
| pruned weights re-trainable? | yes, nothing pushes them back | **yes, and the loss pushes them back** |

FRP's published robustness is consistent with its own setting and predicts little about this one.
**This is a scope statement about a different threat model, not a refutation of their paper** — the
same distinction `ERRATUM_r2` C15 insists on for AdvUnlearn.

> ⚠ **The mechanism above is REFUTED by the diagnostic it declared. See `FINDING44`.** The
> pre-declared check ran: the attack's update was **uniform** across pruned and surviving positions
> (concentration ratio **1.0169** — 3.020 % of update norm on 2.970 % of entries), per-entry RMS
> update identical (1.987e-04 vs 1.970e-04), and post-attack weight mass in formerly-pruned
> positions **0.00005 %**. The adversary did **not** refill the mask; it recovered the concept
> through the **surviving 97 %** of weights. The *outcome* prediction in §5a was right and the
> *mechanism* was wrong, and "the mask is an initial condition, not a constraint" is **withdrawn as
> an explanation**. `FINDING44` §3 gives the corrected reading, which is stronger: the concept is
> re-encodable in weights that FRP's forget-critical scoring declined to prune.

## 3. FRP's erasure/utility point, stated without inflation

At the paper's own ρ = 3 % with the pre-declared primary λ = 0.05, FRP reaches church **0.1333 at
retain 0.9417**, against official AdvUnlearn's **0.400 at 0.992**.

**This is not a strictly better point.** FRP buys 0.27 of erasure depth for roughly 5 points of
retain. Which is preferable depends on how the two axes are weighted, and calling it a win on the
trade-off without saying so would be the kind of claim this project has had to retract before. What
*is* true: it is a genuinely competitive operating point, reached by a published method, that erases
considerably deeper than the robust SOTA — **and it is just as completely reversed.**

Two further honest qualifications on the ladder:

- **The usable window is narrow.** Beyond ρ = 5 % the model is destroyed (retain 0.0750 at 10 %,
  0.0167 at 20 %), so church's "target 0.0000" at those rungs is the degenerate-image artifact, not
  erasure — the same artifact behind `H64`'s van_gogh 1.000 (`FINDING39` §1).
- **Target suppression is NON-MONOTONE in ρ** on both concepts: church 0.1333 → 0.2667 from 3 % to
  5 %; parachute 0.2000 → 0.4667 from 5 % to 10 %. "Prune more, erase more" is false here.

## 4. λ, the hyperparameter the paper never assigns

λ is never given a value anywhere in 2608.25429. `PREREG_H66` §3(c) pre-declared 0.05 as the primary
and committed to reporting the headline at λ = 0.05 **regardless of which λ performed best**. At
ρ = 3 % on church:

| λ | target | retain8 |
|---|---|---|
| 0.00 | 0.2000 | 0.9500 |
| **0.05 (primary, headline)** | **0.1333** | **0.9417** |
| 1.00 | 0.5333 | 0.9500 |

λ matters a great deal — a 4× spread in erasure depth — which makes its absence from the paper a
genuine reporting gap, and makes our choice of it a stated port decision rather than a reproduction.

## 5. What this does and does not close

**Closes:** the competitive-baseline gap by one. Four remain unrun (Meta-Unlearning 2410.12777,
IMMA 2311.18815, GIFT 2507.13598, From Dormant to Deleted 2505.22310), and **none of the five has
usable released code** (`ERRATUM_r2` C19).

**Does not close:** the mandate. No arm, ours or published, has reached retain ≥ 0.85 with
max-over-valid H ≤ 0.40. The count of methods reversed by this adversary is now **six** — UCE, ESD,
official AdvUnlearn, damage+distil, the `frag` allocation, and FRP.

**Does not establish:** that pruning-based erasure is inherently fragile in general. One port, one
model, two concepts, one attack family, and a λ we chose ourselves.

## 6. Scope

Our port from the paper, not the authors' code, which has not been released. MLP-only scope mapped
to SD's 32 feed-forward linears; our calibration construction; λ chosen by us. Any disagreement with
the authors' reported numbers is attributable to the port or the domain change, not to their method.
Parachute's selected arm sat at retain 0.7417 — below the 0.85 floor before any attack ran — so it
could not have met the bar regardless, and its reversal is reported for completeness rather than as
the test.
