# Finding 57 — the decisive re-zero test is done: putting FRP's pruning mask *back* into the attacked model removes nothing. The matched random control shows this is about **redundancy, not about FRP's mask**.

**Date:** 2026-10-24 · **Closes:** the follow-up `FINDING44` §4 recorded as *planned, not done*
**Evidence:** `outputs/rezero_church_r3.json`, `outputs/rezeroctl_church_r3.json`,
`outputs/rezero_parachute_r5.json` (the builds) and `outputs/gen_REZERO_frp_church_r3.json`,
`outputs/gen_REZEROCTL_frp_church_r3.json`, `outputs/gen_REZERO_frp_parachute_r5.json` (the
evaluations)
**Status correction:** `FINDING44` §4 says this test *"is recorded here as planned, not done."*
**That sentence is now stale** — the test ran, including a control `FINDING44` did not describe.

---

## 1. The test

`FINDING44` established, from a weight-space accounting identity, that the relearning adversary did
**not** refill FRP's pruned positions: the update was uniform (concentration ratio 1.0169), and
post-attack weight mass in formerly-pruned positions was 0.00005 %. §4 named the one measurement
that would turn that accounting argument into a direct one:

> *"re-zero the pruned positions in the attacked checkpoint and re-measure the target. If the
> concept survives that, the surviving weights alone carry it and the conclusion is direct."*

That is exactly what was run — plus a **matched random control** that brackets it: re-zero the same
*number* of feed-forward entries at *random* positions instead of FRP's chosen ones.

## 2. The result

| checkpoint | positions re-zeroed | **church (target)** | retain8 | utility (11) |
|---|---|---|---|---|
| attacked, **FRP's own mask re-applied** | 4 415 427 | **1.000** | 0.9667 | 0.9758 |
| attacked, **matched random mask** (seed 1234) | 4 415 427 | **1.000** | 0.9667 | 0.9758 |
| attacked, **FRP's mask re-applied** (parachute, ρ=5 %) | 7 434 307 | **1.000** | 0.9583 | 0.9636 |

(Distinct checkpoints and distinct runs — `md5sum` differs, and CLIP alignment differs at the fourth
decimal, 0.25805 vs 0.25895. This is **not** the label collision that produced bit-identical rows
earlier in this phase; that failure mode was checked for explicitly.)

**The concept survives re-zeroing at full strength, with retained capability essentially untouched.**
`FINDING44` §3's corrected mechanism is now **directly confirmed rather than inferred**: after the
attack, the surviving 97 % of feed-forward weights carry the concept on their own. FRP's pruning is,
post-attack, **functionally irrelevant** — you can put the entire mask back and lose nothing, on
either concept.

## 3. ⚠ But the control narrows the claim, and it narrows my own framing

`FINDING44` §3 phrased the mechanism as the concept being *"re-encodable in weights that the
forget-critical scoring did not select"* — a sentence whose force comes from an implied contrast
with FRP's **scoring**.

**The random control removes that contrast.** Re-zeroing 4 415 427 *arbitrary* feed-forward entries
is equally harmless — identical target, identical retain, to three decimals. So this measurement
supports:

- ✅ the concept is recoverable from the weights FRP left alone — **confirmed**;
- ❌ anything specific about FRP's mask, its forget-critical ranking, or its choice of positions —
  **not supported**. Any 3 % of feed-forward weights is this redundant at this scale.

The honest statement is the weaker one: **this is a redundancy result about SD-1.4's feed-forward
layers, not a critique of FRP's scoring.** I am recording that explicitly because the stronger
phrasing was mine, it was more interesting, and the control I ran does not support it. `FINDING44`
§3 should be read against this section.

## 4. The timing is the substantive point

The mask was **not** always irrelevant. Before any attack, FRP at ρ = 3 % drove church to **0.1333**
from the base model's 1.000 — the pruning did real work at erasure time. After 300 gradient steps on
16 images, that same mask removes **nothing**.

So the finding is not "pruning never mattered". It is that **the adversary does not restore the
pruned route; it relocates the function**, leaving the defence's intervention intact and inert. The
erasure is undone without being reversed.

**The control that would complete this and was not run:** a random-mask ρ = 3 % prune of the *clean*
model. Pre-attack, FRP's selection plainly mattered (0.1333 vs 1.000 for the unpruned model), but
without a random-prune arm at matched sparsity on the clean model I cannot quantify *how much* of
that erasure came from FRP's ranking versus from removing 3 % of feed-forward capacity at all. That
is a real gap and is recorded as not done rather than glossed.

## 5. Bearing on `ERRATUM_r2` C17, which still stands

C17 — no never-trained oracle, so *reactivation of residual memory* and *relearning from the
attacker's data* are not distinguishable — is the phase's deepest open limitation. This result adds
to the evidence `FINDING44` §3 began: the concept returns through positions the defence never
touched, and re-closing the removed route costs nothing. That is more consistent with relearning
than with reactivation along the pruned path.

**It does not resolve C17.** A redundant encoding already latent in the surviving weights would
produce exactly this signature too, and nothing here separates "latent and re-exposed" from "learned
afresh". C17 stands, with a second piece of mechanistic evidence attached.

## 6. Scope

Two arms (`frp_church_r3_l05`, `frp_parachute_r5_l05`), one attack configuration each (the
highest-recovery point: church seed 0 / lr 1e-5, parachute seed 1 / lr 1e-5), one random control on
church only, one random seed for that control. The re-zero is applied to FRP's scope — the 32
feed-forward linears — and says nothing about the other 160 transformer-block linears. The CLIP
evaluation classifier's revision is unpinned, as everywhere in this line.
