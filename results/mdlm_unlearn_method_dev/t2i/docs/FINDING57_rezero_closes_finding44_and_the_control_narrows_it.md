> ### ⚠ CORRECTION NOTICE (audit of `0b435427`, remediation r5)
> The three interventions and their numbers stand. **Narrowed:** they establish final-coordinate dispensability for the **final evaluated model on this panel** — not dispensability during the attack trajectory, not localisation to surviving feed-forward weights, and not a claim about every 3% mask (one seed, count-matched but not magnitude-matched). See `ERRATUM_interpretations_r5.md` §2. The persistent-mask intervention that addresses the remaining alternatives is experiment **E5**, not run.
>
> Original text below is **unaltered**. Register: `docs/CORRECTION_REGISTER_r5.md`.

---

# Finding 57 — FRP's pruning mask is **decisive before the attack and irrelevant after it**. Putting the mask back into the attacked model removes nothing; a random mask of the same size erases nothing on the clean model. The adversary does not just undo the erasure — it makes the defence's weight selection stop mattering.

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

## 3. ⚠ The *post-attack* random control narrows the claim, and it narrows my own framing

(Read with §4: the random control on the **clean** model, run later, shows the mask was decisive
*before* the attack. This section concerns only the **post-attack** comparison.)

`FINDING44` §3 phrased the mechanism as the concept being *"re-encodable in weights that the
forget-critical scoring did not select"* — a sentence whose force comes from an implied contrast
with FRP's **scoring**.

**The random control removes that contrast.** Re-zeroing 4 415 427 *arbitrary* feed-forward entries
is equally harmless — identical target, identical retain, to three decimals. So this measurement
supports:

- ✅ the concept is recoverable from the weights FRP left alone — **confirmed**;
- ❌ anything specific about FRP's mask, its forget-critical ranking, or its choice of positions —
  **not supported**. Any 3 % of feed-forward weights is this redundant at this scale.

The honest statement, **scoped to the post-attack model**, is the weaker one: once the attack has
run, this is a redundancy result about SD-1.4's feed-forward layers, not a critique of FRP's
scoring. I am recording that explicitly because the stronger phrasing was mine, it was more
interesting, and this control does not support it. `FINDING44` §3 should be read against this
section.

⚠ **Do not generalise this sentence past the attacked model.** §4's clean-model control shows the
opposite holds pre-attack: a random mask erases *nothing* (target 1.0000) where FRP's erases to
0.1333. "Any 3 % is equally redundant" is true of the **attacked** network and false of the
**clean** one.

## 4. The timing is the substantive point

The mask was **not** always irrelevant. Before any attack, FRP at ρ = 3 % drove church to **0.1333**
from the base model's 1.000 — the pruning did real work at erasure time. After 300 gradient steps on
16 images, that same mask removes **nothing**.

So the finding is not "pruning never mattered". It is that **the adversary does not restore the
pruned route; it relocates the function**, leaving the defence's intervention intact and inert. The
erasure is undone without being reversed.

**The control that would complete this** — a random-mask ρ = 3 % prune of the *clean* model — **has
now run, and it is decisive.**

| arm | target (church/parachute) | retain8 | utility |
|---|---|---|---|
| **random mask, ρ=3 %, clean model, seed 0** | **1.0000** | 1.0000 | 1.0000 |
| **random mask, ρ=3 %, clean model, seed 1** | **1.0000** | 1.0000 | 1.0000 |
| **random mask, ρ=5 %, parachute, clean** | **1.0000** | 1.0000 | 1.0000 |
| FRP's own mask, same sparsity (`FINDING43`) | **0.1333** | 0.9417 | — |
| untouched base model | 1.000 | 1.000 | 1.000 |

**Removing a random 3 % of the feed-forward weights does nothing whatsoever** — the concept is fully
intact, retention and utility are untouched, the model is indistinguishable from base on every
measured axis. FRP's mask at identical sparsity drives church to **0.1333**.

**So FRP's forget-critical ranking does essentially all of the work.** The erasure is not an artifact
of removing capacity; it is attributable to *which* weights are chosen. This is the clean version of
the claim §3 above had to withdraw, and it sharpens the finding rather than softening it:

| | does the choice of mask matter? |
|---|---|
| **before the attack** | **Decisively.** FRP's mask → 0.1333; a random mask of the same size → 1.0000. |
| **after 300 attack steps** | **Not at all.** Re-zeroing FRP's mask → 1.000; re-zeroing a random mask → 1.000. |

**The adversary does not merely undo the erasure — it renders the defence's weight selection
irrelevant.** Which 3 % you remove is the entire ballgame beforehand and makes no measurable
difference afterwards. That is a stronger and more precise statement than either §2 or §3 alone, and
it is the mechanistic core of this phase's negative result.

⚠ What §3's narrowing still holds onto: *post-attack*, FRP's positions are no more special than
random ones. The random control does not rescue the §3 phrasing about scoring; it establishes that
the scoring mattered **at erasure time** and stopped mattering **after the attack**. Those are two
different claims and both are now measured.

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
