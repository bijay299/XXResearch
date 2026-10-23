# Finding 39 — ❌ H64 is FALSE: the forget-critical × retain-sparing allocation collapses utility, exactly as pre-declared

**Date:** 2026-10-23 · **Pre-registration:** `PREREG_H64_frag_allocation.md` · **Recipe:** pools v2
**Status:** **K2 fires on both concepts.** Not a candidate regardless of any robustness number.

---

## 1. The result

`undo_t2i.py --protect frag`, per-parameter damage scale ∝ `normalise(F_forget) × (1 −
normalise(F_retain))`, α=0.3, β=0.1, AdvUnlearn teacher, 6000 distillation steps, pools recipe v2:

| arm | target suppression | **retain8** | utility_all_nontarget (n=11) | K2 (≥ 0.60) |
|---|---|---|---|---|
| `h64v2_frag_a0.3_church` | 0.000 | **0.0000** | 0.0909 | ❌ **FAIL** |
| `h64v2_frag_a0.3_parachute` | 0.067 | **0.0833** | 0.1455 | ❌ **FAIL** |

Per-concept, the church arm generates **nothing recognisable at all**: all twelve concepts at 0.000
except `van_gogh` at 1.000. The parachute arm is the same picture with `van_gogh` 0.867 and `monet`
0.400.

**That apparent "survival" of the two painting classes is an artifact, not retained capability.** A
destroyed model emits painterly noise, and a 12-way closed-set CLIP argmax has to put it somewhere —
"van gogh painting" and "monet painting" are where degenerate images land. The same artifact is
already on record (`FINDING35` §5: the α=0.7 arm generating nothing showed a *smaller* CLIP-align
drop than a working erasure). It is why `utility_all_nontarget` is 0.09–0.15 while `retain8` reads
0.00–0.08: different denominators over the same ruined model.

## 2. This is the outcome the pre-registration declared, by the route it declared

`PREREG_H64` §3, written before the arm existed:

> | retain < 0.60 | inherits `forget_fisher`'s collapse — the retain-sparing factor is insufficient
> to protect the overlapping weights |

That is what happened. **K2** — *"retain8 < 0.60 ⇒ not a candidate regardless of robustness"* —
fires on both concepts, so **K3 (replication) and the robustness bar are moot**: there is no
candidate to replicate.

The mechanism reading the pre-registration offered is the one supported: the **forget-critical factor
dominates and the retain-sparing factor does not rescue it.** Multiplying by `(1 − F_retain)` is not
enough to spare the retain-critical weights, because the weights that most encode the concept are
largely the same ones general generation needs — `PREREG_H51` §7's conclusion, now reached from the
intersection rather than from the forget-only half.

## 3. ⚠ A confound I must not paper over: this is NOT a clean comparison to H51 or H47

| arm | allocation | pool recipe | retain8 |
|---|---|---|---|
| `h47_adv_a0.3` | `retain_fisher` | **v1** | **0.8667** |
| `h51_ff_a0.3` | `forget_fisher` | **v1** | **0.0917** |
| `h64v2_frag_a0.3_church` | **`frag`** | **v2** | **0.0000** |

**H64 differs from both in two variables at once** — the allocation *and* the pool recipe. v1 pools
contained the evaluation strings verbatim (`FINDING38` §2), so `h47`'s 0.8667 was measured partly
**in-sample** while H64's 0.0000 is measured fully out-of-sample. I therefore **cannot attribute
H64's collapse to the `frag` allocation** on this evidence. Three readings remain open:

1. the `frag` allocation genuinely destroys utility (the pre-declared reading);
2. the v2 pool change removed an in-sample inflation that was holding `retain_fisher`'s number up,
   and *every* damage+distil arm is worse than published once measured out-of-sample;
3. both.

This is exactly the error pattern `ERRATUM_r2` C8 corrects in our own earlier work — attributing a
utility change to one variable when the teacher, the pool and the construction all moved together.
I am not going to repeat it one document later.

## 4. The decisive control, queued

> **A matched v2 rebuild of `retain_fisher`** at the same α, teacher, step count and pools.
>
> - if it lands near **0.85**, the v2 pools are innocent and the **`frag` allocation** is the cause
>   — reading (1), and H64 is cleanly false;
> - if it also **collapses**, the **pool repair** is the cause — reading (2), and a much larger
>   correction follows: every published damage+distil retain number in this phase is an in-sample
>   artifact, and `FINDING38` §3's withdrawal of the cross-family ranking would need to extend to
>   the within-family α-ladder as well.

Reading (2) would be the more consequential outcome and it is the one that would cost me more, which
is a reason to run the control rather than to leave the attribution comfortable. Queued behind
`H66` (the competing-baseline gap binds every claim and takes priority), at ~35 GPU-minutes.

## 5. The attacks are still running, and why that is not wasted

K2 has already disqualified the arm, so the 18 attacks (9 per concept) cannot change the verdict.
They are allowed to finish for two reasons: the standing instruction not to interrupt running jobs,
and because issue 6 requires the **gated and ungated** record on every arm. `H51`'s analogous run
produced **0 of 3 gate-passing** — "the model is too broken to attack" is itself a recorded
observation about damaged arms, and the same measurement on `frag` completes the ladder.

Any robustness number that emerges will be reported **as inadmissible under K2**, never as evidence
that the arm resists.

## 6. What this retires

`FINDING35` §3 diagnosed that resistance appears only in damaged arms at low utility, leaving "make
the damage selective" as the single untried response. That lever is now exhausted in all three
forms:

| allocation | targets forget-critical | spares retain-critical | retain8 | verdict |
|---|---|---|---|---|
| `retain_fisher` | ✗ | ✓ | 0.8667 (v1, in-sample) | utility fine, **no robustness** |
| `forget_fisher` | ✓ | ✗ | 0.0917 (v1) | **destroyed** |
| `frag` | ✓ | ✓ | 0.0000 (v2) | **destroyed** |

Selectivity, implemented three ways, does not produce robustness at usable utility in this model.
UNDO's own ablation anticipated it — *"changes in mixing coefficient (α) produce a more pronounced
effect on robustness than the noise scale (β)"* — i.e. the noise **distribution** is not where the
leverage lives. The published principle from 2608.25429 holds as a *predictor* claim about LLM
pruning updates; **as a noise-allocation rule inside perturb-then-distil on a diffusion model it
does not transfer**, and that is the honest scope of this negative.

Whether FRP — the same principle as *pruning*, which is how its authors instantiate it — transfers
is a separate question, pre-registered in `PREREG_H66` and running.
