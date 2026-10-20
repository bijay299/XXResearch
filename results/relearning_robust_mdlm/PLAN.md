# New-method sprint: relearning-robust unlearning

**Started 2026-10-20 ~10:05 ET. GPUs 2 and 3 only** (0 and 1 belong to another user).
Clean restart. **Nothing from the three discarded lines is reopened**: no SPEX
(student-as-own-reference), no ARC (reference caching / pooled states), no TD-SSD
(twin-difference `zmix`), no token banning. The mechanism here is unrelated to the
reference pass and to `zmix` algebra.

## Why a new axis at all

The last sprint's own numbers close off the obvious axis. On the 128-fact probe the
*existing* control (anchor-KL / SSD family, fresh states) reached:

```
forget disclosure 0.049    retain 1.000    degenerate outputs 0
```

That is nearly saturated. **There is no headroom left on single-shot disclosure for this
task**, so "a method that forgets more" cannot be demonstrated here and I will not pretend
otherwise. A new method needs an axis where the strong control *measurably fails*.

The failure I think is real, and that my own clean-forgetting work points straight at:
**forgetting is shallow**. A model at 0.049 disclosure may sit on a *steep slope* of the
adversary's objective — high forget loss, but large gradient — so a handful of fine-tuning
steps on a few leaked facts rolls it straight back, including facts the adversary never
supplied.

## Phase 0 — the kill gate (first, before any method exists)

This is the lesson from the three failures: **build the probe that can kill things before
inventing anything.** No method is written until the axis is proved real.

**Relearning-robustness probe.** Adversary gets **8 leaked forget facts** (of 64) and runs
plain masked-diffusion fine-tuning. Disclosure is measured on the **56 held-out forget
facts the adversary never supplied**, at k ∈ {0, 2, 5, 10, 20} steps.

Three starting points, identical adversary:

| start | meaning |
|---|---|
| θ₀ (memorized) | upper bound — the facts are fully present |
| **θ_u** (unlearned control, 0.049) | the thing under test |
| **θ_base** (never memorized) | **floor** — cannot recover facts it never knew |

The 64 forget facts are *independent* fictional entity→answer bindings, so the 8 leaked
facts carry **zero semantic information** about the held-out 56. Any recovery of the 56 is
therefore evidence of **residual knowledge in θ_u being unlocked**, not of learning.
θ_base is the control that makes that inference valid.

### Pre-registered gate (written before the run)

> **AXIS REAL** if θ_u's held-out disclosure at k=10 exceeds θ_base's at k=10 by
> **≥ 0.25**.
> **AXIS DEAD** otherwise → I pivot immediately to sequential/continual unlearning
> (unlearn request 1, then 2, then 3; measure collateral accumulation), same probe shape,
> and this document is amended with the negative result rather than quietly dropped.

Measured on the intersection of the held-out 56 with the 51 facts θ₀ actually recalled, so
every probe is a fact that was genuinely there to begin with.

## Phase 1 — theory, only if the axis survives

The first-order statement the method would rest on. Adversary descends the memorization
loss `L_mem`; from the unlearned point θ_u,

```
L_mem(θ_u − η∇L_mem(θ_u)) = L_mem(θ_u) − η‖∇L_mem(θ_u)‖² + O(η²)
```

so **initial relearning speed is governed by ‖∇L_mem(θ_u)‖²**, not by the value of
`L_mem(θ_u)`. Standard unlearning maximises the *value* and says nothing about the
*gradient* — it lands on a steep slope. The target is instead a point that is **high and
stationary**: a local maximum or saddle of the adversary's landscape.

To be derived and machine-checked in `theory/THEORY.md` (and, per the standing rule, any
derivation I get wrong stays in the file alongside the correction):
- **L1** the first-order relearning-rate identity above, with its remainder bound
- **L2** conditions under which a high-loss stationary point exists at all without
  collapsing the retain distribution (this is where it could fail)
- **L3** whether the finite-difference surrogate is a valid approximation of the
  gradient-norm penalty at the step sizes used, and at what cost

A literature check runs in parallel so I do not re-derive something standard and call it
new. **No novelty claim either way.**

## Phase 2 — candidate method

```
min_θ   −L_mem(θ)  +  λ_R · L_retain(θ)  +  λ_G · ‖∇_θ L_mem(θ)‖²
```

The third term needs a Hessian-vector product (double backward) or a finite-difference
ascent step. **Cost will be measured, never assumed**, and the smoke gate aborts if the
gradient-norm term is not actually reducing ‖∇L_mem‖ relative to the control.

## Phase 3 — head-to-head

Candidate vs the same strong control on **both** axes: single-shot disclosure/retention
(where the control is at 0.049/1.000 and the candidate must not regress) **and** relearning
robustness (the axis the control is predicted to fail). 2 seeds.

### Pre-registered verdict rules

- **SUPPORTED** — candidate's held-out disclosure at k=10 is ≥ 0.20 *lower* than the
  control's, **and** its single-shot disclosure is not worse by > 0.05, **and** retain not
  worse by > 0.05.
- **COST-ONLY / NOT SUPPORTED** — otherwise, reported as such with the numbers.

A candidate that wins by output collapse does not count: **clean forgetting** (non-degenerate
*and* non-disclosing) is reported for every arm, and a candidate whose win comes with
degenerate outputs is recorded as a failure.

## Standing constraints

Synthetic data only; no new model-weight downloads; no paid APIs; GPUs 2–3 only; other
users' processes untouched; all prior published work left unchanged; no claim of certified
deletion, benchmark superiority, or established novelty.
