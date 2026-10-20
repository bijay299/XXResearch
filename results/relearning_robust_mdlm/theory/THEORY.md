# Theory: why relearning speed is a gradient-norm property, and where the method breaks

Sprint started 2026-10-20 10:05 ET. Every derivation below is machine-checked by
`scripts/verify_theory.py`. **Per the standing rule, any derivation I get wrong stays in
this file next to its correction** — the prior sprint's Prop 7 was wrong twice and both
wrong versions are still published.

Notation. `L(θ)` is the adversary's objective: the masked-diffusion memorization loss of
the forget facts (the thing the adversary *minimises* to bring a fact back). `L_R` is the
retain loss. `g₀ := ∇L(θ_u)` at the unlearned point. `M` is a smoothness constant:
`‖∇L(θ) − ∇L(θ')‖ ≤ M‖θ − θ'‖` on the region of interest.

---

## Lemma 1 — one adversary step, exactly

For a single gradient step `θ⁺ = θ_u − η g₀`, Taylor with remainder gives

```
L(θ⁺) = L(θ_u) − η‖g₀‖² + (η²/2)·g₀ᵀ H(ξ) g₀      for some ξ on [θ_u, θ⁺]
```

and under `M`-smoothness,

```
L(θ_u) − L(θ⁺)  ≥  η‖g₀‖²·(1 − ηM/2)                                   (L1)
```

**The value `L(θ_u)` does not appear.** For any `η < 2/M` the one-step recovery is
proportional to `‖g₀‖²`. Two models can sit at *identical* forget loss and relearn at
completely different speeds.

This is why "disclosure 0.049" is not evidence of removal. It is a statement about the
*value* of the objective, and the adversary's first step is governed by its *slope*.

**Status: elementary and certainly not novel as calculus.** What matters is whether it is
the right diagnostic for unlearning, which is an empirical question (Phase 0), not a
theoretical one.

---

## Proposition 2 — a robustness radius

Because the adversary's reachable set is limited by how far it can move in parameter
space, the useful statement is a *lower bound on the loss inside a ball*. Taylor with
`M`-smoothness, for any `θ` with `‖θ − θ_u‖ ≤ r`:

```
L(θ)  ≥  L(θ_u) − ‖g₀‖·r − (M/2)·r²                                     (P2)
```

Invert it. To achieve a loss drop of `Δ` the adversary must travel at least

```
r  ≥  ( −‖g₀‖ + sqrt(‖g₀‖² + 2MΔ) ) / M                                 (P2′)
```

Two regimes:

| | required displacement |
|---|---|
| steep slope, `‖g₀‖² ≫ 2MΔ` | `r ≳ Δ/‖g₀‖` — **shrinks as the slope steepens** |
| stationary, `‖g₀‖ → 0` | `r ≳ sqrt(2Δ/M)` — **independent of the loss value** |

So driving `‖g₀‖ → 0` raises the adversary's required parameter displacement from
`O(Δ/‖g₀‖)` to `O(sqrt(Δ/M))`. That is the entire mechanism, and it is a bound, not an
intuition.

**Honest limitation, stated up front.** P2 bounds what is reachable *within a ball*. It
does **not** bound what happens once the adversary leaves that ball — I can control
`‖∇L‖` at `θ_u`, never along the adversary's whole path. So the method can only ever buy
*more adversary steps*, never immunity. Any claim of immunity would be false, and I will
not make one.

---

## Proposition 3 — the penalty has a spurious attractor at the memorized solution

**This is the method's main failure mode and it follows from the objective itself.**
The candidate objective is

```
J(θ) = −L(θ) + λ_R·L_R(θ) + λ_G·‖∇L(θ)‖²
```

Its gradient is

```
∇J = −∇L + λ_R·∇L_R + 2λ_G·H(θ)·∇L(θ)                                   (P3)
```

Set `λ_R = 0` and consider any critical point of `L` itself, i.e. `∇L(θ*) = 0`. Then the
penalty term vanishes **and so does its gradient**, so `∇J(θ*) = 0`.

The critical points of `L` include its **global minima** — which is exactly `θ₀`, the
fully memorized model. **The gradient-norm penalty is perfectly satisfied by putting the
fact back.** `‖∇L‖² = 0` is achieved both at a high-loss stationary point (what we want)
and at the memorized optimum (the worst possible outcome).

### P3a — the drift is decided by the *starting basin*, not by `λ_G`

**My first version of this was wrong and is preserved here.** I claimed: *"there is a
`λ_G` above which the penalty dominates and the optimiser drifts back downhill toward
memorization,"* and the verifier tested `λ_G = 50`. **It failed.** At `λ_G = 50` the
dynamics are merely unstable (oscillating about the memorized minimum, then diverging at
`λ_G ≳ 200`) — not drifting. The size of `λ_G` was the wrong variable.

Measured instead, on `L(x) = x²(x−3)²` — memorized minimum at `x = 0`, target high-loss
stationary point at `x = 1.5`:

| start | `λ_G = 1` | `λ_G = 5` | `λ_G = 20` | `λ_G = 50` |
|---|---|---|---|---|
| `x₀ = 0.45`, **inside** the memorized basin | `L→0.0000` | `L→0.0000` | `L→0.0000` | unstable |
| `x₀ = 1.40`, **outside** it (post-unlearning) | `L→5.0625` | `L→5.0625` | `L→5.0625` | `L→5.0625` |

**The fact returns completely — `L → 0.0000` — for every moderate `λ_G`, when the
optimiser starts inside the memorized basin.** The curvature explains it: `L''(0) = +18`
(convex at the memorized minimum) versus `L''(1.5) = −9` (concave at the target). The
penalty's gradient `2λ_G·L''·L'` therefore points *back toward the minimum* throughout the
convex region, so the penalty actively fights the escape.

**Design consequence, and it is a hard constraint:** unlearning starts at `θ₀`, which *is*
the memorized solution — the worst possible initialisation for this penalty. So the
mechanism **cannot be a one-stage objective applied from `θ₀`**. It has to be a **second
stage applied to an already-unlearned model**, which is outside that basin. The method is
therefore a *flattening* stage, not an unlearning stage.

### P3b — the first term cannot be raw ascent

A second constraint, found the same way. In 1-D, ascent on `L` lands on a local maximum,
which is automatically stationary, so the penalty looks pointless — `λ_G = 0` already
reaches `|L'| = 1.4e−08`. That is an artefact of one dimension. In 40 dimensions raw
ascent **diverges**: `‖θ‖ → ∞` within 3000 steps (verified). High loss and zero gradient
coincide in 1-D and come apart in high dimensions, which is exactly why the penalty could
have value — and also why the first term must be a **bounded, anchored** forget loss (the
exclusion-target / NPO form that MDU, EMDU and SSD all already use), never `−L`.

### Guards this forces, pre-registered

1. **Monitor `L(θ)` every step; abort the arm if `L` falls below its value at `θ_u`.** A
   run that reduces disclosure *and* reduces the adversary's loss is recovering the fact.
2. **Stage-2 only.** Never initialise this penalty from `θ₀`.
3. **A `λ_G` sweep is mandatory** — one value cannot separate "the penalty helped" from
   "the penalty was too weak to matter."

**Testable prediction that falsifies the whole method:** if the gradient-norm term cannot
reduce `‖∇L‖` at matched disclosure, there is nothing here. Gated in Phase 2.

---

## Proposition 4 — the finite-difference surrogate, and its cost

`‖∇L‖²` needs a Hessian-vector product (double backward). The standard
first-order surrogate takes an ascent step of size `ρ` along the normalised gradient:

```
L̃(θ) := L(θ + ρ·g/‖g‖),        g = ∇L(θ)
```

Then `L̃(θ) = L(θ) + ρ‖g‖ + O(ρ²)`, so **the finite-difference surrogate approximates
`‖g‖`, not `‖g‖²`**. Penalising `L̃ − L` therefore penalises the gradient *norm*, a
different (weaker, non-smooth at 0) penalty than the squared norm. They share the same
zero set, so P2's conclusion is unaffected, but the two are **not interchangeable in the
gradient**, and reporting one as the other would be wrong.

Cost, to be **measured not assumed**: the surrogate needs one extra forward+backward per
step (≈2× the control); the exact HVP needs a double backward (≈3×). The Phase-2 gate
records measured `s/step` for both and the sprint uses whichever actually reduces `‖g‖`.

---

## What would make me abandon this

Stated now, so it cannot be rationalised later:

- **Phase 0:** if the unlearned control does not relearn faster than the never-memorized
  floor (gap < 0.25 at k=10), the axis is not real → pivot to sequential unlearning.
- **Phase 2:** if the penalty cannot reduce `‖∇L‖` at matched disclosure, or if the
  guard in P3.2 fires (the arm drifts back toward memorization), the method is dead.
- **Phase 3:** if robustness improves only through output degeneracy, it is recorded as a
  failure — clean forgetting is reported for every arm.

**No novelty claim.** A literature check is running in parallel precisely because the
mechanism in P2 is natural enough that it may already be published; if it is, this becomes
a replication on a masked-diffusion model and will be labelled as such.
