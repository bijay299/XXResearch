# CANDIDATES — three mechanism-track specifications and the selection

Day-1 checkpoint (2026-10-19, all three tracks reported within ~25 min of each other). Full specifications are
preserved verbatim, unedited by the coordinator, in `agents/B_candidate_spec.md`, `agents/C_candidate_spec.md`,
`agents/D_candidate_spec.md`, with investigation logs `agents/{B,C,D}_analysis.md` and scripts `agents/{B,C,D}_scripts/`.
Every track reported **method-level novelty = NONE**. That is recorded, not softened.

## Summary of the three specifications

| | B — loss / output geometry | C — shared parameters / optimizer | D — trajectories / timesteps |
|---|---|---|---|
| Hypothesis | H-B1: at GEM's default eta=1, free output-space flow turns an offset eps from b into transverse collateral sqrt(S0²−L²)/2 ≈ sqrt(Lε/2) | C-H1 (VAU-AdamW): correcting the *actual* AdamW step in Adam's own metric keeps zero first-order retained drift with more erasure progress than Euclidean post-projection under anisotropy; fresh-batch verification certifies the budget | H-D: a prefix-only (σ 0.77–1.0) local loss reaches the endpoint only via carry-over/basin crossing; uniform 11-of-28 coverage should erase ≥ as well at equal compute |
| Proved (conditional) | I1 GEM(η=1)=relu(2s−L)·L/(D⁺+D⁻); I2 conserved D⁺+D⁻ ⇒ T1 | T1 pre-Adam projection non-neutral iff null(J) not D-invariant; T2 metric projection; T3 Hoeffding certificate under adaptivity | D-T1 endpoint = transported integrated velocity error, exact constant for Gaussian data; D-P2 IS unbiased only with weights |
| Counterexample | stiff-transverse metric diag(1,100): r_end=4.86L vs 0.007L predicted | stale J: gate holds budget but progress 0.09–0.18 < penalty 0.32; LoRA gauge step | bimodal data: first-order prediction 86–94 % wrong |
| Closest prior / simple alternative | axial margin / truncated anchor (already closed CPR controls) | Adam-NSCL (UNVERIFIED), scaled gradient projection, TRPO/CPO backtracking / tuned retention penalty | EraseFlow, TRACE (UNVERIFIED) / `uniform t` one-flag change |
| Prospective test | P2: GEM/axial collateral ratio < 2 at stiffness ≤ 1, ε/L ≤ 1e-3 falsifies | ratio H/Euclid progress < 1.2 at ≥ 2 decades, or tuned penalty > 10 % better, falsifies | P-D4 already INCONCLUSIVE (6/0/6); P-D5 untested |
| Native cost | 0 extra passes | k VJPs / K steps + Rmax fresh forwards + 3P state | 0 extra passes (same compute) |
| Why not selected / selected | Its only "method" is an already-closed control; distinction lives in an unmeasurable Jacobian metric. Retained as a **diagnostic** in the test suite (family `G5`). | **SELECTED** (see below) | Its repair is a one-flag ablation with an inconclusive toy; already covered by GEM's own on-policy rollout. Retained as a recommended **ablation arm** for the real-model protocol. |

## Selection: C-H1, narrowed — "VAU" (Verified Adam-metric Update) for AdamW-trained LoRA erasure

**Why.** (i) *Mathematical plausibility:* the core distinction is provable, not only observed (DERIVATION.tex §3).
(ii) *Meaningful distinction:* GEM (verified, `A_gem_code_facts.md` §5–6) trains LoRA with plain AdamW and **no retention
term at all**; zero-init B makes step 1 an `lr·sign(g)` step whose retained effect does not shrink with any projection
applied *before* Adam (C T1). The closest published primitive that acts on the Adam update (Adam-NSCL, UNVERIFIED) projects
Euclidean-ly; a coordinator pre-check (`code/scratch_checks/sign_check.py`, development draws only) found the Euclidean
post-projection of a preconditioned step yields **negative** first-order erasure progress in 3.5 % of random draws, while the
Adam-metric projection never does (provable, Prop. 2). (iii) *Implementability:* matrix-free with k VJPs; no dense P×P object.

**What is NOT claimed.** No component is new: scaled/metric gradient projection, null-space projection of Adam updates,
backtracking with a constraint check, Hoeffding validation are standard. The claim under test is an **engineering
hypothesis** about one composition in one regime (AdamW-LoRA erasure with retained probes), plus conditional results.
Retained probes are **extra information GEM does not use**; every control receives the same probes.

**Early pivot:** not used (0 of 1). **Rejected specifications preserved:** B and D files above, unmodified.

### Exact candidate (frozen in `configs/candidate.json` before validation)
State: θ∈ℝᴾ (LoRA A,B flattened), AdamW (m, v, t), β₁=0.9, β₂=0.999, ε=1e-8, wd=0.01 (torch defaults; GEM passes only lr),
probe block Sᴶ∈ℝ^{k×P} (k sketched retained-output rows), refresh period K, budget τ², gate size n₂, per-sample bound M,
confidence δ, max retries R.

```
for t = 1..T:
  g  = ∇θ L_erase(θ)                                       # native GEM loss, teachers detached
  m' = β₁m+(1−β₁)g ; v' = β₂v+(1−β₂)g² ; m̂=m'/(1−β₁ᵗ) ; v̂=v'/(1−β₂ᵗ)
  D  = 1/(√v̂+ε)                                            # Adam's diagonal preconditioner, ℝᴾ
  Δ₀ = −lr·D⊙m̂ − lr·wd·θ                                    # the ACTUAL AdamW step
  if t ≡ 1 (mod K): Sᴶ = Σ·J_r(θ)   (k VJPs of fresh Gaussian sketch Σ∈ℝ^{k×N_r d} on retained batch R₁)
  G  = Sᴶ diag(D) Sᴶᵀ + μI ,  μ = 1e-6·tr(G₀)/k
  Δ₁ = Δ₀ − D⊙(Sᴶᵀ G⁻¹ Sᴶ Δ₀)                               # min ‖x−Δ₀‖_{D⁻¹} s.t. Sᴶx≈0
  if g·Δ₁ ≥ 0: Δ₁ = 0                                       # no first-order erasure progress
  for s in 1, ½, …, 2^{−(R−1)}:                             # verification on FRESH retained batch R₂ each try
       ρ̂ = mean_{x∈R₂} min(‖f_x(θ+sΔ₁) − f_x(θ₀)‖², M)
       if ρ̂ + M·sqrt(ln(TR/δ)/(2n₂)) ≤ τ²: accept s; break
  else s = 0
  if s>0: θ ← θ+sΔ₁ ; (m,v,t) ← (m',v',t)      else: (θ,m,v,t) unchanged (rollback), data advances
```
Ablations required by the protocol: `VAU-noGate` (s=1 always), `VAU-Euclid` (Δ₁ with D replaced by 1 in the projector, i.e.
Adam-NSCL-style), and every control with the **same** gate (so the gate is not credited to the metric).
