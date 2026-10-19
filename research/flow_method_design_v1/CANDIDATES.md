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
applied *before* Adam (C T1). The closest published primitive that acts on the Adam update (Adam-NSCL; code-verified by Agent F at `ShipengWang/Adam-NSCL@a2f39b4`, `optim/adam_svd.py:78-90,110-114`) projects Euclidean-ly. *[Narrowed after review F, wording from F_REVIEW §4.2:]* For β₁ = 0 (or AdamW step 1) and λ_wd = 0, the Euclidean post-projection of the Adam step can have negative first-order progress and the D-metric projection cannot (Prop. 2). With momentum both can (Counterexample 4). Reversals occurred in 3.5 % of draws at P = 6, k = 2 and in 0 % at P = 400 for k/P ≤ 0.66 (F_REVIEW f3), so they are not expected at LoRA scale (k/P ≈ 1e-5). (iii) *Implementability:* matrix-free with k VJPs; no dense P×P object.

**What is NOT claimed.** No component is new: scaled/metric gradient projection, null-space projection of Adam updates,
backtracking with a constraint check, Hoeffding validation are standard. The claim under test is an **engineering
hypothesis** about one composition in one regime (AdamW-LoRA erasure with retained probes), plus conditional results.
Retained probes are **extra information GEM does not use**; every control receives the same probes.
Under the change of variables y = D^{-1/2}θ, VAU's correction is the Euclidean (Adam-NSCL/GPM-style) projection applied in Adam's whitened coordinates, i.e. a scaled gradient projection / SQP step with metric D^{-1} (F_REVIEW §3).

**Narrowed hypothesis C-H1 (post-review; this is what the frozen validation tests).** In AdamW-LoRA erasure where fresh probes span the retained response (low effective rank, k ≥ rank), D-metric post-projection gives ≥ 1.2× the erasure progress of Euclidean post-projection at drift budgets ≤ 5 % of plain AdamW's, after lr tuning of both, and beats a same-probe tuned penalty and the null-A LoRA control at matched erasure. Development evidence before freeze (F's toy, not validation): held at 1 % budget (1.38×), failed at 5 % (1.12×); a same-probe penalty beat VAU 1.7–2× in base/rich/stale/β₁=0.99/sketch regimes.

**Early pivot: considered and declined (0 of 1 used).** After F's objection 2 (in GEM's zero-init regime the D metric routes the correction through A and >99.9 % of the actual retained change is the bilinear dB·dA term), the coordinator tested a factor-wise exact alternative (FNS: per LoRA layer dA·U = 0 and dB·(A·U) = 0 on the retained feature span U, giving exact finite-step retention on probed inputs; `code/scratch_checks/factor_exact_check.py`: drift ≤ 2.4e-13 up to step scale 10, vs 8.9e2 for A-only projection). It **collapses** in GEM's regime: with B = 0 and rank(A·U) = r the constraints force dB = 0 forever, so nothing is erased (`factor_collapse_check.py`: max|B| = 1e-15 after 50 steps). The construction that works keeps A's rows orthogonal to span(U) and trains B — the known null-A / InfLoRA-style LoRA (InfLoRA: UNVERIFIED this sprint; related null-space edits SPEED, AlphaEdit). Adopting it as the candidate would relabel published work, so it enters as the control `na_lora` instead, using the same retained probes. **Rejected specifications preserved:** B and D files above, unmodified.

### Exact candidate (frozen in `configs/candidate.json` before validation)
State: θ∈ℝᴾ (LoRA A,B flattened), AdamW (m, v, t), β₁=0.9, β₂=0.999, ε=1e-8, wd=0.01 (torch defaults; GEM passes only lr),
probe block Sᴶ∈ℝ^{k×P} (k sketched retained-output rows), refresh period K, budget τ², gate size n₂, per-sample bound M,
confidence δ, max retries R.

```
θ ← θ₀; (m, v) ← 0; c ← 0                                   # c = AdamW step counter, advanced ONLY on commit
for t = 1..T:
  g  = ∇θ L_erase(θ)                                       # native GEM loss, teachers detached
  m' = β₁m+(1−β₁)g ; v' = β₂v+(1−β₂)g² ; c' = c+1 ; m̂=m'/(1−β₁^{c'}) ; v̂=v'/(1−β₂^{c'})
  D  = 1/(√v̂+ε)                                            # Adam's diagonal preconditioner, ℝᴾ
  Δ₀ = −lr·D⊙m̂ − lr·wd·θ                                    # the ACTUAL AdamW step (torch semantics)
  if t ≡ 1 (mod K): Sᴶ = Σ·J_r(θ)   (k VJPs of fresh Gaussian sketch Σ∈ℝ^{k×N_r d} on retained batch R₁)
  if Sᴶ is empty: Δ₁ = Δ₀                                   # no probes ⇒ bitwise AdamW
  else: G = Sᴶ diag(D) Sᴶᵀ ; μ = max(1e-6·tr(G)/k, 1e-12) ; Δ₁ = Δ₀ − D⊙(Sᴶᵀ (G+μI)⁻¹ Sᴶ Δ₀)
  if g·Δ₁ ≥ 0: continue                                     # reject: no gate, no state commit, data advances
  for s in 1, ½, …, 2^{−(R−1)}:                             # optional gate; FRESH i.i.d. retained batch R₂ per try, never reused
       ρ̂ = mean_{x∈R₂} min(‖f_x(θ+sΔ₁) − f_x(θ₀)‖², M)
       if ρ̂ + M·sqrt(ln(TR/δ)/(2n₂)) ≤ τ²: accept s; break   # T, R, τ, M, δ, n₂ fixed in advance
  else: continue                                            # all tries failed: (θ,m,v,c) unchanged
  θ ← θ+sΔ₁ ; (m,v,c) ← (m',v',c')                          # state from raw g; NOT an AdamW trajectory
```
Ablations/controls required by the protocol: `vau_nogate` (s=1 always), `post_euclid` and `post_euclid_reject` (Adam-NSCL-style, without/with the same reject rule), `vau_blocksplit` (metric normalised per LoRA factor block), `pre_proj` (GPM-style), `penalty` (same probes), `na_lora` (null-A LoRA, same probes), plain `adamw` (native GEM) and its lr/step-norm matched version, and every control with the **same** gate (so the gate is not credited to the metric).
