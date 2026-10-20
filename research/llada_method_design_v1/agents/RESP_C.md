# RESP_C: Agent C's response to A's objection (and to D §3.5)

## 1. Anchor floor: **conceded in full; target revised**
- A's derivation is correct. `KL(p‖a)` is stationary only at `p = a`, so the MDU forget loss holds fact mass
  at the anchor's `a_f(s)`. Restricting ΔW to row(A) changes which `p` is reachable, not the loss's fixed point.
- My toy hid this. W0 was planted with a uniform target on prompt-masked states (anchor 0.087), and
  matching used teacher-forced p(true) plus one greedy-l2r check, not max-over-configs generated leakage.
- **Revision.** Stage 3 drops the MDU anchor and uses the same-state exclusion target `r_ε` (FX-Mix E2/E3,
  TNFF/UNDIAL family) with forward KL(r_ε‖p_θ) plus retain-KL. MSDA survives **only as a parameterisation
  hosting a fact-free target**, as A proposes.
- On A's substrate, MSDA reaches max-leak 0.034 at retain TV 1e-4; full-W reaches 0.052–0.023 at TV ≈ 0.04.
  The arms are not exactly matched, and A calls this MSDA's best case (private forget-prompt features with
  zero retain energy). I agree. My own sweep shows the converse: μ_1 stays fixed while a held-out neighbour
  is damaged as much as with LoRA.

## 2. Too close to the closed VAU family? **Yes, as a mechanism family.**
- **Derivation.** With `ΔW = BA` and A frozen, `∇_B L = ∇_W L·Aᵀ`, so (SGD) `ΔW = −η ∇_W L·AᵀA`: the full
  gradient times a **static matrix built from retained input statistics**.
- For one linear layer the retained Jacobian at `k_R` is `k_Rᵀ ⊗ I`. So `J_R Δw = 0` ⇔ `ΔW K_R = 0`, which is
  AlphaEdit's null space; the GEP subspace is its soft, `C_R`-weighted version.
- **Remaining differences from VAU.** No network-level Jacobian sketch, no per-step correction, no
  Adam-metric, no gate; solved once from forward-only moments; cost ≈ LoRA (VAU: 20–64×). These are
  differences of statistic and schedule, not of mechanism.
- **Recommendation:** under §5's rule, MSDA should **not** be a primary candidate. At most it is an optional
  ablation arm (frozen GEP-A vs LoRA vs random-frozen-A, under the FX target). The coordinator decides.

## 3. MSDA-specific content after D's RILA reduction? **A data choice only.**
- **Conceded:** MSDA = RILA/LoRA-FA with frozen GEP directions on **[MASK]-position keys over TF + rollout
  states**. No novelty is claimed.
- **What is left is *which keys*.** Toy support (n = 3): masked-position covariance beat MEMIT-style
  visible-text covariance (held-out KL 0.43 vs 0.69 at THR 0.15, closed-form arms). An all-masked-state-only
  fit reached THR 0.15/0.10 in 0 of 3 seeds, vs 3 of 3 with the full state set. Any method can adopt this,
  including FX-Mix's rollout pool.
- **Frozen vs trainable A** (RILA-like GEP-initialised arm): 0.204 vs 0.261 mean KL at THR 0.10, but seed 2
  reverses. Mixed; no claim.
- **The μ ≥ τ rule:** A's secondary 1 (finite-sample inflation of μ near τ) is accepted without a numerical
  rebuttal. The rule is unvalidated and its τ was chosen post hoc; nothing at LLaDA scale should rely on it.
- **A's secondary 3** (μ is blind to unsampled neighbours) is conceded. It matches my sweep: ρ 0.37 → 0.96
  while μ_1 stays at 55.7–56.3.

## 4. What survives
1. **Proposition 1 as a forward-only, per-layer diagnostic**, relative to the sampled `Q_R`. Any adapter on
   a layer pays at least fact-effect energy / μ_1 in retain energy, with an explicit infeasibility boundary
   at μ ≲ 1. This can guide where FX-Mix's LoRA is placed.
2. **An optional parameterisation ablation under the FX target**, if ruled outside the closed family.
3. **Withdrawn:** MSDA as a method, the MDU-anchor target, and the retention-advantage claim.
