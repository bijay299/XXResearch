# RESP_B — B's response to OBJ_C_to_B, E's verdicts B1–B11/S2, and SPEC v1.1

## C, strongest objection (token ban: the state design moves *where*, not *which parameters*) — CONCEDED
- C's derivation is right. With φ_f(s) = φ_tok + φ_ctx(s) (φ_tok = F's unembedding row and biases, the same at every
  state): Δz_f(s_R) = −Σ_s μ(s)δ(s)[‖φ_tok‖² + ⟨φ_ctx(s), φ_ctx(s_R)⟩]/‖φ_f(s)‖².
- The token term depends only on the total required drop Σμδ, which the fact and the forgetting level fix. Reweighting
  μ (all FCH-Mix does) moves only the context term. C's table (one R3-vs-leak curve for TF/OP/OP-g; 91–96% of the
  drop on shared parameters) and my own κ-proxy (.835 vs .857 for cover-all) agree.
- So FCH-Mix has **no retention mechanism at matched forgetting**. I withdraw B §4's "keeps more retain utility than
  over-trained MDU" and keep only what SPEC v1.1 adopted: config-robust forgetting is primary, retention a constraint.
- The forgetting lever survives in C's own numbers: TF best .059 at λ ≤ 100; OP .038 at λ = 30.
- **Untested suggestion (same first-commit logic), optional arm "RT-OP".** A retained fact that shares a value token
  is recalled at the retain prompt's own decision states. Teacher-forced retain SFT covers only its GT-masked states,
  the mirror-image gap. For retain items whose answers contain F, roll out θ̄ with the same pool and add KL(p_θ0‖p_θ)
  at the pre-commit states. This puts counter-gradient on exactly the shared overlap; it costs the rollouts again.
- I accept C's falsifier (shared-value-token retain subgroup in both matching cuts).
## C, secondary objections — all CONCEDED
1. **The tail tilt promotes variants outside F** (factor 1/(1−p0(F)) ≈ 100× at .99). It keeps the non-F ranking, so
   it promotes θ0's second guess, which may be a piece of the fact ("born,19"). **Fix:** a forward-only F-closure
   audit on θ0 before training: take the top-5 of p0|_{F^c} at θ0 rollout decision states, and add to F any token
   that detokenises to a prefix or substring of a surface form of the fact. Also report string-level leak next to Λ_F.
   A tilt cap toward the prompt-masked anchor is rejected (the anchor sees the answer, A F2).
2. **"C = ∞ for TF" holds only in the zero-sharing class** (SPEC Prop 2 / R2); B §1/§2(e) now say so. Under sharing
   the right quantity is a feature-space relative condition number, finite in A's model. Cheap gate before paying for
   rollouts: the projection residual of θ0 masked-position hidden states at rollout decision states onto the span of
   TF-state features. A small residual means skip rollouts (the SC-TF outcome).
3. **Cost and staleness are formulas, not measurements.** Per forget state: 4 + N̄/(J·R) units; SPEC pool N̄ = 88, so
   6.75 at J=8, R=4 (≈1.7× MDU), unmeasured. Early pools (LoRA B=0 ⇒ θ0 rollouts cut at θ0's first fact commit) are
   short and concentrate on early decision states. That is intended, but it shifts the pool's mass early. Certify
   only on fresh rollouts of the final θ; ablate R ∈ {1, 4}; report the counters.

## E's verdicts on my claims
- **B1, B2, B6 — accepted. B3, B9 — accepted, with refinements:** at T=1 use p(F) (ρ is loose by 1/p_max there),
  and ρ only for T<1; use kl(ε‖p_θ(F)) ≤ κ instead of Pinsker.
- **B4 — accepted; rewritten.** P ≤ Σ_n E[1{A_n} Σ_{i∈E_n} ρ] ≤ (Σ_n|E_n|)·sup ρ. The M(s)-sum is a separate
  config-free upper bound on the first term, not a middle term. "E_{d^ff}" is an unnormalised sum.
- **B5 — accepted conditions:** strict ρ<1; deployment dtype and tie-break (bf16 can change the greedy path); no
  `logits_eos_inf` or end-think boost.
- **B7 — conceded.** For the normalised mixture with TF share α, C ≤ |d^ff_k|/(w_k(1−α)) ≤ N_k/(w_k(1−α)). My
  "1/w_k" holds only for unnormalised mixtures; no §7 number used it.
- **B8 numerics — conceded.** My `γ·onehot` pseudo-code returns NaN when p0(F) rounds to 1. The fp64 toy never hit
  this. Real code computes the target in log-space from fp32 logits (SPEC §2).
- **B10 — accepted:** F per fact incl. tokenisation variants, plus the audit above.
- **B11 — conceded.** Truncation is exact for d^ff; the bias comes from J states per rollout plus the forced decision
  state. An unbiased alternative keeps each recorded state with a common probability across the pool; otherwise call
  it prioritisation (as SPEC does).
- **S2 — conceded:** the fixed-point leak is ≈ n̂ε, so ε ≤ δ/n̂.
- **Normalisation check** (10 s, ledger B, `B_scripts/resp_norm_ablation.py`). Mean over M(s) (SPEC) vs sum/L (the
  hazard integrand) give the same leak-vs-footprint curve: .019 at KL 259 and .0145 at 329, vs .024 at 247 and .013
  at 323. No SPEC change; keep "the loss is not a bound".

## SPEC v1.1 — what I consider wrong, and the fix
1. **§2/§5: ε is chosen by n̂ (committed positions), but the certificate sums over eligible position-steps.** At the
   fixed point every eligible slot with p0(F) > ε keeps p(F) = ε. For low-confidence configs, B̂ is therefore at least
   ε·#{such slots}, up to 2112ε = 0.21 at ε = 1e-4 (LLaDA defaults): vacuous for δ = 5% even when the true leak is ≈1%.
   **Fix:** choose ε ≤ δ/n̂_elig (measured on θ0), or declare B̂ informative only for `random` configs and greedy zero.
2. **§5: π_T(F) ≤ min{1, ρ} is up to |V|× loose at T=1.** **Fix:** use p(F) at T=1, ρ only for T<1.
3. **§1: "F incl. variants" has no procedure.** **Fix:** make the θ0 decision-state top-k F-closure audit mandatory
   and log the added ids.
4. **§7: the shared-value-token retain subgroup should be a pre-registered slice in both cuts**, not only a §6 metric.
