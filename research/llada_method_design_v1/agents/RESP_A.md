# RESP_A — Agent A's answers to B's objection and E's verdicts (no new runs)

## B → A
**B-main (my max-leak numbers are in-sample; the pool misses random-order parallel states). CONCEDED and revised.**
- My toy maximum was over the same C1–C6 configs that produced the rollouts. "Only FFX-OP reaches ≤0.05" should have
  read "inside 𝒟". B's held-out leak of 0.20 for my LLaDA pool stands.
- *Why (derivation).* At state s, a random-order step that commits a pair (j,k) fills it w.p. p(x_j|s)p(x_k|s). A
  one-token-per-step config reaches the same state w.p. p(x_j|s)p(x_k|s,x_j). The visitation density ratio is
  Π p(x_k|s)/p(x_k|s,x_j). It is unbounded for anti-correlated pairs ("1952 EOS" vs "1952 Paris"). Full support at
  T=1 makes coverage finite, but not small.
- **Revision.** The pool must contain (gen,gen,rnd,1) and a single-block random config with the largest
  tokens-per-step k the evaluator may use. Held-out k above the pool's maximum is a pre-registered extrapolation
  test. SPEC v1's P2/P3 already adopt B's fix.
**B-1 (KL form of P2 is false for 0<T<1). CONCEDED.** Re-derived: π_0.7(F) = 0.179/(0.223+0.179) = 0.445 > 0.30.
Replacement for T∈[0,1]: π_T(F) ≤ min(1,ρ) with ρ = p(F)/p_max, since Σ_f p_f^β/Σ_v p_v^β ≤ Σ_f (p_f/p_max)^β ≤ ρ
for β ≥ 1. The T=1 statement and the strict T=0 statement stay.
**B-2 (the toy does not test the spec). CONCEDED.** The toy used all masked positions and summed DAgger visitation,
not E_t with a ½/½ mixture of n_op states. B shows E_t vs M(s) changes nothing (0.2006 both), and SPEC v1 uses the
M(s) mean. The toy numbers are relabelled "toy estimator".
**B-3 (ε=0 has no stationary point under Adam). CONCEDED.**
- The F-logit gradient p_θ(F) > 0 never vanishes. Adam's normalisation gives O(lr) steps on forget-dominated
  directions, so z_F drifts without bound.
- My toy hid this: L-BFGS on a convex problem with a prox term μ‖θ−θ0‖² has a finite minimiser, so the prox set the
  operating point. The R3 drop (0.95→0.87) is that cost along shared directions.
- **Fix:** ε>0 (gradient (p_θ(F)−ε)₊ vanishes at ε), or ε=0 with the step budget as declared knob (SPEC v1 has both).

## E's verdicts on A1–A12
- **A1 CORRECT.** Agreed (Csiszár I-projection / TNT target; no novelty claimed). **A2 CWC. Conceded:** LLaDA defines no
  joint (E2: TV 0.40 between orders), so q* is not an LLaDA target; only per-position r_ε is (q* → tabular remarks).
- **A3 CWC. Agreed.** The claim covers the token event Λ_F only. It is silent if the prompt shows an F token (those
  items are reported separately) and blind to string pieces outside F.
- **A4 CORRECT. Agreed,** plus a consequence: under parallel commits, mean-field can put mass on p0-null pairs
  (E2: 0.714; my C6 junk is 0.167). This is a fluency risk, reported per parallel config.
- **A5 (arithmetic right, interpretation wrong). Conceded.** Leak-relevant pairs are *all* (fact-free state, masked
  position) pairs, since F can be committed at non-fact positions; under sharing, any term matters. Re-worded as R5:
  "SFT-weighted share on (fact position, fact-free state) = (1−ε^k)/((1−ε)L)".
- **A6 CWC. Agreed:** add cfg=0, logits_eos_inf=False, end-think 0; needs realisability; a corollary of Prop 1.
- **A7 INCORRECT as printed. Conceded (typo).** The decision state s_{t*} is fact-free, so the sum runs over t ≤ t*,
  as in my Appendix A1. The L=1 counterexample (leak 0.3, printed bound 0) stands.
- **A8 CORRECT (random only). Agreed:** low_confidence needs the eligible-set form; random uses k_n/|E_n|.
  **A9 (T=1 only): conceded**, replaced by the ρ form (B-1). **A10 CWC: agreed** (strict; ties/eos_inf/dtype out of scope).
- **A11 UNSUPPORTED. Conceded and withdrawn.**
  - The loss is a position-mean over a stale ½TF+½OP mixture. The bound needs the per-config sum over the eligible
    position-steps of fresh current-model rollouts.
  - At gen 128 / block 32 / 128 steps, Σ|E_n| = 4·528 = 2112, so 5% needs per-position KL ≲ 2.4e-5. Vacuous for
    training; only the post-hoc certificate B̂_π survives (SPEC §5).
- **A12 CORRECT. Agreed:** selectivity only at θ0; for θ≠θ0 it is distillation to p0 (a retention anchor).

## SPEC v1 — what I consider wrong or under-specified, with fixes
1. **§7 matching axis.** "Held-out leak at matched retention" lets the chosen axis pick the winner. Fix:
   pre-register the full (held-out max leak, retention) Pareto curve over the λ/step grid and report both cuts (the
   assignment's default is retention at matched forgetting).
2. **§3 coverage in k.** By the density-ratio derivation, held-out random configs with more tokens per step than the
   pool's maximum are uncovered. Fix:
   - LLaDA: add (32,128,rnd,1) (k=4) to the pool, or name it as the extrapolation cell.
   - Toy: flag H6 (k=4, pool k≤2) as extrapolation, and report the max over H1–H5 alongside the max over H1–H6, which
     H6 will likely decide alone.
3. **§2 numerics.** Compute r_ε from fp32 reference logits. A bf16 logsumexp can misplace the p0(F) ≤ ε switch at
   ε=1e-4.
4. **§5/§6 certificate.** Compute B̂_π with the deployment dtype and tie-break, with Λ_F over all positions. Report
   its MC standard error: it is noisiest exactly where the leak is small.