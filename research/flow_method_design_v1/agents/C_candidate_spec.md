# Agent C candidate spec: VAU-AdamW (actual-update, H-metric corrected, two-point verified AdamW step)

VERDICT: method-level novelty = NONE. This is a composition of established parts (Adam-NSCL-style projection of the
Adam update [UNVERIFIED, from memory, arXiv unreachable], preconditioned/metric projection, TRPO/CPO-style backtracking
on the actual step, held-out validation). It is kept only as ONE testable specification (hypothesis C-H1) because the
computed regimes in C_analysis.md show where it differs from the pre-Adam projection and the Euclidean post projection.

## Hypothesis C-H1 (two sentences)
For AdamW-trained LoRA erasure with a retained-set drift budget, (i) correcting the ACTUAL tentative AdamW update
(momentum, diagonal preconditioner and decoupled WD included) by the minimum-H-norm projection, H = diag(sqrt(vhat)+eps),
gives exactly zero first-order retained drift while keeping materially more erasure progress than the Euclidean
post-update projection when per-parameter gradient scales are anisotropic (>= 1 decade); and (ii) accepting the step only
if a fresh-batch two-point forward check of the cumulative retained drift passes (else scale, roll back m, v, t, theta)
certifies the drift budget with probability 1-delta with no Taylor/Jacobian assumptions. It does NOT beat a tuned ordinary
retention penalty by a large margin (toy: 6-12% progress at equal drift) and may lose to it when the Jacobian is stale.

## Exact update rule (state and dimensions)
theta in R^P (all LoRA A,B flattened, P = sum_l r(d_in+d_out)); AdamW state m,v in R^P, step t; hyper lr, b1, b2, eps, wd.
Per step:
1. g = grad of erasure loss (one fwd+bwd on erasure batch; GEM teachers a,b detached as native).
2. Tentative state: m' = b1 m + (1-b1) g; v' = b2 v + (1-b2) g^2; t' = t+1; mhat, vhat bias-corrected.
   Dg = 1/(sqrt(vhat)+eps) in R^P (diagonal); Delta0 = -lr Dg*mhat - lr*wd*theta   (the actual AdamW step).
3. Retained Jacobian block S J in R^{k x P} (k probe directions; refreshed every K steps, hence stale in between):
   Delta1 = Delta0 - (Dg*(SJ)^T) (SJ diag(Dg) (SJ)^T + mu I)^{-1} (SJ Delta0)      [min ||x-Delta0||_{H}, H=diag(1/Dg), mu small ridge]
4. Reject if g . Delta1 >= 0 (no first-order erasure progress): Delta1 := 0, commit state anyway only if rule 6 says so.
5. Two-point verification on a FRESH retained batch R2 (n2 samples, independent of Delta1, outputs of frozen base model
   cached): rho_hat = mean_x ||f_x(theta+s*Delta1) - f_x^base||^2 ; accept iff rho_hat + M sqrt(ln(1/d')/(2 n2)) <= tau^2,
   where M = assumed per-sample bound, d' = delta/(T*Rmax). Else s <- s/2 (max Rmax tries), else s=0.
6. Commit: theta += s*Delta1; (m,v,t) <- tentative state (state is committed from raw g; alternative "scaled" variant
   m <- scaled is a design toggle, untested). On s=0 roll back to old (theta,m,v,t) and advance data only.

## Pseudocode
```
for t in range(T):
  g = erasure_grad(theta)
  d0, st = adamw_peek(state, g, theta)          # no mutation
  if t % K == 0: SJ = retained_jacobian_probes(theta, R1, k)   # k VJPs (or JVPs) on retained batch R1
  d1 = d0 - Dg*SJ.T @ solve(SJ@(Dg[:,None]*SJ.T)+mu*I, SJ@d0)
  if g@d1 >= 0: d1 = 0
  for s in (1, 1/2, 1/4, ...)[:Rmax]:
      if two_point_ok(theta + s*d1, R2_fresh()): break   # 1 forward on n2 retained samples
  else: s = 0
  if s>0: theta += s*d1; adamw_commit(state, st)
```
## Cost per step (P LoRA params; L layers; k probes; K refresh; Rmax tries)
- Baseline GEM step: 1 student fwd+bwd on erasure batch + 2 no-grad teacher fwds (as native).
- Extra: k VJP/JVP passes on retained batch every K steps (amortised k/K backward-equivalents); SJ memory k*P floats
  (k=64, P=5e7 -> 12.8 GB fp32: DENSE, not hidden; use per-layer blocks or k<=16, or CG matrix-free = 2 passes per CG iter);
  Gram k^2 P flops + k^3 solve per step; 1..Rmax extra forwards on R2 (n2 samples; base outputs cached offline);
  trial copies theta' (P) plus tentative m',v' (2P). No dense Jacobian over P x outputs is ever formed.
## Closest published method / strongest simple alternative
Closest: Adam-NSCL (projects Adam's candidate update onto approx null space of feature covariance; UNVERIFIED).
Also TRPO/CPO line search on constraint; GPM/OGD (pre-optimizer gradient projection); natural-gradient / GN metrics.
Strongest simple alternative: retention penalty lam/2 ||J(theta-theta0)||^2 (or output-space distillation to the base) with
ordinary AdamW and lam tuned on a held-out drift; plus fresh-batch early-stop. In the toy it was within 6-12% of H-correction.
## Numerical prediction and falsification (toy, then future real run)
Prediction P1 (computed, C_scripts/c2,c4): pre-Adam projection leaves 48-100% of raw per-step drift (not neutral); Euclid
post projection loses progress with anisotropy (0.67 -> 0.08 of raw at 0..4 decades); H post projection keeps 0.77 -> 0.24.
Falsify C-H1(i) if, with exact J and anisotropy >= 2 decades, progress(H) / progress(Euclid post) < 1.2 on >= 8 seeds, or if
a tuned penalty beats H-correction by > 10% at matched cumulative drift while using the same Jacobian information.
Falsify (ii) if the empirical rate of accepted steps with true population drift > tau exceeds delta (needs valid M), or if
the budget-matched erasure efficacy under the gate is not above the penalty + early-stop baseline.
Real-model requirement: native FLUX LoRA + AdamW state access, a retained set R1/R2 split disjoint from erasure prompts,
cached base outputs. Not run here; no real-model claim.
