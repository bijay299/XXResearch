# Agent C analysis: output-space gradient -> LoRA update -> retained drift
Tags: DERIVED (proof here), COMPUTED (script in C_scripts, run via meter.py, ledger labels C_*), UNVERIFIED.

## 0. Bottom line
The order-of-operations observation is TRUE and quantified (T1, C-C1), but acting on the actual update is what Adam-NSCL
already does (UNVERIFIED: arXiv blocked, WebFetch ENOTFOUND). Metric (H) projection and step rejection are established
families. Surviving content = one conditional specification (C_candidate_spec.md) + the regime map below. Method novelty: NONE.

## 1. Jacobian chain through LoRA (DERIVED)
Layer y = (W + BA)x. With output-space gradient G = dL/dY (N x d_out): g_B = G^T (X A^T), g_A = B^T (G^T X).
Parameter step (dA,dB) changes the layer map by dW = dB A + B dA + dB dA (EXACT, no remainder beyond the bilinear term).
B=0 at init => g_A = 0 exactly (COMPUTED c1: max|gA| = 0). Consequence for A at step 1: AdamW update = 0/(0+eps) = 0 plus
decoupled WD only (observed 4e-6 = lr*wd*|A|). At step 2, |g_A| ~ |B||G| = O(lr) but Adam normalises: max|dA| = 7.5e-4 vs
SGD 2.8e-6 (x265), cross term dB dA drift 2.5e-4 vs 1e-7 (x2400) (COMPUTED c1, lr=1e-3). Adam suppresses A only if
|g_A| <~ eps (eps sweep 1e-8..1e-4: no suppression here because |g_A|~1e-2). So the "vanishing A gradient" is a step-1
fact only; under Adam the second-order bilinear term becomes non-negligible from step 2, which first-order controls miss.
Decoupled WD on LoRA: both factors scale by (1-eta), eta = lr*wd: dW = ((1-eta)^2-1) BA = -(2eta-eta^2) BA (DERIVED);
it is a retained-output change of size ~2*eta*||BAx|| that no gradient projection removes.
AdamW first step: mhat/sqrt(vhat) = sign(g) (|g|>>eps), so step 1 is lr*sign(g) per entry, independent of gradient scale.

## 2. Order of operations (core result)
Notation: J retained Jacobian (m x P), Pi = I - J^+ J, D_t = diag(1/(sqrt(vhat_t)+eps)).
**T1 (DERIVED, exact).** Pre-Adam projection with beta1 arbitrary and fixed J: m_t in null(J) since it is a convex-weighted
sum of Pi g_s; the step is -lr D_t m_t (-lr*wd*theta). Retained drift = lr ||J D_t m_t||. For any scalar d: J D m = J (D-dI) m
because J m = 0, so drift <= lr ||J|| (Dmax-Dmin)/2 ||m|| on the support. Drift = 0 for all g iff D null(J) subset null(J)
(iff null(J) is D-invariant); for D with distinct diagonal entries this means null(J) is a sum of coordinate/eigen-subspaces.
Proof: J D Pi = 0 for all g iff D(null J) subset null J; invariant subspaces of a diagonal matrix are sums of subspaces of its
eigenspaces. Verified: identity to 1e-16, uniform D -> 5e-13, coordinate-aligned null(J) -> 5e-17, generic -> 0.17 vs bound 4.3
(COMPUTED c3). Plus WD term lr*wd*J theta is untouched by gradient projection. Step 1: drift = lr||J sign(Pi g)||,
independent of gradient scale and of how exact the projection is (c3: 0.096..0.098 for residual 1e-2..1e-8).
**Magnitude (COMPUTED c2, P=300,m=100, beta1=.9, 10 seeds, exact J):** pre-Euclid projection leaves 0.48-0.71 of the raw
per-step drift (wd=0), ~0.82-1.0 with wd=1 (WD dominates); pre-H(prev state) 0.55-0.69.
**T2 (DERIVED, standard).** For ANY tentative AdamW step Delta0 (momentum, WD, any state), Delta1 = Delta0 - Dg J^T (J Dg J^T)^{-1} J Delta0
satisfies J Delta1 = 0 (J full row rank), is the unique minimiser of ||x-Delta0||_H s.t. J x = 0 with H = diag(1/Dg), and
||Delta1||_H <= ||Delta0||_H. Computed: residual 2e-17 vs raw 2.7e-2, H-norm ratio 0.84 (c3). With mhat=g the first-order
progress is -g.Delta1 = lr ||(I-Q) Dg^{1/2} g||^2 >= 0 (Q = projector onto range(Dg^{1/2} J^T)), zero iff g in range(J^T)
(exact first-order infeasibility). With momentum mhat != g progress may be negative -> rule "reject if g.Delta1>=0".
**Euclidean vs H post projection (COMPUTED c2/c4):** equal drift (0), progress/raw 0.67 vs 0.77 (iso), 0.30 vs 0.53 (2 dec),
0.08 vs 0.24 (4 dec). Strongest simple alternative (penalty with lambda tuned using the exact J, oracle): progress 0.645 vs 0.687
(1 dec), 0.320 vs 0.359 (3 dec) at drift <= 5% of raw cumulative drift. So H-correction beats Euclid post projection decisively
in anisotropy but is only 6-12% above the penalty; this is a variant of the established metric-projection family (not novel).
Caveat: penalty uses oracle exact J and tuned lambda over 28 values and the toy is linear in theta; WD-containing rows in c2 have
progress ratios contaminated by the WD term (only drift columns meaningful).

## 3. Feasibility of simultaneous erasure/retention
First order: exact neutrality needs null(J_r) non-trivial and g_e not in range(J_r^T) (T2). Rank: J_r has up to N_r*d_out rows;
a LoRA layer has P = r(d_in+d_out) (e.g. 98k-246k at r=16). Generic toy: N=4 (rank 32, null 88) vs N>=16 (rank 120, null 0)
(COMPUTED c3). Thus with a rich retained set exact null-space neutrality is generally EMPTY and a budget tau>0 (soft) is needed;
a low-effective-rank retained response (decaying spectrum) restores approximate null spaces (c3 T4 setting). Real FLUX
spectrum: unknown (UNVERIFIED, not measured). This is a linearised statement only; not claimed for nonlinear unlearning
(C-X1: f=th1*th2 at origin has J d=0 but drift 0.09; curved null-space escapes are possible).

## 4. Finite-step behaviour
Construction vs verification (key asymmetry). Let rho(theta)=E_x||f_x(theta)-f_x^base||^2, per-sample in [0,M].
**T3 (DERIVED, complete).** If trial theta' is independent of a fresh iid retained batch R2 (|R2|=n), then
P(rho(theta') > rho_hat + M sqrt(ln(1/d)/(2n))) <= d (Hoeffding). Over T steps with at most Rmax trials each, each trial uses a new
batch (theta' depends only on past batches), union bound with d' = delta/(T Rmax): with prob >= 1-delta every accepted iterate
satisfies rho <= tau^2. No Jacobian, no Taylor remainder, no stale/sketch error enters, because the check evaluates the actual
finite step. Holds for any optimiser; the AdamW-specific part is rolling back (theta,m,v,t) on rejection.
Computed: fresh estimate unbiased (ratio 1.003); with ORACLE M slack/mean = 0.57 at n=200; coverage 1.000 (c3). With
unknown M the bound is not usable: M must be assumed or enforced by per-sample clipping.
**Prediction-side bounds (for choosing the trial scale; DERIVED).** ||f(th+D)-f(th)|| <= ||J D|| + (beta/2)||D||^2 (beta = sup
Hessian-vector curvature on segment); stale J: + L_J ||th-th_s|| ||D||. Sketch lemma: for Gaussian S (k x p, entries N(0,1/k)) drawn
AFTER D, fixed u: P(||Su||^2 <= (1-2sqrt(x/k))||u||^2) <= e^-x (Laurent-Massart), so ||u||^2 <= ||Su||^2/(1-2 sqrt(x/k)) w.p. >= 1-e^-x,
k>4x, independent of P and rank. FAILS if D was constructed to lie in null(SJ): sketched drift 0, true drift not.
Computed: batch-constructed projection has train drift 3e-30, population drift 1.1e-3 (c3) i.e. fresh verification is essential.
**Counterexamples outside assumptions (COMPUTED c3).** (a) Rare subgroup mass p=0.005, n=200: P(unseen)=0.37, T3 silent on it
(T3 certifies the mean, not worst-case prompts). (b) First-order-neutral LoRA gauge step dB=BK, dA=-KA: B dA + dB A = 0 for all x
yet exact drift = ||B K^2 A x|| (|K|=0.3: 0.68, 2.3% of ||BAx||), caught only by a two-point check, not by a JVP gate (C-X2).
(c) f=th1 th2 curvature (C-X1). (d) Stale J: c4 with relative error 0.5/1.0, ungated H-projection drift reaches 2.1x / 3.95x tau
(3 decades); gate holds 1.00 tau at progress 0.18/0.09 of raw, far below the oracle-J penalty (0.32): stale information makes
the method lose to the penalty.
Momentum carry-over: committing m,v from the raw g while applying a corrected step is a design choice; rejection with state
rollback avoids polluting m. Not separately tested (stated as untested).

## 5. Discarded ideas
- "Projection before AdamW is useless": false in general; exactly neutral when D is scalar on null(J) or null(J) coordinate-aligned (T1, c3).
- Sketch-based verification with the construction sketch: invalid (adaptivity).
- Claiming obstruction for nonlinear unlearning: discarded (C-X1).
- Novelty of "post-preconditioner projection": discarded; Adam-NSCL (UNVERIFIED) acts on the Adam update.
- Fisher/GN metric alternative to H: established; not tested.

## 6. Scripts
C_scripts/common.py (AdamW), c1_lora_zero_init.py, c2_order_of_ops.py (+c2_output.txt), c3_theory_checks.py
(+c3_theory_checks_output.txt), c4_pareto.py (+c4_pareto_output.txt). Ledger labels C_e1, C_e2b, C_c3_theory_checks, C_c4b (~5 core-min).
