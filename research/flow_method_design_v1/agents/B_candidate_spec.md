# Agent B candidate specification (loss / output-space geometry)

**Verdict: NO NEW METHOD SURVIVES (new-loss candidate = NONE).** One precise, falsifiable *hypothesis about native GEM*
survives (H-B1, below). Its closest repair is the already-closed axial-margin control, so H-B1 is a *discriminating
test and a diagnosis*, not a novel loss. Generic Taylor/KKT/projection algebra is not claimed as contribution; the
items that are specific (identity (I1), conserved quantity (I2), offset-amplification law (T1)) are exact but are
elementary conic geometry, and I could not check prior literature for them (arXiv blocked; UNVERIFIED novelty).
Evidence tags: DERIVED = proof below, COMPUTED = `B_scripts/b_checks.py` (ledger labels `B_checks`, `B_checks_v2`, ~1.8 core-s each).

## 0. Setting and notation
Per sample i: student output v_i, anchor a_i, target b_i, all in R^d (flattened). Matrices V,A,B in R^{n x d}.
L_i = ||b_i-a_i||, e_i = (b_i-a_i)/L_i, s_i = (v_i-a_i).e_i (axial coordinate, a at 0, b at L), r_i = ||(I-e_i e_i^T)(v_i-a_i)||.
D+ = ||v-a||, D- = ||v-b||, u+ = (v-a)/D+, u- = (v-b)/D-. Exact (eps_n=0) native objective
`J_eta(V) = (gamma/n) sum_i relu(D+_i - eta D-_i)`; gradient row i: `(gamma/n) 1[D+>eta D-] (u+ - eta u-)` (a.e.; undefined at v=b, v=a).
The eps_n all-ones shift is the CLOSED artifact and is deliberately removed here (eps_n -> 0).
Upstream default eta = 1.0 (GEM @ c328fc6, `erasure_gem.py:28`, VERIFIED by `git grep`), so eta=1 is the case that matters.

## 1. Exact structural facts (DERIVED, each COMPUTED)
* **(I1) Gate identity (eta=1).** D+^2 - D-^2 = 2L(s - L/2), hence
  `relu(D+ - D-) = relu(2s - L) * (L / (D+ + D-))`.
  GEM(eta=1) = axial margin (closed control, eta=1) x gate g = L/S in (0,1], S = D+ + D- (eccentricity of the ellipse with foci a,b through v;
  g=1 exactly on the segment [a,b]). max abs error over 2000 random points 1.2e-14 (T1).
* **(I2) Orthogonality / conservation (eta=1).** (u+ - u-).(u+ + u-) = |u+|^2-|u-|^2 = 0, i.e. grad(D+ - D-) is tangent to the
  confocal ellipsoid S = const (T2: max 6.3e-16). So free-output gradient flow v' = -grad J_1 keeps S = D+ + D- constant on the
  active set.
* **(F1) Unbounded flat set.** eta=1: zero set = half-space {s <= L/2}; the loss tends to 0 at fixed s=L (no erasure) as r -> inf:
  loss = sqrt(L^2+r^2) - r ~ L^2/(2r) (T7, ratios 0.83, 0.998, 1.0000 at r=1,10,100). eta<1: zero set = Apollonius ball, centre s_c = -eta^2 L/(1-eta^2),
  radius R = eta L/(1-eta^2) (boundary residual < 4e-15, T6); it contains the truncated-anchor ball radius eta L/(1+eta). eta>1: zero set is the exterior of a ball.
  Orthogonal slack at zero loss: eta L/sqrt(1-eta^2) (0.58 L at eta=.5, 1.33 L at .8, 3.0 L at .95); eta=1: unbounded.
* **(F2) Axis ray beyond b is a stationary plateau at eta=1** (u+ = u- there; loss = L constant, gradient 0; T5 theta=0: flow does not move).
* **(F3) Transverse sign.** Transverse gradient = v_perp (1/D+ - eta/D-). At eta=1 on the active set (D+>D-) it is always negative:
  descent *increases* r (T9: all active rows move away from the axis). So at eta=1 the entire active region repels transversally.
* **(F4) Homogeneity.** J is positively homogeneous degree 1 in (v-a, b-a); gradient norm is scale free; loss/L constant under rescaling (T8). At eta=1
  loss <= L, and for a=b exactly the loss is identically 0 (no anchoring signal), while eta<1 gives (1-eta)D+.

## 2. Hypothesis H-B1 (offset-amplification of transverse noise under GEM(eta=1))
**Statement.** In output space with a student/optimizer whose output-space metric is isotropic (M = J J^T = c I), native GEM at eta=1,
started at a small offset delta (|delta| = eps, angle theta to e, 0<theta<=pi) from b, ends on the bisector hyperplane at transverse
distance exactly `r_end = sqrt(S0^2 - L^2)/2`, with S0 = eps + sqrt(L^2 + 2 eps L cos(theta) + eps^2) (the initial value of D+ + D-).
For delta perpendicular to e, `r_end ~ sqrt(L eps / 2)`: noise of size eps becomes collateral of size sqrt(L eps/2), amplification
sqrt(L/(2 eps)) >> 1 (70.7x at eps=1e-4 L; 7071x at 1e-8 L). The axial-margin control and truncated anchor have r_end <= eps (no amplification).
**Why it matters.** The closed CPR record compared only the leading drift *at* v=b, where the eps_n shift decides the direction and a centered radial
control ties CPR. At any finite offset (bf16 / minibatch / shared-parameter noise, any eps>0) radial repulsion has a distinct, *noise-driven* mechanism
(unstable transverse saddle at b) that does NOT depend on the eps_n artifact and that the centered radial control shares with native GEM but axial repulsion does not.

### Theorem T1 (conditional) and proof
Assume (a) eta=1, eps_n=0; (b) free output gradient flow v' = -c grad J_1(v), c>0, started at v0 = b + delta with 0 < |delta|, v0 not on the axis ray
beyond b, with v(t) staying in the active set until it first reaches {s = L/2}; (c) a,b fixed (detached teachers), L>0.
Claim: S(v(t)) = S0 for all t until the flow hits the zero set; the flow stays in span{e, delta}; at the hit, D+=D-=S0/2 and s=L/2, so r_end = sqrt(S0^2 - L^2)/2.
Proof. dS/dt = grad S . v' = -c (u+ + u-).(u+ - u-) = 0 by (I2). The velocity lies in span{u+,u-} = span{v-a, v-b} which stays in the 2-plane through a,b,v0, so the
trajectory is planar. On the zero-set boundary D+ = D- = S/2 (since D+ - D- = 0 and D+ + D- = S0) and the foci are at distance L, so by Pythagoras at the midpoint
projection r^2 = (S0/2)^2 - (L/2)^2. Existence of the hit: on the ellipse S=S0 the active arc (s > L/2) is bounded away from stationary points except the axis point s=(L+S0)/2 beyond b, which is unstable
(F2/T5); the loss decreases along the flow and the non-axis points have |grad| = |u+ - u-| > 0, with the minimum on the ellipse arc attained on the compact set; hence the flow reaches s=L/2 in finite time for theta in (0,pi]. QED (the finite-time reach is verified numerically in T3/T5, not formalised beyond this argument).
COMPUTED: T3 (eps from 1e-8 to 0.1): r_end vs formula agree to 5e-6 relative; max S drift 1e-12..7e-6 (integrator error); log-log slope 0.5003. T4: identical in d=64 with random orthogonal delta. T5: 7 angles match formula (theta=pi-1e-3 gives r=5e-6, reverse offset gives tiny collateral).

### Explicit counterexamples outside the assumptions (T10, COMPUTED)
| violation | outcome |
|---|---|
| M = diag(1,100) (stiff transverse) | S not conserved (drift 8.8), r_end = 4.86 L (!) vs formula 0.007 L: far worse than T1 |
| M = diag(100,1) (stiff axial) | r_end = 1e-4 = eps (no amplification), s_end 0.27 (overshoot of midplane) |
| eta=0.5, eps=1e-2 | S drifts 0.9%; r_end=0.024 vs eta=1 formula 0.071 (no conservation law; amplification smaller) |
| delta along +e (theta=0) | stuck on plateau: r_end=0, loss L (F2) |
So H-B1 is a statement about isotropic-metric output flow; a real LoRA student has M = J J^T with unknown anisotropy and T10 shows the sign of the effect is governed by the transverse/axial stiffness ratio of M, which is *not* measurable without a real model.

## 3. Candidate-level comparison
* **Closest published method:** native GEM (multimodal-ai-lab/GEM @ c328fc6 `erasure_gem.py:233-249`; the paper's equations UNVERIFIED here). I1 shows GEM(eta=1) is an axial margin times an eccentricity gate; whether the authors know this is UNVERIFIED.
* **Strongest simple alternative:** axial margin relu(L-(1+eta)alpha) (closed control in `losses.py`) and truncated anchor relu(D+ - eta L/(1+eta)). Both have zero transverse repulsion; truncated anchor additionally contracts r (T3: r_end 5e-5 < eps 1e-4).
  Any "fix" of H-B1 is one of these, i.e. **not new**. The gate g=L/S is the only thing GEM adds, and it is exactly what creates F1 (flat at infinity) and H-B1.
* Distinction from CPR: CPR = axial repulsion (r preserved) with radial attraction; H-B1 concerns radial repulsion's finite-offset dynamics. It does *not* show CPR is better downstream; it only isolates a mechanism the closed v=b study could not.

## 4. Matrix-form objective, gradient, pseudocode (for the test harness, not a new loss)
```
# V,A,B: (n,d) float32; dP = ||V-A||, dM = ||V-B|| (eps_n = 0 for analysis; native adds +1e-6 to every entry of the difference)
UP = (V-A)/dP[:,None];  UM = (V-B)/dM[:,None]
act = (dP - eta*dM) > 0
G   = gamma/n * act[:,None] * (UP - eta*UM)          # dJ/dV
S   = dP + dM                                         # diagnostic: conserved at eta=1 under isotropic flow
gate= L/S, L=||B-A||                                  # diagnostic
```
Diagnostics to log per step (cost O(n d), no extra forward passes): S, g=L/S, s, r. Native-implementation: only the already available v,a,b; no extra model calls.

## 5. Concrete numerical prediction and falsification (CPU toy only; no real-model claim)
Prediction P1: in the free-output simulation (`b_checks.py`), r_end/eps follows sqrt(L/(2 eps)) within 1% for eps in [1e-8, 1e-2] L at eta=1 (**already met**, so P1 is a verified theorem, not evidence about FLUX).
Falsifiable real-setting prediction P2 (future protocol, not run): in a LoRA/linear-student toy with *measured* M, collateral r_end/L for GEM(eta=1) exceeds the axial-margin control by factor >= 5 when eps/L <= 1e-3 and M's transverse-to-axial stiffness ratio kappa <= 1.
**Falsified if** the measured ratio r_end(GEM)/r_end(axial margin) is < 2 at eps/L <= 1e-3 with kappa<=1 in the toy, or if on teachers from real flows eps/L at init is >= 0.3 (then amplification <= 1.7x per T3 and the effect is negligible).
Honest status: P1 is arithmetic; P2 untested; the effect may be irrelevant for FLUX because the isotropic-M assumption and the size of eps/L are unknown.

## 6. Cost
Zero extra forward/backward passes; diagnostics O(n d) per step. CPU toy: <2 core-s (ledger).
