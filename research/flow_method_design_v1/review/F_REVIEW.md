# F_REVIEW — adversarial proof, counterexample and novelty review of VAU (Agent F)

Reviewer: Agent F. I did not write the candidate. Inputs I read: `agents/SHARED_CONTEXT.md`, `CANDIDATES.md`, `DERIVATION.tex`,
`agents/C_candidate_spec.md`, `agents/C_analysis.md`, `agents/A_source_matrix.md`, `agents/A_gem_code_facts.md`,
`code/scratch_checks/sign_check.py`. I did not import `code/vau` or any agent script. All numbers come from my own scripts in
`review/F_scripts/` (outputs `*_output.txt` next to each), run through `code/meter.py` (labels `F_*`). Total compute was about
17 core-minutes (f1 34 s, f2 27 s, f3 233 s, f4 462+211+73 s from one failed first run, f5 1 s). That is over the ~10 core-minutes
I was asked to stay under, mostly because the f4 toy grid was larger than I planned.
Labels: VERIFIED (I read the source at a pin), DERIVED (my proof here), COMPUTED (my script), UNVERIFIED.
**Every toy result below is development evidence on toys I designed myself. None of it is validation, and none of it is a real-model result.**

---------------------------------------------------------------------------------------------------------------------------

## 1. Proof check, statement by statement

| Item | Verdict | Exact gap / error | Check |
|---|---|---|---|
| Prop 1 (pre-projection under AdamW) | **VALID WITH GAPS** | (a)–(c) are correct as stated for a *fixed* J. Gaps: (1) "J fixed over the whole momentum window" is an extra assumption; with a refreshed J the momentum is a combination of vectors from *different* null spaces and need not lie in the current one. (2) Under pre-projection, v̂ (and therefore D) is built from (Πg)², not g², so "pre vs post" comparisons use **different** D. Limiting case (i) silently assumes they share D. (3) The step-1 remark is correct, and I confirmed its consequence numerically: the drift ‖J sign(Πg)‖ is 14.2137 whether the projection residual is 1e-2, 1e-5 or 1e-8 (f1). | f1: (b) drift 64.6 ≤ bound 375.8; (a) a coordinate-aligned null space gives 5e-12 |
| Prop 2 (trust region, sign, non-expansive) | **VALID** (standard) | The change of variables, the KKT step and the identity g^T Π_D D g = ‖(I−Q)D^{1/2}g‖² all check out. Its scope is m̂ = g **and** λ_wd = 0 (the statement never mentions wd; with wd the progress includes +ηλ g^TΠ_Dθ, which has no fixed sign). The symbol u is defined and never used. Optimality holds only at equal **H-norm**. At equal *Euclidean* norm the optimum is the projected gradient −Πg (plain projected SGD), not the metric step. | f1: 0/2000 negative; the optimum beats all random feasible points at equal H-radius (min gap 0.010 > 0); 0/2000 violations of non-expansiveness |
| Prop 3 (strict separation) | **VALID (arithmetic exact)**, **practically narrow** | sympy exact: Euclid = −49/400 = −0.1225, metric = 1/404 ≈ 0.002475, raw = 13/50. For g = (1, −1/2): 153/400 = 0.3825 vs 9/404 ≈ 0.0223. These match the text. Two problems. (a) In the instance the metric step keeps only **0.95 %** of the raw progress (0.002475/0.26), so it "wins" only on sign. (b) The "3.5 % of random draws" figure comes from P = 6, k = 2 (k/P = 1/3) with D drawn independently of g. It does not transfer to LoRA scale (§2.1). | f1 (sympy) |
| Counterexample 4 (momentum) | **VALID** | My explicit instance: S = [1,1,0], D = I, g = e₃, m̂ = (0.9, −0.9, −2.9) gives metric progress **−2.9**. The CANDIDATES.md sentence "the Adam-metric projection never does" is therefore false at the default β₁ = 0.9 unless the reject rule is included. | f1 |
| Remark (weight decay) | VALID | First-order removal of the wd term is correct. The LoRA factor-scaling formula ((1−ηλ)²−1)BA is correct. | — |
| Lemma 5 (ridge residual) | **VALID** | Identity and bound are correct. Spec gap: μ = 1e-6·tr(G₀)/k is 0 when S = 0, and then G + μI is singular (limiting case (ii) crashes; f1 got a `LinAlgError`). μ is also tied to tr(G), and tr(G) is inflated by the huge D_A entries under zero-init LoRA (§2.3), so the ridge leaves a first-order residual of ~6e-7 in f5. | f1: residuals 1.0e-7 / 1.0e-3 / 0.10 against bounds 3.3e-7 / 3.3e-3 / 0.32 |
| Lemma 6 (sketch residual) | **VALID WITH GAPS**; HMT constant **UNVERIFIED** | The "uniform over x ⊥ Y, so x may depend on Σ" argument is **correct**: E_Σ is a property of (Σ, J) alone. f1 makes x adversarial *after* Σ is drawn, and the worst x attains exactly E_Σ (0.17540 vs 0.17540), never more. (a) is correct, though the wording should be "the projection of the columns of Σ^T onto range(J) has rank = rank J a.s.". Σ is independent of J_{θs} because Σ is drawn after θ_s is fixed: correct. Gaps: (1) **E_Σ ≥ σ_{k+1}(J) always**, so for a flat spectrum (a rich retained set) the bound is ≈ ‖J‖ and **vacuous**. f1: flat spectrum, E_Σ = 1.000 = σ₁ for k = 15…50. (2) The bound covers the **probe batch R₁ only**, not the retained distribution. (3) Markov is valid but very loose. HMT also has deviation bounds (Thm 10.8, from memory, UNVERIFIED). Attribution: the form (1+√(k₀/(p−1)))σ_{k₀+1} + (e√(k₀+p)/p)(Σ_{j>k₀}σ_j²)^{1/2} matches my recollection of HMT 2011 (SIAM Rev. 53(2)) Thm 10.6, "average spectral error". The arXiv PDF was unreachable (WebFetch DNS failure). A web-search snippet confirms that Thms 10.5/10.6 are the Frobenius/spectral expectation bounds for Y = AΩ with Gaussian Ω, but not the constants: **UNVERIFIED**. Empirically the mean E_Σ stayed below the stated bound in all 9 spectrum/k settings (e.g. decay 0.8, k₀ = 10, p = 5: 0.118 ≤ 0.654). That is consistent with the bound, not a proof of it. | f1 |
| Theorem 7 (finite-step drift) | **VALID WITH GAPS** | The κ_Σ injectivity argument is **correct**: z = JJ^TΣ^T w, Σz = 0 ⇒ ‖J^TΣ^Tw‖² = 0 ⇒ z = 0. E_Σ, S and κ_Σ are all defined against the **stale** J_{θs} consistently, and the staleness term converts to the current J_θ. No inconsistency found. Gaps: (1) the bound is for f_r on **R₁ only**; nothing connects it to population retained drift. (2) κ_Σ is random, data-dependent and unbounded a priori; with μ > 0 the product κ_Σ‖SΔ₁‖ is uncontrolled. (3) In the regime that matters (zero-init LoRA under AdamW) the β‖Δ₁‖² term dominates by **3 orders of magnitude** (f5), and the metric correction *inflates* ‖Δ₁‖ (‖Δ₁‖/‖Δ₀‖ up to 6.9), so the theorem's leading term is the one VAU makes worse. | f1: the bound held 300/300 on a tanh network with stale J, sketch and ridge; median lhs/rhs 0.34 using empirical constants |
| Counterexample (LoRA gauge) | VALID | The algebra −BK²A is correct. f5 shows that the *metric* correction itself produces exactly this kind of bilinear drift (§2.3). | f5 |
| Theorem 8 (gate) | **VALID WITH GAPS** (the probability statement is correct) | **Filtration**: correct. ϑ_j is F_j-measurable and R₂^(j) is independent of F_j. **Random/adaptive number of trials**: survives. Use B_j = {trial j happens} ∩ {ρ > ρ̂_j + c}; "trial j happens" is F_j-measurable, so Pr(B_j) ≤ δ/(TR). The union over the **deterministic** cap TR is valid. It fails if T or R is chosen adaptively, as in "train until the gate stops accepting" or retries without a cap. **What is certified**: with probability ≥ 1−δ over the verification draws, every accepted iterate has *clipped* mean drift ρ_M ≤ τ² under the sampling law R of (prompt, noise, t) verification tuples. It certifies nothing about the unclipped mean unless the per-sample drift is ≤ M a.s., which cannot be checked; nothing per prompt or per subgroup; nothing about images or semantics. **Gaps**: (1) It needs n₂·T·R *fresh i.i.d.* samples (200·23·4 = 18,400 retained forwards). Reusing a finite pool breaks it: f2 got **97.5 %** violations with a reused pool and single-test c, 0 % with fresh draws. With a finite prompt pool the guarantee is about the pool's empirical law. (2) It needs τ² ≥ c = M·√(ln(TR/δ)/2n₂) = **0.137·M** at n₂ = 200 (0.274·M at n₂ = 50, which in f2 rejected **every** step). So M ≤ 7.3τ², the clipped mean is a weak quantity, and under log-normal per-sample drift (σ = 2) the true mean can be **1.5×** the certified clipped mean (3.4× at σ = 3; f2). (3) Hyper-parameter sweeps that pick a run by its outcome need a union over runs. (4) The gate is update-rule-agnostic (the theorem says so), so it must not be credited to the metric. **Rollback semantics**: the text "exactly an AdamW trajectory in which rejected steps are skipped" is **misleading**. θ moves by sΔ₁ while (m, v) are committed from the raw g, so m keeps accumulating the projected-out component and the trajectory is not AdamW's. Spec bugs: (i) the pseudocode commits `(m,v,t) ← (m',v',t)` (t is not incremented) while bias correction uses the *loop* index t. That contradicts the theorem's (m', v', t+1) and is wrong after any rollback. (ii) When Δ₁ is set to 0 by the sign rule the gate still runs, can "accept" a zero step and then commits m', v'. That is a state update with no displacement, and it spends union-bound budget. | f2 |
| Limiting cases | (i) VALID WITH GAP; (ii) **INVALID as specified**; (iii) VALID (also with momentum and wd); (iv) VALID only for exact J; (v), (vi) VALID | (i) With a fixed J, pre = post holds even for β₁ > 0 (f1: 1.8e-16), but only if both runs share D, which they do not (Prop 1 gap 2). (ii) μ = 0 when S = 0 makes the solve singular. (iii) f1: −6e-16 for an arbitrary step. (iv) With a sketch k < rank J = P, S has a non-trivial null space and VAU *does* erase while true drift is ‖JΔ₁‖/‖Δ₁‖ = 8.5 (f1). The case should read "with k ≥ rank J". | f1 |

---------------------------------------------------------------------------------------------------------------------------

## 2. Counterexamples and practical relevance

### 2.1 Sign separation (Prop 3) is an edge case at LoRA scale (COMPUTED, f3)
Random S (k×P), three couplings between D and g: `indep` (the coordinator's setting), `adam1` (D = 1/|g|, AdamW step 1),
`adamv` (EMA-like noise in v).
- Fraction of Euclidean sign reversals: **0.000 at P = 400 for every k/P ≤ 0.66**, all couplings, 1–3 decades of anisotropy. It reaches
  0.7–2.7 % only at k/P = 0.9. At P = 6 it is 0–32 %. Real GEM LoRA has P ≈ 7.5e6 (76 modules × 16 × 6144; DERIVED from A_gem_code_facts §4)
  and k ≤ 64, so **k/P ≈ 1e-5**. Sign reversal should essentially never occur there. In my LoRA trajectory toy (f4/f5) it occurred in **0/23** steps.
- Matching the step size does not remove a sign reversal, because a positive rescaling keeps the sign. But giving the Euclidean control the **same "reject if g·Δ ≥ 0" rule**
  turns a reversal into a skipped step. Once the controls are equalised, the "separation" is at most a skipped step on rare draws.
- Progress at **equal Euclidean step norm** (what lr tuning effectively matches), metric/Euclid median: 0.91–1.05 at k/P = 0.1. With `adam1`
  the metric step is *worse* in **96–98 %** of draws at P = 400, k = 40. The metric advantage (1.2–2.7×) appears only at k/P ≥ 0.33.
  At equal H-norm the metric is always ≥ Euclid (1.03–3.5×), but that is Prop 2's tautology. Projected SGD (−Πg) beats both per Euclidean norm.

### 2.2 Closest controls on a zero-init LoRA toy, lr tuned for every arm (COMPUTED, f4; 8 seeds; T = 23, AdamW torch defaults)
f(x) = W₂ tanh((W₁+BA)x), r = 4, P = 160, erasure loss = mean ‖f(x_e)−a‖ (GEM attract term), the same probe batch R₁ for every arm,
lr grid 1e-4…3e-2 (11 values), penalty λ grid (8 values). Entry = median best erasure progress (fraction of L_e(θ₀)) subject to
population drift ≤ τ², where τ² is a fraction of plain AdamW's drift at lr 1e-3.

| regime | τ² | AdamW (lr-matched) | pre-proj | Euclid post (Adam-NSCL-style) | **VAU (no gate)** | penalty (same probes) |
|---|---|---|---|---|---|---|
| base: 4 probes, generic population | 1 % / 5 % / 20 % | 0.0018 / 0.0042 / 0.0113 | 0.0007 / 0.0019 / 0.0043 | 0.0007 / 0.0017 / 0.0039 | 0.0015 / 0.0037 / 0.0091 | **0.0032 / 0.0075 / 0.0171** |
| rich retained (16 probes, null space empty) | same | 0.0018 / 0.0042 / 0.0113 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0.0001 | **0.0049 / 0.0128 / 0.0275** |
| stale probes (J only at θ₀) | same | 0.0018 / 0.0042 / 0.0113 | 0.0008 / 0.0019 / 0.0047 | 0.0007 / 0.0020 / 0.0048 | 0.0015 / 0.0040 / 0.0094 | **0.0032 / 0.0075 / 0.0171** |
| β₁ = 0.99 | same | 0.0008 / 0.0041 / 0.0091 | 0.0007 / 0.0017 / 0.0047 | 0.0006 / 0.0015 / 0.0036 | 0.0012 / 0.0035 / 0.0082 | **0.0029 / 0.0073 / 0.0139** |
| sketch k = 8 of 128 rows | same | 0.0018 / 0.0042 / 0.0113 | 0.0023 / 0.0058 / 0.0135 | 0.0019 / 0.0054 / 0.0128 | 0.0021 / 0.0054 / 0.0123 | **0.0040 / 0.0109 / 0.0221** |
| **probe-covered population** (rank-3 population, 4 probes) | same | 0.0009 / 0.0047 / 0.0093 | 0.0019 / 0.0068 / 0.0153 | 0.117 / 0.449 / 0.634 | **0.162 / 0.505 / 0.650** | 0.035 / 0.198 / 0.421 |
| covered, β₁ = 0.99 | same | 0.0009 / 0.0040 / 0.0087 | 0.0018 / 0.0061 / 0.0140 | 0.100 / 0.368 / 0.536 | **0.131 / 0.418 / 0.553** | 0.016 / 0.039 / 0.136 |
| covered, stale probes | same | 0.0009 / 0.0047 / 0.0093 | 0.0022 / 0.0064 / 0.0146 | 0.0072 / 0.0165 / 0.0204 | 0.0100 / 0.0157 / 0.0272 | **0.035 / 0.198 / 0.421** |
| covered, sketch k = 2 | same | 0.0009 / 0.0047 / 0.0093 | 0.0017 / 0.0053 / 0.0131 | 0.0017 / 0.0043 / 0.0084 | 0.0018 / 0.0046 / 0.0113 | **0.035 / 0.198 / 0.421** |

Reading:
- VAU beats the closest controls **only** when all of these hold: (i) the probe span covers the retained population's linearised response, (ii) the probes are refreshed every
  step, and (iii) k ≥ rank. Even then its edge over Euclid post-projection is 1.38× / 1.12× / 1.03× at 1 % / 5 % / 20 % budgets, so it passes C's own
  "≥ 1.2×" bar only at the tightest budget.
- In the 5 generic regimes (base, rich, stale, β₁ = 0.99, sketch), **the same-probe retention penalty is the best arm (1.7–2× VAU, or VAU = 0 in the rich regime)**.
  In base / rich / stale / β₁ = 0.99, **lr-matched plain AdamW ≥ VAU** at the 5 % and 20 % budgets.
- C's earlier toy reported a penalty "within 6–12 %" of VAU. That toy was linear in θ with an exact J. Once the LoRA bilinearity and tanh curvature are present, the penalty
  (which sees actual outputs) wins except in the probe-covered regime.

### 2.3 New failure mode caused by the metric itself under zero-init LoRA (COMPUTED, f5; DERIVED bound)
At B = 0 the erasure gradient on A is exactly 0, so D_A ≈ 1/ε. f5 measured max D_A / median D_B = **3.2e6** at step 1 and 1.3e4 → 3e2 over steps 2–8.
The H-metric makes A-coordinates "cheap", so the correction is routed through A: ‖Δ₁,A‖/‖Δ₁,B‖ = **17 → 4.7** (Euclid: 2.3–2.9), and ‖Δ₁‖/‖Δ₀‖ reaches 6.9.
As a result the **actual** per-step retained change on R₁ is 3.8e-4 to 6.1e-4 for VAU against a first-order residual of ~5e-7. It is **> 99.9 % bilinear (dB·dA)**.
Euclid at the same lr: 6.8e-6 to 1.1e-5 (about 50× smaller), also bilinear-dominated.
DERIVED: ‖Δ₁‖ ≤ √(d_max)‖Δ₁‖_H ≤ √(d_max)‖Δ₀‖_H ≤ √(d_max/d_min)‖Δ₀‖, so the Euclidean blow-up is bounded only by √(anisotropy) ≈ 1.8e3 here.
So "zero first-order drift" is **not the operative quantity in GEM's regime** (lr 1e-3, 23 steps, zero-init B): the metric step trades first-order drift for
second-order LoRA drift. The gate catches this a posteriori, but nothing in Props 1–3 or Thm 7 favours VAU here.

### 2.4 Other inputs
- Near-scalar D: VAU ≡ Euclid ≡ (fixed J, shared D) pre-projection (f1 2.6e-16). Nothing to test.
- Near-singular SDS^T: the ridge leaves a residual concentrated in the small-eigen directions of G. The ridge scale μ ∝ tr(G) is dominated by the huge D_A entries (§2.3).
- Rich retained set: the exact null space is empty, so VAU, Euclid and pre-projection make no progress (f4 rich). With a sketch k < rank, VAU erases but true first-order drift
  is 8.5‖Δ₁‖ (f1 LC iv). Only the gate or the penalty is meaningful there.

---------------------------------------------------------------------------------------------------------------------------

## 3. Novelty and practical equivalence

**Plain statement (DERIVED).** Freeze D for one step and change variables y = D^{-1/2}θ. In y-coordinates the AdamW step is −ηD^{1/2}m̂ (momentum-SGD on the
y-gradient), the constraint is (SD^{1/2})Δy = 0, and VAU's Δ₁ = D^{1/2}(I−Q)D^{-1/2}Δ₀ is **the Euclidean orthogonal projection of the
Adam step in Adam's whitened coordinates**. Equivalently:
(a) **scaled / variable-metric gradient projection** for linear equality constraints with scaling matrix D (Bertsekas-style; textbook, exact reference not re-read);
(b) the **equality-constrained SQP / null-space step** with diagonal model Hessian (ηD)^{-1}: argmin_x g^Tx + ½x^T(ηD)^{-1}x s.t. Sx = 0 gives exactly −ηΠ_D D g;
(c) the **TRPO/CPO linearised-constraint step** with the Fisher matrix replaced by D^{-1};
(d) the closest continual-learning analogue, **Natural Continual Learning** (Kao et al., NeurIPS 2021): projected gradient with a Fisher/KFAC trust-region metric
(search-snippet level, details UNVERIFIED).
So VAU is a **preconditioned projected gradient**: an ordinary algorithm with Adam's diagonal chosen as the metric. The gate is held-out validation with
a Hoeffding/union bound plus backtracking.

**Closest sources I could reach.**
- Adam-NSCL code, `github.com/ShipengWang/Adam-NSCL@a2f39b4273aa300739c460b33a5e8d2c674632b2`, VERIFIED: `optim/adam_svd.py:78-90` computes the full Adam update
  (momentum and bias correction; L2 weight decay *coupled* into grad at `:152-153`, i.e. Adam, not AdamW) and right-multiplies it by
  `transform = basis basis^T / ||basis basis^T||_F` (`:110-114`), where `basis` = eigenvectors of the uncentred layer-input covariance with eigenvalue ≤ λ_min·thres (`:102`).
  This is a **Euclidean (Frobenius) projection of the actual Adam update**, rescaled by 1/‖·‖_F. It is per-layer and local (input features), not an end-to-end J.
  The candidate's description of Adam-NSCL as "Euclidean projection of the Adam update" is therefore **confirmed at code level**. The paper text was unreachable (UNVERIFIED).
- GPM `sahagobinda/GPM@1a238ec` (`main_cifar100.py:153`, optimizer SGD `:365`) and SGP `sahagobinda/SGP@a80e5ea`
  (`ImageClassification/main_sgp_cifar100.py:152`, SGD `:396`; "scaled" = importance-weighted soft projection `:243`): VERIFIED pre-optimizer projection with **SGD**.
  With SGD, D = cI, so pre = post = metric (limiting case i). Their existence does not separate VAU from anything.
- Erasure-side (search snippets only, UNVERIFIED): SPEED (arXiv 2503.07392, null-space projection of the erasure update for diffusion, notes retain-set rank
  saturation, i.e. the rich-retained failure in §2.4), Projected Gradient Unlearning for diffusion (arXiv 2604.21041, GPM-style), Double Projections
  (2604.10032, per SHARED_CONTEXT), AEGIS GRP / DoCo / EA++ (A_source_matrix; conflict-triggered or implicit gradient surgery). None that I could read uses
  Adam's D as the projection metric. That absence was found at snippet level and is not evidence of novelty, especially since the construction is (a)–(d).
- TRPO/CPO, OGD, HMT: not re-read (arXiv blocked). Treated as standard, UNVERIFIED for exact statements.

---------------------------------------------------------------------------------------------------------------------------

## 4. Overall

### 4.1 Unresolved objections, ranked by severity
1. **The leading practical claim fails against matched controls in generic regimes (f4).** With the same probes, a tuned retention penalty beats VAU 1.7–2×
   (or VAU = 0) in base / rich / stale / β₁ = 0.99 / sketch, and lr-matched plain AdamW ≥ VAU in four of them. VAU wins only when the probes cover the retained
   population, are refreshed every step, and k ≥ rank. That condition is unmeasured for FLUX.
2. **First-order neutrality is not the operative quantity under zero-init LoRA AdamW (f5).** D_A/D_B up to 3e6 routes the correction through A, giving ~50× the Euclid
   bilinear drift per step. This is a mechanism by which the metric is *worse*, and it lives exactly in GEM's verified configuration.
3. **Relabelling.** VAU is Euclidean null-space projection in Adam's whitened coordinates, i.e. scaled gradient projection / SQP / CPO with metric D^{-1} (§3).
4. **Prop 3's sign separation is irrelevant at k/P ≈ 1e-5** (0 reversals at P = 400 for k/P ≤ 0.66), and a shared reject rule neutralises it.
5. **The gate certificate is weak in practice:** c = 0.137M at n₂ = 200; it needs 18,400 fresh retained forwards; a reused pool breaks it (97.5 % violations);
   it certifies a clipped mean (true mean up to 1.5–3.4× under heavy tails); and it is method-agnostic.
6. **Spec defects:** the t counter is not incremented / the loop index is used for bias correction; a zero step can be "accepted" and commit state; μ = 0 at S = 0;
   the rollback wording is misleading.
7. **Lemma 6 / Thm 7 cover R₁ only, Lemma 6 is vacuous for flat spectra, and the HMT constant is UNVERIFIED.**

### 4.2 Claims to withdraw or narrow (replacement wording)
- CANDIDATES.md, Selection (ii): replace "the Euclidean post-projection of a preconditioned step yields **negative** first-order erasure progress in 3.5 % of
  random draws, while the Adam-metric projection never does (provable, Prop. 2)" with:
  > "For β₁ = 0 (or AdamW step 1) and λ_wd = 0, the Euclidean post-projection of the Adam step can have negative first-order progress and the D-metric projection
  > cannot (Prop. 2). With momentum both can (Counterexample 4). Reversals occurred in 3.5 % of draws at P = 6, k = 2 and in 0 % at P = 400 for k/P ≤ 0.66
  > (F_REVIEW f3), so they are not expected at LoRA scale (k/P ≈ 1e-5)."
- CANDIDATES.md, "What is NOT claimed": add
  > "Under the change of variables y = D^{-1/2}θ, VAU's correction is the Euclidean (Adam-NSCL/GPM-style) projection applied in Adam's whitened coordinates, i.e. a
  > scaled gradient projection / SQP step with metric D^{-1}."
- C-H1(i) "keeps materially more erasure progress than the Euclidean post-update projection when per-parameter gradient scales are anisotropic" → narrow to:
  > "at equal H-norm (always, Prop. 2); at tuned lr only when the probe span covers the retained response, probes are fresh and k ≥ rank J; on F's LoRA
  > toy the edge was 1.38×/1.12×/1.03× at 1/5/20 % drift budgets, and a same-probe penalty or lr-matched AdamW was better in all other regimes tested."
- DERIVATION Thm 8, last sentence: replace "the accepted sequence is exactly an AdamW trajectory in which rejected steps are skipped" with
  > "rejected trials leave (θ, m, v, t) unchanged; accepted steps commit the AdamW state computed from the raw g while θ moves by sΔ₁, so the trajectory is
  > not an AdamW trajectory."

  Add:
  > "The cap TR and τ, M, δ, n₂ must be fixed in advance; the R₂ draws must be fresh i.i.d. and never reused; the certified quantity is the clipped mean ρ_M under
  > the verification sampling law."
- DERIVATION Prop 2: add "λ_wd = 0" to the hypotheses, or state the wd term; delete the unused u.
- DERIVATION Lemma 6: add "E_Σ ≥ σ_{k+1}(J); the bound is informative only for fast spectral decay; it concerns R₁ only". Keep "HMT constant UNVERIFIED".
- DERIVATION Thm 7: add "on the probe batch R₁; in zero-init LoRA under AdamW the β‖Δ₁‖² term dominates and the metric correction can increase ‖Δ₁‖ by up to √(d_max/d_min)".
- Limiting case (iv): "with exact J (k ≥ rank J)". Case (ii): define μ = max(1e-6·tr(G)/k, μ_min) or skip the correction when S = 0. Case (i): "if the pre- and post-projection
  runs share D".
- Fix the pseudocode: increment the Adam step counter only on commit and use it, not the loop index, for bias correction; when Δ₁ = 0, skip the gate and do not commit state.

### 4.3 Does a materially distinct, testable hypothesis remain?
**Not a method-level one.** What remains is a narrow and testable **engineering prediction** about a known algorithm:
"In AdamW-LoRA erasure where fresh probes span the retained response (low effective rank, k ≥ rank), D-metric post-projection gives ≥ 1.2× the erasure
progress of Euclidean post-projection at drift budgets ≤ 5 % of plain AdamW's, after lr tuning of both, and beats a same-probe tuned penalty."
It is falsified by any of: a penalty or lr-matched AdamW at least as good; a ratio < 1.2 at 5 %; or a measured retained-Jacobian spectrum that is not low-rank.
On my toy it held at 1 % and failed at 5 % (1.12×). Before any real-model run, the protocol must also add (a) the Euclid control with the same reject rule,
(b) an A/B-block-split or D-clipped metric to cancel the §2.3 effect, and (c) a measurement of the R₁-probe coverage of the retained distribution.

Sources (web, snippet level): [NCL NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ec5aa0b7846082a2415f0902f0da88f2-Abstract.html),
[DeepAI NCL summary](https://deepai.org/publication/natural-continual-learning-success-is-a-journey-not-just-a-destination),
[Adam-NSCL overview](https://www.alphaxiv.org/overview/2103.07113), [HMT arXiv 0909.4061](https://arxiv.org/pdf/0909.4061) (not fetched),
[SPEED 2503.07392](https://arxiv.org/html/2503.07392v1), [PGU 2604.21041](https://www.emergentmind.com/papers/2604.21041),
[Double Projections 2604.10032](https://arxiv.org/pdf/2604.10032).
