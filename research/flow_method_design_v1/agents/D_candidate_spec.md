# Agent D candidate specification (trajectories, timesteps, state coverage)

**Verdict: NONE (no new method).** GEM at c328fc6 already trains on states visited by its own current student: an on-policy rollout refreshed every iteration (`EG:33,169-170,180-189`). It sums deterministically over a fixed prefix of 11 of 28 steps, sigma in [0.77, 1].
* On-policy/DAgger coverage, stale-refresh variants, noised-data states and contrast-proportional timestep sampling are each either native GEM, equivalent to it in expectation, or an objective change with no shown benefit.
* Endpoint-sensitivity weighting is the linearised endpoint (backprop-through-sampler) objective, which is established (UNVERIFIED citations).

What survives is **one hypothesis about native GEM (H-D)** with proved conditional results (D-T1, D-C2, D-P1, D-P2) and a cheap real-model **ablation**. It is not presented as a contribution. Its "fix" is the simple alternative: uniform coverage of all 28 student steps.

## 1. Setting
Rectified flow: x_sigma = (1-sigma) x0 + sigma eps, and the velocity target is eps - x0.
The sampler integrates dx/dsigma = f(x, sigma) from sigma=1 down to 0. With the exact posterior-mean field, u(x,sigma) = (x - E[x0 | x_sigma = x]) / sigma.
Tweedie gives d_x E[x0|x] = (1-sigma) Cov[x0|x_sigma=x] / sigma^2. Hence
**d_x u = (I - (1-sigma) Cov[x0|x] / sigma^2) / sigma** (DERIVED; check for Gaussian N(m, s^2 I): d_x u = d log sigma_tot / d sigma with sigma_tot^2 = (1-sigma)^2 s^2 + sigma^2).

Student on the target prompt: v = u + Delta, where u is the frozen target teacher b and Delta is the LoRA-induced change. Reference flow map: Phi_{sigma->0}. Student trajectory: x^(sigma). Reference trajectory from the same eps: x(sigma).

## 2. Hypothesis H-D (two sentences)
GEM's per-step loss on the student's own prefix states (sigma >= 0.77) reaches the generated endpoint only through (a) LoRA parameter sharing that carries the velocity change into steps 11..27, or (b) the state crossing a basin boundary inside the prefix. Otherwise the late target-teacher flow contracts the edit by the factor exp(int_0^sigma1 mu) (= s/sigma_tot(sigma1) for Gaussian data).
Therefore, at matched forward compute, covering all 28 student steps (a random 11-of-28 subset per iteration) erases at least as well as the prefix. It erases strictly better when the concept commits late (low sigma) or when timestep sharing is weak.

## 3. Proved conditional results
### D-T1 (endpoint = integrated local error on student states; explicit constant)
**Assumptions.**
* (A1) u is C^1 in x on [sigma_min, 1] (sigma_min > 0, or data with a non-degenerate density so that u is finite at sigma -> 0).
* (A2) Both ODEs have solutions on [0,1] from the same eps.
* (A3) Delta is measurable and bounded along x^.

**Claim.** x^(0) - x(0) = -int_0^1 J(sigma) Delta(x^(sigma), sigma) dsigma, with J(sigma) = d_y Phi_{sigma->0}(y) evaluated at y = x^(sigma). Hence
||x^(0) - x(0)|| <= int_0^1 exp(int_0^sigma mubar(tau) dtau) ||Delta(x^(sigma),sigma)|| dsigma,
where mubar(tau) = sup_x mu_2(-d_x u(x,tau)) = sup_x (-1 + (1-tau) lambda_max(Cov[x0|x_tau=x]) / tau^2) / tau.
The Lipschitz-Gronwall bound is the same expression with mubar replaced by sup ||d_x u||.

**Proof.**
1. Let psi(sigma) = Phi_{sigma->0}(x^(sigma)). Then psi(1) = Phi_{1->0}(eps) = x(0) and psi(0) = x^(0).
2. The flow property Phi_{sigma->0}(Phi_{1->sigma}(z)) = Phi_{1->0}(z) is independent of sigma. Differentiating it gives d_sigma Phi_{sigma->0}(y) + d_y Phi_{sigma->0}(y) u(y,sigma) = 0 for every y.
3. By the chain rule, d psi / d sigma = d_sigma Phi(x^) + d_y Phi(x^) (u(x^) + Delta(x^)) = J(sigma) Delta(x^(sigma), sigma).
4. Integrate from sigma=1 to 0.
5. For the bound: Z(tau) = d_y Phi_{sigma->tau}(y) solves dZ/d(-tau) = -d_x u(Phi_{sigma->tau}(y), tau) Z with Z(sigma) = I. The log-norm inequality ||Z(0)|| <= exp(int_0^sigma mubar) follows (standard: d||Z w||/d(-tau) <= mu ||Z w||).
6. mu_2 of the symmetric matrix -d_x u is its largest eigenvalue, which is the stated formula. QED.

COMPUTED (`D_bounds` S1a), Gaussian with s in {0.3, 0.1, 0.03} and Delta = c on [0.772, 1]:
* The log-norm bound equals the exact error to <2e-3 relative.
* The Gronwall bound over-estimates by 9.5x, 79x and 870x, approximately (sigma_tot(sigma1)/s)^2.
* **When D-T1/Gronwall is loose:** contractive flows (small posterior covariance, mid/low sigma), where mu << ||d_x u||. Gronwall is never tight for rectified flow toward concentrated data.

**Corollary (prefix-only edits).** If Delta = 0 for sigma < sigma1, then ||x^(0) - x(0)|| <= sup_{sigma >= sigma1} exp(int_0^sigma mubar) * int_{sigma1}^1 ||Delta||. For Gaussian data the factor is s/sigma_tot(sigma1) (0.13 at s=0.1, sigma1=0.772).
COMPUTED on the discrete 28-step Flux schedule (S1b): endpoint/state-shift ratios 0.36, 0.10, 0.016 against bounds 0.40, 0.13, 0.04.

**Counterexample outside A1-linear regime (S1c).** Take bimodal data (separation 6, s=0.1) and constant offset c on the prefix. The first-order prediction (J evaluated on the reference trajectory, the "off-policy" linearisation) has relative error 0.007 at c=0.01 but 0.86-0.94 for c >= 0.1, with 0.5-20% mode flips.
Near a basin boundary mubar > 0, and the local-to-endpoint map is neither linear nor contractive. This is where erasure must act, so no local-loss certificate of endpoint erasure follows from D-T1 alone.

### D-C2 (exposure gap inside GEM's window is bounded)
For sigma in W = [sigma1, 1], applying D-T1 with Phi_{sigma'->sigma} gives
||x^(sigma) - x(sigma)|| <= C_W int_sigma^1 ||Delta|| <= C_W (1-sigma1) sup_W ||Delta||, with C_W = exp(int_W max(0, mubar)).
On GEM's window, (1-tau)/tau^2 <= 0.228/0.772^2 = 0.383. So **C_W = 1 whenever lambda_max(Cov[x0|x_tau]) <= tau^2/(1-tau), which is >= 2.61 on W.**
* For any per-state loss with x-Lipschitz constant K_l, |E l(on-policy) - E l(teacher-rollout)| <= K_l C_W 0.228 sup_W ||Delta||.
* Whether FLUX latents satisfy lambda_max <= 2.61 at sigma >= 0.77 is UNVERIFIED. Correlated latent directions may violate it.
* COMPUTED: toy endpoint scores for on-policy, teacher-rollout and noised-data states agree within seed spread (<= ~0.1).

### D-P1 (noised data vs teacher rollout)
Suppose the teacher field is the exact marginal velocity of its data law and the sampler is exact. Then the teacher-rollout state at sigma has law p_sigma = law of (1-sigma)x0 + sigma eps.
Proof: the continuity equation (standard rectified-flow marginal-preservation property).
So for a per-state loss l(x, sigma) the two coverages give identical expected objectives.
For GEM they differ only through D-F4 (rollout guidance 3.0 vs loss 1.0), student-vs-teacher drift (D-C2), and discretisation. This is a statement about expectations; noise pairing is irrelevant because the loss is pointwise in x.

### D-P2 (timestep importance sampling)
Let J = sum_k p_k l_k(theta), with g_k = grad l_k and gbar = sum p_k g_k. Sample k ~ q.
* (a) The estimator (p_k/q_k) g_k is unbiased if and only if q_k > 0 wherever p_k g_k != 0. Its trace variance is sum_k p_k^2 ||g_k||^2 / q_k - ||gbar||^2, minimised by q_k proportional to p_k ||g_k|| (Cauchy-Schwarz).
* (b) The unweighted estimator g_k has expectation grad J_q, a different objective unless q = p.
* (c) For GEM's hinge, the output gradient norm |u+ - eta u-| lies in [0, 1+eta] independent of the contrast L_k = ||b-a|| (B-D7). So q proportional to L_k is not the optimal q. As L_k -> 0 with g_k != 0, the variance term p_k^2 ||g_k||^2 / q_k -> infinity.
* (d) If the minimiser set Z = intersection of {l_k = 0} is non-empty, then every reweighting with q_k > 0 has the same global minimiser set Z. The objective change acts only through the finite-step path (23 AdamW steps) and through conflicts under finite capacity.

COMPUTED: relative variance uniform 0.48%, contrast-q 0.67%, gradient-norm-q 0.45%. Unweighted contrast-q cosine 0.99999.
GEM sums all 11 steps, so its timestep-sampling variance is exactly 0.

## 4. Objective, estimator, pseudocode for the ablation (native loss unchanged)
Inputs: target/anchor embeddings, a LoRA student, the frozen base (teacher = adapter off), N=28, budget B_F in forward-equivalents.
State per iteration: one noise eps (as in GEM; the toy used 32) and its student rollout.
```
for it in range(I):                         # I = floor(B_F / F_iter)
    eps = randn(latent_shape)               # train_generator, as GEM
    K   = sorted(random.sample(range(28), 11))         # native: K = [0..10]
    with no_grad, adapter ON: traj = rollout(eps, steps 0..max(K), guidance=3.0)   # same as EG:180-189
    for chunk in chunks(K, 6):              # EG:202-204
        x = stack(traj[k+1] for k in chunk); t = sigma_idx(chunk)   # keeps GEM pairing (latent after step k at index k)
        a = teacher(x, t, anchor); b = teacher(x, t, target)        # no_grad, adapter off, guidance 1.0
        v = student(x, t, target)
        loss = gamma * relu(||v-a|| - eta*||v-b||).sum() / len(K); loss.backward()
    adamw.step(); zero_grad()
```
Dimensions: x has shape [|chunk|, 1024, 64] at 512 px. v, a and b have the same shape, flattened per sample for the norms. Nothing else changes.

## 5. Cost (symbolic; F = one FLUX transformer forward on one latent at 512 px; backward ~2F is ASSUMED)
| scheme | rollout F | loss-pass F | F / iter | iters at native compute |
|---|---|---|---|---|
| native prefix (K = 0..10) | 11 | 11 x (2 teacher + 3 student) = 55 | 66 | 23 (budget 250 timesteps) -> ~1518F |
| uniform 11-of-28 | E[max K]+1 ~ 26-28 | 55 | ~82 | 18 |
| full 28 window | 28 | 140 | 168 | 9 (budget 250) |
| stale refresh R | 11/R | 55 | 55 + 11/R | saving <= 17% |
| IS with m of 11 | 11 | 5m | 11 + 5m | variance cost per D-P2 |
| endpoint-sensitivity weights | 11 + 17 per probe | +>=51 per weighted step (VJP through 17 late steps) | >= 2x native | established linearised endpoint objective |

No GPU timing is claimed.

## 6. Predictions and falsification
* **Toy (already run, pre-registered P-D4, `D_matched_compute`).** At matched forwards, uniform 11-of-28 beats the prefix by > 0.05 in >= 7 of 12 cells. Outcome: 6 wins, 0 losses, 6 ties, so **INCONCLUSIVE** (falsification threshold <= 4 not met).
  The earlier P-D2 (commitment-window ordering) and P-D3 were **FALSIFIED** (D_analysis 2.4).
* **Remaining CPU-testable prediction P-D5 (not run).** Use the same toy with a 2-layer MLP student (shared over t, input [x, sigma]) instead of an explicit Delta.
  * Measure the late-step transport ratio rho_late = mean_{k>=11} ||Delta|| / mean_{k<=10} ||Delta|| along student trajectories after native training.
  * Prediction: across 8 seeds x 3 cases, prefix-minus-uniform score difference is positively correlated with (1 - rho_late), Spearman >= 0.5.
  * Falsified if Spearman < 0.2, or if the prefix beats uniform11 by > 0.05 in any cell with rho_late < 0.5.
* **Real-model protocol (future, not run).** Hold GEM fixed at c328fc6 defaults. Compare end_at=10 against end_at=27 with a random 11-of-28 subset at matched forwards (23 vs 18 iterations).
  * Log rho_late = ||v - b|| on steps 11..27 relative to steps 0..10, along student rollouts.
  * H-D predicts the uniform arm's concept-detector rate is <= the prefix arm's at equal retention.
  * It also predicts that the gap grows as rho_late falls.
  * Falsified if the prefix arm is better beyond seed CI at matched retention, or if rho_late >= 0.8 while the gap is large (then the mechanism is not D-T1 contraction).

## 7. Closest published methods / strongest simple alternative
* Native GEM student-mode prefix rollouts (VERIFIED code; paper text UNVERIFIED).
* DAgger (Ross, Gordon, Bagnell, AISTATS 2011; well known, not re-verified in this sprint).
* EraseFlow (trajectory objective) and TRACE (late-step trajectory-aware fine-tuning claimed for Flux) are A's UNVERIFIED abstract-level threats. TRACE directly covers "move supervision to late steps".
* DRaFT / AlignProp-type backprop through the sampler for endpoint objectives (UNVERIFIED ids). Alekseev-Groebner formula (classical; textbook section UNVERIFIED).
* **Strongest simple alternative:** uniform coverage of all 28 student steps (a GEM hyperparameter), at matched forwards.

## 8. Strongest objection
* The toy uses an explicit correction Delta with hand-picked timestep sharing (LOCAL/SMOOTH/SHARED). Outcomes swing with lr (0.05 vs 0.2) and sharing. FLUX LoRA on q/k with timestep modulation could share edits across sigma strongly, in which case the prefix suffices (D-T1 route a).
* The toy has no retention measure, so "erases more" may simply mean "edits more".
* The only real-model-relevant output is a one-flag ablation, not a method.
