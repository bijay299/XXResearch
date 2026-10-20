# Objection A → C (MSDA): strongest objection + secondary

## Strongest objection — MSDA as specified trains toward a target whose fixed point leaks, so its only winning regime (strict forgetting) is unreachable on the assignment's metric

**Claim.** C's Stage 3 trains B with MDU's forget loss `KL(p_θ(·|x,y_t) ‖ p_θ0(·|m,y_t)^τ/Z)`. The parameterisation
cannot move the model below the anchor's fact mass. The loss *resists* forgetting below that level, the anchor sees
the visible answer tokens, and LLaDA/TOFU answers restate the subject. On the assignment's metric (generated-answer
leakage, max over ≥2 native configs) strict matched forgetting is therefore infeasible for MSDA. By C's own §7, strict
forgetting is the only regime where MSDA wins; at loose forgetting it loses to LoRA/FT in all seeds. C's toy hides this
because its W0 is *planted* with a uniform target on prompt-masked states (anchor p(true) 0.087 vs chance 0.083), and
because matching is done on teacher-forced p(true), not on generated leakage across configs.

**Derivation (per state s, masked position i; a := anchor^τ/Z, p := p_θ(·|s,i)).**
- `∂KL(p‖a)/∂z_v = p_v (log p_v − log a_v − KL(p‖a))`. The unique minimiser is p = a.
- If p agrees with a up to normalisation except that p_f < a_f, the gradient on the fact logit z_f is negative.
  Descent then raises p_f back toward a_f.
- So the forget loss puts a **floor** of a_f(s) on the fact mass at every trained state. This floor is the anchor's
  prediction given the visible answer tokens. It is 0 only if the anchor is fact-free.
- For τ<1 the tilt keeps the anchor's ranking. Wherever the anchor's arg-max is the fact, greedy decoding still
  commits the fact.
- Restricting ΔW to row(A) cannot remove this floor. It only adds approximation error, in either direction, and is
  not controlled.

**Exact check** (`A_scripts/objection_msda_anchor.py` → `objection_msda_anchor_out.json` / `_stdout.txt`; ledger
agent A, 109 s, 1 thread). Setup:
- Model: my shared-parameter log-linear mask predictor (`sharing_check.py`). Its logits are *exactly linear* in a key
  vector φ (m=188 one-hot features), so MSDA is implemented literally: GEP `C_F u = μ C_R u` with shrinkage ρ=1e-2,
  C's rank rule r=#{μ≥4}, giving r=7 (μ: 1725, 1444, 1426, 1304, 18.4, 18.0, 17.6, 3.6, …).
- States: Q_F = forget TF states plus θ0 rollouts under **all 6** configs, which is generous to C. Q_R = retain TF
  states plus a rollout.
- Training: B only.
- Answers restate the subject X. Anchor fact mass at the fact-free TF states is 0.21–0.44 (mean 0.325), against a
  conditional fact mass of 0.70–0.94.

Max-over-6-configs leakage (configs C1…C6), retain decode-TV, and fact mass at the fact position on fact-free TF states:

| arm | λ=3 | λ=30 | λ=100 | p_fact @λ=100 | retain TV @λ=100 |
|---|---|---|---|---|---|
| **MSDA + MDU τ=1 (C as specified)** | 1.00 | 1.00 | **1.00** (C3 = 1.00, C2 .45, C5 .50) | **0.328 ≈ anchor mean 0.325** | 0.0001 |
| FULL-W + MDU τ=1 | 1.00 | 1.00 | 1.00 | 0.326 | 0.042 |
| MSDA + exclusion target (FFX/TNFF, fact-free states) | 0.24 | 0.075 | **0.034** | 0.006 | **0.0001** |
| FULL-W + exclusion target | 0.18 | 0.052 | 0.023 | 0.003 | 0.044 |

What the table shows:
- With C's loss, MSDA's fact mass converges to the anchor's own fact mass: 0.328 against an anchor mean of 0.325.
- Under that loss, maximum leakage stays at 1.00 for every λ, so ≤0.10 is never reached.
- On the same substrate, the coordinator's §7 probe reports that MDU τ=1 still emits forgotten value tokens
  (leak_any 0.10–0.54).

**Concession (in fairness to C).** The same table shows MSDA's *parameterisation* is very strong here. With a leak-free
target it reaches max-leak 0.034 at retain TV 1e-4, against 0.033–0.044 for full-W at max-leak 0.052–0.023; R3 recall is
0.952 → 0.952, against 0.857 for full-W.
- This toy is the best case for MSDA. The forget prompt has *private* features with zero retain energy: 35% of A's row
  energy loads on prompt-identity features (μ≈1300–1700), the analogue of TOFU's forget-only authors.
- So the damaging point is specific: **MSDA's selectivity is only useful if it is paired with a fact-free target. As
  specified, it is paired with one that is not.**
- C's evidence does not test the pairing that matters. C's toy has a uniform anchor and matches on teacher-forced
  p(true), with a single "l2r" generation check.

**Falsifier for this objection (cheap, on Hyperplane, no training).**
- Measure the anchor's fact-token mass p_θ0(fact|m, y_t) on fact-free TF states of forget10. If it is ≈ chance, the
  floor is negligible and this objection fails.
- Otherwise, MSDA must replace its loss before its retention claim can be tested on generated leakage. It can then
  be compared against FULL/LoRA under the *same* fact-free target, at matched max-over-configs leakage.

## Secondary objections (one line each)
1. **Finite-sample C_R inflates μ exactly at the rank-rule threshold.** Under Marchenko–Pastur, Ĉ_R's smallest
   eigenvalues shrink by (1−√(m/n_eff))². At n_eff = 4m (m=12288, n_eff ≈ 5·10⁴ effective keys; masked keys within a
   state share the [MASK] component, so n_eff ≪ #keys), that factor is 0.25. A true-μ=1 direction then shows μ̂≈4,
   which is C's τ. ρ=1e-2 shrinkage barely changes this (0.25 → 0.26), so the rule can admit pure-noise directions.
2. **Closed-family proximity.** The first step is ΔW = −ηGKᵀ·AᵀA: the full gradient times a *static projector built from
   retained statistics*, i.e. the soft generalised-eigen version of AlphaEdit's null-space projector. The remaining
   distinction from the closed VAU family (input second moments vs Jacobian probe; no gate) is a choice of statistic,
   not of mechanism. This is the coordinator's call, but it should be decided before any GPU spend.
3. **The advantage is a private-identity effect that μ cannot certify for neighbours.** Where it works (my toy; TOFU
   forget-only authors), row(A) ≈ forget-identity directions, so MSDA acts as an "identity-conditioned logit bias".
   C's own failure-boundary sweep shows μ_1 does not move (55.7 → 56.3) while a held-out same-subject neighbour is
   damaged as much as with LoRA. On RWKU-style neighbour sets the go/no-go statistic is therefore blind by construction.
