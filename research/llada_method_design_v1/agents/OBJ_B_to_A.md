# Objection B → A (FFX-OP)

## Strongest objection: FFX-OP's evidence of config-robust forgetting is in-sample, and its proposed pool misses random-order parallel decision states

**Claim.** The assignment requires forgetting to hold under more than one native decoding configuration. A's evidence
("only FFX-OP reaches max-leak ≤ 0.05", §4/§7) maximises over **the same six configs that generated the on-policy
training states**. `A_scripts/sharing_check.py` L168 (`onpolicy_ff_states` loops over `CFGS`) and L146 (max-leak
over `CFGS`) use one list, `CFGS = T.CONFIGS` (C1..C6). No config is held out. Prop P2 covers only 𝒟 by
construction, so neither the toy nor the bound says anything about a config the evaluator picks outside 𝒟. When I
test held-out configs exactly, the proposed pool fails on them by a wide margin.

**Exact check.** `B_scripts/objection_A_coverage.py` → `objection_A_coverage_out.json` / `_stdout.txt` (25 s,
ledger agent B, 1 thread). Setup:
- World: B's tabular world (4096 states; exact LLaDA sampler; DP validated against the pinned `generate()` by Monte Carlo).
- Method: A's recipe. ½ fact-free TF states + ½ fact-free on-policy states of the current student; A's exclusion
  target (ε → 1e-9); A's spec E(s) = E_t. The same mirror-step training caricature is used for every arm.
- Evaluation: 41 native configs, with leak reported separately for configs inside and outside the pool.
- A's proposed LLaDA pool, mapped to L=4: (128,32,lc,0)→(4,2,lc,0); (64,128,lc,0)→(2,4,lc,0);
  (128,32,rnd,0.7)→(4,2,rnd,0.7); (128,128,lc,1)→(4,4,lc,1).

| FFX-OP variant | max leak, in pool | max leak, held out | mean leak, held out | worst held-out cells |
|---|---|---|---|---|
| A's toy pool C1..C6, all masked positions (= A's toy) | 0.0036 | **0.066** | 0.0066 | steps 2 / block 2, T=1 (lc and rnd) |
| A's LLaDA pool, E_t only (= A's spec) | 0.0004 | **0.201** | 0.045 | steps 2 / block 4 / rnd, T 0.5–1: 0.19–0.20 |
| A's LLaDA pool, all masked positions | 0.0004 | 0.201 | 0.045 | same |
| A's LLaDA pool + (2,2,rnd,1.0) | 0.0003 | 0.174 | 0.026 | steps 2 / block 4 / rnd: 0.17 |
| reference: B's OP_mix {(4,4,lc,0), (4,4,rnd,1), (2,2,rnd,1)} (B_proposal §7) | — | 0.011 | — | steps 2 / block 4 / rnd, T=1 |

**Mechanism of the failure.**
- In-pool leak is about 0.0004, while held-out leak is about 0.20: roughly 500×.
- The failing cells commit two tokens per step in random order. That creates "any two positions filled jointly
  from per-position marginals" states (product-of-marginals pairs). No config in A's pool visits these states:
  - its only random-remasking config is semi-AR (block = L/4, one token per step);
  - its single-block T>0 config is low-confidence, so it fills confident positions first;
  - its two-tokens-per-step config is greedy (argmax pairs only).
- By A's own P2 / my Prop. 1(e), the coverage coefficient C_{π|μ} for these cells is huge or infinite. Hence the
  certificate is silent, and training never touches the states the first fact commit comes from.
- The E_t restriction is **not** the cause: 0.2006 with E_t vs 0.2006 with all masked positions.
- A block-2 parallel config does not repair it (0.174). What repairs it, in B's runs, is a random-order rollout over
  the whole answer region, plus a parallel-commit schedule.
- Few-step decoding (steps < gen_length, i.e. several tokens per step) is the regime LLaDA users run for speed. So
  these cells are not exotic.

**What this does and does not show.**
- The tabular world has zero parameter sharing, which is A's pessimistic case. With sharing, generalisation may
  close part of the gap.
- But then A's own §6.1 applies: if sharing covers the gap, rollouts are unnecessary and FFX-OP reduces to FFX-TF.
  A's shared toy already shows a gap of only 0.079 vs 0.094.
- Either way, the toy's "only FFX-OP reaches ≤ 0.05" is a property of training and testing on the same configs, not
  evidence of robustness across decoding configs.
- The fix is cheap: put at least one single-block random-remasking T=1 config, and one random-order multi-token-per-step
  config, into 𝒟, and hold configs out in every table. Until then FFX-OP's robustness claim should be stated as
  "inside 𝒟 only".
- The Hyperplane plan's single held-out config ("32 steps, block 8", remasking/T unspecified) would miss this failure
  if it is greedy low-confidence. Require held-out random-remasking cells with ≥2 tokens per step at T ∈ {0.7, 1}.

## Secondary objections

1. **The KL form of P2 is false for 0 < T < 1.** This includes A's own pool config and Hyperplane G3 (T=0.7).
   Exact counterexample: p = (0.35, fact 0.30, tail 0.35 spread over the rest of V). Then KL(q̂‖p) = 0.357, so A's
   bound gives 1−e^{−KL} = 0.30, but π_{0.7}(F) = **0.44** at |V| = 126k (π_{0.5}(F) = 0.39 at |V| = 8). Sharpening
   raises a second-ranked fact's share. A valid replacement for all T ≤ 1 is π_T(F) ≤ p(F)/max_v p(v) (B Prop. 1c).
2. **The toy does not test the specified method.**
   - Spec (§2/§3): E(s) = E_t for on-policy states, a ½ρ_TF + ½ρ_OP mixture, and n_op sampled states.
   - Toy: `excl_rows` (L229–234) applies the target at **all** masked positions, and states are aggregated by
     summed visitation over 3 DAgger rounds.
   - This is harmless here (E_t made no difference above), but the reported numbers belong to a different estimator.
3. **ε = 0 exclusion + forward KL + AdamW has no finite stationary point** (derivation, not run). The F-logit gradient
   at every forget state equals p_θ(F|s,j) > 0, and it is never zero. On forget-specific directions (e.g. LoRA
   components driven mainly by forget states), Adam's normalisation turns this vanishing gradient into steps of size
   O(lr). The result is unbounded drift of the shared fact-token direction, with the operating point fixed by the step
   count rather than by the target. A's own R3 recall drops from 0.95 to 0.87. A floor ε > 0 (or a margin) restores a
   stationary point at p(F) = ε.
