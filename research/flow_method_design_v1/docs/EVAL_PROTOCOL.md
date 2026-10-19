# EVAL_PROTOCOL — synthetic validation of VAU (C-H1), to be frozen before any validation run

Author: Agent G. Machine-readable twin: `configs/eval_protocol.json`. If the two disagree, **the JSON is binding**.
Status: DRAFT_FOR_FREEZE. The protocol becomes binding once `configs/FREEZE.json` records the sha256 of the JSON,
of `configs/candidate.json` and of the `code/vau/` tree. After that, any edit is a labelled amendment. An amendment
cannot change a threshold for runs that have already been made.
Scope: CPU toys only (models a/b/c from Agent E). Nothing here is evidence about FLUX.

Ground rules carried over from the sprint:
- The method has no novelty (see CANDIDATES.md). What is tested is the engineering hypothesis C-H1.
- Retained probes are extra information that GEM never uses. **Every control gets the same probes R1 and the same
  gate batches R2.**
- A negative result is an acceptable outcome.

## 1. What is observed

| quantity | models a, b | model c (tiny flow) |
|---|---|---|
| **Erasure (primary)** | `E_hinge` = mean over **held-out** target inputs of `p_i = 1 - relu(D+ - eta*D-)/L_i`, with eps_n = 0 at evaluation. p = 0 at θ0 (v = b); p = 1 once the input is past the GEM hinge. | `E_end`: the same p, computed on the **endpoints of the student's own full 28-step sampler**. a and b are the base-sampler endpoints for the anchor and target conditions from the same noise. Held-out conditions × 64 fixed eval noises. |
| Erasure (robustness) | `E_axial` = mean clip(s/(L/2), 0, 1). It guards against B-F1: the hinge can drop through transverse escape without any erasure. | `E_end_axial`; `E_mode` if the fixture has labelled modes |
| Reported only | `E_pass`, collateral r/L | local prefix GEM loss (never a success criterion) |
| **Retained drift (primary)** | `rho_mean` = mean over R_test of ‖f_x(θ) − f_x(θ0)‖² / Z, where Z = mean L² on target_test | endpoint drift of the full sampler on held-out retained conditions (same noises); velocity drift at all 28 σ is secondary |
| Worst case | `rho_max`, `rho_q99` over R_test; `rho_rare` (rare subgroup, mass 0.005); `rho_shift` (shifted retained distribution) | held-out retained conditions only |

Splits are drawn independently and checked for disjointness by hashing:
- target: train 32 / sel 128 / test 512
- retained: R1 probes 32; R2 gate batches of 256, drawn fresh on every try; R_sel 512; R_test 2048; R_shift 1024; R_rare 512

R_test, R_shift and R_rare are never used for probes, for the gate, or for selection. The probes, the penalty and the
gate in model c act on velocities at (base retained-trajectory state, σ drawn uniformly from all 28 steps). So the gap
between what the gate certifies (velocity) and the endpoint drift that is evaluated is measured, not assumed away.

## 2. Arms and equal tuning budget

Arms, 13 in total:
- **candidate:** `vau`
- **ablation:** `vau_nogate`
- **ungated controls:** `adamw` (native GEM), `adamw_stepmatch` (AdamW rescaled every step to the H-norm of the paired
  VAU step; zero when VAU rejects), `penalty` (λ/2 · mean over R1 of the output drift, ordinary AdamW), `pre_proj`,
  `post_euclid` (VAU-Euclid / Adam-NSCL-style)
- **gated controls:** `adamw+gate`, `penalty+gate`, `pre_proj+gate`, `post_euclid+gate`, `penalty_earlystop`

Every gated arm uses **the identical gate**: same τ, n2 = 256, clip M = 2τ², δ = 0.05, R_max = 4, the same R2 batch
sequence and the same rollback. This keeps the gate from being credited to the metric.

**Budget.** Every arm gets exactly **12 configurations**: 4 learning rates (lr0 × 10^{−1, −0.5, 0, 0.5}) × 3 values of
its secondary knob:

| arm | secondary knob (3 values) |
|---|---|
| adamw | wd |
| adamw_stepmatch | c (step-norm multiplier) |
| penalty | λ |
| projections | ridge |
| gated arms | τ² |

G8 is reduced to 8 configurations per arm over 9 arms, the same for every arm. `penalty_oracle` (28 λ × 4 lr) runs
only in the G1 exact-J cells, as an upper reference that is not budget-equal.

**Calibration** of lr0, λ0 and τ0 uses development seeds 0..7 only. The values are recorded in FREEZE.json.

## 3. Matched-efficacy comparison

1. For each (family, cell, seed, arm, config), find the first crossing of `E_sel = e` along the checkpointed
   trajectory. Interpolate every test metric linearly between the two checkpoints that bracket it.
2. For each arm, select the config with the lowest **R_sel** drift at the crossing. Then report its **test** metrics.
   No selection ever uses a test set.
3. The **best control** for a seed is the control, across all configs of all 11 control arms, with the lowest R_sel
   drift. This is deliberately conservative: the controls get 11× the candidate's configurations.
4. The levels are e = 0.8 (primary), 0.5 and 0.95. A predeclared fallback (0.8 → 0.5 → 0.2) applies only if both VAU
   and the best control miss e in ≥ 6 of 12 seeds.
5. An arm that does not reach e has ρ = +∞ at that seed.
6. The tradeoff curve is the lower envelope over configs of (E, ρ) for each arm, given as the median and IQR across seeds.

## 4. Test families

Every family has an applicability rule and a substitute; see the JSON for the exact construction. The classes are:
- **Advantage** families, where C-H1 predicts a gain: G1, G2, G5, G6, G7, G8.
- **Safety** families, where VAU only has to be non-inferior.
- **Diagnostic** families, which are not candidate criteria.

| id | class | model | cells | why |
|---|---|---|---|---|
| G1 | advantage (d ≥ 2), safety (d = 0), falsification | a | anisotropy 0/1/2/3 decades × {exact J, sketch k = 16, K = 5} | the core metric claim C-H1(i) |
| G2 | advantage (φ ≠ 0), safety (φ = 0) | a | κ ∈ {1e2, 1e6} × null-space rotation {0, π/8, π/4, Haar} | C-T1: aligned null space makes pre_proj exact, so VAU must not be credited there |
| G3 | safety | a | null fraction cos ψ ∈ {1, 0.1, 0.01, 0}; rich retained set with null(J) = {0} | incompatible constraints; reject rule, gate, no NaN |
| G4 | safety | a (6 cells) + b (1 cell) | ‖b−a‖/L_ref ∈ {1, 1e-2, 1e-4, 1e-6}; eps_n native vs 0 | the v = b degeneracy; the Adam step-1 sign step does not shrink with the contrast |
| G5 | advantage | b | tanh gain {0.5, 1, 2}; per-lr curves; LoRA gauge injection (dB = BK, dA = −KA) | finite steps, curvature, C-X2 |
| G6 | advantage (ρ_shift), safety (ρ_rare) | evaluation-only, on the G5 gain = 1 and G1 d = 2 runs | held-out shifted and rare retained inputs | T3 certifies the mean, not the rare subgroup |
| G7 | advantage (K ≤ 5), robustness | b | (K, k) ∈ {(1,32), (5,32), (20,8), (∞,32)} | stale or sketched probes; C c4 predicts a loss at K = ∞ |
| G8 | advantage, **required for READY** | c | GEM prefix loss, full-sampler endpoint metrics | end to end |
| GD-B | diagnostic | free flow + a | ε/L × stiffness κ | H-B1 P2: GEM/axial collateral ratio ≥ 5 predicted; falsified if < 2 |
| GD-D | diagnostic | c | prefix 0..10 (23 iterations) vs uniform 11-of-28 (18 iterations, matched forwards), adamw and vau at fixed configs | D P-D5 Spearman ≥ 0.5; falsified if < 0.2 |

## 5. Seeds and randomness

- **Development seeds 0..99:** debugging, integrity checks and calibration only.
- **Validation seeds 7000..7011:** 12 per family. They are never run before the freeze; the runner refuses them unless
  FREEZE.json matches. No other range is ever used. Post-repair retests reuse 7000..7011.
- **Derivation:** all randomness comes from `SeedSequence([20261019, FAMILY_CODE, seed]).spawn(8)`. The streams are:
  fixture, splits, training noise, sketch Σ (drawn at k_max = 64 and truncated), gate batches (the j-th try of a run
  uses batch j), eval noises, population draws, and a spare.
- **Pairing:** cells reuse the same draws. Arms share streams 1–5, so all comparisons are paired.

## 6. Statistics and thresholds (decided now)

**Pairing and test.** The per-seed statistic is the mean over primary cells of log(ρ_test(vau) / ρ_test(best control))
at e = 0.8. The geometric-mean ratio is **GMR**. The test is an exact one-sided sign-flip permutation test with 12 seeds
(4096 flips). The confidence interval is a paired percentile bootstrap (10 000 resamples, seed 424242). **Holm**
correction is applied across the 6 advantage families at α = 0.05. Everything else is exploratory.

**An advantage family qualifies only if all of Q1–Q4 hold.**

| criterion | condition |
|---|---|
| Q1 | GMR(ρ_mean) ≤ **0.90**, and Holm-adjusted p < 0.05, and bootstrap upper bound < 1.0, and VAU wins ≥ 9/12 seeds |
| Q2 | GMR of ρ_max, ρ_q99 and ρ_rare each ≤ **1.05** |
| Q3 | median \|ΔE_test\| ≤ 0.03, and GMR ≤ 0.95 when the matching uses E_axial / E_end_axial |
| Q4 | VAU crashes in ≤ 1/12 seeds; no control crashes in > 1 seed; the gate-coverage test is not falsified |

**A safety family passes** if GMR ≤ 1.10, the bootstrap upper bound is ≤ 1.25, VAU never crashes, and there is no
coverage violation.

**Compute.** Costs are in forward-equivalents (FE); one backward is 2 FE. Every gate try is charged, including
rejected ones. Probe refresh is charged as k · 3 · n1 / K, the penalty as 3 · n1 per step, and a model-c iteration as
66 FE per noise. READY requires the median FE overhead, measured as FE needed to reach e = 0.8 relative to the best
control, to be ≤ **4.0** in G5 and G8.

**Falsification (from C_candidate_spec).**
- **F1:** In the G1 exact-J cells, ungated at lr0, with P = held-out hinge progress after 10 steps: if
  P(vau_nogate) / P(post_euclid) < **1.2** in ≥ 8/12 seeds at d = 2 or at d = 3, C-H1(i) is falsified.
- **F2:** If the best of {penalty, penalty+gate, penalty_earlystop} has GMR(penalty/vau) ≤ **0.90** (penalty more than
  10% better) and wins ≥ 8/12 seeds, C-H1 is falsified in that family. If `penalty_oracle` flips G1, the verdict is
  capped at UNRESOLVED.
- **F3:** Gate coverage. Count the runs, over all runs of gated arms in G1–G7, in which any accepted iterate has a
  population clipped drift above τ² (20 000 fresh draws). If this fraction exceeds δ = 0.05 (one-sided binomial
  p < 0.01), C-H1(ii) is falsified.
- **F4:** If VAU is not below penalty+gate and penalty_earlystop in G5 and G8, the gate's value claim is falsified.

## 7. Sprint verdict

| verdict | rule |
|---|---|
| **NO_SUPPORTED_CANDIDATE** | any of: F1; F2 in G1 or G8; F3; no advantage family with point GMR ≤ 0.95; safety failure in ≥ 2 safety families |
| **READY_FOR_SMALL_REAL_MODEL_TEST** | no NO condition; G1 **and** G8 qualify; ≥ 2 of {G2, G5, G6, G7} qualify; all safety families pass; F2 and F4 nowhere; compute rule met; all families RUN; integrity clean; no repair touching vau, the gate or a threshold |
| **HYPOTHESIS_UNRESOLVED** | every other outcome, including UNRUN families, a budget stop, a penalty_oracle flip, or G8 being inapplicable |

## 8. Stopping, failures, repairs

- **Single pass.** Families run in the order G1 → G2 → G3 → G4 → G5 (+G6) → G7 → G8 → GD-B → GD-D. Every unit runs once.
  There is no early stopping on results.
- **Crashes.** NaN, Inf, an exception, or a timeout above 10× the median run time count as a failure: E = 0 from that
  point on and ρ = +∞. This applies equally to every arm.
- **No reruns.** No new seeds, cells, configs or arms are added after any validation result is seen.
- **Repairs.** Each repair is logged as `REPAIR-n` in `results/validation/REPAIRS.md`, with its diff and reason. The
  original results stay in the report, and the retest on the same seeds is reported separately. A repair that touches
  the vau arm, the gate or any threshold caps the verdict at HYPOTHESIS_UNRESOLVED.
- **Integrity checks I1–I12** must pass on development seeds before the freeze. They are listed in full in the JSON.
  Examples: finite-difference gradients; vau with k = 0 and no gate is bitwise equal to adamw; with D constant, vau
  equals post_euclid; the gate with τ = ∞ is bitwise equal to the ungated arm; rollback restores (θ, m, v, t); model c
  at LoRA = 0 reproduces the base endpoints.

## 9. Confounds and what rules each out

| an apparent VAU gain could be | ruled out by |
|---|---|
| weaker erasure | matched efficacy on held-out target inputs; E_test reported at the crossing (\|Δ\| ≤ 0.03); model c judged on the full-sampler endpoint, not the local loss |
| hinge drop from transverse escape (B-F1) | Q3: the result must survive matching on E_axial; collateral r/L logged |
| extra retained information | every control receives the same R1 probes, sketch rows and R2 gate batches; penalty uses the *full* R1 outputs (more information than the sketch) |
| the gate, not the metric | every control is also run with the identical gate; penalty_earlystop; vau_nogate vs post_euclid (F1) |
| smaller or rarer steps (implicit lr) | adamw_stepmatch; the lr grid extends one decade below lr0 |
| extra computation | FE accounting: probe VJPs, every gate try including rejected ones, and penalty passes; compute rule ≤ 4×; FE-vs-erasure curves |
| tuning asymmetry | 12 configs per arm; the best control is chosen over all control configs (conservative) |
| selection on test | selection on target_sel and R_sel only |
| probe overfitting | ρ on R1 vs ρ on R_test (rho_R1_gap); G6 shifted and rare sets |
| measurement failure | I5/I12 (zero drift at θ0); disjointness hashes; eps_n = 0 at evaluation; L floor in G4 |
| lucky seeds | fresh seeds 7000..7011; paired permutation test; Holm; win counts |

## 10. Cost

Assumed per-run times are 0.1 s (a), 0.5 s (b) and 3 s (c). That gives 156 runs per (cell, seed), or 72 in G8.
Estimated cost per family, in core-hours:

| G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | diagnostics + oracle |
|---|---|---|---|---|---|---|---|---|
| 0.42 | 0.42 | 0.26 | 0.57 | 0.78 | 0.05 | 0.56 | 0.72 | 0.14 |

**The total is about 3.9 CPU-core-hours**, roughly 1 hour wall-clock on 4 cores. Timings are measured on development
seed 0 before the freeze. If the projection exceeds 4.0, the reduction ladder L1–L4 in the JSON is applied before the
freeze. The hard stop is 6.0 metered core-hours, and any family not finished by then is UNRUN.
