# EVAL_PROTOCOL — synthetic validation of VAU (narrowed C-H1), frozen before any validation run

Author: Agent G. The machine-readable twin is `configs/eval_protocol.json` (version 1.2). Where the two differ, **the
JSON is binding**.

**Status.** DRAFT_FOR_FREEZE. The protocol becomes binding when `configs/FREEZE.json` records the sha256 of three
things: the JSON, `configs/candidate.json` and the `code/vau/` tree. After that, any change is a labelled amendment, and
an amendment can never change a threshold for runs already made.

**Amendment A1-prefreeze (2026-10-19, made before any validation run).** This version folds in review/F_REVIEW.md
§4.1–4.3 and the scratch checks `code/scratch_checks/factor_exact_check.py` and `factor_collapse_check.py`. The changes
are:
- narrowed hypothesis, with a measurement of probe coverage
- new arms `post_euclid_reject`, `vau_blocksplit`, `na_lora` and `na_lora+gate`
- new falsifier F5
- new family G9
- configurations cut from 12 to 8 per arm
- new integrity checks I13–I15

**Amendment A2-prefreeze (2026-10-20, made before any validation run).** The coordinator decided Agent E's four open
items:
1. **I4** keeps the frozen ridge. The residual ‖JΔ1‖/‖JΔ0‖ must be ≤ 1e-8 with the ridge forced to 0, and ≤ 1e-5
   (relative) with the frozen ridge 1e-6·tr(G0)/k.
2. **Naming** is as in v1.1: `post_euclid` has no reject rule, `post_euclid_reject` has it, and F1 uses
   `post_euclid_reject`.
3. **`na_lora`'s secondary knob** is the standard GPM-style feature-energy threshold ∈ {0.99, 0.999}, i.e. the top
   singular vectors that hold that fraction of the R1 layer-input energy. `na_lora+gate` and the G8 centre use 0.999.
   The exact tol-1e-10 span is reported as a diagnostic only.
4. **The compute rule** is unchanged. FE is reported per arm against both the best control and native adamw.

No other rule changed. The JSON is now version 1.2.

**Scope.** CPU toys only (models a/b/c from Agent E). Nothing here is evidence about FLUX. The method has no novelty: by
F §3 it is Euclidean null-space projection in Adam's whitened coordinates, i.e. a scaled gradient projection. Every
control receives the same probes R1 and the same gate batches R2.

## 0. The hypothesis under test, narrowed per F §4.3

The claim is: in AdamW-LoRA erasure where fresh probes span the retained response (low effective rank, k ≥ rank), the
D-metric post-projection does two things after lr tuning:
- it keeps at least 1.2× the erasure progress of the Euclidean post-projection that uses the **same reject rule**, at
  drift budgets ≤ 5% of plain AdamW's;
- it has lower drift at matched efficacy than a same-probe tuned penalty.

**No advantage is claimed outside the covered regime.**

### Probe coverage (measured for every family, cell and seed)

Coverage is measured at θ0, before training. It is a property of the fixture and does not depend on outcomes. It is
re-measured at every probe refresh.

- **C_probe**: the share of held-out retained response energy that lies in the probe row space,
  ‖J_held P_S‖²_F / ‖J_held‖²_F. Here J_held is a 256-row Gaussian sketch of the R_test response Jacobian, and P_S
  projects onto rowspace(S_J). A version weighted by D is also reported.
- **C_feat**: the share of held-out retained layer-input energy that lies in the span of the R1 layer inputs (the span
  that `na_lora` uses).
- **eff_rank_99**, reported against k.

Each cell is classified from the medians over seeds:

| class | rule |
|---|---|
| COVERED | median C_probe ≥ 0.90 and k ≥ eff_rank_99 |
| PARTIAL | 0.50 ≤ median C_probe < 0.90 |
| UNCOVERED | median C_probe < 0.50 |

**The advantage criteria (Q1–Q4) and falsifiers F2 and F5 are evaluated on COVERED primary cells only.** PARTIAL and
UNCOVERED cells are reported in full, with no criterion attached. An advantage family with no COVERED primary cell is
NOT_TESTABLE. Families G1 (d ∈ {2, 3}), G5 and G8 carry a `population ∈ {generic, covered}` cell axis. Here "covered"
means a retained subspace of rank q with q·d_out ≤ k. This axis only shapes the fixture; the measured class decides
which cells count.

## 1. What is observed

| quantity | models a, b | model c (tiny flow) |
|---|---|---|
| **Erasure (primary)** | `E_hinge`: the mean over **held-out** target inputs of p_i = 1 − relu(D+ − ηD−)/L_i, with eps_n = 0 at evaluation | `E_end`: the same p computed on the **endpoints of the student's own full 28-step sampler**. a and b are the base endpoints for the anchor and target conditions, from the same noise. |
| Erasure (robustness) | `E_axial`: guards against B-F1, transverse escape | `E_end_axial`, and `E_mode` where it applies |
| Reported only | `E_pass`, collateral r/L | the local prefix loss |
| **Drift (primary)** | `rho_mean` over R_test, normalised by Z = mean L² | endpoint drift on held-out retained conditions |
| Worst case | `rho_max`, `rho_q99`, `rho_rare` (mass 0.005), `rho_shift` | held-out conditions |
| Mechanism (models b, c) | per-step nonlinear share β_t, LoRA bilinear share γ_t, D_A/D_B ratio, ‖Δ1_A‖/‖Δ1_B‖, ‖Δ1‖/‖Δ0‖ (defined in G9) | same |

**Splits.** All splits are disjoint, which is checked with hashes:
- target: train 32 / sel 128 / test 512
- R1 = 32
- R2 = 256, fresh i.i.d. on every gate try and never reused
- R_sel = 512, R_test = 2048, R_shift = 1024, R_rare = 512

R_test, R_shift and R_rare never feed the probes, the gate or any selection step.

## 2. Arms and equal tuning budget

There are **16 arms on models b and c and 14 on model a**. On model a, `na_lora` and `na_lora+gate` are NOT_APPLICABLE,
because the model has no factor structure.

| group | arms |
|---|---|
| candidate | `vau`, which follows candidate.json including the F §4.2 fixes: t counts commits only; a zero step skips the gate; ridge floor |
| ablations (not controls) | `vau_nogate`; `vau_blocksplit`, which divides D by its mean within each LoRA factor block and so cancels the A/B imbalance of F §2.3 |
| ungated controls | `adamw` (native); `adamw_stepmatch`; `penalty`; `pre_proj`; `post_euclid` (Adam-NSCL-style, no reject rule); **`post_euclid_reject`** (the same plus VAU's "g·Δ1 ≥ 0 → 0" rule); **`na_lora`** (a KNOWN, InfLoRA-style null-A LoRA: A's rows are made orthogonal to the span of the R1 layer inputs and frozen, then B is trained; retention is exact on R1 at any step size) |
| gated controls (same gate) | `adamw+gate`, `penalty+gate`, `pre_proj+gate`, `post_euclid_reject+gate`, `penalty_earlystop`, **`na_lora+gate`** |

**Budget.** Every arm gets **8 configs** on models a and b: 4 learning rates (lr0 × 10^{−1, −0.5, 0, 0.5}) × 2 values of
its one secondary knob. The knobs are wd, the step-match factor c, λ, the ridge, τ², or the feature-energy threshold of
na_lora; see the JSON. On model c (G8) every arm gets **4 configs** (4 learning rates, with the knob fixed at its
calibrated centre).

A diagnostic `penalty_oracle` (28 λ × 4 lr) runs only in the G1 exact-J covered cells.

Calibration of lr0, λ0 and τ0 uses development seeds 0..7 only.

## 3. Matched-efficacy comparison

Arms are compared where they erase equally well:
1. Take the first crossing of `E_sel = e`, and interpolate every test metric linearly between the bracketing checkpoints.
2. For each arm, choose the config with the lowest **R_sel** drift. Report that config's **test** metrics.
3. The **best control** is chosen per seed over every control config. This includes na_lora(+gate) and
   post_euclid_reject(+gate). It is conservative: the controls get about 13× as many configs as the candidate.
4. The levels are e = 0.8 (primary), 0.5 and 0.95. If both arms miss e in 6 or more seeds, fall back 0.8 → 0.5 → 0.2.
5. An arm that never reaches e has ρ = +∞.

## 4. Test families

**Advantage families** count only on their COVERED primary cells:

| id | model | cells | purpose |
|---|---|---|---|
| G1 | a | anisotropy 0/1/2/3 decades × {exact J, sketch k = 16, K = 5}, generic; plus covered population at d = 2, 3 (12 cells) | the metric claim; F1 |
| G2 | a | κ ∈ {1e2, 1e6} × null-space rotation {0, π/8, π/4, Haar} | φ = 0 is a safety cell, because there pre_proj is exact (C-T1) |
| G5 | b | tanh gain 0.5 / 1 / 2 generic, and gain 1 covered (4 cells); per-lr curves; LoRA gauge injection | finite steps and curvature; C-X2 |
| G6 | evaluation only, on the G5 gain-1 runs and G1 d = 2 | ρ_shift is the advantage metric; ρ_rare is a safety metric | held-out conditions |
| G7 | b, covered | (K, k) ∈ {(1,32), (5,32), (20,8), (∞,32)} | stale or sketched probes |
| G8 | c | population generic / covered; GEM prefix loss; full-sampler endpoint | end to end; **required for READY** |

**Safety families:**

| id | model | cells | purpose |
|---|---|---|---|
| G1 d = 0, G2 φ = 0 | a | — | VAU must not be credited where it adds nothing |
| G3 | a | null fraction cos ψ ∈ {1, 0.1, 0.01, 0}, plus a rich cell with null(J) = {0} | incompatible constraints |
| G4 | a ×6, b ×1 | ‖b−a‖/L_ref down to 1e-6; eps_n native vs 0 | the v = b degeneracy |
| **G9** | b | **GEM zero-init regime**: B = 0, GEM hinge starting at v = b, AdamW lr 1e-3, wd 0.01, T = 23 | **F objection 2**, see below |

**G9 measurements.** One measurement config per arm, run at lr 1e-3, records every step:
- actual retained change δ_t on R_test
- first-order part ‖J sΔ‖², from an exact JVP
- nonlinear share β_t
- LoRA bilinear share γ_t = Σ‖dB dA x‖² / ‖(dB A + B dA + dB dA)x‖²
- the imbalance ratios D_A/D_B, ‖Δ1_A‖/‖Δ1_B‖ and ‖Δ1‖/‖Δ0‖
- the first-order residual

F's prediction (from f5) is reported but is not a criterion. It says VAU's δ_t is at least 10× that of
post_euclid_reject over steps 1–8, with γ_t > 0.99, and that vau_blocksplit brings the ratio below 3×.

G9's safety rule: on the full 8-config grid with T = 23, VAU's matched-efficacy drift must be non-inferior to the best
control. The same mechanism quantities are logged in every model b and c run.

**Diagnostics** (no candidate criterion):
- GD-B, B's prediction P2: the collateral ratio is predicted ≥ 5; P2 is falsified if it is below 2.
- GD-D, D's prediction P-D5: prefix vs uniform 11-of-28 at matched forwards.

The applicability rule and substitute for each family are given in the JSON.

## 5. Seeds

- **Development seeds 0..99** are used only for debugging, the integrity checks and calibration.
- **Validation seeds 7000..7011** give 12 per family. They are refused unless FREEZE.json matches, and no other range is
  ever used.
- **Derivation:** all randomness comes from `SeedSequence([20261019, FAMILY_CODE, seed]).spawn(8)`. There are fixed
  streams for the fixture, the splits, the training noise, the sketch, the gate batches, the eval noises, the population
  draws and the coverage sketch.
- **Pairing:** cells reuse the same draws and arms share the same streams, so every comparison is paired.

## 6. Statistics and thresholds

**Test statistic.** For each seed, take the mean log(ρ_test(vau)/ρ_test(best)) over the COVERED primary cells, at
e = 0.8. Its exponential gives the geometric-mean ratio, **GMR**. Significance comes from an exact one-sided sign-flip
permutation test over 12 seeds; confidence intervals come from a paired bootstrap with 10 000 resamples. Holm correction
is applied over the testable advantage families at α = 0.05.

**An advantage family qualifies only if all four conditions hold:**
- **Q1 (drift):** GMR ≤ **0.90**, and Holm p < 0.05, and the bootstrap upper bound < 1.0, and VAU wins at least 9 of
  the 12 seeds.
- **Q2 (worst case):** the GMR of ρ_max, of ρ_q99 and of ρ_rare are each ≤ **1.05**.
- **Q3 (efficacy check):** the erasure gap |ΔE_test| ≤ 0.03, and GMR ≤ 0.95 when arms are matched on axial erasure
  instead.
- **Q4 (stability):** VAU crashes in at most 1 of 12 seeds, no control crashes in more than 1, and F3 is not triggered.

**A safety family passes** when GMR ≤ 1.10, the bootstrap upper bound is ≤ 1.25, VAU never crashes, and no gate
violation occurs.

**Compute** is counted in forward-equivalents (FE), with every gate try and probe VJP charged. VAU's FE overhead
relative to the best control must be ≤ **4.0** in G5 and G8.

**Falsifiers:**
- **F1.** Cells: G1 exact J, d = 2 and 3, covered population. Arms: ungated, at lr0. Statistic: the ratio of held-out
  progress after 10 steps, P(vau_nogate)/P(**post_euclid_reject**). C-H1(i) is falsified if this ratio is below **1.2**
  in at least 8 of 12 seeds, at either d. The ratio against post_euclid without the reject rule is also reported.
- **F2.** The best penalty variant (penalty, penalty+gate, penalty_earlystop) has GMR(penalty/vau) ≤ **0.90** and wins
  at least 8 of 12 seeds, in the COVERED cells of a family. The claim is then falsified in that family. If
  penalty_oracle flips the G1 result, the verdict is capped at UNRESOLVED.
- **F3.** Over all gated runs in G1–G7 and G9, count the runs in which some accepted iterate has population clipped
  drift above τ². If that share exceeds δ = 0.05 (binomial p < 0.01), F3 is triggered.
- **F4.** VAU does not beat both penalty+gate and penalty_earlystop in G5 and G8.
- **F5 (new).** The best of {na_lora, na_lora+gate} matches or beats VAU at matched efficacy, meaning
  GMR(ρ_vau/ρ_na) ≥ 0.95, in the COVERED cells of **both G5 and G8**. If either family has no COVERED cell, its generic
  cells are used instead. **F5 on its own yields NO_SUPPORTED_CANDIDATE.**

## 7. Sprint verdict

| verdict | rule |
|---|---|
| **NO_SUPPORTED_CANDIDATE** | any one of: F1; F2 in G1 or G8; F3; **F5**; no testable advantage family reaches a point GMR ≤ 0.95; safety failures in 2 or more safety families (G9 counts) |
| **READY_FOR_SMALL_REAL_MODEL_TEST** | all of: no NO condition; G1 and G8 are testable and qualify; at least 2 of {G2, G5, G6, G7} are testable and qualify; every safety family passes, including G9; F2, F4 and F5 never trigger; the compute rule holds; every family has run; integrity is clean; no repair touched vau, the gate or a threshold. This verdict is **conditional**: the FLUX test launches only if FLUX's measured C_probe is ≥ 0.90 (REAL_MODEL_PROTOCOL C9). |
| **HYPOTHESIS_UNRESOLVED** | everything else, including G1 or G8 being NOT_TESTABLE, unrun families, a budget stop, or a penalty_oracle flip |

## 8. Stopping, failures, repairs, integrity

- **Order.** Coverage measurement first, then G1, G2, G3, G4, G5 (with G6), G7, G9, G8, GD-B and GD-D. It is a single
  pass with no early stopping on results.
- **Crashes.** A NaN, an Inf, an exception, or a run longer than 10× the median counts as a failure: E = 0 and ρ = +∞
  from that point.
- **Nothing is added after results are seen.** No new seeds, cells, configs or arms.
- **Repairs.** A fix after the freeze is logged as `REPAIR-n` in `results/validation/REPAIRS.md`. The original results
  are kept and the retest is reported separately. A repair that touches vau, the gate or a threshold caps the verdict
  at UNRESOLVED.
- **Integrity checks I1–I15** must pass on development seeds before the freeze. The ones added for F's spec defects are:
  - **I13:** the step counter t increments only on commit and drives bias correction.
  - **I14:** a zero step skips the gate and consumes no batch; S_J = 0 does not crash.
  - **I15:** na_lora leaves R1 outputs unchanged to 1e-12 at any lr while the target output changes; vau_blocksplit
    equals vau on a model with a single block.
  - **I3 (extended):** with D held constant, vau equals post_euclid and vau_nogate equals post_euclid_reject.

## 9. Confounds and what rules each out

| An apparent VAU gain could come from | Ruled out by |
|---|---|
| weaker erasure | matching on held-out erasure; reporting E_test at the crossing; judging model c on the full-sampler endpoint |
| transverse escape (B-F1) | Q3: the gain must survive matching on axial erasure; collateral is logged |
| extra retained information | every control gets the same R1, sketch and R2; the penalty sees the full R1 outputs; na_lora uses the same R1 inputs |
| the gate rather than the metric | every control is also run with the identical gate; penalty_earlystop |
| the reject rule rather than the metric | **post_euclid_reject**, also used as the F1 comparator (F §2.1) |
| a known factor-wise method doing as well | **na_lora(+gate)**, which enters best-control selection, plus F5 |
| the A/B preconditioner imbalance helping or hurting | **vau_blocksplit**, and the G9 mechanism measurements (F §2.3) |
| an artefact of the probe-coverage regime | measured C_probe / C_feat, with criteria evaluated on COVERED cells only |
| smaller or rarer steps | adamw_stepmatch, and an lr grid reaching one decade below lr0 |
| extra compute | FE accounting that counts rejected tries; the ≤ 4× rule |
| unequal tuning or selecting on test data | 8 configs per arm; selection on sel/R_sel only |
| probe overfitting or measurement failure | rho_R1_gap; the shifted and rare sets; I5/I12; disjointness hashes |

## 10. Cost (estimate; per-run timings are assumed until measured on development seed 0)

| G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 (T_iters 28) | G9 | diagnostics + oracle + coverage |
|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 0.30 | 0.19 | 0.43 | 0.85 | 0.05 | 0.48 | 0.90 | 0.10 | 0.16 |

Values are core-hours, computed from assumed per-run times of 0.1 s (model a), 0.5 s (b), 0.2 s (b with T = 23) and
2.1 s (c). The total is **about 3.9 CPU-core-hours**, which is roughly 1 hour on 4 cores. If the projection exceeds 4.0,
the reduction ladder L1–L5 is applied before the freeze. The hard stop is 6.0 metered core-hours.
