# AMENDMENT 01 — documentation-only corrections to the closeout at `7fde561`

Date 2026-10-20. Documentation only: **no experiment, test, rerun, recalculation of results, code change, or change to
any frozen artifact** (`configs/eval_protocol.json`, `configs/candidate.json`, `configs/FREEZE.json`, `code/`, `results/`
raw outputs, `review/`, `agents/`, `CANDIDATES.md`, `DERIVATION.tex` are unchanged). The only reads were of existing
outputs and source files; one count of existing raw records by `kind`/`status` was made to reconcile §6. Historical
statements are corrected here by erratum; the current summary documents (`START_HERE.md`, `results/RESULTS.md`,
`docs/INTERPRETATION_NOTE.md`, `NOVELTY_ASSESSMENT.md`, `COST_MODEL.md`, `docs/EVIDENCE_INDEX.md`) were edited to agree
with this amendment and point here. The original sprint budget and deadline remain binding.

**The computed verdict is unchanged: `NO_SUPPORTED_CANDIDATE`** (reasons recorded by the frozen runner: F1; safety
failures in G1, G2, G3, G4, G6, G9).

## 1. The bilinear explanation — corrected against the existing validation outputs
**Frozen definition** (`code/run_validation.py:570-588`, decision D18 `:98-100`; aggregation `:1320-1334`). Per step t,
on the mechanism input set (model b: full R_test; model c: fixed 256-query subset), with LoRA layer inputs X at θ_t and the
applied step (dA, dB):
γ_t = Σ_layers ‖X (s·dB·dA)ᵀ‖² / Σ_layers ‖X (s·(dB·A + B·dA + dB·dA))ᵀ‖²  (a ratio of sums over layers and inputs);
δ_t = mean over inputs of ‖f(θ_t + sΔ) − f(θ_t)‖² / Z. For each G9 measurement run (lr 1e-3, T = 23), `gamma_1_8` and
`delta_1_8` are means over steps 1–8; the reported value per arm is the **median over the 12 seeds**, and the drift ratio is
median `delta_1_8` of the arm / that of `post_euclid_reject`.

**Existing results** (`results/validation/summary.json → families.G9.mechanism`):
| Arm | `gamma_1_8` | Drift ratio versus `post_euclid_reject` |
|---|---:|---:|
| VAU | 0.00004155088312525 | 1.68535667282 |
| Euclidean reject (`post_euclid_reject`) | 0.00003786092937525 | 1 |
| VAU blocksplit | 0.00002757748937525 | 1.75772630047 |

VAU's γ ≈ **0.00416 %**. γ is a **layer-level squared-norm ratio** of the bilinear part of the LoRA map change to the total
LoRA map change, summed over layers; it is **not** a fraction of semantic damage, perceptual damage or total network-output
error.

**Contrary evidence, stated explicitly.** The frozen G9 prediction (`eval_protocol.json families.G9.prediction_from_F_f5`:
"vau delta_t >= 10x post_euclid_reject at steps 1-8 with gamma_t > 0.99") was **not supported by these validation
aggregates**: drift ratio 1.69 (not ≥ 10), γ ≈ 4.2e-5 (not > 0.99); `vau_blocksplit` did not reduce the ratio (1.76). This
diagnostic was not a candidate-verdict criterion; the verdict is unchanged.

**Withdrawn generalization.** `review/F_scripts/f5_lora_diag.py:16-18` computes the mixed nonlinear output finite difference
`f(dA+dB) − f(dB) − f(dA)` on one particular toy (F's f4 toy, lowdim_pop3, N_r = 4, lr 1e-3). That quantity does not isolate
the layer term dB·dA (it also contains tanh cross-curvature between the two factor displacements). The development output
(`f5_lora_diag_output.txt`) and F's original text are preserved. **Withdrawn:** the generalization of that toy result into
"> 99.9 % bilinear (dB·dA) damage in GEM's setup", "the metric routes the correction through A and > 99.9 % of the actual
retained change is bilinear" as a fact about GEM's configuration, and any use of it as a validated explanation of VAU's
failure. Affected historical text (preserved, superseded here): `review/F_REVIEW.md:75-76, :126`;
`review/F_FULL_ALGORITHM_NOVELTY.md:46-49, :63`; `CANDIDATES.md:35` ("F's objection 2 … > 99.9 %"); `DERIVATION.tex`
remark after Thm 7 ("(review F, f5: > 99.9 % … bilinear term)"); the coordinator's chat report of 2026-10-20 ("strongest
objection … second-order term then makes up over 99.9 %"). No method is recommended on the premise that bilinear
domination was confirmed; the large `DA_DB_step1` imbalance (≈ 1.86e8) is recorded but its consequence for drift was not
shown to be bilinear.

## 2. Statistical and observable claims
- **"Tied" withdrawn.** Replace "VAU tied the best controls" with "**VAU did not establish the preregistered qualifying
  advantage**" (Q1 GMR 1.011, CI [0.77, 1.46]; 0.946, CI [0.83, 1.08]; 0.984, CI [0.67, 1.43] in G1 / G6 / G7). These
  intervals do not establish equality or equivalence.
- **F1's observable.** F1 measures **cumulative held-out hinge progress** P = E_test(hinge) after T_p = 10 steps, or at the
  first step either arm's E_test reaches 0.95 (`eval_protocol.json falsification.F1_CH1_i_metric`; runner D-decision
  text `run_validation.py:84-85`). It is **not** the first-order quantity −g·Δ. Wording that called it "first-order erasure
  progress" is corrected.
- **F1 counts preserved, wording corrected.** Covered cells: 9/12 seeds (d = 2) and 11/12 (d = 3) fell below 1.2×. Some
  seeds exceeded 1.2× (d = 2: 1.34, 1.27, 1.35; d = 3: 1.22). Wording implying that every individual gain was below the
  threshold is withdrawn; the tables in `results/RESULTS.md` §4 are unchanged.
- **Three distinct things, not one.** (a) *Safety qualification failure* = a family failing the frozen rule (GMR ≤ 1.10 and
  bootstrap upper ≤ 1.25, no VAU crash); (b) *failure to reach matched erasure* = an arm whose configurations never reach
  E = 0.8, scored by the frozen convention as ρ = ∞ / log-ratio ±log 100 (`run_validation.py:1075-1083`, D21);
  (c) *measured drift* = interpolated ρ at the crossing. GMRs dominated by (b) — e.g. G4 47.5, G8 fallback values 0.0099 —
  are **convention-based ratios, not direct measurements of physical damage**.

## 3. Mathematical claims — narrowed
- **D-invariance equivalence.** "VAU ≡ `post_euclid_reject+gate` iff range(Sᵀ) is D-invariant" holds for the **idealized,
  unregularized projectors** (μ = 0, S of full row rank): Π_D = Π ⇔ range(DSᵀ) = range(Sᵀ). It is **not** automatically an
  "if and only if" theorem for the implemented positive-ridge (μ > 0), stateful, gated procedure; there, the condition is
  sufficient for the idealized projectors to coincide and approximately for the ridge ones, but equivalence of the full runs
  was not proved.
- **dim null(S) = 1.** This gives **collinearity of the projected proposals** at that step. It does **not** establish that
  complete Adam trajectories are equivalent after learning-rate tuning (the scale factor varies by step and interacts with
  the reject rule, gate and state).
- **Optimal direction and the metric.** "VAU is the optimal feasible direction **iff** post-projection drift is measured in
  Adam's metric" is withdrawn. Correct statement: if the step budget is measured in the metric D⁻¹, the D-metric projection is
  the optimal feasible direction (conditional **sufficient** result, Prop. 2). Global proportionality of a damage metric to
  D⁻¹ is **not necessary**: optimal directions under two metrics can coincide when the metrics agree on the feasible
  subspace, or for a particular gradient.
- **Results retained, qualified.** Retained as valid under their stated assumptions: the conditional KKT/trust-region
  characterisation (Prop. 2, β₁ = 0 or m̂ = g, λ_wd = 0), the explicit sign example (Prop. 3), the ridge-residual identity
  (Lemma 5), the sketch-residual statement with its R₁-only scope (Lemma 6, HMT constant UNVERIFIED), and the clipped-mean
  acceptance certificate under fresh i.i.d. batches and a fixed trial cap (Thm 8). Blanket wording such as "mathematics sound"
  / "DERIVATION results hold within their scopes" is replaced by: *the listed conditional results are correct under their
  stated assumptions; Thm 7's constants are unmeasurable for real networks; the equivalence and optimality statements are
  narrowed as above.*
- **Declined factor-wise variant.** Its collapse (Prop. "collapse under zero-init B") holds for **its particular constraints**
  dA·U = 0 and dB·(A·U) = 0 **together with** B = 0 **and** rank(A·U) = r. Zero-initialized LoRA alone is insufficient; e.g.
  if rank(A·U) < r, or the constraints are imposed jointly rather than factor-wise, the argument does not apply. Wording
  "provably collapses under zero-init LoRA" (START_HERE) and "collapses in GEM's regime" (`CANDIDATES.md:35`) is narrowed
  accordingly.

## 4. Coverage and the verdict
- The exact computed verdict and recorded reasons are retained.
- **F1 and the safety failures each support stopping independently of the disclosed F4/F5 ambiguities**: F1 was evaluated on
  measured-COVERED cells and is a NO condition; ≥ 2 safety failures is a separate NO condition; neither uses F4 or F5.
- **All families ran, including tiny-flow G8.** What remained untested in G8 (and G2, G5) was the **narrowed hypothesis within
  the qualifying COVERED regime**, because no G8 cell met the frozen COVERED definition (C_probe 0.82 / 0.79 < 0.90). **No
  actual FLUX experiment ran.**
- **Withdrawn:** "the only open question this sprint leaves is whether the covered regime is reachable" (START_HERE,
  chat report) and the suggestion of a coverage-measurement study. No coverage-only study is launched or proposed by this
  amendment.

## 5. Accounting and failure attribution
- **Metered consumption: 17,413.191 recorded core-seconds ≈ 4.84 core-hours**, after removing the nested self-meter
  `E_run_validation` (contained in `VALIDATION_PHASE_v1_2`), including **1,682.043 core-seconds across five unsuccessful
  attempts** (D_window_toy, F_f4_lora_toy, F_f5_diag first attempt, first E_run_dev, and the 0-s validation launch failure).
  This is **metered consumption with disclosed unmetered work** (interactive shell/Python reads, pip installs, git, agent
  reasoning), **not an exhaustive total**.
- **Memory.** 1.53 GiB is the reported **peak RSS of the largest metered child** (`ru_maxrss` of `RUSAGE_CHILDREN`), not
  aggregate concurrent process-tree RSS. `code/meter.py` sets a per-process `RLIMIT_AS` of 2 × 12 = **24 GiB** by default — an
  **address-space** limit that does **not** enforce a 16-GiB aggregate RSS ceiling.
- **Storage.** ≈ 158 MiB is the **final directory size**, not whole-sprint peak additional storage. Not in that figure: the
  venv (~243 MiB in /tmp), the aborted torch download cache (~2.7 GB, deleted), the reference worktree and external clones
  (GEM, Adam-NSCL) in the session scratchpad.
- **No resource overrun was demonstrated**; incomplete measurement is not proof of a breach either way.
- **84 VAU timeouts.** All 84 occurred in coverage-instrumented runs (the one VAU run per unit with `cov_refresh`) in exact-J
  (G1) and K = 1 (G7) cells; this is **associated with** and **consistent with** instrumentation overhead, but the cause was
  **not conclusively isolated**. Their frozen failure scoring (counted as VAU crashes; `vau_crash` in safety_G1) is preserved.
- **Record reconciliation (corrected).** Raw validation records by kind (read from `results/validation/raw/`):
  regular arm records 66,576 = **60,540 ok + 84 failed + 5,952 NOT_APPLICABLE**; G9 measurement records **192** (all ok);
  G5 gauge-check records **24** (12 `vau`, 12 `vau_nogate`; these carry `gauge_check`, no status field). Arm-bearing total
  **66,576 + 192 + 24 = 66,792**. The figure "72 G5 gauge records / 66,840" given in the amendment request does **not** match
  the stored records (24 gauge lines exist), so 66,792 is reported. Separately counted: 528 unit-header records, 432 GD-B diagnostic records, and 48 GD-D diagnostic rows (GD-D rows carry
  arm/status fields but are diagnostic, not arm-run records, and are excluded from the arm-bearing total as in the request).
  The previous `results/RESULTS.md` row "ok | 66,576 … total" mislabelled the total as ok and is corrected.

## 6. What is not changed
Frozen code, configurations, freeze manifest, raw outputs, development scripts, agent reviews, failure records, the ledger,
earlier studies, branches, tags and release assets. No validation rerun, VAU repair, new control, CPR+VAU combination,
FLUX load, or further assessment.
