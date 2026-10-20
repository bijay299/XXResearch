# RESULTS — frozen synthetic validation (protocol v1.2, freeze `aec0f38`)

Source of every number: `results/validation/summary.json` (computed by the frozen runner `code/run_validation.py`), raw
per-run records `results/validation/raw/<family>/<cell>__s<seed>.jsonl.gz`, per-unit CPU `results/validation/units.jsonl`.
Seeds 7000–7011 (12 per family), run once, after the freeze commit. **Synthetic evidence only**: models (a) linear
reference, (b) LoRA tanh MLP, (c) tiny conditional rectified-flow attention model. Nothing here measures object removal,
image quality or FLUX behaviour.

## 1. Computed verdict (unchanged; produced by the frozen runner)
```
{"verdict": "NO_SUPPORTED_CANDIDATE",
 "reasons": ["F1", "safety failures ['G1', 'G2', 'G3', 'G4', 'G6', 'G9']"]}
```
Logic: `run_validation.py:1585-1603` — any NO condition ⇒ NO_SUPPORTED_CANDIDATE. F5 was **not** among the reasons.
Basis, rules quoted and ambiguities: `docs/INTERPRETATION_NOTE.md`.

## 2. Completion
636/636 units completed (rc 0; 3593 s wall). Inside units, individual arm runs:
| status | count | where | cause |
|---|---|---|---|
| ok | 66,576 run records total incl. below | all families | — |
| NOT_APPLICABLE | 1152+1152 (G1), 768+768 (G2), 480+480 (G3), 576+576 (G4) | `na_lora`, `na_lora+gate` on the linear model | as specified (no LoRA) |
| failed | 72 (G1), 12 (G7) | **`vau` only** | `RunFailure: timeout` (frozen per-run CPU cap = 10× median run time, `run_validation.py:1933`). All 84 are the single coverage-instrumented VAU run per unit (`cov_refresh` recorded, `:725`) in exact-J (G1) and K=1 (G7) cells; the same instrumented run succeeded in sketch/K5 cells; no plain VAU run failed. Counted as VAU crashes by the frozen rules (sets `vau_crash` in safety_G1); **not rerun**. |

## 3. Probe coverage at θ₀ (frozen definition: COVERED iff median C_probe ≥ 0.90 and median k ≥ eff_rank_99)
| family | cell | C_probe | C_probe (D-weighted) | eff_rank_99 | k | C_feat | class |
|---|---|---|---|---|---|---|---|
| G1 | d0/d1/d2/d3 exact generic | 0.999–1.000 | 0.999–1.000 | 41–46 | 48 | — | COVERED |
| G1 | d2/d3 exact covered | 0.999 | 0.999 | 16 | 16 | — | COVERED |
| G1 | d2/d3 sketch covered | 0.997 | 0.998 | 16 | 16 | — | COVERED |
| G1 | d0/d1 sketch generic | 0.392 / 0.431 | 0.594 / 0.629 | 46 / 45.5 | 16 | — | UNCOVERED |
| G1 | d2/d3 sketch generic | 0.500 / 0.560 | 0.686 / 0.729 | 43.5 / 41 | 16 | — | PARTIAL |
| G2 | κ=1e2, φ∈{0,π/8,π/4,Haar} | 0.615–0.664 | 0.726–0.812 | 18–26 | 16 | — | PARTIAL |
| G2 | κ=1e6, φ=Haar | 0.578 | 0.695 | 14 | 16 | — | PARTIAL |
| G2 | κ=1e6, φ∈{0,π/8,π/4} | 0.298–0.386 | 0.373–0.517 | 11–11.5 | 16 | — | UNCOVERED |
| G3 | cosψ∈{0,0.01,0.1,1} / rich | 0.859 / 0.618 | 0.891–0.916 / 0.843 | 23 / 32 | 16 | — | PARTIAL |
| G4 | model a (6 cells) | 0.400 | 0.549–0.660 | 46 | 16 | — | UNCOVERED |
| G4 | model b | 0.512 | 0.726 | 57 | 16 | 1.000 | PARTIAL |
| G5 | gain1 covered | 0.808 | 0.937 | 32 | 16 | 0.998 | PARTIAL |
| G5 | gain0.5 / gain1 / gain2 generic | 0.596 / 0.498 / 0.477 | 0.815 / 0.782 / 0.669 | 47 / 58.5 / 63.5 | 16 | 0.999 | PARTIAL / UNCOVERED / UNCOVERED |
| G7 | K1, K5, K∞ with k=32 | 0.960 | 0.991 | 32 | 32 | 0.998 | COVERED |
| G7 | K20 with k=8 | 0.567 | 0.801 | 32 | 8 | 0.998 | PARTIAL |
| G8 | covered / generic | 0.823 / 0.789 | 0.901 / 0.924 | 22.5 / 25.5 | 16 | 1.000 | PARTIAL / PARTIAL |
| G9 | zero-init | 0.534 | 0.851 | 56 | 16 | 0.999 | PARTIAL |
| G6 | G1 d2 sketch covered | 0.997 | 0.998 | 16 | 16 | — | COVERED |
| G6 | G1 d2 sketch generic / G5 gain1 covered / generic | 0.500 / 0.808 / 0.498 | — | — | 16 | — | PARTIAL / PARTIAL / UNCOVERED |

**Testability of advantage families** (≥ 1 COVERED primary cell): G1 TESTABLE (6 cells), G6 TESTABLE (1 cell), G7
TESTABLE (2 primary cells K1, K5); **G2, G5, G8 NOT_TESTABLE** (no COVERED primary cell). Because G8 is NOT_TESTABLE,
`READY_FOR_SMALL_REAL_MODEL_TEST` was unavailable under the frozen rules irrespective of any other outcome. No coverage
measurement was missing; all families ran.

## 4. Falsifier F1 (C-H1(i), metric progress) — evaluated in COVERED cells
R_s = P(vau_nogate)/P(post_euclid_reject), first-order erasure progress at step 10, lr0, ungated, G1 exact-J. Falsified
if R_s < 1.2 in ≥ 8 of 12 seeds at d=2 or d=3 in the covered population.
| cell | R_s per seed (7000…7011) | seeds < 1.2 | |
|---|---|---|---|
| d2 covered | 1.08 1.02 1.34 1.15 1.18 1.13 1.08 1.17 1.27 1.08 1.35 1.05 | **9** | triggers |
| d3 covered | 1.04 1.01 1.14 1.06 1.07 1.12 1.04 1.06 1.12 1.08 1.22 1.08 | **11** | triggers |
| d2 generic (not a criterion) | 1.71 1.40 3.22 1.84 1.37 1.63 1.44 2.27 1.82 1.54 3.00 1.48 | 0 | reported |
| d3 generic (not a criterion) | 1.83 1.55 3.76 2.09 0.99 1.61 1.61 2.48 1.93 1.56 3.10 1.63 | 1 | reported |
The metric always gave ≥ 1× the Euclidean progress in
covered cells but by less than the pre-declared 1.2×; the larger gains appear only in generic (uncovered) cells, which
are outside the narrowed hypothesis.

## 5. Advantage families (TESTABLE ones; GMR = geometric-mean ratio ρ_vau / ρ_best-control at matched erasure E=0.8)
| family | Q1 GMR | 95 % CI | wins / needed | Holm p | qualifies | F2 GMR (penalty/vau) | F5 GMR (vau/na) |
|---|---|---|---|---|---|---|---|
| G1 | 1.011 | [0.77, 1.46] | 8 / 9 | 0.937 | no | 24.5 (VAU ≪ penalty) | n/a (na_lora not applicable; value is a floor artifact) |
| G6 | 0.946 | [0.83, 1.08] | 6 / 9 | 0.660 | no | 1.92 | n/a |
| G7 | 0.984 | [0.67, 1.43] | 6 / 9 | 0.937 | no | 13.3 | 0.964 |
NOT_TESTABLE families, reported only: G8 generic report GMR 80.6, CI [52.3, 100], wins 0/10 — VAU's matched drift far above
the best control's where comparisons exist. G5 fallback: F5 GMR 19.4 (na_lora far lower drift than VAU), F4 GMR 0.021.
G8 fallback: F5 GMR 0.0099 and F4 GMR 0.0099 — **convention values**, because `na_lora`, `na_lora+gate`, `penalty+gate`,
`penalty_earlystop` never reached the matched erasure level in G8 while VAU sometimes did (`log_ratio` returns −log 100,
`run_validation.py:1075-1083`); they are not evidence of lower VAU drift. F5 required ≥ 0.95 in **both** G5 and G8, so it
did not trigger.

## 6. Safety families (pass iff GMR ≤ 1.10 and bootstrap upper ≤ 1.25, no VAU crash)
| family | GMR | 95 % CI | passes |
|---|---|---|---|
| G1 (d=0 cells) | 2.08 | [0.50, 6.96] | no (also `vau_crash` from the instrumented-run timeouts) |
| G2 (rotation 0) | 1.002 | [0.77, 1.28] | no (CI upper 1.28 > 1.25) |
| G3 | 4.98 | [1.94, 12.98] | no |
| G4 | 47.5 | [28.3, 76.9] | no |
| G6 (rare subgroup) | 3.64 (ρ_max 3.57) | [2.14, 6.36] | no |
| G9 (GEM zero-init) | 12.0 | [4.26, 33.8] | no |
Descriptive post-hoc table (`results/validation/posthoc_arm_table.{py,json}`, not a decision input; its "best" column
includes VAU-family rows): gated VAU frequently failed to reach E = 0.8 (e.g. G4 model-a cells 0–1/12 seeds, G8 covered
1/12, generic 0/12), which the frozen matched-efficacy convention scores at log 100 — the dominant source of the large
safety GMRs. This is the "gate stall" already seen in development and not acted on.

## 7. Gate certificate F3 and oracle
F3: 0 / 24,032 gated runs violated the clipped population drift budget (not triggered). Unclipped violation fraction 0.194;
rare-subgroup violation fraction 0.744 — the certificate covers the clipped mean only, as stated in DERIVATION Thm 8.
penalty_oracle flip: no (oracle GMR 0.988).

## 8. Diagnostics (no candidate criterion)
GD-B (Agent B, H-B1 P2): not falsified on its SGD cells (GEM/axial collateral ratio 7.3–4.9e6 under SGD); under AdamW the
same ratio is only 1.28–1.81. GD-D (Agent D, P-D5): falsified (Spearman 0.011 < 0.2).

## 9. Measured compute
Validation phase: 13,920.5 core-s (3.87 core-h; projection 3.69, hard stop 6.0), 3593 s wall, peak RSS 1.53 GiB.
FE per arm (median per run, protocol units) e.g. G1: adamw 996, penalty 4,433, post_euclid_reject+gate 24,213, vau 29,245;
vau / best control 1.43 (G1), 19.9 (G5), 27.1 (G7), 2.07 (G8); vau / adamw 20–64×. Details: `COST_MODEL.md`.
