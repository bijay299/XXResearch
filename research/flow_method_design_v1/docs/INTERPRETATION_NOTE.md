# Interpretation note (post-freeze; does not modify any frozen artifact)

> **Amended by [`../AMENDMENT_01.md`](../AMENDMENT_01.md)** (documentation only). The computed verdict and its recorded
> reasons are unchanged; wording in §3(4), §4 and §5 is corrected below.

Written by the coordinator after the freeze commit `aec0f3837aba5842c2e30eb4f758bfa9750fb99d`, following the user's
2026-10-20 instruction to preserve the runner's computed verdict, disclose specification ambiguities, and scope the
scientific interpretation separately. Nothing here changes code, thresholds, cell classification or decision rules.

## 1. Frozen-record wording relevant to the verdict (quoted from `configs/eval_protocol.json` v1.2, hashed in `configs/FREEZE.json`)

**A. Coverage restriction** — `probe_coverage.use`:
> "the advantage criteria Q1-Q4 and falsifiers F2, F5 are evaluated ONLY on COVERED primary cells; PARTIAL and UNCOVERED
> cells are reported (all metrics, all arms) with no candidate criterion; an advantage family with no COVERED primary cell
> is NOT_TESTABLE (does not qualify)"

**B. F5's own definition** — `falsification.F5_known_null_A`:
> "in the COVERED primary cells of G5 AND of G8: best of {na_lora, na_lora+gate} matches or beats vau at matched efficacy,
> i.e. GMR(rho_mean vau/na) >= 0.95 in both families -> F5 triggered -> NO_SUPPORTED_CANDIDATE on its own (VAU wins
> elsewhere are irrelevant). If G5 or G8 has no COVERED cell, F5 is evaluated on its generic cells and, if triggered
> there, also yields NO_SUPPORTED_CANDIDATE"

**C. F4's definition** — `falsification.F4_gate_value`:
> "C-H1(ii) value claim FALSIFIED if in G5 and G8 vau does not have lower rho_mean than penalty_earlystop and penalty+gate
> at matched e (GMR >= 1.0)"

**D. Verdict rules** — `verdict_rules`: NO_SUPPORTED_CANDIDATE if any of F1; F2 in G1 or G8 (covered cells); F3;
"F5 triggered (sufficient alone)"; "no testable advantage family has point GMR(rho_mean) <= 0.95 in its covered cells";
safety failure in ≥ 2 safety families. READY requires, among others, "G1 and G8 testable (>= 1 COVERED primary cell) and
qualify". HYPOTHESIS_UNRESOLVED: "every other outcome (incl. G1 or G8 NOT_TESTABLE, …)".

## 2. Ambiguities present in the pre-freeze records (disclosed, not resolved)

1. **F5 precedence.** A restricts F5 to COVERED primary cells; B permits a fallback to generic cells when G5 or G8 has none.
   The two are not reconcilable by reading alone. The frozen runner implements B's fallback:
   `code/run_validation.py:1472-1479` (computes `F5_GMR`, `F4_GMR` and the compute rule on all primary cells of a
   NOT_TESTABLE G5/G8) and `:1595-1596` (F5 as a sufficient NO_SUPPORTED_CANDIDATE condition). Which reading the authors
   of the protocol intended to take precedence cannot be established from the pre-freeze records.
2. **F4 conjunction.** C says "in G5 and G8"; the runner's recorded decision D13 (`code/run_validation.py:86-87`, applied at
   `:1617`) triggers F4 if the condition holds "in G5 OR in G8 (the value claim needs both)". F4 only enters the READY
   checklist, not the NO_SUPPORTED_CANDIDATE list.
3. **NO_SUPPORTED_CANDIDATE vs HYPOTHESIS_UNRESOLVED ordering.** The runner evaluates NO conditions first
   (`:1585-1603`), so a fallback-cell F5 trigger yields NO_SUPPORTED_CANDIDATE even when D's UNRESOLVED clause ("G1 or G8
   NOT_TESTABLE") would also apply. The verdict rules list NO conditions as "if any" without stating precedence over the
   UNRESOLVED catch-all; the runner's order is a reasonable reading but not one the text forces.

Note on item 3 (added after reading the results, wording unchanged above): `verdict_rules.HYPOTHESIS_UNRESOLVED` reads
"every other outcome", which most naturally means outcomes matching neither the NO nor the READY lists, so the runner's
NO-first order is the better-supported reading. The parenthetical "(incl. G1 or G8 NOT_TESTABLE …)" leaves residual
ambiguity, which is disclosed rather than resolved.

## 3. The computed verdict and exactly what produced it
Runner output (`results/validation/summary.json` → `verdict`), preserved unchanged:
`NO_SUPPORTED_CANDIDATE`, reasons `["F1", "safety failures ['G1', 'G2', 'G3', 'G4', 'G6', 'G9']"]`.

| rule | cells used | outcome | coverage regime of those cells |
|---|---|---|---|
| F1 (`falsification.F1_CH1_i_metric`; runner D-decision "F1 judged on the covered exact-J cells") | G1 d2_exact_covered, d3_exact_covered (both measured COVERED: C_probe 0.999, eff_rank 16 = k) | R_s < 1.2 in 9/12 (d2) and 11/12 (d3) seeds ⇒ triggered | **COVERED** — a test inside the narrowed hypothesis' own regime |
| safety (`verdict_rules`: "safety failure in >= 2 safety families"; per-family rule GMR ≤ 1.10 and CI upper ≤ 1.25) | all cells of each safety family (safety families are not restricted to COVERED cells by the frozen text) | 6 of 6 fail | mostly PARTIAL/UNCOVERED (G2, G3, G4, G9); G1 d=0: exact-J cell COVERED, sketch cell UNCOVERED; G6 mixed |
| F5 | G5, G8 had no COVERED cell ⇒ runner fallback to their generic/primary cells (`:1472-1479`) | not triggered (G5 19.4 ≥ 0.95 but G8 0.0099 < 0.95) | fallback cells; G8 value is the −log 100 unreached-control convention (`:1075-1083`) |
| F2, F3, oracle flip | as specified | not triggered | — |

**Sensitivity to the disclosed ambiguities (reasoning only; nothing recomputed).**
(1) F5 precedence: under the strict reading A, F5 is not evaluable (no COVERED cells in G5/G8) and therefore also not
triggered; under the runner's reading B it was evaluated on fallback cells and not triggered. The verdict does not depend
on this ambiguity. (2) F4 "and/or": F4 only enters the READY checklist, and READY was already unavailable (G8
NOT_TESTABLE), so the verdict does not depend on it. (3) NO-vs-UNRESOLVED precedence: if one insisted that "G8
NOT_TESTABLE ⇒ UNRESOLVED" overrides the NO list, the label would read HYPOTHESIS_UNRESOLVED; the runner did not do that,
the more natural reading of "every other outcome" supports the runner, and the computed verdict stands. This is
disclosed as a limit on how strongly the label can be read, not replaced.
(4) The 84 VAU `timeout` failures all occurred in the coverage-instrumented run (associated with and consistent with
instrumentation overhead; the cause was not conclusively isolated). They set `vau_crash` in
safety_G1, but G1 also fails on GMR 2.08 alone, five other safety families fail, and F1 uses ungated runs, so the
verdict does not depend on them either. They remain counted as failures, as the frozen rules require.

## 4. Scientific interpretation (separate from the label)
- **What was tested in the covered regime and failed.** The narrowed hypothesis' first clause (C-H1(i): the Adam-metric
  projection gives ≥ 1.2× the cumulative held-out hinge progress (10-step or earlier-cutoff; not the first-order −g·Δ) of the Euclidean post-projection where probes cover the retained response)
  was tested in COVERED linear-reference cells and **did not meet its pre-declared threshold** (gains 1.01–1.35×; seed medians ≈ 1.14× at d=2 and
  ≈ 1.07× at d=3; 9/12 and 11/12 seeds below 1.2×, a few individual seeds above it). At matched erasure, VAU **did not
  establish the preregistered qualifying advantage** in any TESTABLE family (Q1 GMR 1.011 G1, 0.946 G6, 0.984 G7; none
  qualified; the CIs do not establish equivalence). On this synthetic evidence the covered-regime advantage claim is not supported.
- **What was not tested.** All families ran. In G2, G5 and G8 — including the tiny flow model, which carries the GEM-like
  prefix objective — no cell met the COVERED definition, so the narrowed hypothesis was **not tested within the qualifying
  COVERED regime** there. No FLUX experiment ran. Their results (e.g. G8 generic GMR 80.6; G5 na_lora
  far lower drift) describe VAU outside its claimed regime and **must not be read as falsifying the covered-regime claim**.
  They do support the weaker statement that **this sprint did not establish a useful candidate**.
- **Safety failures** (qualification failures under the frozen rule) are largely driven by gated VAU failing to reach
  matched erasure E = 0.8, which the frozen convention scores at log 100; such ratios are convention-based, not direct
  measurements of drift or damage. They show that VAU as frozen (its τ grid, gate) is not a safe drop-in at matched erasure in those regimes;
  they are not evidence about the metric component in covered cells.
- **Observed but outside the hypothesis (post hoc, no criterion):** the metric's progress gain was larger in generic,
  uncovered G1 cells (1.4–3.8×). This was not predicted as the claim and is not evidence for it; any use of it would be a
  new hypothesis requiring fresh, pre-registered tests.
- **Not inferable from any of this:** object removal, image quality, FLUX behaviour, or any real-model ranking.

## 5. What the computed label licenses
`NO_SUPPORTED_CANDIDATE` here means: the frozen synthetic validation produced a covered-regime falsification of C-H1(i)
(F1) and ≥ 2 safety-family failures, and no advantage family qualified. It does not mean the listed conditional mathematical results are wrong
(they hold under their stated assumptions; equivalence/optimality claims are narrowed in AMENDMENT_01 §3), and it does
not test the narrowed claim within the COVERED regime on the flow model. F1 and the safety failures each support stopping
independently of the F4/F5 ambiguities in §2. The G9 mechanism diagnostic contradicted the bilinear explanation
(AMENDMENT_01 §1); no explanation of VAU's failure is claimed as validated. The
real-model protocol (`docs/REAL_MODEL_PROTOCOL.md`) is retained as written but, under this label, is **not recommended
for launch**.
