# Interpretation note (post-freeze; does not modify any frozen artifact)

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

## 3. Results-dependent sections
See §4–§6 below (filled after validation completed; numbers copied from `results/validation/summary.json`).
