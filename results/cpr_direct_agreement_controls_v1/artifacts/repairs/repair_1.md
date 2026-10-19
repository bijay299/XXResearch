# Repair cycle 1 (post-freeze) — synthetic-gate fixture construction defect

* Start of engineering: 2026-10-19T18:38Z (approx.; ledger clock records the cycle).
* Failed launch: `gate.synthetic` (CPU only, 0 GPU-h, 32 s), `SYNTHETIC_GATE_FAIL: fixture mixed d=256: only 0/6 rows after 40000 tries`.
  The failure occurred in FIXTURE CONSTRUCTION, before any native-vs-reference comparison. No loss agreement/disagreement was observed.
* Diagnosis (3000 FP64-reference draws with the frozen d=256 generator): 0 rejected by hinge-boundary margin, 0 by active/inactive sign,
  3000/3000 rejected by the optional gradient-conditioning filter `min_grad_frac=0.01` (the minimum |element| of a 256-dim near-Gaussian gradient
  is essentially never >= 1% of the maximum).
* Patch (one, `code/cprd2/gate_core.py`): the d=256 fixture is built with `min_grad_frac=0.0`. Fixture seeds, distributions, margin (0.1), row counts,
  tolerances (rtol/atol), objectives, arms and decision rules are unchanged. Gradient comparison still uses the frozen rtol=1e-4/atol=1e-7, so
  near-zero gradient elements are compared absolutely rather than being excluded. The d=32, d=64 fixtures keep the filter.
* Disclosure: this relaxes a fixture-conditioning filter that could not produce any row; it was chosen from fixture-feasibility diagnostics
  only (no loss comparison outcome informed it).
* Retest: one rerun of `gate.synthetic` after the additive repair commit.
