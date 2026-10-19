# Repair history

Policy (frozen in `RUN_MANIFEST.md` §5): at most **two** focused
implementation-repair/retest cycles after the initial suite, within the same
caps. Failures and diffs are preserved. **Fixtures, `eta`, `epsilon` and
tolerances are not changed to manufacture superiority.** Corrections are
timestamped when made and never backdated.

---

## Attempt 01 — initial suite

- **Executed:** 2026-10-19T02:42:47Z (manifest sealed 02:42:26Z, before execution)
- **Result:** `CPU_PROTOTYPE_INCOMPLETE` — 92/92 declared configurations
  recorded, 0 missing, 0 undeclared, **2 failures**.
- **Resources:** 0.83 s elapsed, 5.23 CPU-core-s, 486,608 KiB peak RSS.
- **Preserved verbatim:** `artifacts/attempt_01/results.json`,
  `artifacts/attempt_01/resource_ledger.jsonl`.

### Failures

| Case | Recorded | Diagnosis |
|---|---|---|
| `C1.hinge_active` | `loss = 0.0`, expectation `loss > 0` | **Test-expectation defect.** At `v = (0.01,0)`, `a = (0,0)`, `b = (1,0)`, `eta = 1/2`: `D_plus ≈ 0.01`, `D_minus ≈ 0.99`, so the hinge argument is `≈ -0.485` — the configuration is **INACTIVE**, and it was mislabelled "active". Placing `v` *near the anchor* makes the hinge inactive, not active. |
| `C3.hinge_exactly_zero` | `hinge_argument = 1.5497e-06`, expectation `\|loss\| <= 1e-6` | **Test-expectation defect.** The root `t = eta/(1+eta) = 1/3` is the zero of the **unshifted ideal** hinge. Native GEM's all-ones `eps_n` shift *raises* `D_plus` and *lowers* `D_minus` asymmetrically along this path, so the **actual** shifted hinge's zero sits at a slightly different `t`. The suite used the ideal root where the actual one is required. |

### What these failures are *not*

Neither failure is a defect in the native reference, in CPR, or in any control.
Neither touches any CPR-versus-control comparison: group F (the entire
matched-progress analysis) and groups A, B, D, E, G, H all passed unchanged,
and every Section-5 prediction declared in the manifest was reproduced. No
superiority claim depends on C1 or C3.

---

## Repair cycle 1 — 2026-10-19T02:43:17Z

**Changes (both confined to `run_suite.py`, group C):**

1. `C1.hinge_active` — replace the mislabelled near-anchor point with a point
   that is **genuinely** hinge-active: `v = (0.9, 0)`, giving
   `D_plus ≈ 0.9`, `D_minus ≈ 0.1`, hinge argument `≈ 0.85 > 0`. The case now
   also records the hinge argument so the activity claim is auditable, and it
   remains distinct from `C5.v_equals_b`.

2. `C3.hinge_exactly_zero` — locate the zero of the **actual native shifted
   hinge** along the collinear path `v(t) = (t, 0)` using the **already-frozen
   bounded bisection policy** (strict bracket, `max_iter = 200`,
   `xtol = 1e-12`), instead of assuming the unshifted ideal root `1/3`. The
   case records the ideal root, the actual root, their difference, the residual
   hinge argument and the gradient, and asserts the residual is at the FP32
   evaluation floor.

**Not changed:** no fixture tensor, no `eta`, no `eps_n`, no `tau_L`, no
tolerance governing any CPR-versus-control comparison, and no group-F content.
The bisection policy reused in C3 is the one frozen in the manifest for the
margin-truncated control; it is not a new or tuned procedure.

**Diff:** `artifacts/repair_01.diff` (preserved).

### Retest result — attempt 02

Recorded in `artifacts/results.json`, `artifacts/attempt_02/`, and summarised
in `FINAL_REPORT.md`.
