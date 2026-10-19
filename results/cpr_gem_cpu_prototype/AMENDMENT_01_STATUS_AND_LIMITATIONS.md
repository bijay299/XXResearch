# AMENDMENT 01 — corrected status and limitations

**Issued:** 2026-10-19 (UTC). **Documentation-only.** No code was changed, no
test was run, no experiment was executed, and no prototype file was repaired.

**Amended status:**

> ## `CPU_PROTOTYPE_INCOMPLETE — PARTIAL CORRECTNESS EVIDENCE`

This **supersedes the interpretation** of the earlier `CPU_PROTOTYPE_VERIFIED`
headline. It does **not** alter any recorded result. The string
`CPU_PROTOTYPE_VERIFIED` remains, unchanged and correct as an executable
output, inside `artifacts/results.json`, `artifacts/attempt_02/results.json`
and the ledgers: it is the verdict **the suite computed about itself** under
its own declared criteria. Those criteria are now known to be narrower than the
requested contract, so the verdict is **an output of an incomplete suite**, not
a finding that the contract was met.

**What the record still says, and still means.** The suite recorded **92
declared configurations, 0 missing, 0 undeclared, 0 failures**, after one of
two permitted repair cycles, with a first attempt that failed 2/92 and is
preserved in full. None of that is withdrawn. What is withdrawn is the
inference from it: **passing a suite whose safety checks and coverage are
incomplete does not establish the requested contract.**

Protected evidence — `code/**`, `RUN_MANIFEST.md`, `REPAIR_HISTORY.md`,
`artifacts/**` — is **byte-for-byte unchanged**; see
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md). Where
`RUN_MANIFEST.md` or `REPAIR_HISTORY.md` state something this amendment
corrects, those files are immutable published records and are **superseded by
this document**, not edited.

---

## 1. Findings that survive

These principal observations remain supported by the recorded numbers.

| Shared map | Recorded outcome | Status |
|---|---|---|
| `favorable` (`J_c=J_a=I`) | CPR leading drift **lower**, `F` 1.000000 vs 1.299118, ratio 0.7698 | **retained** |
| `adverse_alignment` (`J_a=[1,1]`) | **native GEM wins** by 4.87× via `B·rho < 0` cancellation | **retained** |
| `progress_rejection` (`J_c=[[1,0],[2,1]]`) | native `POSITIVE_PROGRESS_NOT_FEASIBLE`, denominator −0.093833 | **retained as recorded** |
| `near_singular` (`J_c=[[1,0],[1.8266,1]]`) | native `REJECT_NEAR_SINGULAR_DENOMINATOR`, denominator 1.002e−03 | **retained as recorded** |
| `no_leading_advantage` (`J_a=[1,0]`) | **equal** drift; `K_ac·u = 0` and `gamma_cross = 0` | **retained** |

**The §5.1 safety defect does not invalidate these rows.** Every one of the 15
recorded `F`-group `g_parallel` values is **positive** — native +0.646446586,
CPR +1.5, centered +1.0 — so no recorded row relied on the defective
`abs(g_parallel)` acceptance, and none implies a negative learning rate.

### The dominant scientific limitation stands, and stays prominent

**At `v = b`, the centered smooth radial control reproduces CPR's favourable
leading drift** — the two agree to **5.6e−13**. Native GEM's transverse
component there is an artifact of `pairwise_distance`'s all-ones `eps_n` shift,
and both CPR and the control reach `rho ≈ 0` by removing it.
**Directional repulsion's advantage is not isolated.**

Gradient differences recorded elsewhere (groups C and D) are **observations at
frozen fixtures**. They do **not** establish better preservation, and they are
not offered as a substitute for the isolation that is missing.

**No FLUX utility, no semantic unlearning, no novelty and no paper readiness is
established by anything in this directory.**

---

## 2. Native first-stop matching is only boundary matching

### `FIRST_STOP_MATCH_NOT_ESTABLISHED`

`matched_k_t` locates a root `t*` of the CPR hinge interior and sets
`k_t = eta·((b − v(t*))·e_tau)`. That establishes **equality at a selected CPR
boundary**: at `t*` the CPR hinge and the truncated-anchor control both vanish.

It does **not** establish that the two controls reach zero at the **same first**
stopping point. The recorded evidence is a **strict endpoint bracket**
(`phi(0) = +1.0000009536743164`, `phi(L) = −0.4999985992908478`) plus the root
and its residual. An endpoint bracket constrains the function only at the two
endpoints; **truncated-anchor regression can reach zero earlier** than `t*` if
`D_plus` dips to `k_t` anywhere in `(0, t*)`.

**Sufficient condition (stated, not verified here).** In real unit-direction
arithmetic, if attraction `D_plus(v(t))` is *decreasing* on `[0, t*]` and its
initial value `D_plus(v(0))` is above the threshold `k_t`, then no earlier
crossing exists and the first stopping points coincide. **No record in this
directory establishes that monotonicity for the actual numerical direction**
(FP32, native all-ones-shifted `D_plus`), and **no new root scan or calculation
is authorized**, so the condition is reported as a condition.

Accordingly the generic implementation is described as **boundary-matched**,
and the label is `FIRST_STOP_MATCH_NOT_ESTABLISHED` unless and until existing
definitions or records are shown to rule out an earlier crossing.

**This also qualifies the coordinator's original specification.** The frozen
contract directed root-finding on the CPR hinge interior to obtain "a stronger
native collinear match". That procedure yields boundary matching; the stronger
first-stop reading is not delivered by the root alone. The specification, not
only its implementation, is qualified here.

**Preserved unchanged:** `t_star = 0.666667312383197`,
`root_residual = 8.940696716308594e-08` (tolerance `1e-5`),
`bracket_width = 9.094947017729282e-13`,
`k_t_matched = 0.3333336412906647`, exceeding the ideal `1/3` by
`3.0795733135802195e-07`. The qualified **ideal unshifted collinear** result is
retained in the form given in §3.8 below. **Not every recorded fixture is
invalidated by this** — the root and residual observations stand as recorded.

---

## 3. Unfixed safety and coverage limitations

All of the following are **present in the published code and UNFIXED**. They
are documented, not repaired; no code change is authorized.

### 3.1 `geometry.analyse` accepts negative longitudinal gradients — UNFIXED

`code/cpr_gem/geometry.py:133` tests `abs(g_parallel) < MARGIN_G_PARALLEL`
rather than requiring `g_parallel > MARGIN_G_PARALLEL`. A sufficiently negative
`g_parallel` therefore **passes** the gate, and the matching that follows would
require a **negative learning rate** to deliver positive first-order target
progress. This is a real safety defect in the diagnostic.

**Scope of impact: none on the record.** All 15 recorded `F`-group
`g_parallel` values are positive (minimum +0.646446586), so no published row
exercised the defect.

### 3.2 Groups D6–D8 do not test the required epsilon orientations

D6/D7/D8 perturb the **student** `v` by `±1e−3` parallel, antiparallel and
perpendicular to the contrast, while `eps_n`'s own direction relative to the
contrast is held fixed. The contract asked for `epsilon` **parallel,
antiparallel and perpendicular to the contrast**. The recorded gradients remain
valid observations about student displacement; they are **not** the requested
epsilon-orientation coverage, which is **missing**.

### 3.3 Group G measures the wrong remainder

Group G records the **vector Taylor residuals** `‖c(ψ) − c_lin‖` and
`‖r(ψ) − r_lin‖`. The contract asked for the **scalar target-progress error**
and the **squared allowed-drift remainder**. The recorded slope (1.99994) and
bounded ratios (0.1409, 0.09327) are correct statements about the quantities
actually measured, and nothing more. Additionally the declared neighbourhood
bound `M = 2.0` was **asserted, not justified** over that neighbourhood.

### 3.4 Group F omits three declared controls

Group F compares **only** native GEM, CPR and the centered smooth radial
control. The **ordinary anchor regression**, **axial margin** and
**margin-truncated anchor** controls received **no shared-map comparison**.
Those comparisons are missing from the requested contract.

### 3.5 Group F accepts any rejection as a pass and asserts no ordering

`code/run_suite.py:501` sets `ok = True` for **any** rejection status, so a
rejection fired for the wrong reason would still pass. Group F also **never
asserts the expected cross-arm ordering**: the favourable/adverse/equal
verdicts were computed and printed, but no check would have failed had the
ordering come out reversed. The ordering agreed with the frozen prediction —
that agreement is an observation, not an enforced test.

### 3.6 A2 skips the zero-gradient case

`code/run_suite.py:96` forces the error to `0.0` when the FP64 oracle gradient
is identically zero, instead of checking that the **implementation's** gradient
is also zero. **9 of 27** A2 configurations took that branch
(`status = ALL_INACTIVE`). For those 9, no gradient agreement was actually
tested. The remaining 18 are genuine, with maximum relative error
`3.519119903620753e-07`.

### 3.7 C3 and C5 assertions are weaker than their labels

`C3.hinge_exactly_zero` asserts a valid bracket, an approximate boundary
residual (`|arg| ≤ 1e−6`) and **finite** gradients. It does **not** assert the
exact zero-gradient behaviour its name implies. Its **logged** values remain
valid observations — `hinge_argument = 0.0` and `grad = [0.0, 0.0]` were
recorded — but they were observed, not asserted.

`C5.v_equals_b` is an **unconditional `"pass": True`**
(`code/run_suite.py:309`). It records numbers; it tests nothing.

### 3.8 E4's "verified" identity claim is withdrawn

`E4.cpr_stopping_point_identity` evaluates `L.cpr_loss`, which uses the
**shifted** native `D_plus`, while the case is labelled
`IDEAL_UNSHIFTED_DOMAIN` and was reported as verifying the ideal unshifted
collinear identity. **That exact verification claim is withdrawn.** The
recorded value is unchanged and remains a valid observation: the CPR hinge
evaluated at `alpha = L/(1+eta) = 2/3` returns
`cpr_loss_at_stop = 9.834766387939453e-07`, i.e. small but **not** exactly
zero — consistent with the shift, and consistent with the ideal identity
holding only approximately under FP32 shifted arithmetic.

### 3.9 Native scaling is algebraically reassociated

`code/cpr_gem/losses.py:91` computes `loss * (gamma_scale / n_latents)`.
Upstream GEM computes `(config.gamma * loss) / n_latents`. These are
algebraically equal but **not bit-identical** in FP32 evaluation order. The
supported claim is therefore **mathematical and tolerance agreement with the
upstream formula**, not **bit-exact upstream fidelity**.

Similarly, `_validate` rejects the specific malformed inputs it enumerates
(shape mismatch, sub-2-D input, nonfinite student or teacher, non-positive
`n_latents`). It is **not** a universal malformed-input rejector, and no such
claim is supported.

### 3.10 Forward-parity maximum corrected

The previous report quoted **4.3e−08** for forward parity. That was the error
of a **single** fixture, not the maximum. The correct maximum over all 9
recorded `A1` configurations is:

> **`2.915990400741569e-07`** (at `A1.parity.native.seed20261019.generic`)

still far inside the declared tolerance `1e-4`. **This corrects a summary
statement from existing records. It does not change any result and parity does
not fail.**

---

## 4. Corrected chronology, fixtures and accounting

### 4.1 "No fixture tensor changed" is withdrawn

Repair cycle 1 **did** change two fixture tensors:

- `C1`: `v = (0.01, 0)` → `v = (0.9, 0)`
- `C3`: an ideal root `eta/(1+eta) = 1/3` → a **computed shifted root**
  (`0.33333231508777317`)

The earlier claim that "no fixture tensor … was changed" is **withdrawn as
written**. What remains accurate and is **not** withdrawn: these were
**documented correctness repairs** to two mislabelled group-C cases, the
**core loss functions were not changed**, and **group F was not changed**.
Both the pre-repair source and the diff are preserved
(`artifacts/attempt_01/run_suite.py`, `artifacts/repair_01.diff`).

### 4.2 Pre-freeze numerical geometry development — disclosed

Before the freeze seal, the principal-fixture gradient geometry was computed
interactively (the native gradient at `v = b`, the resulting `u = (0,−1)` and
`rho_native ≈ 0.5469`), and those values were then written into
`RUN_MANIFEST.md` as "expected signs". The seal at **02:42:26Z** therefore
preceded **the recorded suite**, not **every numerical calculation**.

Stated precisely, without inflation in either direction: the literal frozen
requirement was **freeze-before-suite**, and that requirement **was met** — no
broader violation is claimed or implied. Equally, it is **not** accurate to
describe all of the manifest's predictions as **unseen** before freezing: the
group-F gradient quantities had been computed. The §5 algebra and the
expected orderings were derived analytically; the gradient numbers were
checked numerically first.

### 4.3 Resource accounting, corrected and separated

The previously reported **4.89 core-seconds / 0.92 s / 481,036 KiB** describes
**attempt 02 only**. It is **not** cumulative task usage.

**Ledger arithmetic over genuine existing records** (full detail and method in
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md)):

| Recording point | CPU core-s | Elapsed (ledger intervals) | Max RSS |
|---|---|---|---|
| Two **unique** `suite_end` ledger records | **10.125** | **1.758 s** | **486,608 KiB** |
| Two underlying **result-summary** snapshots | **10.119** | **1.749 s** | 486,608 KiB |
| attempt 02 alone (previously reported) | 4.891 / 4.888 | 0.923 / 0.919 s | 481,036 KiB |

The two rows differ because they are **different recording points**, a few
milliseconds apart within each run — the ledger entry is written after the
summary snapshot. Neither is wrong; they are not interchangeable.

**No double counting.** `artifacts/resource_ledger.jsonl` is opened in **append**
mode, so it accumulated **both** runs (228 lines).
`artifacts/attempt_02/resource_ledger.jsonl` is a **byte-identical copy** of it
(same SHA-256 `015993a08395f46d…`), and `artifacts/attempt_01/…` is a strict
114-line prefix. Summing the copies would double-count; only the **two unique
`suite_end` records** are counted above.

### 4.4 Full-task totals are INCOMPLETE / UNKNOWN

Two activities consumed resources and have **no published accounting**:

1. **Pre-freeze numerical geometry development** (§4.2) — interactive, no
   ledger was active.
2. **A staged reproduction run** executed from the repository copy at
   `results/cpr_gem_cpu_prototype/code/` to confirm one-command
   reproducibility. Its artifacts directory was deleted after the check and
   **was never published**, so no ledger for it exists.

These cannot be resolved from genuine existing records. **No entry is
fabricated, no zero cost is inferred, and nothing is rerun to recover them.**
Accordingly:

> **Full-task resource totals are INCOMPLETE / UNKNOWN.** Only the two recorded
> suite runs are accounted for.

**No budget overrun is demonstrated by the available records.** The accounted
usage (10.125 core-seconds against a 12 core-hour cap; 486,608 KiB against
8 GiB) is orders of magnitude inside the caps, and the two unaccounted
activities were of the same small scale — but that last point is an
expectation, not a measurement, and is not offered as evidence.

### 4.5 Timestamp discrepancy — retained, not backdated

`REPAIR_HISTORY.md` and the previous `FINAL_REPORT.md` give the initial suite's
execution time as **02:42:47Z**. The **first ledger record** —
`ledger_init`, `watchdog_start` and `suite_start` — is timestamped
**02:42:34Z**.

A search of every file in this directory finds `02:42:47Z` **only in those two
prose statements**; **no record supports it**. Its meaning therefore **cannot be
established from existing evidence**, and the discrepancy is **retained and
disclosed** rather than resolved. **Nothing is backdated.** The
ledger-authoritative start of the initial suite is **02:42:34Z**; the freeze
seal at **02:42:26Z** precedes it either way.

### 4.6 Watchdog enforcement was per-run only

The three watchdog layers (`ulimit -v` / `ulimit -t`, external `timeout`, and
the in-process poller) are established **per run**. **Aggregate task-budget
enforcement across runs was never established** — nothing summed usage across
processes and stopped on a cumulative cap.

Two further limits stand as previously disclosed and are repeated because they
qualify the word "enforced": **`ulimit -v` bounds virtual address space, not
RSS**, and the **2-second in-process poller can miss a sub-second spike**.

---

## 5. Automatic continuation is withdrawn

The previous report's recommendation — to launch a further CPU-only
discrimination sweep separating CPR from the centered smooth radial control —
is **withdrawn as an automatic next step**. **No new probe, repair cycle,
successor task or sweep is authorized.**

The reasoning behind the withdrawal matters, because the earlier framing
overstated what such a sweep could deliver:

- A **non-win on one bounded synthetic path would not establish global
  equivalence** of CPR and the centered control.
- It would **not** establish that centering the smoothing is cheaper in
  compute.
- It would **not** supply a universal reason to close the CPR line.

**This amendment neither declares CPR novel nor closes the candidate as
impossible.** The isolation question raised in §1 remains **open and
unresolved**, and resolving it requires a separate authorization that does not
exist.

---

## 6. What this amendment changed

| File | Change |
|---|---|
| `AMENDMENT_01_STATUS_AND_LIMITATIONS.md` | **new** — this document |
| `PRESERVATION_AND_ACCOUNTING.md` | **new** — mechanical preservation verification and ledger arithmetic |
| `FINAL_REPORT.md` | status corrected to `CPU_PROTOTYPE_INCOMPLETE — PARTIAL CORRECTNESS EVIDENCE`; corrected claims; pointer to this amendment |
| `EVIDENCE_INDEX.md` | regenerated at the amendment payload commit |
| `../README.md` | pointer row updated |

**Unchanged, byte-for-byte:** `code/**` (all 8 files, including comments),
`RUN_MANIFEST.md`, `REPAIR_HISTORY.md`, `artifacts/**` (all 9 files, both
attempts, both ledgers, freeze seal, repair diff). Verified mechanically
against reviewed HEAD `8387fb80c589b1149370c1084924d061c25ac7ee` in
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md).

**Original published claims remain recoverable** in Git history at
`aa4f5c926ee2e231692924162848b116a15e3edd` (payload) and
`8387fb80c589b1149370c1084924d061c25ac7ee` (reviewed HEAD).
