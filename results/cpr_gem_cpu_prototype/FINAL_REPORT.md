# CPR-GEM CPU-only correctness prototype — final report

> ## Status: `CPU_PROTOTYPE_INCOMPLETE — PARTIAL CORRECTNESS EVIDENCE`
>
> **Corrected by [`AMENDMENT_01_STATUS_AND_LIMITATIONS.md`](AMENDMENT_01_STATUS_AND_LIMITATIONS.md)
> (2026-10-19), which governs wherever it and this report differ.**
>
> The suite recorded **92 declared configurations, 0 missing, 0 undeclared,
> 0 failures**, after one of two permitted repair cycles, with a first attempt
> that failed 2/92 and is preserved in full. **None of that is withdrawn.**
> What is withdrawn is the inference: the suite's **safety checks and coverage
> do not establish the full requested contract**, so passing it does not mean
> the contract was met.
>
> The historical executable verdict `CPU_PROTOTYPE_VERIFIED` remains unchanged
> inside `artifacts/results.json` and the ledgers. It is correct as what it is —
> **the verdict the suite computed about itself under its own declared
> criteria** — and those criteria are now known to be narrower than the
> contract. It is **an output of an incomplete suite**, not a finding of
> contract satisfaction.
>
> Unfixed safety and coverage gaps are enumerated in **Amendment 01 §3**;
> corrected chronology, fixture and accounting claims in **§4**; the withdrawn
> continuation in **§5**.

| | |
|---|---|
| Frozen (manifest sealed) | 2026-10-19T02:42:26Z |
| Initial suite (attempt 01) | ledger-authoritative start **02:42:34Z** — `CPU_PROTOTYPE_INCOMPLETE`, 2 failures |
| Repair cycle 1 | 2026-10-19T02:43:17Z |
| Retest (attempt 02) | 2026-10-19T02:44:29Z — executable verdict `CPU_PROTOTYPE_VERIFIED` |
| CPU, **two recorded runs** | **10.125 core-s** (ledger) / 10.119 (summaries), cap 12 core-hours |
| Elapsed, **two recorded runs** | **1.758 s** (ledger intervals) / 1.749 s (summaries) |
| Peak RSS, **max over both runs** | **486,608 KiB** ≈ 0.46 GiB (cap: 8 GiB) |
| Full-task totals | **INCOMPLETE / UNKNOWN** — see Amendment 01 §4.4 |
| New disk | ~0.4 MiB of text/JSON; **nothing installed** (cap: 10 GiB) |
| GPU | **zero** — `CUDA_VISIBLE_DEVICES=""`, no CUDA API call, no weights, no images |

The earlier headline figures **4.89 core-s / 0.92 s / 481,036 KiB** describe
**attempt 02 only** and were wrongly presented as task usage. A prose timestamp
of `02:42:47Z` for the initial suite appears in this report's earlier revision
and in `REPAIR_HISTORY.md`; **no record supports it**, and the discrepancy is
retained rather than backdated (Amendment 01 §4.5).

---

## 1. What was established, and what was not

**Established.** The native GEM target loss, the CPR candidate and four
controls agree with an independent FP64 oracle within declared tolerances, and
behaved as the frozen contract predicted on the configurations that were
actually checked — including the cases where CPR *loses*. The corrected
matched-first-order-progress algebra of §5 reproduces its closed form to FP64
identity (relative error ≤ 2.1e-16), and each declared rejection fired with the
declared reason recorded.

**Not established — and this is the correction.** The suite's checks are
**narrower than the contract**. Group F omits three declared controls and
asserts no cross-arm ordering; group G measures vector Taylor residuals rather
than the requested scalar progress error and squared drift remainder; D6–D8 do
not test the required epsilon orientations; A2 skips 9 of 27 gradient
comparisons; C5 is an unconditional pass; and `geometry.analyse` contains an
**unfixed** safety defect that would accept a negative longitudinal gradient.
The claim that "every declared rejection fired for the declared reason" is
weakened accordingly: group F accepts **any** rejection as passing, so a
rejection fired for the wrong reason would not have been caught. Full list:
Amendment 01 §3.

**Still not established.** This is **executable conditional mathematics**. It is
not novelty, not semantic erasure or preservation, not a FLUX improvement, and
not paper readiness. No model was loaded; no image was generated; no real-model
Jacobian was computed. Whether FLUX satisfies the favourable conditions remains
**unknown** and requires a separate task and budget.

---

## 2. Headline result — and the finding that qualifies it

### 2.1 The corrected hypothesis is confirmed, conditionally

**These rows survive Amendment 01 and are retained.** All 15 recorded
`g_parallel` values are **positive** (native +0.646446586, CPR +1.5, centered
+1.0), so the unfixed `abs(g_parallel)` safety defect (Amendment 01 §3.1)
**did not affect any row below**, and none implies a negative learning rate.
Note that the orderings were **computed and printed, not asserted**: group F
would not have failed had they come out reversed (Amendment 01 §3.5).

At one common detached point (`d = 2`, `a = (0,0)`, `b = v = (1,0)`,
`eta = 1/2`, `P = I`, `p = 1e-3`), with gradients taken from the **implemented
losses** and the common transverse direction `u` taken from the **actual**
native gradient:

| Shared-parameter map | `A` | `B` | `C` | `gamma_cross` | `F` native | `F` CPR | Verdict | CPR/native |
|---|---|---|---|---|---|---|---|---|
| `favorable` (`J_c=J_a=I`) | 1 | 0 | 1 | 0 | 1.299118 | 1.000000 | **CPR lower** | 0.7698 |
| `adverse_alignment` (`J_a=[1,1]`) | 1 | −1 | 1 | 0 | 0.205285 | 1.000001 | **NATIVE lower** | 4.8713 |
| `progress_rejection` (`J_c=[[1,0],[2,1]]`) | 1 | −2 | 5 | −2 | — | 1.000000 | `POSITIVE_PROGRESS_NOT_FEASIBLE` (native) | n/a |
| `near_singular` (`J_c=[[1,0],[1.8266,1]]`) | 1 | −1.8266 | 4.3365 | −1.8266 | — | 1.000000 | `REJECT_NEAR_SINGULAR_DENOMINATOR` (native) | n/a |
| `no_leading_advantage` (`J_a=[1,0]`) | 1 | 0 | 0 | 0 | 1.000000 | 1.000000 | **EQUAL** | 1.0000 |

Measured frame: `e = (1,0)`, `u = (0,−1)`, `rho_native = +0.546917`,
`rho_cpr = −6.67e−7` (reported as measured, not idealized to zero),
`g_parallel`: native 0.646447, CPR 1.500000.

Both halves of the corrected hypothesis hold. Directional repulsion **does**
reduce leading allowed-output drift at equal positive first-order target
progress under favourable geometry (−23% here), and it **does** lose — by a
factor of 4.87 — when the allowed-output responses align so that native GEM's
transverse component **cancels** (`B·rho < 0`, alignment `B/√(AC) = −1.0`).
The `B` term, not the sensitivity norms `A` and `C`, decides the ordering: `A`
and `C` are identical (1 and 1) in the favourable and adverse rows, and only
`B` flips from 0 to −1.

**The adverse row is a passing check of a limitation, not a defect.**

### 2.2 The finding that qualifies it: a control reproduces the advantage

At the principal fixture point, **the centered smooth radial control is
indistinguishable from CPR** in leading drift:

| Map | `F` CPR | `F` centered | abs diff |
|---|---|---|---|
| `favorable` | 1.0000000000004 | 1.0000000000010 | **5.6e−13** |
| `adverse_alignment` | 1.000001333 | 1.000002000 | 6.7e−07 |
| `no_leading_advantage` | 1.0 | 1.0 | 0.0 |

Native GEM's transverse component at `v = b` is **an artifact of
`pairwise_distance`'s all-ones `eps_n` shift**, not of repulsion geometry: with
`v = b` the raw difference is zero, so the repulsion direction is set entirely
by `eps_n·ones`, giving the 45° vector `(0.7071, 0.7071)` and hence
`native_grad = (0.6464, −0.3536)`. CPR removes that transverse component by
projecting onto `e_tau`; the centered control removes it by centering the
smoothing. **Both reach `rho ≈ 0`, so at this point the diagnostic cannot
attribute CPR's advantage to directional repulsion rather than to deleting an
`eps`-shift artifact.**

This remains the **dominant scientific limitation**: the advantage is
**control-equivalent at the principal point**, so **directional repulsion's
advantage is not isolated**. Control-equivalent behaviour is an informative
result, and it stays prominent rather than buried. Amendment 01 corrects the
surrounding framing — partial correctness evidence was obtained, **not**
established correctness against the full contract.

The gradient differences recorded away from `v = b` (groups C and D, below)
are **observations at frozen fixtures**. They do **not** establish better
preservation and are not offered as a substitute for the missing isolation.

The arms *do* separate away from `v = b`, where `||v − b|| > 0` makes the
repulsion directions genuinely different (group D, `epsilon = 1e−3`):

| Perturbation | native grad | CPR grad | `alpha` |
|---|---|---|---|
| parallel `(+ε,0)` | `(0.5000, −0.000498)` | `(1.5000, 9.99e−7)` | −1.000e−3 |
| antiparallel `(−ε,0)` | `(1.5000, −0.000500)` | `(1.5000, 1.00e−6)` | +9.999e−4 |
| perpendicular `(0,+ε)` | `(0.9995, **−0.4990**)` | `(1.5000, 1.00e−3)` | 0.0 |

Native's parallel-direction gradient swings between 0.5 and 1.5 under a `±1e−3`
displacement, and a purely perpendicular displacement induces a transverse
component of −0.499 in native but only `1e−3` in CPR. These are **recorded
observations at frozen fixtures**, not a claim of general stability.

---

## 3. Verification of the native reference and the candidate

| Check | Result |
|---|---|
| Forward parity, FP32 implementation vs independent FP64 oracle, `(8, 65536)`, 9 fixtures | max rel err **2.915990400741569e−07** (tol 1e−4) — *corrected; the earlier 4.3e−08 was one fixture, not the maximum* |
| Autograd vs independent FP64 analytic gradient, 3 arms × 9 fixtures | all within tol 1e−4 |
| FP64 central differences vs FP64 analytic, away from hinges (min hinge margin 0.562) | rel err **5.5e−10** (tol 1e−7) |
| Teacher detachment | anchor grad `None`, target grad `None`, student grad present |
| FP32 promotion | FP64 input → `torch.float32` output |
| Shape/batch handling | `(8,4,8,16)` flattens identically to `(8,512)`; scalar output |
| Independent divisor | `n_latents = 5 ≠ batch 8`; loss scales exactly as `1/n_latents` |
| Scaling and reduction | `gamma = 100` scales exactly ×100; reduction is `.sum()` |
| Rejection of the **enumerated** malformed inputs | shape mismatch, 1-D input, nonfinite `v`, nonfinite teacher, `n_latents = 0` → all `ContractViolation` |

**Qualifications (Amendment 01 §3.6, §3.9).** The A2 row above covers **18 of
27** configurations: the other 9 had an all-zero FP64 oracle gradient and the
suite forced the error to `0.0` **without checking the implementation's
gradient was also zero**. The maximum genuine A2 relative error is
`3.519119903620753e-07`.

Native scaling is computed as `loss * (gamma_scale / n_latents)` whereas
upstream computes `(gamma * loss) / n_latents` — algebraically equal, **not
bit-identical** in FP32 evaluation order. The supported claim is **mathematical
and tolerance agreement with the upstream formula, not bit-exact upstream
fidelity**. Likewise the rejection row covers the **enumerated** malformed
inputs; **no universal malformed-input rejection** is claimed.

**`eps_n` semantics verified, not assumed.** `F.pairwise_distance(x, y, p=2)`
computes `||x − y + eps_n·ones||₂`: measured `pairwise_distance(0,0) =
1.7320508e−06 = eps_n·√3`. Preserved exactly, argument order included.

### Hinge and norm geometry

- **Active** (`v=(0.9,0)`): hinge argument 0.850001, gradient `(1.5, −3.9e−6)`.
- **Inactive** (`v=a`, `eta=2`): loss exactly 0, gradient exactly 0.
- **Exactly zero hinge**: the *actual* shifted root is `t = 0.33333232`, which
  differs from the unshifted ideal `eta/(1+eta) = 1/3` by **−1.018e−6** — the
  `eps_n` shift raises `D_plus` and lowers `D_minus`. At that root the hinge
  argument `0.0` and the gradient `(0,0)` were **logged** — valid observations,
  consistent with `torch.relu`'s zero boundary derivative, and `clamp` was not
  substituted. **But C3 asserted only** a valid bracket, `|arg| ≤ 1e−6` and
  **finite** gradients: the exact zero-gradient behaviour its name implies was
  **observed, not asserted** (Amendment 01 §3.7).
- **Native shifted-norm minimum**: `D_minus = 1.33e−08` at `v = b − eps_n·ones`
  versus `1.414e−06` at `v = b` — the minimum is **not** at `v = b`. The
  gradient there is finite, `(1.5, 0)`. Note this value is determined by FP32
  rounding residue in `b − eps_n`, so the repulsion *direction* near the shifted
  minimum is numerically fragile in native GEM; recorded as an observation.
- **`v = a`**, within its domain: hinge inactive at `eta = 1/2, L = 1`; loss 0.

### Contrast degeneracy

`L = 0`, `L = tau_L/2`, `L = tau_L`, `L = 10·tau_L`: `e_tau` is **exactly** zero
and `alpha` **exactly** zero at and below `tau_L`, live above it. All outputs
and gradients finite throughout. At `L ≤ tau_L` repulsion is off while
**attraction remains** — the loss equals `D_plus` alone, confirmed to 1e−6. The
safe denominator prevents any eager `0/0`.

### Controls

- **Centered smooth radial**: `delta_d = eps_n·√d = 1.4142136e−06` matches
  native's `D_minus` at `v = b` (`1.4142136e−06`) to 1e−12. Value matched,
  directional shift removed. A **diagnostic control, not complete causal
  isolation** — it changes the smoothing geometry globally, not only at the
  degenerate point.
- **Anchor regression**: bit-identical to native at `eta = 0`.
- **Axial margin**: in-domain value 1.0 as predicted; on degenerate contrast
  correctly returns `NOT_APPLICABLE` rather than a repaired number.
- **Margin-truncated, ideal `k_t`**: `k_t = 1/3`; CPR's hinge at the predicted
  stopping point `alpha = L/(1+eta) = 2/3` evaluates to
  `9.834766387939453e−07`. **The claim that this *verifies* the ideal unshifted
  collinear identity is WITHDRAWN** (Amendment 01 §3.8): the case evaluates
  `cpr_loss`, which uses the **shifted** native `D_plus`, while being labelled
  `IDEAL_UNSHIFTED_DOMAIN`. The recorded value is unchanged and remains a valid
  observation — small but **not exactly zero**, consistent with the shift and
  with the ideal identity holding only approximately in FP32.
- **Margin-truncated, matched `k_t`** — **`FIRST_STOP_MATCH_NOT_ESTABLISHED`**:
  bounded bisection on the *actual* CPR hinge interior gave a strict **endpoint**
  bracket (`phi(0) = +1.0000009536743164`, `phi(L) = −0.4999985992908478`),
  `t* = 0.666667312383197`, **root residual 8.940696716308594e−08**
  (tolerance 1e−5), bracket width 9.094947017729282e−13,
  `k_t = 0.3333336412906647`, exceeding the ideal `1/3` by
  **3.0795733135802195e−07** — the FP32 normalization identity is approximate,
  as declared. Degenerate contrast correctly returned `NOT_BRACKETED`.

  **This establishes equality at a selected CPR boundary — boundary matching —
  not a shared first stopping point.** A strict endpoint bracket constrains the
  function only at the endpoints; truncated-anchor regression **can reach zero
  earlier** if `D_plus` dips to `k_t` inside `(0, t*)`. A sufficient condition
  in real unit-direction arithmetic — `D_plus` decreasing on `[0, t*]` with the
  initial value above threshold — is **stated, not verified**: no record
  establishes that monotonicity for the actual numerical direction, and no new
  root scan is authorized. **This also qualifies the coordinator's original
  specification**, which directed root-finding to obtain "a stronger native
  collinear match" (Amendment 01 §2).

### Nonlinear remainder and LoRA toy

- **Nonlinear map** (FP64, separate from the FP32 losses), frozen shrinking
  steps `h ∈ {1e−2, 5e−3, 2.5e−3, 1.25e−3}`: first-order remainder /`h²` is
  stable at **0.1409 → 0.1410** (target) and **0.09327** (allowed), both well
  under the declared bound `M = 2.0`; observed log-slope **1.99994 ≈ 2**. The
  smallest remainder is 2.2e−07, far above the FP64 floor, so no asymptotic
  slope below numerical precision was demanded.
  **Qualification (Amendment 01 §3.3):** these are **vector Taylor residuals**,
  `‖c(ψ) − c_lin‖` and `‖r(ψ) − r_lin‖` — **not** the contract's **scalar
  target-progress error** and **squared allowed-drift remainder**, which were
  not measured. The bound `M = 2.0` was **asserted, not justified** over that
  neighbourhood.
- **LoRA toy** (rank 2, 4→3, seeded Kaiming `A`, zero `B` — the *actual*
  initialization): effective `ΔW = BA` and adapter output are **exactly zero**,
  while `∂/∂B` has max magnitude **0.4410** and `∂/∂A` is **exactly 0**. The
  relevant first-order Jacobian is nonzero despite a zero effective update. No
  PEFT dependency, no AdamW trajectory.

---

## 4. Repair history

One of two permitted cycles was used. Both initial failures were
**test-expectation defects in group C**, not defects in the native reference,
CPR, or any control, and neither touches any CPR-versus-control comparison.

| Case | Attempt 01 | Diagnosis | Repair |
|---|---|---|---|
| `C1.hinge_active` | `loss = 0.0` | Point `v=(0.01,0)` is near the **anchor**, so the hinge is **inactive** (`≈ −0.485`); it was mislabelled active. | moved to `v = (0.9,0)`, hinge argument now recorded |
| `C3.hinge_exactly_zero` | hinge argument `1.5497e−06` | The ideal root `1/3` is the zero of the **unshifted** hinge; the `eps_n` shift moves the actual zero. | root of the **actual** shifted hinge via the already-frozen bisection policy |

**CORRECTION (Amendment 01 §4.1).** The earlier statement that "no fixture
tensor … was changed" is **WITHDRAWN as written**. Repair cycle 1 **did** change
two fixture tensors: `C1` from `v = (0.01,0)` to `v = (0.9,0)`, and `C3` from
the ideal root `1/3` to a **computed shifted root** `0.33333231508777317`.

What remains accurate and is **not** withdrawn: these were **documented
correctness repairs** to two mislabelled group-C cases; `eta`, `eps_n` and
`tau_L` were not changed; **the core loss functions were not changed**; and
**group F was not changed**. Attempt 01's `results.json`, ledger and source are
preserved verbatim in `artifacts/attempt_01/`, with the diff in
`artifacts/repair_01.diff`.
Full detail: [`REPAIR_HISTORY.md`](REPAIR_HISTORY.md) (immutable; superseded on
this point by Amendment 01).

### Pre-freeze development — disclosed (Amendment 01 §4.2)

Before the seal, the principal-fixture gradient geometry was computed
interactively (native gradient at `v = b`, `u = (0,−1)`, `rho_native ≈ 0.5469`)
and those values were written into `RUN_MANIFEST.md` as "expected signs". The
seal at `02:42:26Z` therefore preceded **the recorded suite**, not **every
numerical calculation**. The literal frozen requirement was
**freeze-before-suite**, and **that requirement was met** — no broader
violation is claimed. Equally, it is **not** accurate to call all of the
manifest's predictions unseen before freezing: the group-F gradient quantities
had been computed, while the §5 algebra and expected orderings were derived
analytically.

---

## 5. Budget and enforcement — what was actually in force

**Corrected per Amendment 01 §4.3–§4.6.** The figures below are for the **two
recorded suite runs only**; full-task totals are **INCOMPLETE / UNKNOWN**.

| Cap | Limit | Accounted actual (2 recorded runs) | Enforcement |
|---|---|---|---|
| Elapsed (task) | 6 h; science by hour 5 | **INCOMPLETE / UNKNOWN** (ledger intervals total **1.758 s**) | operator clock + ledger, **per run** |
| CPU | 12 core-hours | **10.125 core-s** (ledger) / 10.119 (summaries) | `ulimit -t 43200` + in-process watchdog, **per run** |
| Processes | 1 per run | 1 per run | single-process suite |
| Threads | ≤ 4 | 4 | `torch.set_num_threads(4)`, `OMP/MKL/OPENBLAS=4` |
| Memory | 8 GiB | **486,608 KiB ≈ 0.46 GiB** max over both runs | `ulimit -v 8388608` (**virtual, not RSS**) + 2 s in-process RSS poller |
| New disk | 10 GiB | **~0.4 MiB** | nothing installed; text/JSON only |
| GPU | zero | zero | `CUDA_VISIBLE_DEVICES=""` asserted at start (hard exit 2); CPU tensors asserted; no CUDA API calls; no idleness polling |

Arithmetic, sources and the double-count hazard:
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md) §3.

**Two activities have no published accounting** and cannot be resolved from
genuine existing records: **pre-freeze numerical geometry development**, and a
**staged reproduction run** from the repository copy whose artifacts were
deleted unpublished. No entry is fabricated, no zero cost inferred, nothing
rerun. **No budget overrun is demonstrated by the available records.**

Three **real, unprivileged** watchdog layers were active **per run**: shell
`rlimit`s (verified compatible with torch before the freeze), an external
`timeout --kill-after=30 1800`, and an in-process thread polling wall/CPU/RSS
every 2 s with `os._exit(97)` on breach.

**Disclosed enforcement gaps (not papered over):**
1. **Aggregate task-budget enforcement across runs was never established** —
   nothing summed usage across processes and stopped on a cumulative cap.
   Watchdogs bound each run individually, not the task.
2. `ulimit -v` bounds **virtual address space, not RSS**; RSS is bounded only
   by the in-process poller, which samples every 2 s and **can miss a
   sub-second spike**.
3. `getrusage(RUSAGE_CHILDREN)` accounts only for **reaped** children. Each run
   is single-process, so the self figures are operative — but the ledger cannot
   see an unreaped child.
4. The 6-hour elapsed cap is enforced by the operator clock and the ledger, not
   by a kernel mechanism.

**No environment was built or rebuilt.** The pre-existing
`/data/bijaypandey/cuig_pilot/venv-cuig` (Python 3.10.12, PyTorch 2.4.0) was
reused as-is; no package was installed, so the disk budget is essentially
untouched and no root-filesystem write was attempted (`/` is 100% full).

---

## 6. Limitations

0. **Incomplete coverage and an unfixed safety defect** — see
   **Amendment 01 §3** for the full enumerated list: `geometry.analyse` accepts
   negative longitudinal gradients (**UNFIXED**); D6–D8 do not test the required
   epsilon orientations; group G measures the wrong remainder; group F omits the
   anchor, axial and truncated-anchor controls, accepts any rejection as a pass
   and asserts no ordering; A2 skips 9 of 27 gradient comparisons; C3 under-asserts
   and C5 is an unconditional pass; E4's identity claim is withdrawn; native
   scaling is reassociated relative to upstream FP32 evaluation.
1. **Control-equivalence at the principal point** (§2.2) is the dominant
   **scientific** limitation: at `v = b` the centered smooth radial control
   matches CPR's leading drift to 5.6e−13, so this diagnostic does **not**
   separate directional repulsion from removal of the `eps_n`-shift artifact —
   **directional repulsion's advantage is not isolated.** Gradient differences
   recorded elsewhere do **not** establish better preservation.
1b. **First-stop matching is only boundary matching** —
   `FIRST_STOP_MATCH_NOT_ESTABLISHED` (Amendment 01 §2). This qualifies the
   coordinator's original specification as well as its implementation.
2. **One point, one geometry.** The §5 comparison is evaluated at a single
   common detached state with `d = 2`, `P = I`, and five hand-declared maps.
   It is an **analytical diagnostic, not a replacement for native AdamW**.
   Finite nonlinear target progress differs from `p` by a remainder (measured:
   `0.141·h²`). **Equal leading drift does not mean equal cumulative model
   behaviour.**
3. **`tau_L = 1e−6` is a frozen design choice**, not an empirically validated
   scale, and no sensitivity analysis of it was authorized or performed.
4. **FP32 normalization and projector identities are approximate** — measured:
   the matched `k_t` departs from the ideal by 3.08e−07, and `rho_cpr` is
   −6.67e−07 rather than exactly 0.
5. **The ideal threshold is not an exact native match off-axis**, and no
   off-axis equivalence is claimed.
6. **Synthetic tensors only.** Nothing here speaks to diffusion scores, FLUX,
   semantic erasure, preservation, or image quality.
7. **Prior art — DVE.** DVE already uses the contrast direction. The
   distinction offered here is only that the direction enters the **training
   hinge** rather than a post-hoc edit. That distinction, plus the standard
   local analysis in §5, **does not by itself establish a contribution**. No
   literature search or candidate shortlist was performed — none was
   authorized — so this paragraph is a limitations statement, not a prior-art
   survey.
8. **Gemini's note was not supplied.** No Gemini note accompanied this task and
   none exists in this repository or on this host; the only trace is a negative
   reference in `results/modern_flux_dp_uce/sequential_diagnostic_protocol/SOURCE_AND_ASSUMPTION_LEDGER.md`
   stating that no claim is made that Gemini's protocol was evaluated.
   Coordinator-side Mac paths were not assumed to exist on Hyperplane, and this
   task was not blocked on the note. Nothing here depends on it.

---

## 7. Recommendation — WITHDRAWN

**The previous recommendation to run a further CPU-only discrimination sweep
separating CPR from the centered smooth radial control is WITHDRAWN as an
automatic next step** (Amendment 01 §5). **No new probe, repair cycle,
successor task or sweep is authorized.**

The earlier framing also overstated what such a sweep could deliver. A non-win
on one bounded synthetic path would **not** establish global equivalence of CPR
and the centered control, would **not** establish that centering the smoothing
is cheaper in compute, and would **not** supply a universal reason to close the
CPR line. The earlier text asserted all three; each is withdrawn.

**This amendment neither declares CPR novel nor closes the candidate as
impossible.** The isolation question in §2.2 remains **open and unresolved**,
and resolving it requires a separate authorization that does not exist.

Real-model work (FLUX Jacobians, adapter training, image generation, scoring)
remains out of scope.

---

## 8. Reproduction

```bash
cd results/cpr_gem_cpu_prototype/code && ./run_suite.sh
```

Single command, CPU-only, ~1 second, no network, no installation. Exit code 0
means the suite's own executable verdict `CPU_PROTOTYPE_VERIFIED` — which, per
the banner above, is **an output of an incomplete suite**, not contract
satisfaction. Override the interpreter with `CPR_PYTHON=...` if `venv-cuig` is
unavailable; any CPU PyTorch ≥ 2.0 should reproduce the results within the
declared tolerances. **This amendment ran no such reproduction.**

## 9. Files

| Path | Contents |
|---|---|
| `AMENDMENT_01_STATUS_AND_LIMITATIONS.md` | **corrected status and limitations — governs where it differs from this report** |
| `PRESERVATION_AND_ACCOUNTING.md` | mechanical preservation verification and ledger arithmetic |
| `RUN_MANIFEST.md` | frozen contract: formulas, fixtures, seeds, tolerances, expected signs/rejections, pins, budget controls (immutable; superseded in part by Amendment 01) |
| `FINAL_REPORT.md` | this report |
| `REPAIR_HISTORY.md` | attempt 01 failures, diagnosis, repair cycle 1 (immutable; superseded in part by Amendment 01) |
| `EVIDENCE_INDEX.md` | SHA-256 of every committed file, from Git-committed bytes |
| `code/cpr_gem/losses.py` | attributed native reference, CPR, controls, thresholds |
| `code/cpr_gem/oracle.py` | independent FP64 oracle (separate from FP32 losses) |
| `code/cpr_gem/geometry.py` | §5 matched-progress diagnostic with rejection gates |
| `code/cpr_gem/fixtures.py` | frozen fixture declarations and budget assertions |
| `code/cpr_gem/ledger.py` | append-only resource ledger and in-process watchdog |
| `code/run_suite.py`, `code/run_suite.sh` | the 92-case suite and its one-command runner |
| `artifacts/results.json` | complete machine-readable results (attempt 02) |
| `artifacts/resource_ledger.jsonl` | append-only resource ledger |
| `artifacts/FREEZE_SEAL.txt` | pre-execution timestamp and code hashes |
| `artifacts/repair_01.diff` | the repair diff |
| `artifacts/attempt_01/` | preserved failing run: results, ledger, source |
| `artifacts/attempt_02/` | the verified run |

---

### Attribution

The native baseline is transcribed by reading from
[`GEM`](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/gem/operators/erasure/erasure_gem.py)
@ `c328fc6bec687efe60223498d1b6bed2e5fbcffd`, lines 233–249. GEM is third-party
work by its authors. No GEM operator or pipeline module is imported anywhere in
this prototype, and no GEM code is vendored into this repository.
