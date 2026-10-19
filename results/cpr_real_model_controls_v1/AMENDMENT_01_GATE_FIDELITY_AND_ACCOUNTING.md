# AMENDMENT 01 — gate fidelity, measurement interpretation and accounting

**Issued:** 2026-10-19 (UTC). **Documentation-only.** No code was fixed, no test
ran, no scorer was substituted, no model was loaded, no image was generated or
scored, no label was assigned, and the pilot was not restarted.

**This amendment supersedes contradictory statements wherever they appear**,
including in the protected `RUN_MANIFEST.md`, which is immutable and is **not
edited**. Original claims remain recoverable in Git history at
`a462b94800d44441c6c056c4cd220350862741ad` (initial payload),
`5ea308a356938b44a5aee031b80193fa8953e296` (verification payload) and
`5253df00f184dfed772c9c04d3270a86f8a872af` (reviewed HEAD).

---

## 1. Status — retained unchanged

> ## `ENTRY_GATE_FAILED — SCORER_NOT_QUALIFIED`

**No production CPR training, no development operating-point selection, no
held-out comparison and no CPR-versus-control result exists.
CPR improvement remains untested.**

This is **not** a method `NO_GO`, **not** `NO_ADVANTAGE_DEMONSTRATED`, **not** a
practical success and **not** an equivalence finding. The pilot's question is
untouched.

**"No production training" is not "no training of any kind."** Multiple native
bring-up training segments ran. The ledger records **three completed
`train_profile` segments** — pids 1737911, 1740615 and 1742802, each **22
processed timesteps and 2 optimizer updates** — plus further launches that
failed before reaching that point. Real gradient steps were taken against the
FLUX backbone.

All raw verdicts and observations are preserved unchanged: **78 rows, 76
passes, 0 recorded failures, 2 NAs**, and all 64 scorer predictions and visual
judgments. Only their interpretation is corrected.

---

## 2. Measurement interpretation — corrected without rescoring

All 64 predictions and all 64 visual judgments are preserved byte-for-byte.
Nothing was rescored and no alternative threshold was evaluated.

### 2.1 Ten unsuccessful negative decisions are not ten false positives

| Reference | n | Scorer `absent` | Scorer `present` | Scorer `uncertain` |
|---|---|---|---|---|
| Positive | 32 | — | **32** | 0 |
| Negative | 32 | **22** | **6** | **4** |

The previous report said *"All ten errors are false positives."* **Withdrawn.**
There are **six** definite false positives and **four** uncertain decisions.
Under the frozen rule the four uncertain decisions are not resolved in the
scorer's favour and so count against the gate, but they are **not** positive
misidentifications and must not be described as such.

Frozen conservative qualification, unchanged: **22/32 = 0.6875 < 0.80**.

### 2.2 Per-object fractions, correctly labelled

| Probe | Negatives n | absent | present | uncertain | Observed correct-negative fraction | Positives n |
|---|---|---|---|---|---|---|
| backpack | 11 | 3 | **6** | 2 | 3/11 = 0.2727 | 16 |
| bicycle | 11 | 10 | 0 | 1 | 10/11 = 0.9091 | 8 |
| violin | 10 | 9 | 0 | 1 | 9/10 = 0.9000 | 8 |

These are **observed correct-negative fractions on 10–11 negatives each**, not
qualified specificities. The previous report's claim that *"bicycle and violin
would have cleared the gate"* is **withdrawn**: the gate requires **≥ 20
definite positives and ≥ 20 definite negatives**, and no single object supplies
either (8 positives each for bicycle and violin; 10–11 negatives each).
**No object is independently qualified.** The gate verdict rests on the
aggregate 22/32.

Backpack negatives are **three absent, six present and two uncertain of 11** —
not "eight false positives".

### 2.3 Scores are single deterministic decisions, not probabilities

The previous report narrated softmax outputs as if they were measured
likelihoods — *"called a potted plant a backpack at p=0.715"*, *"27% of the
time"*. **Withdrawn.** Each image was scored **once**; `p_present` is a
single deterministic two-way softmax output of the frozen rule, not an
empirical frequency, and **3/11 is a count across eleven different images**,
not a repeated-trial probability for any one image.

### 2.4 Scope of the finding

The result concerns **this frozen decision rule evaluated against this
reference set**. Weak handling of negation in the present-vs-absent phrase
contrast is a **plausible explanation, not an isolated causal finding**, and
nothing here demonstrates that CLIP generally cannot measure object presence.
No ablation isolating negation was run.

**The reference labels are Claude's model-assisted contact-sheet judgments,
not independent human ground truth.** The gate therefore compared one model's
labels to another model's outputs.

### 2.5 The threshold claim is withdrawn

The previous report and release body asserted that raising `threshold_present`
to ≈ 0.85 *"would have passed the gate on this data."* **This is false and is
withdrawn.**

Under the frozen rule a case is labelled `absent` **only** when
`p_present ≤ threshold_absent = 0.40`. Raising `threshold_present` cannot
increase the number labelled `absent`; it only moves some `present` labels into
`uncertain`. The true-negative count stays **22/32** and specificity stays
**0.6875**. The four uncertain negatives already sit at p = 0.4135–0.4592, all
above 0.40. Moving positives into the uncertain band does not repair
specificity.

No alternative threshold was evaluated, and none should be inferred from this
correction.

---

## 3. Correctness and native-fidelity qualifications

The 78 rows (76 passes, 0 failures, 2 NAs) are retained exactly. They are
**partial correctness evidence for the formulas and for gradient/engineering
behaviour — not full-contract verification.**

### 3.1 `G3.boundary.native_gem` does not test an exact boundary

The record is `hinge_argument = -8.940696716308594e-08`, `loss = 0.0`,
`grad_max_abs = 0.0`. The hinge argument is **negative**, so this is an
**inactivity** test, not an exact-boundary test. The assertion used
`|arg| ≤ 1e-6`, which **does not require a zero hinge argument**. Calling it a
boundary case that "asserts exactly zero loss and exactly zero gradient" at the
boundary overstates it. **Equivalent boundary checks for the other hinge-bearing
arms are missing.**

### 3.2 The axial inactivity case was wrongly marked inapplicable

`G3.inactive.axial_margin` was recorded `INAPPLICABLE` on the stated ground
that the point does not deactivate a `Dplus` hinge. That reasoning does not
apply to arm 6, whose hinge does not involve `Dplus`. At the fixture
`a = (0,0)`, `b = (1,0)`, `v = a`, `eta = 1`: `L = 1`, `e = (1,0)`,
`alpha = (b-v)·e = 1`, so the axial hinge argument is **`1 − 2·1 = −1`** — well
inside the domain and a perfectly valid inactivity check. **The check was
omitted, not inapplicable.** The raw NA row is left unchanged; this entry
discloses the omission.

### 3.3 Coverage is not uniform across the six arms

Several reduction, divisor, `gamma`-scaling and dtype-promotion checks were run
**once on `native_gem` only**, or on a shared code path, and the report's
framing implied independent six-arm coverage of every property. **Withdrawn.**
What is genuinely per-arm: forward parity, gradient agreement, teacher
detachment, and the small-contrast/no-NaN behaviour. Genuinely active gradient
checks and the separate FP64-oracle evidence stand as recorded.

### 3.4 Native-fidelity deviation in trajectory sampling

The harness calls `transformer.eval()` before sampling the student trajectory
and `transformer.train()` afterwards. Pinned native GEM samples the trajectory
**after `wrapper.train()`**, with the module in training mode. `torch.no_grad()`
**does not disable LoRA dropout**. At the native `lora_dropout = 0.1` this
changes both the sampled trajectory and the RNG stream relative to the pinned
source.

Accordingly the blanket assurances **"same RNG", "numerically inert" and "no
algorithmic change"** applied to the bring-up patches are **withdrawn** as
stated. Patch P3/P4 memory placement remains placement-only; but the eval-mode
switch introduced alongside them is a **behavioural deviation from the pinned
native procedure** and is recorded as such. The patch descriptions in
`artifacts/bringup_patches.md` are preserved unchanged; this entry supersedes
their "numerically inert" characterisation for that specific change.

### 3.5 The custom LoRA has no demonstrated parity with native PEFT

`cprlib/lora.py` re-implements LoRA rather than using PEFT. The recorded
observations — 76 modules across 19 dual blocks, 0 single-block leak,
7,471,104 parameters, r=16/α=16/dropout=0.1/scaling=1.0, and a zero initial
effective update — are **useful observations, not an equivalence proof**.
**No forward, backward or optimizer-update parity against native PEFT was
demonstrated**, and none is claimed.

### 3.6 Serialization observations are scoped to the custom implementation

The matching restored state hash and the eval-mode fixed-input maximum
difference of **0.0** remain valid **serialization** observations **within this
custom implementation**. They do **not** establish native training
reproduction, merged/unmerged bit-equivalence, or any erasure efficacy.

### 3.7 Some bring-up flags were recorded, not enforced

`bringup.py` records most findings into a results dict rather than asserting
them. An **earlier attempt whose fixed-input equality check failed
(`fixed_input_forward_bit_identical: false`, max diff 0.0703) still ran to
completion and printed `FITS`.** A later passing probe does **not**
retroactively make that gate binding. The gate structure was weaker than the
report implied.

**None of these defects was fixed.** No test was implemented, no mode changed,
no LoRA repaired and no serialization check rerun.

---

## 4. Freeze chronology — corrected, not backdated

All original manifest, seal and ledger bytes are preserved.

**What the seal does support.** All **14** seal entries match the final
committed bytes exactly (verified mechanically; see
`PRESERVATION_AND_ACCOUNTING.md`). The seal timestamp **04:04:41Z** precedes
calibration generation (**04:05:20Z**) and the visual judgments
(**04:11:58Z**). That genuinely supports the claim that the code and
configuration were snapshotted **before any calibration outcome existed**.

**What it does not support.** It does **not** establish the complete pre-test
freeze the manifest describes. Two manifest statements are contradicted by the
ledger:

| Manifest claim | Ledger record | Status |
|---|---|---|
| F1 — cases frozen 03:52, *"before any image was generated or any model was loaded"* | bring-up `job_start` **03:48:50Z**; first M0 decode **03:51:54Z** | **contradicted** — a model was loaded and an image decoded before 03:52 |
| F2 — CPU gate at 03:56 | `cpu_entry_gate` job_start/job_end **03:41:20Z** | **contradicted** — the gate ran ~15 min earlier than stated |

These are corrected here. **Nothing is backdated and no earlier freeze is
invented.** The F1/F2 timestamps in `RUN_MANIFEST.md` are wrong as written and
are superseded by this table.

**These contradictions alone do not prove outcome-driven selection.** No
evidence of fixture, threshold or tolerance changes made in response to
outcomes was found; the scorer configuration hash in the seal matches the one
the qualification used.

---

## 5. Retries and the retry cap — breach acknowledged

The ledger records **13 `job_start` and 7 `job_end`** events, with **8
bring-up launches** (03:48:50, 03:50:45, 03:51:35, 03:52:50, 03:53:46,
03:56:50, 03:58:29, 03:59:02) — one initial run plus **seven relaunches**.

The previous report claimed **"0 of 1 GPU retries"** used. **Withdrawn.**
Seven relaunches **exceeded the one-technical-retry allowance**. That the
relaunches were driven by small compatibility fixes (tokenizer, device type,
OOM/placement, probe shape, probe mode) does **not** exempt them from the cap:
the cap governs relaunches, not their cause.

The existing patch descriptions in `artifacts/bringup_patches.md` are
preserved. **Per-attempt diffs and failure traces were not captured at the
time and no longer exist**; they are disclosed as unavailable rather than
reconstructed or fabricated.

---

## 6. Resource accounting — corrected

### 6.1 The published figures predate release verification

`FINAL_REPORT.md` reported **0.1161 GPU-hours** and **0.123 CPU-core-hours**.
Those were read **before** the release-verification job ran.

**Final counters** (`artifacts/counters.json`):

| Quantity | Value |
|---|---|
| GPU device-seconds | **431.473** = **0.1198536 h** |
| — of which release verification | **13.645 s** (`gpu_device_seconds_eval`) |
| — entry/bring-up | 417.828 s |
| CPU core-seconds | **459.661** = **0.1276836 h** |
| Decoded images | 68 |
| Attempts | 13 |

### 6.2 Those counters are themselves incomplete

Costs are folded into the cumulative counters **only on normal job close**.
**Six bring-up attempts have a `job_start` but no `job_end`** (pids 1733463,
1734765, 1735624, 1736953, 1737911, 1742123), so **their GPU and CPU costs were
never folded in**.

**Lower-bound reconstruction.** For each distinct PID, take the **maximum
cumulative snapshot** that PID ever wrote to the ledger, then sum across PIDs.
Because each snapshot is cumulative *within* its own process, taking one
maximum per PID avoids double-counting, and summing distinct PIDs avoids
counting any process twice:

| Quantity | Lower bound |
|---|---|
| GPU device-seconds | **545.611** = **0.1515586 h** |
| CPU core-seconds | **594.584** = **0.1651622 h** |

**These are lower bounds, not exact reconstructed totals.** A killed process
stops writing, so any cost incurred after its last ledger line is invisible.
**Full costs remain unknown where unrecorded.** Nothing was rerun to estimate
them.

### 6.3 Images

**68 decoded images stands**: 64 calibration + **four** decodes of the **single**
M0 case `dev.target.00` (03:51:54, 03:53:08, 03:54:05, 03:57:09) across
repeated bring-up attempts, yielding **65 unique primary PNGs**.

### 6.4 "Reserve untouched" is withdrawn

Release verification consumed **13.645 GPU-seconds** recorded against
`gpu_device_seconds_eval`. The claim that the 8-hour evaluation reserve was
**untouched** is **inaccurate**; a small amount of it was used.

### 6.5 What compliance can and cannot be claimed

**No numerical GPU, CPU or image ceiling overrun is demonstrated** — even the
lower bounds sit far below every ceiling. **That does not establish complete
compliance**, because the accounting is incomplete (§6.2) and several controls
were not enforced (§6.6). **The retry-cap breach (§5) is separately supported
and is a real breach.**

### 6.6 Enforcement gaps

* **No cross-process locking.** `counters.json` is read-modify-written by
  independent processes; concurrent updates could be lost.
* **Costs fold only on normal close**, so failed-job costs are lost (§6.2).
* **CPU and RSS checks are per-process**, narrower than the declared
  process-tree ceilings.
* **Storage, the 4-hour entry sub-cap and the 8-hour reserve were tracked but
  never enforced** — no mechanism would have halted a breach of them.
* **The published launchers do not substantiate every shell-timeout and
  `rlimit` claim** in the manifest; `runenv.sh` sets caches, threads and
  telemetry, and the timeouts used were ad-hoc per invocation.
* **~56 MiB is not a measured whole-task storage peak.** It was a directory
  size at one moment. **Full storage usage is unknown**; no genuine measurement
  of the peak exists.

### 6.7 The 8.41-hour projection

The arithmetic is correct given its inputs (1.384 s per processed timestep,
4.79 s per decoded image, 12.7 s load, ×1.30). But it **extrapolates from a
single 22-timestep profile on one arm** and does not qualify the complete
workflow (scoring, analysis, checkpoint I/O, 36 trajectories' divergent
behaviour). The 8-hour reserve **cancels on both sides** of the `fits`
expression and therefore adds no safety to that test.

**"Fits comfortably" and any readiness assurance are withdrawn.** The estimate
is retained as **provisional**.

---

## 7. Target-scope deviation: `bicycle`

The manifest judged `bicycle` "not a genuine collision" with closed SD-1.5
study `seq29`, where it was a **retention panel class**. That judgment applied
a **narrower interpretation than the fresh-target/panel rule** the contract
set. It is recorded here as a **design-scope deviation**.

It is **not** evidence that the old experiment was rerun: **no held-out bicycle
run exists** (no held-out stage ran at all), and **no reuse of any old asset is
established**. **No substitute object is selected now.**

---

## 8. Recommendation — withdrawn

The previous report's recommendation to qualify an instrument as a follow-up
task, and specifically to score candidate instruments against the published
calibration set at zero GPU cost, is **withdrawn**.

**The calibration set is now seen data.** Its images, the scorer's outputs on
them and the reference labels are all published, and the failure mode has been
analysed in public. Using it to choose a future scorer would make it
**development material, not an untouched qualification test**, and any
instrument selected on it could not then be said to have passed an independent
gate.

**No automatic re-attempt, scorer selection, or successor task is recommended
or authorized.** The code limitations in §3 are left **unfixed and documented**.
**CPR's practical value remains unanswered.**
