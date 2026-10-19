# CPR real-model controls pilot v1 — final report

> ## Terminal status: `ENTRY_GATE_FAILED — SCORER_NOT_QUALIFIED`
>
> The prespecified **measurement-qualification gate failed**. Under the
> contract ("Failed qualification ends the task; no scorer shopping or
> threshold retuning after inspecting failures") scientific work stopped at the
> entry gate.
>
> **No production CPR training, no development operating-point selection, no
> held-out comparison and no CPR-versus-control result exists.** Therefore
> **no** scientific decision — `PROMISING_FOR_FRESH_REPLICATION`,
> `NO_ADVANTAGE_DEMONSTRATED` or `INCONCLUSIVE` — is available or claimed, and
> this is **not** a method `NO_GO`, a practical success or an equivalence
> finding. **CPR improvement remains untested.**
>
> **"No production training" is not "no training of any kind":** three
> completed native bring-up training segments ran (22 processed timesteps and
> 2 optimizer updates each), plus further launches that failed earlier.
>
> **Corrected by
> [`AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md`](AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md)
> (2026-10-19), which governs wherever it and this report differ.** Several
> claims below are withdrawn there — notably "ten false positives", the
> threshold-raising claim, "bicycle and violin would have cleared the gate",
> "fits comfortably", "0 of 1 GPU retries", "reserve untouched", and the
> resource figures in §3.

**The question it would have answered (unanswered):** at comparable actual
target suppression, does CPR preserve allowed content better than native GEM
and the strongest simple controls?

---

## 1. What happened

The entry gate has four parts. Three passed; the fourth failed.

| Gate | Result (as recorded) | Qualification (Amendment 01) |
|---|---|---|
| (a) Model-free arm correctness | **PASS** — 78 rows, 76 pass, 0 fail, 2 `INAPPLICABLE` | **partial** evidence only; one "boundary" case tests inactivity, one check was wrongly marked NA, coverage is not uniform across arms (§3.1–§3.3) |
| (b) Bring-up, native training, compact save/reload | **PASS** — reload bit-identical, 76/76 `qk_dual` modules | fidelity deviation in trajectory sampling; **no LoRA parity vs PEFT**; flags recorded, not enforced (§3.4–§3.7) |
| (c) **Measurement qualification** | **FAIL** — specificity 0.6875 < 0.80 | **stands; this is the binding gate** |
| (d) Budget projection | recorded `FITS` — 8.41 projected GPU-hours of 24 | **provisional**; extrapolated from one 22-timestep profile, and the reserve cancels in the `fits` test (§6.7) |

The gate that failed is the one that decides whether any semantic number this
pilot could produce would mean anything.

### The failure, exactly

64 pristine-M0 calibration images (32 intended-present, 32 unrelated
negatives) were generated. Visual judgments were recorded **before the scorer
ran and before any scorer output existed** — all 64 were unambiguous, 32
present / 32 absent / 0 uncertain. Then the frozen scorer ran.

| Metric | Value | Required |
|---|---|---|
| Definite positives | 32 | ≥ 20 ✓ |
| Definite negatives | 32 | ≥ 20 ✓ |
| **Sensitivity** | **1.000** (32/32) | ≥ 0.80 ✓ |
| **Specificity** | **0.6875** (22/32) | ≥ 0.80 ✗ |

**Verdict `SCORER_NOT_QUALIFIED`.**

### Where the unsuccessful decisions fall

**CORRECTED (Amendment 01 §2).** The breakdown of the 32 negative references is
**22 `absent`, 6 `present`, 4 `uncertain`** — so there are **six definite false
positives and four uncertain decisions**, not "ten false positives". Under the
frozen rule the uncertain decisions count against the gate, but they are not
positive misidentifications.

| Probe | Negatives n | absent | present | uncertain | Observed correct-negative fraction | Positives n |
|---|---|---|---|---|---|---|
| **backpack** | 11 | **3** | **6** | **2** | 3/11 = 0.2727 | 16 |
| bicycle | 11 | 10 | 0 | 1 | 10/11 = 0.9091 | 8 |
| violin | 10 | 9 | 0 | 1 | 9/10 = 0.9000 | 8 |

These are **observed correct-negative fractions on 10–11 negatives each, not
qualified specificities**. The earlier claim that **"bicycle and violin would
have cleared the gate" is WITHDRAWN**: the gate needs ≥ 20 definite positives
*and* ≥ 20 definite negatives, and no single object supplies either (8
positives each for bicycle and violin). **No object is independently
qualified**; the verdict rests on the aggregate 22/32.

Each image was scored **once**. `p_present` values are single deterministic
outputs of the frozen rule, **not measured probabilities**, and 3/11 is a count
across eleven different images, **not a repeated-trial frequency** for any one
image. The earlier phrasings "called a potted plant a backpack at p=0.715" and
"27% of the time" are **withdrawn** as statistical mischaracterisations.

**`backpack` is the development target**, where all six arms' learning rates
and checkpoints would have been selected and the ≥ 80%-suppression floor
applied. With a 3/11 correct-negative fraction there, the selection stage —
and everything downstream — was unmeasurable.

The decision rule contrasts present-phrases against explicit absent-phrases
("a photo with no backpack"). **Weak handling of negation is a plausible
explanation, not an isolated causal finding**: no ablation isolating negation
was run, and nothing here shows that CLIP generally cannot measure object
presence. The result concerns **this frozen rule against this reference set**.
Qualifying the instrument against held-apart
images is exactly what exposed it, *before* it could launder noise into
suppression and preservation percentages.

### What was deliberately not done

- **No scorer shopping.** LLaVA-1.6-vicuna-7b and CLIP ViT-L/14-336 are both
  cached locally and were not tried.
- **No threshold retuning.** Thresholds were frozen in `cprlib/scorer.py` and
  hashed into `artifacts/freeze_seal.json` (`8177b47cb5b71f8b…`) before any
  calibration image existed. **CORRECTION (Amendment 01 §2.5):** the earlier
  claim that raising `threshold_present` to ~0.85 *"would have passed the gate"*
  is **false and withdrawn**. A case is labelled `absent` only when
  `p_present ≤ threshold_absent = 0.40`, so raising `threshold_present` cannot
  increase the true-negative count — it only moves `present` labels into
  `uncertain`. Specificity stays 22/32 = 0.6875. No alternative threshold was
  evaluated.
- **No prompt or template edits**, no re-generation of calibration cases, no
  dropping of the backpack probe.
- **No production run.** Zero production training trajectories were executed —
  though three native bring-up training segments did run (Amendment 01 §1).

---

## 2. What the passing gates did establish

These stand on their own and are reusable, but they are **engineering
qualification, not scientific results**.

### (a) Partial correctness evidence — 78 rows, 76 passes, 0 failures, 2 NAs

**Heading corrected (Amendment 01 §3): this is partial correctness evidence
for the formulas and for gradient/engineering behaviour, not full-contract
verification of all six arms.** The raw rows are unchanged.

`artifacts/cpu_gate_results.json`. Against an **independent FP64 oracle**
written from the contract's formulas rather than by calling the production
code: forward parity (≤ 1e-5), autograd vs independent analytic gradients, FP64
central differences away from hinges (≤ 1e-7), active/inactive behaviour,
reduction and scaling order, the independent divisor (`n_latents = 11 ≠ batch`),
teacher detachment, FP32 promotion from FP64 and BF16, finite gradients and
safe small-contrast handling, axial-domain handling, and rejection of malformed
shapes, nonfinite inputs and invalid divisors.

**Four qualifications, all from the existing records:**

1. **`G3.boundary.native_gem` does not test an exact boundary.** Its recorded
   `hinge_argument` is **−8.940696716308594e−08** — negative — so the zero loss
   and zero gradient test **inactivity**, not boundary behaviour; the assertion
   (`|arg| ≤ 1e-6`) never required a zero hinge argument. **Equivalent boundary
   checks for the other hinge-bearing arms are missing.** The earlier sentence
   claiming an exact-boundary assertion is withdrawn.
2. **The axial inactivity case was wrongly marked `INAPPLICABLE`.** At
   `a=(0,0)`, `b=(1,0)`, `v=a`, `eta=1` the axial hinge argument is
   `1 − 2·1 = −1`, squarely in-domain and a valid inactivity check. It was
   **omitted, not inapplicable**. The raw NA row is left unchanged.
3. **Coverage is not uniform across arms.** Several reduction, divisor,
   `gamma`-scaling and dtype-promotion checks ran on `native_gem` only or on a
   shared path. **No independent six-arm coverage of every property is
   claimed.** Genuinely per-arm: forward parity, gradient agreement, teacher
   detachment, small-contrast/no-NaN behaviour.
4. Two cases are recorded `INAPPLICABLE` and are **not** counted as passes.

### (b) The native GEM procedure runs and the snapshot round-trips

**Heading corrected (Amendment 01 §3.4–§3.6):** "reproduces correctly"
overstated what was shown.

`artifacts/bringup_results.json`:
- Pinned FLUX revision `3de623fc…` resolves from the pristine read-only cache;
  resolved snapshot basename **equals the pin**.
- `qk_dual` resolves to **exactly 76 modules = 19 dual blocks × 4**
  (`add_q_proj`, `add_k_proj`, `to_q`, `to_k`), **0 single-block leak**,
  7,471,104 trainable parameters, r=16 α=16 dropout=0.1 scaling=1.0.
- Adapter is **inert at initialisation** (max effective delta exactly 0.0).
- Training runs: 22 processed timesteps, 2 optimizer updates, loss
  2943.19 → 2892.51, updates finite and nonzero.
- Compact snapshot is **28.55 MiB** (no transformer copy): reloads 152/152
  keys with no missing/unexpected/nonfinite/shape mismatch, **identical state
  hash**, and a fixed-input forward that is **bit-identical after reload
  (max abs diff 0.0)** — with a perturbation control confirming the test has
  power. Unmerged/merged bit-equivalence is **not** asserted.

A real defect was caught here: the first round-trip reported *not* bit-identical
(max diff 0.0703). The cause was **LoRA dropout p=0.1 still active in train
mode**, making the probe stochastic — not a serialization bug. Fixed by running
the probe in eval mode (patch P6).

**Qualifications (Amendment 01 §3.4–§3.7):**

- **Native-fidelity deviation.** The harness calls `transformer.eval()` before
  sampling the student trajectory; pinned native GEM samples after
  `wrapper.train()`, and `no_grad` does **not** disable LoRA dropout. At
  `dropout = 0.1` this changes behaviour **and the RNG stream** versus the
  pinned source. The blanket assurances **"same RNG", "numerically inert" and
  "no algorithmic change"** are **withdrawn** for this change.
- **No LoRA parity.** The custom LoRA has **no demonstrated forward, backward
  or update parity against native PEFT**. Module inventory, shapes, scaling and
  the zero initial effective update are observations, **not an equivalence
  proof**.
- **Serialization scope.** The matching restored hash and the eval-mode
  fixed-input max diff of 0.0 are valid **within this custom implementation**.
  They do **not** establish native training reproduction, merged/unmerged
  bit-equivalence, or any erasure efficacy.
- **Recorded, not enforced.** Most bring-up findings were written to a results
  dict rather than asserted. An **earlier attempt whose fixed-input equality
  check failed still completed and printed `FITS`.** A later passing probe does
  not retroactively make that gate binding.

### (c) Measured performance — provisional

1.384 s per processed timestep · 4.79 s per decoded 512×512 image · 12.7 s
pipeline load ⇒ the full 36-trajectory / 1,992-image matrix projects to
**8.41 GPU-hours** including a 30% margin, against a 24-hour ceiling.

**CORRECTED (Amendment 01 §6.7):** the arithmetic is right given its inputs,
but it extrapolates from a **single 22-timestep profile on one arm** and does
not qualify the complete workflow (scoring, analysis, checkpoint I/O, 36
trajectories' divergent behaviour). The 8-hour reserve **cancels on both sides**
of the `fits` expression, so it adds no safety there. **"Fits comfortably" and
any readiness assurance are withdrawn**; the estimate stands as **provisional**.

---

## 3. Resource usage against every ceiling

**CORRECTED (Amendment 01 §5–§6).** The figures first published here (0.1161
GPU-h, 0.123 CPU-core-h) were read **before** the release-verification job ran,
and the counters are themselves **incomplete**: costs fold in only on normal
job close, and **six failed bring-up attempts never closed**.

| Ceiling | Limit | Final counters | Lower bound incl. unclosed jobs |
|---|---|---|---|
| GPU device-hours | 24 | **0.1198536** (431.473 s) | **≥ 0.1515586** (545.611 s) |
| — release verification | (8 h reserve) | **13.645 s used** — reserve **not** untouched | — |
| CPU core-hours | 48 | **0.1276836** (459.661 s) | **≥ 0.1651622** (594.584 s) |
| Newly decoded images | 2,200 | **68** → **65 unique primary PNGs** | — |
| Elapsed | 7 days | ~1.0 h | — |
| Workers | 8 | 8 | — |
| New project storage | 200 GiB | **UNKNOWN** — ~56 MiB was a directory size at one moment, not a measured peak | — |
| GPUs | ≤ 4 genuinely available | 2 only (GPUs 2,3) | — |
| Implementation-repair cycle | 1 | 0 used | — |
| **Technical GPU retry** | **1** | **BREACHED** — 8 bring-up launches = **7 relaunches** | — |

**Lower-bound method:** per distinct PID take the maximum cumulative snapshot
that PID wrote, then sum across PIDs — no double counting, no PID missed.
**These are lower bounds, not exact totals**; a killed process stops writing,
so later cost is invisible. **Full costs remain unknown where unrecorded**, and
nothing was rerun to estimate them.

**68 decoded images** = 64 calibration + **4** decodes of the **single** M0 case
`dev.target.00` ⇒ **65 unique primary PNGs**. (The earlier "1 + 3 re-decodes"
phrasing counted the same case twice over; the totals are unchanged.)

**"0 of 1 GPU retries" is WITHDRAWN.** The ledger shows 13 job starts, 7 ends
and **8 bring-up launches** — one initial plus **seven relaunches**, exceeding
the one-retry allowance. Small compatibility fixes do not exempt their
relaunches from the cap. Per-attempt diffs and failure traces were not captured
and **no longer exist**; they are disclosed as unavailable, not reconstructed.

**Enforcement gaps (Amendment 01 §6.6):** no cross-process locking on the
counter file; costs fold only on normal close; CPU/RSS checks are per-process,
not process-tree; storage, the 4-hour entry sub-cap and the 8-hour reserve were
tracked but **never enforced**; and the published launchers do not substantiate
every shell-timeout/`rlimit` claim in the manifest.

**No numerical GPU, CPU or image ceiling overrun is demonstrated** — but that
**does not establish complete compliance**, given the incomplete accounting and
unenforced controls. **The retry-cap breach is separately supported.**

No paid API, no cloud compute, no external messages, no reviewer recruitment.
Telemetry and W&B disabled. No GPU-idleness polling.

---

## 4. Complete and missing matrix

| Planned | Status |
|---|---|
| 12 development trajectories (6 arms × 2 LRs × 1 seed) | **NOT RUN** |
| 864 development edited images + 24 M0 | **NOT RUN** (1 M0 case generated during bring-up) |
| 24 held-out endpoints (6 arms × 2 objects × 2 seeds) | **NOT RUN** |
| 960 held-out edited images + 80 M0 | **NOT RUN** |
| 64 calibration images | **COMPLETE** |
| Development selection, comparator choice, margins | **NOT REACHED** |
| Held-out comparison, bootstrap, ambiguity bounds | **NOT REACHED** |

Nothing was quietly removed: controls, concepts, seeds and evaluation all
remain in `configs/cases.json` exactly as frozen before any generation.

---

## 5. Limitations

1. **Visual judgments are Claude's.** They are **model-assisted judgments, not
   independent human ground truth**. The gate therefore compared one model's
   labels against another model's labels. A human-labelled reference could move
   the measured specificity in either direction — though a 0.273 specificity on
   plainly unrelated scenes (potted plant, beach, forest path) is not a
   marginal call.
   *(Amendment 01 §2: "0.273 specificity" is an observed correct-negative
   fraction over 11 images, not a specificity estimate, and six of the ten
   unsuccessful negative decisions were definite false positives — the other
   four were uncertain.)*
2. **The gate used 64 images and three probe objects**, with only 8–16
   positives and 10–11 negatives per object. **No object is independently
   qualified** under the ≥ 20/≥ 20 requirement. The aggregate 22/32 = 0.6875
   against a 0.80 threshold is what the frozen rule decides on.
3. **This result is about one decision rule against one reference set, not
   about CLIP generally.** Weak negation handling is a **plausible
   explanation, not an isolated causal finding** — no ablation was run. A
   differently-constructed classifier, a detector, or a VLM judge might
   qualify. None was tried, by design.
4. **Nothing here speaks to CPR's practical value.** No production training, no
   operating-point selection, no comparison. **CPR improvement remains
   untested.** Three native bring-up training segments did run. The closed CPU
   prototype (`results/cpr_gem_cpu_prototype/`, status
   `CPU_PROTOTYPE_INCOMPLETE`) remains the only CPR-specific evidence and is
   explicitly not a practical result.
5. **The `bicycle` target is a design-scope deviation** (Amendment 01 §7): its
   prior role as a retention panel class in closed SD-1.5 study `seq29` was
   accepted under a **narrower interpretation than the fresh-target/panel
   rule**. This is not evidence the old experiment was rerun — **no held-out
   bicycle run exists** and no old-asset reuse is established. **No substitute
   object is selected now.**
6. **GPU-second accounting charges wall-clock × devices held**, which
   over-counts idle time inside a job and cannot see other users' processes.
7. **Known code defects are documented and left unfixed** (Amendment 01 §3):
   the missing boundary checks, the omitted axial inactivity check, the
   `eval()`-mode trajectory-sampling fidelity deviation, the unproven LoRA
   parity, and the recorded-not-enforced bring-up flags.

---

## 6. Recommendation — WITHDRAWN

**The previous recommendation is withdrawn in full** (Amendment 01 §8). It
proposed qualifying a replacement instrument as a follow-up task and,
specifically, scoring candidate instruments against the published calibration
set at zero GPU cost.

**The calibration set is now seen data.** Its images, the scorer's outputs on
them and the reference labels are all published, and the failure mode has been
analysed in public. Choosing a future scorer against it would make it
**development material, not an untouched qualification test**, and any
instrument selected on it could not be said to have passed an independent gate.

**No automatic re-attempt, scorer selection, or successor task is recommended
or authorized.** The code limitations in §2(a)/§2(b) and Amendment 01 §3 are
left **unfixed and documented**.

**Not authorized and not done here:** no new probe, no scorer substitution, no
threshold change, no production training, no expansion of concepts or seeds, no
test implementation, and no further correction cycle.

**CPR's practical value remains unanswered.**

---

## 7. Reproduction

```bash
source results/cpr_real_model_controls_v1/code/runenv.sh
$CPR_PY code/gate/run_cpu_gate.py                       # model-free, CPU, ~1 min
CUDA_VISIBLE_DEVICES=2 $CPR_PY code/gate/bringup.py     # GPU bring-up, ~1 min
CUDA_VISIBLE_DEVICES=2 $CPR_PY code/gate/gen_calibration.py   # 64 images, ~5 min
```

Scoring and qualification reproduce from the published
`artifacts/calibration_visual_judgments.json` and the frozen scorer config.

---

## 8. Publication and release verification

Release **`cpr-real-model-controls-v1-entrygate`**, target commit
`a462b94800d44441c6c056c4cd220350862741ad`.

All six assets were **downloaded back from GitHub** and verified:
`sha256sum -c` **OK on every asset**; counts confirmed (**64** calibration
PNGs, **1** M0 dev PNG, **8** contact sheets); and the adapter was
**reconstructed from the downloaded bytes** on a fresh FLUX.1-dev load —
152/152 keys, no missing/unexpected/nonfinite/shape mismatch, all finite,
recomputed state hash `0b3e4536…` **matching the stored hash**.

**Disclosed verification limits.** The adapter check proves the snapshot
reloads into the intended effective weights; it does **not** assert
unmerged/merged bit-equivalence. The PNGs are verified by archive hash and
file count, not by re-generating them (re-generation would consume image
budget and is unnecessary — they are the published primary artifacts). The
download used the same host and token as the upload.

---

## 10. Amendment 01

This report is **corrected and superseded in part** by:

| Document | Contents |
|---|---|
| [`AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md`](AMENDMENT_01_GATE_FIDELITY_AND_ACCOUNTING.md) | retained status; measurement-interpretation corrections; correctness and native-fidelity qualifications; freeze, retry and accounting corrections; withdrawn recommendation |
| [`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md) | mechanical preservation verification, ledger arithmetic, release-asset metadata |

The amendment **governs wherever it and this report differ**, and also
supersedes contradictory statements in the protected, unedited
`RUN_MANIFEST.md` (notably its F1 and F2 freeze timestamps). All raw records —
the 78 gate rows, the 64 predictions and judgments, counters, ledgers, seals
and checksums — are preserved unchanged.

---

### Attribution

Native GEM arithmetic and training procedure are **re-implemented by reading**
[`GEM@c328fc6`](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/gem/operators/erasure/erasure_gem.py);
GEM is third-party work by its authors and **no GEM module is imported**.
Backbone `black-forest-labs/FLUX.1-dev@3de623fc` under the FLUX.1 [dev]
Non-Commercial License — **no model weights are redistributed**; published
assets are generated images and a ~28 MiB LoRA adapter trained here, for
non-commercial research use.
