# RUN_MANIFEST — CPR real-model controls pilot v1

**Question.** At comparable actual target suppression, does CPR preserve
allowed content better than native GEM and the strongest simple controls?

**This is a screening pilot.** Distinct formulas, favourable synthetic examples
or matching hinge boundaries do not establish practical improvement, novelty or
paper readiness.

---

## 0. Freeze history (append-only)

| # | UTC | What was frozen | Evidence |
|---|---|---|---|
| F1 | 2026-10-19T03:52Z | `configs/cases.json` — the complete case/seed/trajectory matrix — written and validated **before any image was generated or any model was loaded** | sha256 `fca4dd02105458686eda80cef508aeaa7c8d456c4385c5b0fff53d986d58bf3c` |
| F2 | 2026-10-19T03:56Z | Model-free arm-correctness gate executed and passed (78 cases, 0 failures) before any GPU work | `artifacts/cpu_gate_results.json` |
| F3 | 2026-10-19T04:02Z | `cprlib/scorer.py` — scorer identity, revision, prompts, thresholds, uncertainty handling, coherence rubric — frozen **before any calibration image existed** | `scorer.config_hash()`, recorded in `artifacts/freeze_seal.json` |
| F4 | 2026-10-19T04:04:19Z | **This manifest and the final runnable code version**, frozen after bring-up and the permitted small fixes, **before any production outcome** | `artifacts/freeze_seal.json` |

**Honest chronology note.** The case matrix (F1) and the arm-correctness gate
(F2) preceded every generation. Bring-up (section 5) then ran and decoded **4
images** (one declared M0 case, re-decoded across failed attempts) while small
path/API/memory fixes were applied; `RUN_MANIFEST.md` itself was finalised at F4,
after bring-up and before any production outcome, as the contract's
"freeze the final runnable version before production outcomes" permits. No
production trajectory, no development outcome and no held-out outcome existed at
F4. Corrections below are append-only and timestamped; nothing is backdated.

---

## 1. Environment, source and model pins

| Item | Pin |
|---|---|
| Host | hyperplane, 4x A100-SXM4-40GB, driver 550.144.03, CUDA 12.4 |
| GPUs used | **2 and 3 only** — GPUs 0 and 1 were occupied by another user and left alone |
| Python | 3.10.12 |
| torch / diffusers / transformers | 2.4.0+cu121 / 0.30.2 / 4.44.2 (pre-existing venv, reused **read-only**) |
| Pilot-owned overlay | `pydeps/` — `sentencepiece==0.2.0`, `protobuf==4.25.3` (4.8 MiB). Required because the FLUX T5 tokenizer config sets `add_prefix_space`, forcing a slow→fast conversion. **The shared venv was not modified.** |
| GEM reference | `multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd` |
| Loss reference | `gem/operators/erasure/erasure_gem.py` lines 233-249 |
| Backbone | `black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| Backbone cache | `/data/bijaypandey/cuig_pilot/hf_home/hub` — pristine, read-only, `HF_HUB_OFFLINE=1`; resolved snapshot basename **equals the pin** |
| Scorer | `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41` |
| Telemetry | disabled (`WANDB_DISABLED`, `HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`) |

No GEM module is imported anywhere. The loss and the training procedure are
**re-implemented from the source read as a reference**. This is a
**source-derived object experiment, not a reproduction of the authors'
benchmark numbers**.

---

## 2. Resolved native configuration

FLUX resolves to the **base `GEMConfig`** (the registry defines only `gem` and
`gem:sd3`; there is no FLUX-specific config), so:

| Parameter | Value | Source |
|---|---|---|
| learning rate | **1e-3** (and 5e-4 = half) | `GEMConfig.learning_rate` |
| `gamma` | 100.0 | contract + `GEMConfig` |
| `eta` | 1 | contract (`GEMConfig` default is also 1.0) |
| `max_step_budget` | 250 **processed timesteps**, not optimizer updates | `GEMConfig` |
| `partial_denoising_end_at_timestep` | 10 → **11 latents** per trajectory | callback appends after steps 0..10 inclusive |
| `partial_denoising_mode` | `student` | `GEMConfig` |
| `partial_denoising_guidance_scale` | 3.0 | `GEMConfig` |
| internal denoising steps | 28 (`partial_denoising_inference_timesteps` is None → `num_inference_steps`) | `MODEL_CONFIGS["flux"]` |
| `predict_noise` guidance | 1.0 | GEM calls it without the argument |
| `max_parallel_timesteps` | **1** | frozen by this contract |
| optimizer | AdamW(lr) | `erasure_gem.py` |
| anchor | `EMPTY_CONCEPT`, exact string **`""`** (empty string) | `gem/operators/utils.py:3` |
| LoRA | `qk_dual`, **r=16, alpha=16, dropout=0.1, bias=none**, scaling 1.0 | `AdaptationConfig.__post_init__` |
| `qk_dual` targets | `transformer` root, **excluding** `single_transformer_blocks`, ending in `add_q_proj`/`add_k_proj`/`to_q`/`to_k` | `pipeline_wrapper_flux._get_modules_for_subset` |
| resolved target count | **76 modules = 19 dual blocks x 4**, 7,471,104 trainable params, 0 single-block leak | measured at bring-up |
| generation | 512x512, 28 steps, guidance 3.5 | contract |
| precision | **BF16 backbone, FP32 loss** (contract; `MODEL_CONFIGS` says float16 — deliberate recorded deviation) | contract |

**Native accumulation preserved.** `loss.backward()` runs once per processed
timestep inside a trajectory and `optimizer.step()` runs **once per outer
iteration**, so gradients accumulate across the trajectory window. The final
iteration terminates mid-window when the 250-timestep budget runs out and still
takes its optimizer step — the native partial-window behaviour. 250 timesteps /
11 per trajectory ⇒ 22 full windows + one 8-timestep window = **23 optimizer
updates**. The budget is never exceeded to manufacture a checkpoint.

**Faithfulness notes recorded, not silently fixed.**
1. GEM pairs `latents_per_step[i]` (the latent *after* step i) with timestep
   index `i` (the noise level *before* step i) — an off-by-one in the pinned
   source. It is **preserved**, not corrected.
2. GEM's `calculate_image_seq_len` uses diffusers' `vae_scale_factor=16` and
   returns 256 for 512x512, whereas the true packed token count is 1024. In the
   source this value only ever feeds the scheduler's `mu` shift, never a tensor
   shape, and the pipeline recomputes its own `mu` during generation. Preserved.

---

## 3. The six arms

Shared: `v` student, `a` detached anchor teacher, `b` detached target teacher;
FP32; all nonbatch dims flattened; `eps = 1e-6`;
`Dplus = ||v-a+eps*1||`, `Dminus = ||v-b+eps*1||`, `L = ||b-a||`,
`e = (b-a)/L` if `L > 1e-6` else exactly zero, `alpha = (b-v).e`.
A safe denominator is formed **before** masking, so no eager `0/0` is evaluated.

| # | Arm | Per-sample loss |
|---|---|---|
| 1 | `native_gem` | `relu(Dplus - eta*Dminus)` |
| 2 | `cpr` | `relu(Dplus - eta*alpha)` |
| 3 | `centered_radial` | `relu(Dplus - eta*sqrt(\|\|v-b\|\|^2 + eps^2*d))` |
| 4 | `margin_truncated` | `relu(Dplus - eta*L/(1+eta))`, threshold **detached** |
| 5 | `anchor_only` | `Dplus` — the native `eta=0` control. **Not MSE, not ESD.** |
| 6 | `axial_margin` | `relu(L - (1+eta)*alpha)`, defined only for `L > 1e-6` |

Reduction, shared and native: `loss = (gamma * per_sample.sum()) / n_latents`,
with the **independently supplied** divisor `n_latents = 11` (not the batch
size) and `torch.relu` boundary behaviour (`clamp` never substituted).

For CPR, small contrast disables repulsion while **attraction is retained**.
An **axial production-domain violation is an explicitly recorded ARM FAILURE**
(`AxialDomainViolation`): never skipped silently, never given a fallback value.

The truncated threshold has **only an ideal unshifted collinear stopping
interpretation**. No claim of equal native first stopping, off-axis equivalence
or empirical suppression is made from it, and no root-matching subproject is
performed.

**Coupling.** All arms share adaptation, optimizer opportunity and paired
seeds; adapter initialisation is seeded identically per trajectory and every
run starts from pristine M0. **Disclosed:** because the partial trajectory is
generated by the *student* (`partial_denoising_mode="student"`), the sampled
trajectories can and will diverge across arms once their adapters differ.

---

## 4. Frozen run matrix

Targets — **development: backpack**; **held-out: bicycle and violin**.

**Collision check against closed work (required).** The complete set of
erasure targets in closed studies is {English Springer, French horn, golf ball,
Frida Kahlo, Katsushika Hokusai, Vincent van Gogh, Bears, Cats, Horses, dog,
sandwich, teapot}. **None of backpack, bicycle or violin is a closed erasure
target.** Two partial overlaps are disclosed rather than waved away:
* `backpack` appears in closed `rece_seq_golfball_englishspringer` only as a
  **scene word inside parachute prompts** ("sky/field/backpack/cargo-plane/canopy"),
  never as a target or a panel class.
* `bicycle` appears in closed `seq29` as a **retention panel class** of an
  SD-1.5 study (40 images/checkpoint). It was never erased there.
Judged **not a genuine collision**: no closed erasure target is reused, and no
closed artefact is reused — different backbone (FLUX vs SD-1.5), different
method, fresh prompts, fresh seeds, fresh images. Recorded so a reader can
disagree.

Allowed panels contain **related but non-target** objects and ordinary
contexts, and exclude near-synonyms (`backpack`: rucksack/knapsack/daypack/
bookbag/packsack; `bicycle`: bike/cycle/pushbike/two-wheeler/ten-speed;
`violin`: fiddle/viola). The dev allowed panel also excludes both held-out
targets.

| Stage | Trajectories | Cases | Images |
|---|---|---|---|
| Development | 6 arms x 2 LRs x 1 seed = **12** | 12 target + 12 allowed = 24, at each of 3 checkpoints | 12x3x24 = **864** edited + **24** M0 = **888** |
| Held-out | 6 arms x 2 objects x 2 seeds = **24** endpoints | 20 target + 20 allowed = 40 | 24x40 = **960** edited + **80** M0 = **1,040** |
| **Primary scientific total** | | | **1,928** |
| Calibration (separate) | | 32 intended-present + 32 unrelated negatives | **64** |
| Technical/retry allowance | | | **208** (not study cases) |

Checkpoints = the first completed optimizer update whose cumulative processed
timesteps reach **63 (25%)**, **125 (50%)**, and the actual native terminal
endpoint at **250**.

Seeds: dev training seed 11; held-out training seeds 21 and 22; evaluation
seeds from a disjoint block (900000+ / 950000+). **Training and evaluation
seeds are disjoint.** Every evaluation case carries its own fixed seed, and
**identical initial-noise SHA-256 hashes are verified across conditions** for
each case — matching seed numbers alone are not accepted as proof of pairing.

Every case is preserved, **including M0 failures**. No prompt or seed is
replaced after viewing outputs.

---

## 5. Entry gate

**(a) Model-free arm correctness — PASSED** (`artifacts/cpu_gate_results.json`):
78 cases, 76 pass, 0 fail, 2 recorded `INAPPLICABLE` (never counted as passes).
Covers, for every arm: source-faithful forward parity against an independent
FP64 oracle; autograd vs independent analytic gradients; FP64 finite
differences away from hinges; active / inactive / exact-boundary behaviour
(boundary asserts **exactly zero loss and exactly zero gradient**); reduction
and scaling order; independent divisor; teacher detachment; FP32 promotion from
FP64 and BF16; finite gradients and safe small-contrast handling; explicit
axial-domain handling; and rejection of malformed shapes, nonfinite inputs and
invalid divisors. Production uses **native AdamW**, not normalized progress
matching; no positive-progress gate is used in production, and the diagnostic
`abs(g_parallel)` defect of the closed CPU prototype is not inherited.

**(b) Bring-up and serialization — PASSED** (`artifacts/bringup_results.json`):
pinned revision resolves from the pristine cache; `qk_dual` resolves to exactly
76 modules across 19 dual blocks with 0 single-block leak; the adapter is
**inert at initialisation** (max effective delta exactly 0); updates are finite
and nonzero; the compact adapter snapshot is **28.55 MiB** (no transformer
copy), reloads with 152/152 keys, no missing/unexpected/nonfinite/shape
mismatch, **identical state hash**, and a fixed-input forward that is
**bit-identical after reload (max abs diff 0.0)** with the perturbation control
confirming the test has power. **Unmerged/merged bit-equivalence is not
asserted.**

**(c) Measurement qualification** — see section 6. **Failed qualification ends
the task.**

**(d) Budget projection** (measured, not assumed): 1.384 s per processed
timestep, 4.79 s per decoded image, 12.7 s pipeline load ⇒ train 3.46 +
generate 2.65 + load 0.36 GPU-hours, x1.30 conservative overhead =
**8.41 projected GPU-hours** against 24, with the 8-hour evaluation reserve
intact. Verdict **FITS**. If it had not, the run would have stopped
`BUDGET_INSUFFICIENT` rather than shrinking the study.

---

## 6. Frozen measurement and decision rules

**Object presence** — frozen in `cprlib/scorer.py` before any calibration image
existed: CLIP ViT-L/14 @ `32bd6428`, three present-templates vs three explicit
absent-templates, mean logits per group, 2-way softmax; **present if
p >= 0.60, absent if p <= 0.40, otherwise uncertain**. Raw CLIP similarity is
not assumed to be calibrated presence; the rule must be qualified.

**Qualification gate:** the 64 calibration images are scored, and separately
compared against visual judgments **recorded without seeing scorer outputs**.
Requires **>= 20 definite positives, >= 20 definite negatives, >= 80%
sensitivity and >= 80% specificity**. Uncertain visual labels are preserved and
excluded from the definite counts; uncertain scorer labels count as errors.
**No scorer shopping, no threshold retuning after inspecting failures.**

**Visual judgments in this pilot are produced by Claude.** They are therefore
**model-assisted judgments, not independent human ground truth**, and every
semantic conclusion is provisional under that instrument.

**Separate recorded fields:** object presence (present/absent/uncertain);
whole-frame coherence (coherent/degenerate/uncertain, rubric frozen in
`COHERENCE_RUBRIC`); local defects. **A distorted face alone is a local defect,
not whole-frame degeneration.** **Target absence does not establish preserved
quality.**

**Eligibility from M0 alone:** target denominator = M0-present cases; allowed
denominator = M0-present **and coherent** cases. Requires **>= 10/12 eligible
development** and **>= 16/20 eligible held-out** cases per panel, else the
comparison is marked **measurement-unqualified** — with no replacements.
All-case counts are reported alongside eligible denominators.

**Metrics.** Suppression = fraction of eligible target cases confidently absent
after editing. Primary preservation = fraction of eligible allowed cases still
present **and** coherent. Target-panel coherence reported separately.
Uncertain/missing edited target output ⇒ unsuccessful suppression;
uncertain/missing allowed output ⇒ unsuccessful preservation.

**Selection (development).** Per arm, among configurations reaching **>= 80%
suppression**, maximise preservation; ties broken by lower training compute,
then by stable configuration ID. If none qualify, retain the
highest-suppression candidate and flag **efficacy-unmatched** — that arm is
never hidden. The primary simple comparator is selected by the same rule.
Native GEM and every control are published separately. **If native GEM or CPR
cannot reach the development efficacy floor, the pilot closes with that finding
rather than proceeding to held-out evaluation to hunt for a win.** Each arm's
selected learning rate and checkpoint, and the primary comparator, are frozen
**before any held-out training or generation**; there is no held-out tuning, no
object-specific stopping selection and no evaluator change.

**Predeclared screening margins (CPR vs the primary comparator):**
CPR loses at most **5 pp** of suppression; CPR gains at least **10 pp** of
allowed-content preservation; CPR loses at most **5 pp** of target-panel
coherence. **Suppression achieved through additional frame degeneration cannot
qualify.**

**Uncertainty, frozen before outcomes.** Paired bootstrap over cases (10,000
resamples, 95% percentile interval) preserving pairing and the training-seed
structure; every object/training-seed result reported individually; plus
conservative ambiguity bounds that assign every uncertain label
**pessimistically for CPR and optimistically for controls**.
**Terminal rule:** wide uncertainty, ambiguity-sensitivity flips, or material
suppression mismatch ⇒ `INCONCLUSIVE`. Favourable point estimates alone are
insufficient. **Two concepts and two seeds cannot establish population-level
equivalence or formal noninferiority.** If another simple control matches CPR,
that constrains the claimed mechanism even if the primary comparator loses.

---

## 7. Budget and enforcement

| Ceiling | Value | Enforcement actually in force |
|---|---|---|
| Elapsed | 7 days; science stops day 6 | operator clock + ledger |
| GPU | 24 device-hours (entry <= 4; **8 reserved** for held-out eval + publication verification) | cumulative cross-process counter in `artifacts/counters.json`, checked by the in-process watchdog |
| Images | 2,200 newly decoded | hard counter; `record_images` **raises** past the ceiling; automatic training previews disabled (trajectories return latents, never decoded) |
| CPU | 48 core-hours; 8 workers | `OMP/MKL/OPENBLAS_NUM_THREADS=8`, rusage in ledger |
| RSS | 128 GiB process tree | watchdog poll |
| Storage | 200 GiB; >= 50 GiB free on `/data` | watchdog poll of `statvfs` |
| GPU count | up to 4 *genuinely available* | GPUs 2,3 only; 0,1 left to another user |

Watchdog layers: shell `rlimit`s + external `timeout` + an in-process thread
polling wall clock, CPU, RSS, cumulative GPU-hours and free disk every 5 s,
`os._exit(97)` on breach. The ledger is append-only JSONL covering parent/child
jobs, setup, attempts, timestamps, device allocation, GPU-seconds, CPU, memory,
disk, decoded images and stop events.

**Disclosed enforcement gaps.** `getrusage(RUSAGE_CHILDREN)` sees only reaped
children. GPU-seconds are charged as wall-clock x devices held by a job, which
**over-counts** idle time inside a job and cannot see another user's processes.
The 7-day elapsed cap is operator-enforced, not kernel-enforced. The in-process
poller samples every 5 s and could miss a shorter spike.

**Repair allowance.** One focused CPU implementation-repair cycle before
production and one technical GPU retry across the whole pilot, within the same
ceilings. Failures and diffs are preserved. **No semantic retries and no
outcome-dependent tolerance changes.** Small path/logging/serialization/
instrumentation fixes are permitted and their patches preserved; those applied
during bring-up are listed in `artifacts/bringup_patches.md`.

---

## 8. Publication plan

Publish under `results/cpr_real_model_controls_v1/`: code, environment/source/
model pins, exact formulas, frozen configs/cases/seeds/decision rules, gate
checks and failures, all scores/labels/logs/counters/run records, the complete
and missing matrix, per-arm results, the full resource ledger and
`FINAL_REPORT.md`. Selected adapters, reconstruction instructions and the
scientific PNGs go out as **new versioned GitHub Release assets**, respecting
upstream and FLUX-dev licensing (non-commercial; no model weights
redistributed — only ~28 MiB LoRA adapters trained here, plus generated
images). Development snapshot hashes and reconstruction information are
published. Published assets are downloaded back into a verification area and
their hashes, counts and adapter reload verified within the same budget.
A noncircular evidence index is built from final committed bytes; an index-only
child commit is acceptable. Live remote SHA, release tag target and downloads
are verified. Failure to publish or verify ⇒ **`PUBLICATION_INCOMPLETE`**, not
completed delivery.
