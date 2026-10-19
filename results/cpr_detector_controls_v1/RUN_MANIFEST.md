# RUN_MANIFEST — CPR detector-defined controls screen v1

**Primary question.** At comparable reduction in target detections, does CPR
retain allowed-object detections better than native GEM and the strongest
simple controls?

**This is a detector-defined screen** — not proof of semantic erasure, scene
coherence, novelty, SOTA or paper readiness.

---

## 0. Freeze status

This is the **prospective freeze**, committed **before any numerical method
check, any detector scoring and any FLUX output**. What legitimately preceded
it, with logging, is confined to what the contract allows: generic
access/import/dependency checks and **annotation-only** selection.

| Preceded the freeze | Evidence |
|---|---|
| network/access checks for both detectors and COCO | `artifacts/preflight.log` |
| dependency check: pilot overlay (`peft` etc.), stack integrity | `artifacts/preflight.log` |
| fresh GEM checkout at the pinned commit | `GEM_src` @ `c328fc6…` |
| detector **revision resolution** and `id2label` inspection (no inference, CPU only) | §3 below |
| **annotation-only** category selection and qualification-case construction | `configs/categories_and_qualification.json` |

**No image was generated, no detector inference was run, and no loss was
evaluated before this freeze.** Any authorized implementation repair gets an
**additive** patch/freeze commit; this first freeze is never backdated or
overwritten.

---

## 1. Pins

| Item | Pin |
|---|---|
| Host | hyperplane, 4x A100-SXM4-40GB. **GPUs 2 and 3 only** — 0 and 1 are another user's |
| Python / torch / diffusers / transformers | 3.10.12 / 2.4.0+cu121 / 0.30.2 / 4.44.2 (shared venv, **read-only**) |
| Pilot overlay | `pydeps/` — `peft==0.12.0`, `sentencepiece==0.2.0`, `protobuf==4.25.3`, installed `--no-deps` (7.1 MiB). A first attempt without `--no-deps` pulled torch 2.14.1 (5.5 GiB) and was **deleted and redone**; the shared venv was never modified. |
| GEM | `multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd`, **fresh checkout** in `GEM_src/` |
| Backbone | `black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` (pristine read-only cache) |
| Primary detector | `PekingU/rtdetr_r50vd` @ **`df939e661d8c52e80608d1ec566561aabd25a4e7`** |
| Secondary detector | `facebook/detr-resnet-50` @ **`70120ba84d68ca1211e007c4fb61d0cd5424be54`** (`no_timm`) |
| COCO | `instances_val2017.json`, sha256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`; zip sha256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| Telemetry | disabled (`WANDB_DISABLED`, `HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`) |

**`no_timm` rationale, frozen before scoring:** both DETR revisions load here,
but `main` uses a timm backbone and the installed timm is 1.0.15 — far newer
than the version DETR was published against. The officially documented
`no_timm` revision removes that third-party compatibility risk entirely.

---

## 2. Evaluators

Both thresholds fixed at **0.30**, never tuned. A class is **detected** iff an
official postprocessed detection of that class reaches the threshold;
otherwise **not detected**. **A technical failure is a missing measurement,
not absence.** Detector scores are **not calibrated presence probabilities**
and the two models' scales are **not comparable**.

**Label spaces differ and are never assumed equal.** Measured on the pinned
revisions: RT-DETR has **80 contiguous ids 0–79** and uses VOC-style names
`motorbike` / `aeroplane`; DETR has **91 ids following COCO ids 1–90 with 11
unused `N/A` slots**. Examples: `stop sign` = **11** (RT-DETR) vs **13**
(DETR); `fire hydrant` = **10** vs **11**; `person` = **0** vs **1**. All
lookups resolve through each model's own `id2label`, keyed by canonical COCO
category id, with a frozen synonym table (`cprd/labels.py`).

Image-level detection and correct-class box matching at **IoU ≥ 0.50** are both
recorded.

---

## 3. Categories — chosen by metadata alone, then frozen

**Closed-category exclusions** (from records of *actually executed* work):
executed erasure targets {English Springer, French horn, Frida Kahlo, golf
ball, Hokusai, van Gogh, Bears, Cats, Horses, dog, sandwich, teapot}; executed
retention-panel classes {cat, dog, sandwich, horse, bird, chair, bicycle}; and
{backpack, violin} for which the v1 pilot actually generated calibration
images. Repository hits for every candidate were checked and found to be
**incidental COCO caption text**, which is not panel membership.

**Eligibility** required a supported canonical label in **both** detectors,
≥ 20 positives (noncrowd box ≥ 5% of image area), ≥ 20 annotation-negatives,
and an available allowed panel. Walking the fixed candidate list in order:

| Candidate | COCO id | pos ≥5% | verdict |
|---|---|---|---|
| traffic light | 10 | **15** | ineligible (< 20) |
| **stop sign** | 13 | 27 | **DEVELOPMENT TARGET** |
| parking meter | 14 | **18** | ineligible (< 20) |
| **bench** | 15 | 121 | **HELD-OUT TARGET 1** |
| **fire hydrant** | 11 | 46 | **HELD-OUT TARGET 2** |

**Allowed panel (identical for all three targets): `person`, `car`,
`motorcycle`, `airplane`.**

**Disclosed deficit.** All three targets are supercategory `outdoor`, and the
only other outdoor classes (traffic light, parking meter) fail the
≥ 20-positive floor. The same-supercategory slots are therefore **empty**, and
the deficit of 2 is filled by continuing down the other-class list in ascending
COCO id. **Consequence: the allowed panels contain no same-supercategory
confounder class**, which weakens the "related but non-target" aspect of the
retention test. This follows deterministically from the frozen rule and the
annotation counts; it is disclosed, not worked around.

**Also disclosed:** `fire hydrant` and `bench` appear in the v1 pilot's
*declared but never executed* allowed panel. The exclusion rule keys on
*actually executed* work, and v1 executed no panel, so they are not treated as
closed. A reader may disagree.

**Scored classes (7):** airplane, bench, car, fire hydrant, motorcycle,
person, stop sign → **280 image/class pairs over 264 unique photographs**
(≤ 600 each). Qualification cases were ordered by
`sha256(project_id | class_id | image_id)` and **never by detector
predictions**. Ten same-supercategory confounder negatives were used for every
class except `person`, whose supercategory has only one class: its deficit of
10 is filled from plain annotation-negatives and disclosed.

### Qualification gate (operational, not a statistical guarantee)

For **every scored class**: ≥ **16/20** positives with a matching detection and
≥ **16/20** negatives without a detection. Counts and uncertainty are
published; **a failing target class is never pooled away**. The same
descriptive check runs for the secondary detector; it cannot replace a failed
primary or adjudicate truth. **Secondary failure alone does not block the
primary-defined comparison but does prevent a corroborated superiority claim
on the affected categories.**

**Stated limits.** These detectors were developed and evaluated on COCO, so
this is a benchmark/implementation check, **not independent out-of-domain
validation**. Annotation-negative is an **imperfect absence reference**.
Real-photo qualification **does not establish accuracy on edited generations**.
**Detector agreement is not ground truth.** Prompts, Claude judgments and the
failed pilot's calibration images are **not** qualification labels.

---

## 4. Native GEM + PEFT with a small loss hook

The **pinned native trainer and wrapper execute**; the training loop, sampler,
teachers, optimizer, divisor and reduction are GEM's own, and adaptation is
**native PEFT**. No LoRA is reimplemented here.

**Resolved from the pinned source** (FLUX uses the base `GEMConfig`; the
registry defines only `gem` and `gem:sd3`):

| Parameter | Value |
|---|---|
| learning rate | **1e-3** native, **5e-4** half |
| `eta` / `gamma` | 1 / 100 |
| `max_step_budget` | 250 **processed timesteps**, not optimizer updates |
| `partial_denoising_end_at_timestep` | 10 → **11 latents** per trajectory |
| mode / guidance | `student` / 3.0 |
| internal steps | 28 (`None` → `MODEL_CONFIGS["flux"]["num_inference_steps"]`) |
| `max_parallel_timesteps` | **1** (contract) |
| LoRA | `qk_dual`, **r=16, alpha=16, dropout=0.1**, bias none |
| anchor | `EMPTY_CONCEPT` = **`""`** |
| precision | **BF16 backbone, FP32 loss** (`MODEL_CONFIGS` says float16 — deliberate recorded deviation) |
| generation | 512×512, 28 steps, guidance 3.5 |

Native partial-window accumulation is preserved: `backward()` per processed
timestep, **one `optimizer.step()` per outer iteration**; 250/11 ⇒ 22 full
windows + one 8-timestep window = **23 optimizer updates**. The budget is never
exceeded to manufacture a checkpoint. Every run starts from pristine M0, and
initialisation/random streams are paired — **without claiming that
student-dependent trajectories stay identical across arms** (the sampler is
`student`, so they will diverge once adapters differ).

### Authorized source patches (exact diffs in `patches/`)

| # | File | Change |
|---|---|---|
| P1 | `erasure_gem.py` | small attributed **loss dispatch**. With no hook, or when the hook returns `None` (the `native_gem` arm), the **original native loss block executes unchanged**, verbatim and clearly delimited. |
| P2 | `base_pipeline_wrapper.py` | **output-only**: skip the preview VAE decode when `return_all_latents=True`. The caller discards it (`_, all_latents_of_trajectory = …`), the block draws **no random numbers** and never touches `latents_per_step`, the sampler, the loss or the optimizer. Without this, ~828 discarded decodes would be charged to the image ceiling. |
| P3 | `erasure_gem.py` | logging/**checkpoint hook** after the native `optimizer.step()`; never alters the loop, budget, sampler or optimizer. |

No replacement trajectory sampler, no train/eval change, no optimizer change,
no silent scientific bug fix. **Mode during student sampling is verified
explicitly: `no_grad` does not mean `eval`,** and native dropout/RNG behaviour
is preserved as the pinned source has it.

### The six per-sample losses

With detached `a`, `b`, FP32 flattened `v`, `eps = 1e-6`,
`Dplus = ||v-a+eps·1||`, `Dminus = ||v-b+eps·1||`, `L = ||b-a||`,
`e = (b-a)/L` when `L > 1e-6` else zero, `alpha = (b-v)·e`,
safe denominator formed **before** masking:

| Arm | Per-sample loss |
|---|---|
| `native_gem` | `relu(Dplus - eta*Dminus)` — **GEM's own block** |
| `cpr` | `relu(Dplus - eta*alpha)` |
| `centered_radial` | `relu(Dplus - eta*sqrt(\|\|v-b\|\|² + eps²·d))` |
| `truncated_anchor` | `relu(Dplus - eta*L/(1+eta))`, threshold **detached** |
| `anchor_only` | `Dplus` — **not MSE, not ESD** |
| `axial_margin` | `relu(L - (1+eta)*alpha)`, requires `L > 1e-6` |

Native scalar order preserved: `(gamma * per_sample.sum()) / n_latents`.
`torch.relu` boundary semantics kept. CPR **retains attraction** at degenerate
contrast. An axial domain violation is an **explicit arm failure** — never a
skipped sample or invented fallback. The truncated threshold has **only** an
ideal-unshifted-collinear interpretation; **no native first-stop matching is
claimed**.

---

## 5. Frozen matrix

| Stage | Trajectories | Cases | Images |
|---|---|---|---|
| Development | 6 arms × 2 LRs × 1 seed = **12** | 12 target + 12 allowed (4 classes × 3 prompts) = 24, at 3 checkpoints | 864 edited + 24 M0 = **888** |
| Held-out | 6 arms × 2 objects × 2 seeds = **24** endpoints | 20 target + 20 allowed (4 × 5) = 40 | 960 edited + 80 M0 = **1,040** |
| **Primary total** | | | **1,928** (+272 technical allowance = 2,200) |

Checkpoints: first full optimizer update reaching **63** (25%), **125** (50%)
processed timesteps, and the **actual native terminal endpoint** at 250.

Seeds: dev training **4101**; held-out training **4201, 4202**; evaluation
seeds from a disjoint 7,000,000+ block, all unique and disjoint from training
seeds. Bootstrap seed **4033946196** (derived from the project id).

Initial-noise tensors/hashes are stored to prove shared inputs; **shared noise
does not establish shared layout**. Every M0 failure and missing/corrupt
result is preserved. **No prompt, seed or category replacement.**

### Eligibility (from primary-detector M0 outputs only)

Development ≥ **10/12** target and allowed cases with ≥ **2/3** per allowed
class; held-out ≥ **16/20** targets and ≥ **4/5** per allowed class.
Insufficient eligibility ⇒ panel **unqualified, without replacement**.
All-case counts are reported too. The secondary detector uses **its own** M0
eligibility plus a common-intersection table, with denominator differences
disclosed.

**S** = eligible target cases no longer detected. **R** = eligible allowed
cases still detected. Detectors receive **pixels only** — no arm/checkpoint
metadata, no negation prompts. **Missing measurements are not successful
nondetection and not retention.** Frozen invalid/constant/black-output checks
run on every image; **a flagged output cannot count as successful
suppression**, and these checks say nothing about scene coherence.

### Development selection

Among each arm's six candidates reaching **S ≥ 0.80**, maximise **R**; ties
broken by fewer processed timesteps, then lower-rate identifier, then stable
id. If none qualify, retain highest S, then R, then stable id, explicitly
**efficacy-unmatched**. The strongest development-qualified simple comparator
is selected by the same rule. **All arms are published.** If native GEM or CPR
misses the development floor, the study **closes with that scoped result**
before held-out expansion — and CPR is **not** globally rejected. An additive
**selection-lock commit** precedes any held-out training or generation; there
is no held-out tuning or stopping selection.

---

## 6. Frozen analysis and terminal rules

Margins: CPR target-detection reduction at most **5 pp** lower; allowed-object
retention at least **10 pp** higher.

**10,000 paired case bootstrap resamples**, 95% percentile interval from the
2.5th/97.5th quantiles, seed **4033946196**. Target and allowed panels are
resampled **separately**; allowed cases are **stratified by class**; the two
fixed training-seed endpoints are **averaged within each resampled case**;
held-out objects are analysed **separately** with individual-seed tables.

These intervals are **conditional on these trained endpoints** — not
population concept/training uncertainty, and **not formal noninferiority**.
Denominators, missingness and detector disagreement are reported.

`DeltaS = S_CPR − S_control`, `DeltaR = R_CPR − R_control`.

**`PROMISING_DETECTOR_DEFINED_TRADEOFF`** requires, against **each of the five
comparators on both held-out objects**: point `DeltaS ≥ −0.05` and
`DeltaR ≥ 0.10`; lower bound `DeltaS ≥ −0.05` and lower bound `DeltaR > 0`;
every individual training-seed row `DeltaS ≥ −0.05` and `DeltaR > 0`; and a
corroborating direction on the qualified secondary/common-intersection
comparison meeting the same minimum eligibility counts. **An unqualified
secondary or material evaluator disagreement prevents a robust positive.**

**`NO_ADVANTAGE_IN_THIS_SCREEN`** requires, for the development-selected
primary comparator on both objects: point `DeltaS ≤ 0.05` **and** upper bound
`≤ 0.05`; point `DeltaR < 0.10` **and** upper bound `< 0.10`; every
individual-seed row meeting those point inequalities; and qualified
secondary/common-intersection corroboration. This rules out the chosen 10-point
retention advantage against that control **on this screen** — it is **not**
global equivalence or zero effect.

Otherwise **`INCONCLUSIVE`**. Access, qualification, budget and publication
failures retain factual blocker statuses. Decision predicates are executable
and frozen before outcomes. **No positive from collapse, missing arms, post-hoc
thresholds or favourable point estimates alone.**

---

## 7. Budget and enforcement

| Ceiling | Value |
|---|---|
| Elapsed | 7 days; science stops day 6 |
| GPU | 24 device-hours; **entry ≤ 4**; **8 reserved** for held-out eval + publication verification |
| Generated images | 2,200 (1,928 primary + 272 technical), counting previews, failures and repeat decodes |
| COCO photographs | 600 unique |
| CPU | 48 core-hours, 8 workers |
| RSS / storage / free space | 128 GiB / 200 GiB / ≥ 50 GiB free on `/data` |
| Repair / retry | **1** CPU implementation-repair cycle, **1** technical GPU retry across the whole project |

`cprd/ledger.py` implements a **parent-side supervisor** with a durable,
**fcntl-locked, interprocess-safe** allocation ledger: reservations are written
**before** launch, children are charged **through confirmed termination**
(success, non-zero exit, exception, OOM, timeout, kill), and `reconcile_all`
charges any still-open reservation at its **full reserved amount** so a failed
job can never vanish from accounting. Aggregate ceilings, the entry subcap and
the **protected evaluation reserve** are refused at reservation time, not
merely forecast. **Every post-allocation technical relaunch counts, including
compatibility fixes.**

Actual enforcement gaps are published verbatim from `cprd/ledger.gaps()` — GPU
charged as wall-clock × devices held; a dead parent leaves a reservation to be
over-charged at reconcile; unreaped CPU uncounted; RSS/workers advisory, not
cgroup-enforced; storage and free space polled, not quota-enforced; the 7-day
cap operator-enforced.

**No semantic retries, no outcome-driven adjustments, and no dropping arms or
cases to finish.**

---

## 8. Publication plan

New commits and **one new versioned release**. Publish: freezes and the
selection lock, source diffs, environment pins, runnable code, gate
checks/failures, case and noise manifests, training counters/logs, **both
detectors' raw boxes and scores**, annotation qualification, analysis tables,
missing work, and the full allocation/retry ledger. Release assets: the
**actual selected native PEFT adapters**, reconstruction metadata, and the
**generated primary PNGs** — not grids or hashes alone. Development snapshots
are preserved with hashes and reconstruction information.

Held-out images are generated from **saved-and-reloaded canonical endpoints**,
consistently across arms, verifying effective-weight restoration **without
assuming merged/unmerged BF16 bit equality**. Redistribution terms are checked
before production: **no pristine base weights and no credentials are ever
republished**; for COCO, image ids, annotation provenance and reproduction
instructions are published **instead of indiscriminately redistributing
photographs**. Published assets are downloaded into a fresh verification area
and their hashes, counts and adapter-state restoration verified within the same
caps, with limits disclosed. A **noncircular index** is built from committed
payload bytes (index-only child acceptable), and the live remote SHA, new tag
and downloads are verified. Only the new directory plus one new
`results/README.md` pointer are staged; closed tracked paths and existing
releases/tags are verified unchanged.
