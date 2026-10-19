# CPR real-model controls pilot v1 — final report

> ## Terminal status: `ENTRY_GATE_FAILED — SCORER_NOT_QUALIFIED`
>
> The prespecified **measurement-qualification gate failed**. Under the
> contract ("Failed qualification ends the task; no scorer shopping or
> threshold retuning after inspecting failures") scientific work stopped at the
> entry gate.
>
> **No production training ran. No development matrix, no held-out matrix, no
> arm comparison exists.** Therefore **no** scientific decision —
> `PROMISING_FOR_FRESH_REPLICATION`, `NO_ADVANTAGE_DEMONSTRATED` or
> `INCONCLUSIVE` — is available or claimed. The pilot's question is **untouched**.

**The question it would have answered (unanswered):** at comparable actual
target suppression, does CPR preserve allowed content better than native GEM
and the strongest simple controls?

---

## 1. What happened

The entry gate has four parts. Three passed; the fourth failed.

| Gate | Result |
|---|---|
| (a) Model-free correctness of all six arms | **PASS** — 78 cases, 76 pass, 0 fail, 2 recorded `INAPPLICABLE` |
| (b) Bring-up, native training, compact save/reload | **PASS** — adapter reload bit-identical, 76/76 `qk_dual` modules |
| (c) **Measurement qualification** | **FAIL** — specificity 0.6875 < 0.80 |
| (d) Budget projection | **PASS** — 8.41 projected GPU-hours of 24 ("FITS") |

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

### Where the failure is concentrated — and why it is fatal here

| Probe object | Sensitivity | Specificity |
|---|---|---|
| **backpack** | 16/16 = 1.000 | **3/11 = 0.273** |
| bicycle | 8/8 = 1.000 | 10/11 = 0.909 |
| violin | 8/8 = 1.000 | 9/10 = 0.900 |

All ten errors are **false positives**, and **eight of them are the `backpack`
probe**. Six fired above `p_present = 0.60` on plainly unrelated scenes — a
potted plant (0.715), a forest path (0.832), a desk with a lamp (0.806), a
stack of books (0.624), a beach (0.610), an open notebook (0.726) — and four
more landed in the uncertain band, which the frozen rule counts as errors
rather than resolving in the scorer's favour.

**`backpack` is the development target.** The development stage is where all
six arms' learning rates and checkpoints get selected, and where the
≥ 80%-suppression efficacy floor is applied. An instrument that calls a potted
plant a backpack 27% of the time cannot measure backpack suppression, so the
entire selection stage — and therefore everything downstream of it — was
unmeasurable. Bicycle and violin would have cleared the gate; that does not
rescue the pilot, because the frozen design selects on backpack first.

This is the warning in the contract — *"Do not assume CLIP similarity is
calibrated object presence"* — materialising concretely. The decision rule
contrasts present-phrases against explicit absent-phrases
("a photo with no backpack"), and CLIP's weak handling of negation means the
absent-phrase still scores highly on any image, so the contrast carries little
information for this object. Qualifying the instrument against held-apart
images is exactly what exposed it, *before* it could launder noise into
suppression and preservation percentages.

### What was deliberately not done

- **No scorer shopping.** LLaVA-1.6-vicuna-7b and CLIP ViT-L/14-336 are both
  cached locally and were not tried.
- **No threshold retuning.** Moving `threshold_present` to ~0.85 would have
  passed the gate on this data; it was not done. The thresholds were frozen in
  `cprlib/scorer.py` and hashed into `artifacts/freeze_seal.json`
  (`8177b47cb5b71f8b…`) before any calibration image existed.
- **No prompt or template edits**, no re-generation of calibration cases, no
  dropping of the backpack probe.
- **No production run.** Zero training trajectories were executed.

---

## 2. What the passing gates did establish

These stand on their own and are reusable, but they are **engineering
qualification, not scientific results**.

### (a) All six arms are correct — 78 cases, 0 failures

`artifacts/cpu_gate_results.json`. Against an **independent FP64 oracle**
written from the contract's formulas rather than by calling the production
code: forward parity (≤ 1e-5), autograd vs independent analytic gradients, FP64
central differences away from hinges (≤ 1e-7), active/inactive/exact-boundary
behaviour, reduction and scaling order, the independent divisor
(`n_latents = 11 ≠ batch`), teacher detachment, FP32 promotion from FP64 and
BF16, finite gradients and safe small-contrast handling, explicit axial-domain
handling, and rejection of malformed shapes, nonfinite inputs and invalid
divisors. The exact-boundary case asserts **exactly zero loss and exactly zero
gradient**. Two cases are recorded `INAPPLICABLE` and are **not** counted as
passes.

### (b) The native GEM procedure reproduces and serializes correctly

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

### (c) Measured performance, for whoever runs this next

1.384 s per processed timestep · 4.79 s per decoded 512×512 image · 12.7 s
pipeline load ⇒ the full 36-trajectory / 1,992-image matrix projects to
**8.41 GPU-hours** including a 30% margin, against a 24-hour ceiling. **The
study fits comfortably.** The blocker is measurement, not compute.

---

## 3. Resource usage against every ceiling

| Ceiling | Limit | Actual | Fraction |
|---|---|---|---|
| Elapsed | 7 days (science by day 6) | **~1.0 h** | 0.6% |
| GPU device-hours | 24 | **0.1161** | 0.5% |
| — of which entry/bring-up | ≤ 4 | **0.1161** | 2.9% |
| — evaluation reserve | 8 held back | **0.0 used** | reserve intact |
| Newly decoded images | 2,200 | **68** | 3.1% |
| CPU core-hours | 48 | **0.123** (441.8 core-s) | 0.26% |
| Workers | 8 | 8 | — |
| Process-tree RSS | 128 GiB | well under | — |
| New project storage | 200 GiB | **~56 MiB** | <0.1% |
| Free space on `/data` | ≥ 50 GiB | ~518 GiB | ok |
| GPUs | ≤ 4 genuinely available | **2 only** (GPUs 2,3; 0 and 1 left to another user) | — |
| Implementation-repair cycle | 1 | **0 used** | — |
| Technical GPU retry | 1 | **0 used** | — |

68 decoded images = 64 calibration + 1 declared M0 dev case + 3 re-decodes
during failed bring-up attempts. 65 PNG files exist on disk; the counter
charges all 68 decodes, failures included.

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
2. **The gate used 64 images and three probe objects.** Per-object
   specificities rest on 10–11 negatives each, so those are coarse estimates.
   The aggregate 0.6875 against a 0.80 threshold is what the frozen rule
   decides on.
3. **This result is about one decision rule, not about CLIP generally.** A
   differently-constructed CLIP classifier, a detector, or a VLM judge might
   qualify. None was tried, by design.
4. **Nothing here speaks to CPR.** No arm was trained. The closed CPU prototype
   (`results/cpr_gem_cpu_prototype/`, status `CPU_PROTOTYPE_INCOMPLETE`) remains
   the only CPR evidence, and it is explicitly not a practical result.
5. **The `bicycle` partial overlap** with closed SD-1.5 study `seq29` (where it
   was a retention panel class, never an erasure target) is disclosed in
   `RUN_MANIFEST.md` §4 and judged not a genuine collision. A reader may
   disagree; no closed artefact was reused.
6. **GPU-second accounting charges wall-clock × devices held**, which
   over-counts idle time inside a job and cannot see other users' processes.

---

## 6. One terminal recommendation

**Qualify an object-presence instrument as a standalone, bounded task before
re-attempting this pilot.** It is cheap — the 64 calibration images already
exist and are published, so candidate instruments can be scored against the
already-recorded visual judgments at **zero GPU cost for generation**. The
specific lesson is that a present-vs-absent *phrase contrast* is the weak
component: CLIP barely represents negation, so an instrument should instead be
built on a discriminative signal (an open-vocabulary detector, or a VLM asked a
direct yes/no question), and it must be qualified **per target object**, since
`backpack` failed at 0.273 while `bicycle` and `violin` cleared 0.90.

That task should also replace the model-assisted reference with human labels on
at least the definite cases, because the present gate compares model to model.

**Not authorized and not started here:** no new probe, no scorer substitution,
no threshold change, no production training, no expansion of concepts or seeds,
and no follow-up correction cycle.

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

### Attribution

Native GEM arithmetic and training procedure are **re-implemented by reading**
[`GEM@c328fc6`](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/gem/operators/erasure/erasure_gem.py);
GEM is third-party work by its authors and **no GEM module is imported**.
Backbone `black-forest-labs/FLUX.1-dev@3de623fc` under the FLUX.1 [dev]
Non-Commercial License — **no model weights are redistributed**; published
assets are generated images and a ~28 MiB LoRA adapter trained here, for
non-commercial research use.
