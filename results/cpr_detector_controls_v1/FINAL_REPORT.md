# CPR detector-defined controls screen v1 — final report

> ## Terminal status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`
>
> A **required** entry-gate correctness condition did not resolve. Per the
> contract ("Unresolved required correctness ends production qualification";
> "On failure, publish the exact blocker without replacements"), **production
> did not start**.
>
> **No development training, no held-out training, no image generation and no
> six-arm comparison exist. Zero images were generated.** No scientific
> decision — `PROMISING_DETECTOR_DEFINED_TRADEOFF`, `NO_ADVANTAGE_IN_THIS_SCREEN`
> or `INCONCLUSIVE` — is available or claimed. **CPR's detector-defined
> retention behaviour remains untested.**

**The question that remains unanswered:** at comparable reduction in target
detections, does CPR retain allowed-object detections better than native GEM
and the strongest simple controls?

---

## 1. The blocker, exactly

The gate: *"The native path with loss dispatch disabled/enabled must agree
under controlled inputs/RNG within a declared tolerance."*

The **declared criterion, frozen in the gate code before the run**, was
**exact equality of the trained native PEFT adapter state hash** after an
identical 22-processed-timestep run from an identical initialisation and seed.

| Run | Adapter state hash |
|---|---|
| dispatch **OFF** (no hook attribute; original code path) | `c05be7b17614d1866962c07408274ba4809b31d2f21ddefe548e0aff7c2641f4` |
| dispatch **ON**, native arm (hook installed, returns `None`) | `ed373817cf779b79d0f4b58a2b7e8e0f0a120da715ad056316752a227888140e` |

**The hashes differ, so the gate FAILED.**

### What the evidence does and does not say

**Code-level argument that the patch is inert** (from the preserved diff,
`code/patches/P1_P3_erasure_gem.diff`): when the hook returns `None`, the
statements that execute are **byte-identical** to the original three lines —

```
dist_pos  = torch.nn.functional.pairwise_distance(pred_flat, pos_flat, p=2)
dist_neg  = torch.nn.functional.pairwise_distance(pred_flat, neg_flat, p=2)
target_loss = torch.relu(dist_pos - config.eta * dist_neg).sum()
```

— re-indented inside an `if`. The only additions on that path are a `getattr`
and one call returning `None`. **No tensor arithmetic differs.**

**Why the run can still differ anyway:** deterministic algorithms were **not**
enabled. Measured in this environment:
`torch.are_deterministic_algorithms_enabled() = False`,
`torch.backends.cudnn.deterministic = False`, `CUBLAS_WORKSPACE_CONFIG` unset.
With non-deterministic attention backward and atomics, plus gradient
checkpointing, two runs of the *same* configuration are not expected to produce
bit-identical weights after 22 optimizer-accumulating timesteps.

**So the most likely explanation is run-to-run GPU non-determinism, not a
non-inert patch — but that is a hypothesis, not a result.** Distinguishing it
requires an OFF-versus-OFF control run, which was **not** performed. Being
"very likely fine" is not the verified agreement the gate required, and this
report does not treat it as one.

### Why it was not re-run

Two reasons, both binding:

1. **The retry allowance is exhausted.** The contract permits **one** technical
   GPU retry across the entire project. The first GPU bring-up job failed on a
   missing optional import; relaunching it consumed that single retry. An
   OFF-versus-OFF diagnostic would be a further GPU launch beyond the cap.
2. **Loosening the criterion after seeing the outcome is forbidden.** The
   contract requires a *declared* tolerance and bars "semantic retries,
   outcome-driven adjustments". The declared criterion was exact equality;
   replacing it with a numerical tolerance now, because exact equality failed,
   would be exactly the outcome-driven adjustment the contract prohibits.

**This is a specification defect on my part.** The contract warned "Do not call
tolerance agreement bit-exact," and I nevertheless froze a bit-exact criterion
for a harness I had not made deterministic. A correctly specified gate would
have declared a numerical tolerance on the adapter delta, or enabled
deterministic algorithms, *before* the run. It did neither.

---

## 2. What passed — all of it before the blocker

These are real results and they are reusable. None of them is a scientific
finding about CPR.

### 2.1 Categories, chosen by metadata alone

Walking the fixed candidate list in order, after excluding actually-executed
closed categories:

| Candidate | COCO id | positives ≥5% | verdict |
|---|---|---|---|
| traffic light | 10 | **15** | ineligible (<20) |
| **stop sign** | 13 | 27 | **development target** |
| parking meter | 14 | **18** | ineligible (<20) |
| **bench** | 15 | 121 | **held-out target 1** |
| **fire hydrant** | 11 | 46 | **held-out target 2** |

**Allowed panel (identical for all three): `person`, `car`, `motorcycle`,
`airplane`.** 7 scored classes, 280 image/class pairs, **264 unique
photographs**.

**Disclosed deficit:** all three targets are supercategory `outdoor`, and the
only other outdoor classes fail the ≥20-positive floor, so the allowed panels
contain **no same-supercategory confounder class**; the deficit of 2 was filled
from other classes in ascending COCO id. This weakens the "related but
non-target" aspect of the retention test and follows deterministically from the
frozen rule. Also disclosed: `fire hydrant` and `bench` appear in the v1
pilot's **declared but never executed** panel.

### 2.2 Both evaluators QUALIFIED

Threshold 0.30, IoU 0.50, gate ≥16/20 on positives and negatives, every class:

| Class | RT-DETR pos / box / neg | DETR pos / box / neg |
|---|---|---|
| stop sign | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| bench | **17/20** · 16/20 · 19/20 | **17/20** · **15/20** · 19/20 |
| fire hydrant | 20/20 · 20/20 · 20/20 | 18/20 · 18/20 · 20/20 |
| person | 20/20 · 20/20 · 19/20 | 20/20 · 20/20 · 19/20 |
| car | 20/20 · 20/20 · 19/20 | 20/20 · 20/20 · 19/20 |
| motorcycle | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| airplane | 20/20 · 20/20 · 20/20 | 19/20 · 19/20 · 20/20 |

**Primary `QUALIFIED`. Secondary `QUALIFIED`.** `bench` is the weakest class on
both detectors and would have been the one to watch.

**The two label spaces differ for all seven classes** — exactly the trap the
contract flags. RT-DETR uses 80 contiguous ids with VOC-style
`motorbike`/`aeroplane`; DETR uses COCO ids 1–90 with 11 unused `N/A` slots:

| Class | RT-DETR id | DETR id |
|---|---|---|
| person | 0 | 1 | 
| car | 2 | 3 |
| motorcycle | 3 | 4 |
| airplane | 4 | 5 |
| fire hydrant | 10 | 11 |
| stop sign | 11 | 13 |
| bench | 13 | 15 |

**Stated limits.** Both detectors were developed and evaluated on COCO, so this
is a benchmark/implementation check, **not independent out-of-domain
validation**. Annotation-negative is an **imperfect absence reference**.
Real-photo qualification **does not establish accuracy on edited generations**.
**Detector agreement is not ground truth.** Scores are not calibrated presence
probabilities and the two scales are not comparable.

### 2.3 Implementation gate — 64 rows, 57 pass, 0 fail, 7 N/A

N/A rows are **not** passes; shared checks are labelled shared. Per-arm
independent values and applicable gradients against an independent FP64
reference; active and inactive behaviour with exactly-zero value and gradient;
teacher detachment; FP32 preparation; native scalar order; CPR **retaining
attraction** at degenerate contrast; and the axial domain violation raising an
**explicit arm failure**.

**Exact-boundary fixtures** were constructed where the **computed hinge
argument is exactly 0.0** — for `axial_margin`, `cpr`, `centered_radial` and
`truncated_anchor`. A tolerance-close inactive point was never accepted as
boundary evidence.

### 2.4 Native GEM + native PEFT

The pinned trainer and wrapper execute; adaptation is **native PEFT**
(`get_peft_model` / `LoraConfig`), and no LoRA is reimplemented.

- FLUX pinned revision `3de623fc…` resolved from the pristine read-only cache.
- `qk_dual` → **exactly 76 modules**, **0 single-block leak**, endswith
  `{add_k_proj, add_q_proj, to_k, to_q}`, **7,471,104** trainable parameters.
- **Base frozen** (0 trainable base parameters); backbone all BF16; LoRA A
  non-zero, B zero, adapter FP32 (native `cast_training_params`).
- Adapter save/reload restores state with a perturbation control; 28.55 MiB.
  **Merged/unmerged BF16 bit equality is not assumed or claimed.**
- **Mode during student sampling verified:** `transformer.training = True`,
  `torch.is_grad_enabled() = False`, `dropout p = 0.1`. **`no_grad` is not
  `eval`**, so native LoRA dropout is active during sampling — preserved, not
  "fixed".
- Finite intended updates through the real native loop (max |Δ| 2.0e-3); peak
  32.34 GiB with gradient checkpointing enabled.
- The CPR dispatch produces a **different** trained adapter than the native
  block under identical seeds, confirming the hook is live.

### 2.5 Budget projection — fits, and was never the blocker

1.350 s per processed timestep ⇒ training 3.37 + generation 2.68 + detectors
0.09 + loads 0.14 GPU-hours, ×1.30 margin = **8.16 projected GPU-hours**
against 16 usable non-reserve hours. A projection **does not replace
enforcement**; the ledger refuses any reservation that would breach a cap.

---

## 3. Resource usage against every ceiling

| Ceiling | Limit | Actual |
|---|---|---|
| Elapsed | 7 days (science by day 6) | **~2.3 h** |
| GPU device-hours | 24 | **0.0466** (167.88 s) |
| — entry/bring-up subcap | ≤ 4 | **0.0466** |
| — protected evaluation reserve | 8 | **0.0 used** |
| Generated images | 2,200 | **0** |
| COCO photographs | 600 | **264** |
| CPU core-hours | 48 | **0.0576** (207.19 s) |
| Workers | 8 | 8 |
| Storage | 200 GiB | ~0.3 GiB (annotations, 264 photos, 2 adapters, overlay) |
| GPUs | ≤ 4 genuinely available | **1** (GPU 2; 0 and 1 held by another user) |
| **CPU repair cycles** | **1** | **1 used** |
| **Technical GPU retries** | **1** | **1 used** |

**Failed jobs did not vanish from accounting.** All three supervised GPU jobs
are charged, two of them failures: `qualify_detectors` ok 58.7 s,
`bringup` exit_1 2.7 s, `bringup_gpu` exit_1 106.5 s — summing to the
167.88 s charged total. **0 unreconciled reservations.**

Enforcement gaps are published verbatim from `cprd/ledger.gaps()` in
`artifacts/allocation_ledger.json` and `ENFORCEMENT_GAPS.md`.

---

## 4. Complete and missing work

| Planned | Status |
|---|---|
| Category selection, qualification case construction | **COMPLETE** |
| Evaluator qualification, both detectors | **COMPLETE — both QUALIFIED** |
| CPU implementation gate | **COMPLETE — 0 failures** |
| Native GEM + PEFT CPU bring-up | **COMPLETE — 0 failures** |
| Native dispatch OFF/ON agreement | **FAILED — the blocker** |
| 12 development trajectories, 888 images | **NOT RUN** |
| 24 held-out endpoints, 1,040 images | **NOT RUN** |
| Development selection, selection-lock commit | **NOT REACHED** |
| Held-out comparison, bootstrap, terminal decision | **NOT REACHED** |

Nothing was quietly dropped: all six arms, both held-out objects, both training
seeds and every case remain exactly as frozen in `configs/cases.json`.

---

## 5. Limitations

1. **The blocker is a gate-specification defect, not a demonstrated code
   defect.** The patch is inert by code inspection; the failure is most likely
   non-determinism. Neither claim was verified by the control run that would
   settle it.
2. **No evidence about CPR.** No training beyond 22-timestep bring-up probes,
   no generation, no comparison.
3. **Qualification is in-domain.** COCO-trained detectors on COCO photographs;
   it says nothing about accuracy on edited FLUX generations, which is the
   regime the study would actually have measured in.
4. **The allowed panels have no same-supercategory confounder** (§2.1), so even
   had the study run, the retention test would have been weaker than intended.
5. **`bench` is the weakest qualified class** (17/20 positives on both
   detectors) and is one of the two held-out targets.
6. **GPU cost is charged as wall-clock × devices held**, which includes
   allocated idle time and cannot see another user's processes.
7. **One device, not four.** GPUs 0 and 1 were in use by another user
   throughout and were left alone.

---

## 6. One terminal recommendation

**If this screen is ever re-attempted, the native-path agreement gate must be
specified correctly before anything runs:** either enable deterministic
algorithms (`torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG`) so exact equality is attainable, **or** declare a
numerical tolerance on the adapter delta **and** include an OFF-versus-OFF
control run so that harness non-determinism is measured rather than inferred.
Without that control, the check cannot distinguish a non-inert patch from
ordinary floating-point noise — which is precisely the position this report is
in.

Everything else in this directory is reusable as-is: the frozen categories, the
qualified evaluators, the label mapping, the implementation gate and the
supervisor/ledger.

**Not authorized and not done here:** no scorer or detector substitution, no
threshold change, no matrix expansion, no production training, no further
retry, and no automatic replication.

---

## 7. Reproduction

```bash
source results/cpr_detector_controls_v1/code/runenv.sh
$CPRD_PY code/scripts/select_categories.py     # annotation-only, no model
$CPRD_PY code/scripts/build_cases.py           # frozen matrix
$CPRD_PY code/gate/cpu_gate.py                 # CPU, ~2 min
$CPRD_PY code/gate/bringup_cpu.py              # CPU, no GPU allocation
CUDA_VISIBLE_DEVICES=2 $CPRD_PY code/gate/qualify_detectors.py
CUDA_VISIBLE_DEVICES=2 $CPRD_PY code/gate/bringup_gpu.py   # reproduces the blocker
```

---

## 8. Publication and release verification

Release **`cpr-detector-controls-v1-entrygate`** (id 417012770), target commit
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f`, six assets.

All assets were **downloaded back from GitHub into a fresh verification area**
and verified: `sha256sum -c` **OK on every asset**; both detectors' raw
qualification files contain **280 rows over 7 classes** each; and the native
PEFT adapter was **restored from the downloaded bytes** on a fresh FLUX build —
**152/152 tensors loaded, recomputed hash `6ae4afdb…` matching the stored
hash**, all finite.

**Disclosed verification limits.** This verifies **effective-weight
restoration** only; **merged/unmerged BF16 bit equality is not asserted**.
There is **no production adapter and there are no generated images** to verify,
because production never started. The download used the same host and token as
the upload. COCO photographs are deliberately **not** redistributed, so their
"counts" are verified as manifest entries (264 unique ids), not as published
image bytes.

---

### Attribution and licensing

Native GEM trainer, wrapper and PEFT adaptation execute from a **fresh
checkout** of
[`GEM@c328fc6`](https://github.com/multimodal-ai-lab/GEM/tree/c328fc6bec687efe60223498d1b6bed2e5fbcffd),
modified only by the attributed patches in `code/patches/` (exact diffs). GEM
is third-party work by its authors and is **not vendored** into this
repository. Backbone `black-forest-labs/FLUX.1-dev@3de623fc` under the FLUX.1
[dev] **Non-Commercial License** — **no base weights are redistributed**.
Detectors `PekingU/rtdetr_r50vd@df939e66` (Apache-2.0) and
`facebook/detr-resnet-50@70120ba8` (Apache-2.0). For COCO, **image ids,
annotation provenance and reproduction instructions are published instead of
redistributing photographs**; `instances_val2017.json` sha256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.
No credentials are published.
