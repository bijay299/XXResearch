# CPR detector-defined controls screen v1 — final report (amended 2026-10-19)

> ## Status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED` — attempt CLOSED
>
> **This is the corrected current report.** It replaces the report first published
> at `9940e94` (15,959 bytes, sha256
> `bfd55931b18a6f9d190f06d14540579964c86038852c249e900fdde28b6a239c`, recoverable
> from Git history). Corrections, and the reasoning for each, are in
> [`AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md`](AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md);
> preservation checks are in [`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md).
> Protected files (`code/**`, `configs/**`, `artifacts/**`, `RUN_MANIFEST.md`,
> `RECONSTRUCTION.md`) are byte-for-byte unchanged, so **their own flags and prose
> (for example `passed`/`QUALIFIED` fields and "stricter" in the repair README) are
> superseded by this report and the amendment where they conflict.**
>
> **No development training, no held-out training, no image generation and no
> six-arm comparison exist. Zero images were generated.** No scientific decision is
> available or claimed. **Whether CPR retains allowed-object detections better than
> native GEM or the simple controls remains untested.**

**Unanswered question:** at comparable reduction in target detections, does CPR retain
allowed-object detections better than native GEM and the strongest simple controls?

---

## 1. The blocker, exactly

Required gate: the native path with loss dispatch disabled and enabled must agree under
controlled inputs and RNG. The coordinator required a **declared numerical tolerance**.
The check as coded (`code/gate/bringup_gpu.py`) instead demanded **exact equality of the
trained native PEFT adapter state hash** after an identical 22-processed-timestep run
from an identical initialisation and seed.

| Run | Adapter state hash |
|---|---|
| dispatch **OFF** (no hook attribute) | `c05be7b17614d1866962c07408274ba4809b31d2f21ddefe548e0aff7c2641f4` |
| dispatch **ON**, native arm (hook installed, returns `None`) | `ed373817cf779b79d0f4b58a2b7e8e0f0a120da715ad056316752a227888140e` |
| dispatch **ON**, CPR arm | `88012e1a199319d9ef3fccd0545f262f8a147fb1a4d43da1bef20378ab61b98d` |

**The native OFF and ON hashes differ, so required agreement is not established.** It is
not converted into a tolerance pass.

### What is and is not known about the cause

**The cause is unresolved.** The earlier "most likely GPU non-determinism" is withdrawn
as a diagnosis; it is one hypothesis.

- Source reading supports that the patch's native-arm statements are the original
  three lines re-indented inside an `if`, with an added `getattr` and a call that
  returns `None`. That is a reading of the diff, not a run-level result.
- The pinned `Operator.__call__` calls `seed_everything`, which sets
  `cudnn.deterministic=True` and `benchmark=False`. A separate measurement of
  `cudnn.deterministic=False` therefore does **not** establish the training-time
  setting. Equally, that helper does not make execution completely deterministic
  (no `torch.use_deterministic_algorithms`, no `CUBLAS_WORKSPACE_CONFIG`).
- Candidate causes **not excluded**: GPU non-determinism, run-order or in-process state
  carry-over (OFF ran first on a fresh build; ON ran second after an adapter-parameter
  restore), or an unrecognised code-path or harness difference. No OFF/OFF run and no
  controlled numerical comparison exists. An OFF/OFF run would be informative but is
  not guaranteed to settle every cause. None was run.
- **A different CPR hash is not a causal hook-effect result**, because native ON already
  differs from native OFF. Hook installation is supported separately by source wiring
  and the CPU checks (`dispatch_native_returns_none`, `dispatch_candidate_returns_tensor`).

### Bring-up probes, and what the script did after the failure

The failed GPU job ran **three 22-processed-timestep training probes** (native OFF,
native ON, **CPR ON**) at 29.8 s (OFF) and 29.5 s (ON). `bringup_gpu.py` records each
condition and **keeps going after a failure**: after the native agreement check failed,
it still ran the finite-update check, the CPR probe and the budget projection, and
returned non-zero only at the end (elapsed 103.0 s). It was not an immediate hard stop.
No production comparison began and no further GPU job was launched. The earlier job
(`bringup`, 2.65 s) failed on a missing optional import and trained nothing.

### Public chronology of the criterion

| Event (Git author time, UTC) | Content |
|---|---|
| `3fba0ff` 05:24:37 | initial freeze: manifest, categories/cases, loss, detector, label and ledger code, patches P1–P3. **No gate code; no exact-hash or tolerance statement.** |
| `67d8b74` 05:27:52 | additive freeze: CPU gate and its repair record only |
| ledger journal ≈05:28:47–05:43:17 | the three recorded GPU jobs |
| `00bc8ae` 07:25:04 | **first public appearance** of the bring-up scripts, `qualify_detectors.py` and the exact-hash criterion |

The criterion existed in local code before the run, but **a detailed prospective public
native-gate freeze is unverified**; nothing is backdated and no hidden selection is
inferred. The exact-hash criterion also departed from the coordinator's requirement of a
declared numerical tolerance — a specification error in this study, not a property of GEM.

---

## 2. What was established, with its limits

None of this is a scientific finding about CPR.

### 2.1 Categories and the panel design — a disclosed deviation

Categories were chosen from COCO annotation metadata alone (stop sign 27 positives at
≥5% area, bench 121, fire hydrant 46; traffic light 15 and parking meter 18 fell under the
≥20 floor).

**The allowed panel for every target is `person`, `car`, `motorcycle`, `airplane`.** The
protocol required two same-supercategory non-target allowed classes plus two others.
All three targets are `outdoor` and no other outdoor class met the floor, so the panels
contain **zero** same-supercategory classes instead of two; the deficit was filled with
other classes in ascending COCO id. The fallback was disclosed prospectively, but it was
an **unauthorized design deviation**: the permitted fallback concerned
qualification-negative photographs, not the retention panel. The earlier claims that this
"follows the frozen rule" and that the categories are "reusable as-is" are **withdrawn**.
No replacement was searched, no eligibility recomputed, and no collision with closed
categories is inferred (candidate-class repository hits were incidental COCO caption text).

### 2.2 Evaluator qualification — corrected, not rescored

Frozen gate: for each class, **≥16/20 box-matched positives (IoU ≥ 0.50, correct class,
threshold 0.30) and ≥16/20 clean annotation-negative cases**. The source evaluated the
image-level detection count instead of the box-matched count; the published `passed` and
`QUALIFIED` flags are superseded.

| Class | RT-DETR: detected / **box-matched** / clean neg | DETR: detected / **box-matched** / clean neg |
|---|---|---|
| stop sign | 20 / **20** / 20 | 20 / **20** / 20 |
| bench | 17 / **16** / 19 | 17 / **15** / 19 |
| fire hydrant | 20 / **20** / 20 | 18 / **18** / 20 |
| person | 20 / **20** / 19 | 20 / **20** / 19 |
| car | 20 / **20** / 19 | 20 / **20** / 19 |
| motorcycle | 20 / **20** / 20 | 20 / **20** / 20 |
| airplane | 20 / **20** / 20 | 19 / **19** / 20 |

- **Primary RT-DETR passes all seven classes** (`bench` at exactly 16/20 and 19/20).
- **Secondary DETR fails `bench`** (15/20 matched; 17/20 detected). Its other six pass.
- **"Both evaluators qualified, 7/7" is withdrawn.**
- The secondary failure did not itself prohibit primary training, but it prevents the
  specified corroborated-positive conclusion on `bench`, a held-out target.
- Preserved: 280 records per detector, 264 unique photographs, 0 missing; all seven
  classes have different label ids across the detectors (RT-DETR 0-based contiguous,
  DETR COCO ids: person 0/1, car 2/3, motorcycle 3/4, airplane 4/5, fire hydrant 10/11,
  stop sign 11/13, bench 13/15).
- The published "raw" detection files hold **derived** scores, IoUs, counts and flags,
  **not** full boxes, labels and scores (descriptions saying otherwise are superseded).
  Counts can be re-aggregated; matching cannot be independently reconstructed.
- Limits stand: both detectors were developed on COCO and tested on COCO photographs;
  annotation-negative is an imperfect absence reference; nothing here shows accuracy on
  edited generations; detector agreement is not ground truth; scores are not calibrated
  probabilities.

### 2.3 CPU implementation gate — 64 rows, 57 pass, 0 fail, 7 N/A

N/A rows are **not** passes (seven native-arm or hinge-inactive rows). Four
exact-zero-hinge boundary fixtures (`axial_margin`, `cpr`, `centered_radial`,
`truncated_anchor`) and the formula, gradient and detach checks are **partial** evidence;
the native checks were deferred to the GPU gate, which failed.

One repair was recorded. `G4.native_scalar_order` had failed (preserved). The repair
replaced an FP32-versus-FP64 oracle at relative tolerance 1e-9 with a **same-dtype bitwise
equality of a duplicate scalar-association computation**. The predicate changed; no loss
implementation changed. The earlier claims "strictly stronger", blanket unchanged-tolerance
compliance, and that 1e-9 is universally impossible for FP32 are **withdrawn**. As
disclosed before, retesting preceded the additive freeze commit.

The axial N/A note ("its inactive point is covered by `G2.boundary`") is corrected: a
boundary is not an inactive point; the separate near-anchor gradient row did check an
all-inactive batch.

### 2.4 Native GEM + native PEFT — genuine progress, with limits

Pinned GEM trainer and native PEFT execute; FLUX at the pinned revision; `qk_dual` is
exactly 76 modules with no single-block leak and 7,471,104 trainable parameters; base
frozen; backbone BF16 with an FP32 adapter; native student sampling occurs with
`transformer.training=True`, `torch.is_grad_enabled()=False` and dropout 0.1 (`no_grad` is
not `eval`; preserved, not "fixed"); the update through the native loop was finite.
Peak GPU memory was 32.34 GiB with gradient checkpointing enabled (disclosed, **not proven
numerically inert**).

The 152/152-tensor reload of the adapter-initialisation asset supports **adapter-parameter
restoration only** — not effective-model parity, forward agreement, merged/unmerged BF16
equivalence or erasure efficacy.

### 2.5 Budget projection — provisional only

8.16 GPU-h = (3.37 training + 2.68 generation + 0.09 detectors + 0.14 loads) × 1.30,
against 16 usable non-reserve hours. Training was briefly profiled (1.3499 s per processed
timestep); **generation at 5 s/image and detector throughput at 0.08 s/image were
assumed**, not measured. This is a provisional projection, not evidence that a workflow
is ready or that the frozen components are reusable as-is.

---

## 3. Resource accounting — corrected

| Item | Value | Qualification |
|---|---|---|
| GPU device-seconds charged | **167.8786015510559 s = 0.0466329449 GPU-h**, one device (GPU 2) | `qualify_detectors` 58.685038566589355 s, `bringup` (import failure) 2.6522905826568604 s, `bringup_gpu` 106.54127240180969 s. Failures included; each below its reservation; 0 open or unreconciled reservations |
| Entry sub-cap / total / evaluation reserve | ≤4 h / 24 h / 8 h | No numeric overrun evidenced; evaluation reserve unused (0 charged) |
| Generated images | 0 | |
| COCO photographs | 264 | |
| Retries | 1 CPU repair, 1 technical GPU retry | |
| Elapsed | about 2.3–2.4 h | The start is from an **unpublished** local `preflight.log` line (05:03:19Z); first-action and time-enforcement evidence is missing. From the first published commit it is 2 h 04 m. Within the one-week cap; no overrun claimed |
| CPU | 207.186047 CPU-s = 0.05755168 core-h | **Only the subtotal of the three supervised GPU children.** Whole-task CPU (setup, gates, downloads, selection, publication) is unknown where unrecorded; not a figure against the 48 core-h ceiling |
| Workers | not measured | the earlier "8" was the ceiling |
| Storage | **no measured peak** | "~0.3 GiB" summed retained artifacts; the manifest discloses a 5.5 GiB dependency attempt that was deleted and redone; sequential footprints are not added and no peak is claimed |

**Supervisor limitations (documented, not repaired).** `settle()` clips an overrunning
charge to its reservation; the exception path kills without a confirmed reap before
settling; `reserve()` enforces only the GPU aggregate, entry sub-cap and protected
reserve; the published source has no RSS/worker watchdog and no disk polling. These do
not invalidate the three observed below-reservation charges and establish no numeric
CPU or storage breach, but blanket complete-enforcement assurances are withdrawn. See
[`ENFORCEMENT_GAPS.md`](ENFORCEMENT_GAPS.md).

---

## 4. Work status

| Planned | Status |
|---|---|
| Category selection, qualification case construction | done, with the §2.1 panel deviation |
| Evaluator qualification | **primary passes 7/7; secondary fails `bench`** (§2.2) |
| CPU implementation gate | 64 rows, 57 pass, 0 fail, 7 N/A; partial evidence (§2.3) |
| Native GEM + PEFT CPU bring-up | ran; see §2.4 limits |
| Native dispatch OFF/ON agreement | **not established — the blocker** (§1) |
| Development trajectories, held-out endpoints, selection, comparison, decision | **not run** |

All six arms, both held-out objects, both seeds and every case remain exactly as frozen in
`configs/cases.json`; nothing was run.

## 5. Limitations

1. The native dispatch agreement is unresolved; neither a non-inert patch nor
   non-determinism is established.
2. No evidence about CPR was produced: bring-up probes only, no generation, no comparison.
3. Qualification is in-domain, and `bench` is marginal for the primary and below the gate
   for the secondary.
4. The retention panels contain no same-supercategory class.
5. The accounting and enforcement limits of §3.
6. One device was used; GPUs 0 and 1 belonged to another user and were left alone.

## 6. Closeout

**This attempt is CLOSED** with `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`.
Implementation, evaluator and watchdog limitations are documented and **left unfixed**.
No follow-up study is selected, scheduled or started, and this report makes no rerun,
scorer-search, replacement-category or retry recommendation.

## 7. Historical commands

The original reproduction commands are preserved in the protected `code/` tree and in
the report at `9940e94`; they are not re-executed or endorsed here. Running the last one
(`bringup_gpu.py`) produced the recorded blocker once.

---

### Attribution and licensing (unchanged)

Native GEM trainer, wrapper and PEFT adaptation execute from a **fresh checkout** of
[`GEM@c328fc6`](https://github.com/multimodal-ai-lab/GEM/tree/c328fc6bec687efe60223498d1b6bed2e5fbcffd),
modified only by the attributed patches in `code/patches/`. GEM is third-party work by
its authors and is **not vendored**. Backbone `black-forest-labs/FLUX.1-dev@3de623fc` under
the FLUX.1 [dev] **Non-Commercial License**; no base weights are redistributed. Detectors
`PekingU/rtdetr_r50vd@df939e66` and `facebook/detr-resnet-50@70120ba8` (Apache-2.0). COCO
image ids and provenance are published instead of photographs; `instances_val2017.json`
sha256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. No credentials
are published.
