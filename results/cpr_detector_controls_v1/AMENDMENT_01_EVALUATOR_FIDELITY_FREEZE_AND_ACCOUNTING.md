# AMENDMENT 01 — evaluator fidelity, public-freeze interpretation and accounting

**Issued:** 2026-10-19 (UTC). **Documentation-only.** No code was repaired, no test
ran, no model was loaded, no GPU was allocated, no detector was run, no image was
generated, scored or labelled, and no closed study was executed or reopened.

**This amendment supersedes contradictory statements wherever they appear** —
including in the protected `RUN_MANIFEST.md`, `RECONSTRUCTION.md`,
`code/**`, `configs/**` and `artifacts/**` (for example the `passed` flags in
`artifacts/qualification.json`, the comments and docstrings in the code, and
`artifacts/repair_01/README.md`), none of which was edited. The original
statements remain recoverable in Git at `3fba0ff3e4eaba65f1f6466249e4dfb939a29a72`
(prospective freeze), `67d8b74d4d571d1641bd840a9adc5f7b4139ca35` (additive freeze),
`00bc8ae0a3d98640cb49cbbe616d2101e827e64f` (payload; release/tag target),
`42614565cd79f911c194d9f7c8d1dea017fbe17c` (verification payload) and
`9940e9403452e2b382c108c2d4d50bd3d054d707` (reviewed HEAD).

**This attempt is CLOSED with its unresolved entry status.** Nothing here selects,
schedules or starts a follow-up study.

Mechanical backing for every number below is in
[`amendment_01_records/mechanical_checks.json`](amendment_01_records/mechanical_checks.json)
(produced by [`mechanical_checks.py`](amendment_01_records/mechanical_checks.py), which parses
records stored at the reviewed HEAD and imports only the standard library).

---

## 1. Status — retained unchanged

> ## `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`

No development selection, no held-out evaluation, no generated image and no
six-arm CPR-versus-control comparison exists. **Zero generated images remains
correct.** Consequently there is **no** improvement finding, **no** equivalence
finding, **no** harm finding, **no** method `NO_GO` and **no** novelty conclusion.
**Whether CPR retains allowed-object detections better than native GEM or the simple
controls is untested.**

### 1.1 "No production comparison" is not "no training whatsoever"

The GPU bring-up job (`bringup_gpu`, ledger pid 1796357, 106.54 device-seconds) ran
**three short training probes of 22 processed timesteps each** (`PROFILE_BUDGET`) —
native dispatch OFF, native dispatch ON, and **CPR ON** — on the real FLUX
backbone with the native PEFT adapter (OFF 29.8 s, ON 29.5 s; the CPR probe's
duration was not recorded separately). The probes were bring-up checks, not
experiments. The number of optimizer updates they performed was not logged and is
not asserted here.

### 1.2 The script did not stop at the failure

`code/gate/bringup_gpu.py` records each condition through `chk()`, which appends a
failure and **continues**; it returns non-zero only at the very end. The native
OFF/ON agreement check failed, yet the script **went on** to the finite-update check,
the **CPR probe**, and the budget projection, and only then exited `1`
(`bringup_gpu.json`: `elapsed_s` 103.0; `failures` =
`["dispatch_off_on_agreement_native_arm"]`; `hash_cpr` present). Any reading that the failure stopped the bring-up
immediately is withdrawn: **it was not an immediate hard stop.** No production comparison began, and no further GPU job was
launched after the failed check. The earlier GPU job (`bringup`, 2.65 s, exit 1)
failed on an import and trained nothing.

---

## 2. Evaluator qualification — corrected without rescoring

**Frozen gate (unchanged):** for every scored class, **≥ 16/20 positives with a
correct-class box at IoU ≥ 0.50** *and* **≥ 16/20 clean annotation-negative
cases**; detector threshold **0.30**. Nothing below changes a threshold, relabels
an image or re-runs inference; the counts are re-aggregated from the published
per-image records and were checked equal to `qualification.json`.

**The source defect.** `code/gate/qualify_detectors.py:112` evaluates
`passed = pos_hit >= MIN_HIT and neg_clean >= MIN_HIT`, where `pos_hit` is the
**image-level detection** count (a detection of the class anywhere in the image).
The frozen gate is about **box-matched** positives, `pos_boxmatch`. The code and
the JSON `passed` / `verdict` flags are preserved byte-for-byte; **their
interpretation is superseded here.**

| Class | RT-DETR image-level / **box-matched** / neg clean | DETR image-level / **box-matched** / neg clean | RT-DETR frozen gate | DETR frozen gate |
|---|---|---|---|---|
| stop sign | 20 / **20** / 20 | 20 / **20** / 20 | pass | pass |
| bench | 17 / **16** / 19 | 17 / **15** / 19 | **pass (at the 16/20 line)** | **FAIL (15/20 < 16)** |
| fire hydrant | 20 / **20** / 20 | 18 / **18** / 20 | pass | pass |
| person | 20 / **20** / 19 | 20 / **20** / 19 | pass | pass |
| car | 20 / **20** / 19 | 20 / **20** / 19 | pass | pass |
| motorcycle | 20 / **20** / 20 | 20 / **20** / 20 | pass | pass |
| airplane | 20 / **20** / 20 | 19 / **19** / 20 | pass | pass |

- **Primary `PekingU/rtdetr_r50vd` passes all seven classes**, with `bench` at
  exactly 16/20 matched positives and 19/20 clean negatives.
- **Secondary `facebook/detr-resnet-50` (`no_timm`) fails `bench`**: 15/20
  box-matched positives despite 17/20 image-level detections. Its other six
  classes pass. The published `verdict: "QUALIFIED"` for the secondary is
  **superseded**.
- **Withdrawn:** "both evaluators QUALIFIED, 7/7 classes" and every restatement of
  it (final report, README pointer, release body).
- **Consequence.** The manifest said a secondary failure does not by itself block a
  primary-defined comparison but **prevents a corroborated superiority claim on the
  affected category.** The secondary failure therefore did **not** itself prohibit
  primary training, but it would have prevented the specified corroborated-positive
  conclusion on **`bench`** (a held-out target). No comparison was run, so this
  never took effect.

**Preserved counts (unchanged):** 280 image/class records per detector,
**264 unique photographs**, **0 missing measurements**; distinct label-map checks
(all seven classes carry different detector label ids; e.g. stop sign 11 vs 13,
bench 13 vs 15); the COCO exposure/domain limitations.

**What the published "raw" records are.** They hold **derived** per-image values —
`detected`, `best_score`, `box_match_iou50`, `best_iou`, `n_detections` and
flags — **not** the full predicted boxes, labels and scores. Counts can be
re-aggregated (and were), but **IoU matching cannot be independently reconstructed
from these records alone**. Earlier descriptions of them as "raw boxes/scores"
are withdrawn. No boxes were fabricated and no inference was re-run.

The protected `RECONSTRUCTION.md` (also a release asset) describes the qualification
archive as "raw per-image boxes, scores and per-class gate counts"; that description is
superseded by the same limitation, and the asset was not altered.

**What qualification establishes — and does not.** A limited benchmark and
implementation check of COCO-trained detectors on COCO val2017 photographs, with
annotation-negative as an imperfect absence reference. It is **not** validation on
edited generations, **not** out-of-domain validation, and **not** semantic ground
truth; detector agreement is not ground truth.

---

## 3. Panel design — claim corrected, design unchanged

The governing protocol required, for each target, **two same-supercategory
non-target allowed classes plus two others**, taking the first valid complete
metadata-only combination.

Actual targets are `stop sign`, `bench`, `fire hydrant`. **Every** allowed panel
is `person`, `car`, `motorcycle`, `airplane`: **zero** same-supercategory classes
instead of two. The fallback (continue down other classes in ascending COCO id) was
disclosed prospectively in `RUN_MANIFEST.md`, but it was an **unauthorized design
deviation**: the permitted fallback concerned qualification-negative photographs and
does **not** authorize substituting the retention panel.

**Withdrawn:** "follows the frozen rule" / "follows deterministically from the
frozen rule", and "categories reusable as-is." The categories, matrix and results
are preserved exactly as published. **No replacement was searched for, no
eligibility was recomputed and category selection was not re-run.** The earlier
disclosure that repository hits for candidate classes were incidental COCO caption
text is not a finding of any closed-category collision either way, and no such
collision is inferred or claimed.

---

## 4. Native fidelity and causal claims — corrected

**Retained as genuine progress:** the actual pinned GEM trainer
(`c328fc6bec687efe60223498d1b6bed2e5fbcffd`) and **native PEFT** execute; the
recorded `qk_dual` inventory (76 modules, 7,471,104 trainable parameters, no
single-block leak); frozen base; FP32 adapter on a BF16 backbone; native student
trajectory sampling with `transformer.training=True`, `torch.is_grad_enabled()=False`
and LoRA dropout 0.1 (`no_grad` is not `eval`). The CPU return check also shows that the native arm's hook returns `None`
(`dispatch_native_returns_none`).

**Not established — the gate is open, not passed.** The trained native OFF and ON
adapter hashes **differ** (`c05be7b1…` vs `ed373817…`). **Required agreement is not
established.** The failed exact-hash check is **not** replaced by a numerical
tolerance and is **not** reinterpreted as passing.

| Earlier statement | Status now |
|---|---|
| "most likely explanation is GPU non-determinism" | **Withdrawn as a diagnosis.** Non-determinism is **one hypothesis.** No OFF/OFF run and no controlled numerical comparison exists, so **the cause is unresolved.** An OFF/OFF run would be informative but would not be guaranteed to settle every candidate cause. Other candidates not excluded by the records include run-order or in-process state carry-over (OFF ran first on a fresh build; ON ran second after an adapter-parameter restore) and an unrecognised code-path difference. None of these is asserted. |
| "deterministic algorithms were never enabled; `cudnn.deterministic=False`" | A **separately observed** `False` does **not** establish the setting during training. The pinned `Operator.__call__` calls `seed_everything(config.train_seed)` (`GEM.__call__` calls `super().__call__` first), which sets `torch.backends.cudnn.deterministic = True` and `benchmark = False` (`gem/utils/seed.py:18-19` in the pinned checkout `c328fc6`, read from the unpublished host copy). By source reading the training-time value is therefore `True`; it was not logged in the run. Conversely, **that helper does not prove complete deterministic execution** (it does not enable `torch.use_deterministic_algorithms` or fix `CUBLAS_WORKSPACE_CONFIG`). |
| "the patch is inert by code inspection" | Supported **only as a source-reading statement**. Source wiring and CPU return checks support that the hook is installed; they do not substitute for the failed run-level agreement. |
| "CPR gives a different adapter ⇒ the hook is live" | **Not a causal hook-effect result.** Native ON already differs from native OFF, so a different CPR hash cannot be attributed to the CPR loss. Hook installation is supported separately by source wiring and `dispatch_candidate_returns_tensor` (CPU). |
| "152/152 tensors loaded, recomputed hash `6ae4afdb…` matching" (release verification, `artifacts/release_verification.json`) | Supports **adapter-parameter restoration** only. It does **not** establish effective-model parity, forward-pass agreement, merged/unmerged BF16 equivalence or erasure efficacy. |
| gradient checkpointing | **Disclosed, not proven numerically inert.** It was enabled for memory and may affect numerics. |

---

## 5. Public freeze chronology and the CPU repair

**What was publicly committed, and when (UTC; from Git author dates):**

| Event | Time | Content |
|---|---|---|
| `3fba0ff` initial freeze | 05:24:37 | manifest, categories/cases, loss, detector, label and ledger code, patches P1–P3. **No gate code.** |
| `67d8b74` additive freeze | 05:27:52 | CPU gate and its repair record **only** |
| GPU jobs (ledger journal) | ≈05:28:47 – 05:43:17 | qualification, failed bring-up, retry bring-up |
| `00bc8ae` payload | **07:25:04** | **first public appearance** of `bringup*.py`, `qualify_detectors.py` and the exact-hash criterion |

The public freeze manifest (identical to the reviewed HEAD's) contains no statement
of the native dispatch-agreement tolerance or of exact-hash equality. **A detailed
prospective public native-gate freeze is therefore unverified.** The earlier wording
that the criterion was "frozen in the gate code before the run" is supported only as
a **local** chronology (the code existed on the host before it ran), not as a
publicly committed fact. This is **not** backdated here, and no hidden outcome-driven
selection is inferred from the publication gap alone. The coordinator's contract
required a **declared numerical tolerance** for agreement, not bit-exact trained-adapter
hashes; the frozen criterion departed from that.

**The CPU gate record, retained.** `artifacts/cpu_gate.json`: **64 rows, 57 pass,
0 fail, 7 N/A** (the seven are native-arm value/active/detach rows and two
hinge-inactive rows; **N/A rows are not passes**). Four computed-exact-zero hinge
boundary fixtures (`G2.boundary.{axial_margin,centered_radial,cpr,truncated_anchor}`)
and the formula/gradient checks are **partial** evidence; the native checks were
deferred to the GPU gate, which failed.

**The CPU repair, corrected.** `G4.native_scalar_order` first failed (preserved in
`artifacts/repair_01/cpu_gate_FAILED.json`). The single repair replaced an
FP32-versus-FP64 oracle at relative tolerance 1e-9 with a **same-dtype
bitwise equality** of the native association against a duplicate scalar
computation (`torch.equal`). **Withdrawn:** "strictly stronger"/"stricter" (the two
predicates test different things), blanket claims that tolerances were unchanged,
and the claim that a 1e-9 FP32-versus-FP64 criterion is universally impossible
for FP32 (it was unsatisfied here; its impossibility in general was not shown).
**No loss implementation changed; the predicate did.** As already disclosed,
retesting preceded the additive freeze commit; the failed and passed files and the
exact diff (`repair_01.diff`) are preserved. The diff header time (05:27:07Z)
precedes the commit (05:27:52Z).

**Axial N/A explanation corrected.** `G2.inactive.axial_margin` is N/A with the note
that its inactive point is "covered by `G2.boundary`". A boundary is **not** an
inactive point. Inactivity of the axial arm was not wholly untested: the separate
`G1.grad.axial_margin.near_a` row checks an all-inactive batch (gradient exactly
0.0) and passes.

---

## 6. Accounting and enforcement — corrected, supervisor not repaired

**Supervised GPU jobs (all on one device, GPU 2):**

| Job | Status | Charged device-seconds | Reserved |
|---|---|---|---|
| `qualify_detectors` | ok | 58.685038566589355 | 3,600 |
| `bringup` (import failure) | exit_1 | 2.6522905826568604 | 5,400 |
| `bringup_gpu` (retry) | exit_1 | 106.54127240180969 | 5,400 |
| **Total** | | **167.8786015510559 s = 0.0466329449 GPU-h** | |

Both failures are included; **all three are below their reservations**; there are
**zero open or unreconciled reservations**. One CPU repair and one technical GPU
retry are recorded. **No numeric breach of the 24-hour total or the 4-hour entry
sub-cap is evidenced**, and the 8-hour evaluation reserve was unused.
The projection's `already_used_gpu_h = 1.517` included the then-open 5,400-second
reservation, not the 0.0466 h actually charged.

**Elapsed time.** The published report's "~2.3 h" has **no tracked basis for its start**. The only
start evidence is an **unpublished host file** (`artifacts/preflight.log` in the
local run directory, not in any commit and not a release asset; sha256
`f628c048380069bb06dc02e177e48fb0b40cf847f21854a4e410c4820a55ec8a`), whose hand-entered
`task start` line reads 05:03:19Z. Measured to the final index commit (07:28:24Z)
that is 2 h 25 m, i.e. **about 2.3–2.4 h** — inside the one-week cap. Measured from
the first published commit (05:24:37Z) it is 2 h 04 m. Complete first-action and
time-enforcement evidence is missing; no overrun and no exact start is claimed.
The same local log header says it precedes the freeze, yet several of its
hand-entered lines (05:30–05:48Z) carry times *later* than the freeze commit
(05:24:37Z) and than the first GPU job (05:28:47Z); those lines cannot be literal
and are not relied on. Machine-generated times (Git author dates, ledger journal,
diff headers) govern.

**CPU.** 207.186047 CPU-seconds = **0.05755168 core-hours is only the subtotal of the
three supervised GPU children.** Whole-task CPU for setup, gates, downloads,
selection and publication is **unknown** where it was not recorded; "0.0576 of 48
core-hours" is not a whole-task figure.

**Storage.** "~0.3 GiB" was a sum of retained artifacts, **not** a full-task peak.
The manifest discloses a 5.5 GiB dependency attempt that was deleted and redone.
Sequential footprints are not added as if simultaneous, and **no measured peak
exists**.

**Supervisor gaps (documented, not fixed).**

1. `settle()` **clips** an overrunning charge to the reservation
   (`ledger.py:200`), contradicting its own docstring ("never charge less than
   observed").
2. On an exception the `except` block sends `SIGKILL` and the `finally` block then
   settles **without a confirmed reap**; the "charged through confirmed termination"
   claim is not substantiated for that path.
3. `reserve()` refuses only the GPU aggregate, the entry sub-cap and the protected
   reserve. The CPU core-hour, elapsed-time, worker, RSS and storage ceilings
   appear only as constants in the `CEILINGS` table and are **never compared with a
   measurement anywhere in the published `ledger.py`**. The image and COCO counters
   add first and raise afterwards (`count_images`, `count_coco`), i.e. they detect
   rather than prevent.
4. The published source contains **no** child RSS or worker watchdog and **no**
   disk-polling code. The sentences in `gaps()` / `ENFORCEMENT_GAPS.md` that RSS and
   worker ceilings are "checked by the in-process watchdog of each child" and that
   storage is "polled" are **not substantiated by the published source** (the
   disclosure that none of this is cgroup- or quota-enforced stands). Likewise the
   report's `Workers | 8 | 8` actual is unmeasured.

These gaps **do not invalidate the three observed below-reservation charges**, and
**no numerical CPU or storage cap breach is established.** They **do invalidate
blanket complete-enforcement assurances**. Withdrawn as unqualified statements: the
report's "the ledger refuses any reservation that would breach a cap"; the manifest's
description of caps being "refused at reservation time"; the report's table header
"Resource usage against every ceiling" insofar as it implies each ceiling was
measured; and "Failed jobs did not vanish from accounting", which is supported
**only for the three observed jobs** (the exception path can settle without a
confirmed reap). No retrospective measurement was invented.

**The 8.16 GPU-hour figure is a provisional projection only.** Training was briefly
profiled (two 22-timestep runs of 29.8 s and 29.5 s, 1.3499 s/processed timestep);
**generation at 5 s/image and detector throughput at 0.08 s/image were assumed**,
not measured. Withdrawn: "fits comfortably", "workflow-ready" and
"everything reusable as-is". Frozen contents are preserved; their fitness for any
later use is not asserted.

---

## 7. Claim-by-claim supersession

| Where | Claim | Status |
|---|---|---|
| final report, README, release body | both evaluators QUALIFIED, 7/7 | **Withdrawn** (§2) |
| final report §2.1, manifest | panel "follows the frozen rule"; categories reusable | **Withdrawn** (§3) |
| final report §1, §5, README, release body | non-determinism "most likely" cause | **Withdrawn as diagnosis** (§4) |
| final report §2.4 | CPR hash differs ⇒ "confirming the hook is live" | **Withdrawn as causal** (§4) |
| final report §1, §2.3, §4 | criterion "frozen in the gate code before the run"; CPU gate "COMPLETE — 0 failures" | **Qualified** (§5) |
| `repair_01/README.md` | "stricter"; "unsatisfiable for FP32" | **Superseded** (§5) |
| final report §3, manifest §7 | cap enforcement; Workers 8; ~0.3 GiB storage; CPU 0.0576 h as the figure against 48 | **Qualified** (§6) |
| final report §2.5, §6 | 8.16 h "fits"; "reusable as-is" | **Withdrawn** (§6) |
| final report §6 | re-attempt recommendation | **Removed** — attempt CLOSED |

---

## 8. Remaining unknowns, and closure

**Unknown and left unresolved:** why the native OFF and ON hashes differ; whether the
native dispatch is numerically equivalent; whole-task CPU; peak storage; exact first
action time; the behaviour of CPR against any control; detector raw boxes.

**CLOSED.** The attempt is closed with `ENTRY_GATE_UNRESOLVED —
NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`. This amendment makes **no** rerun,
scorer-search, replacement-category or retry recommendation, and does not select,
schedule or start any follow-up. Implementation, evaluator and watchdog limitations
are documented here and left **unfixed**.

---

## 9. What was not done

No tests; no `torch` import, candidate/test/scorer module import or checkpoint load;
no GPU allocation or polling; no training, generation, scoring, detection or image
label; no dependency installation; no benchmarking; no literature search; no
reviewer recruitment; no execution of any closed study. Operations were limited to
read-only Git/source/log inspection, JSON parsing, arithmetic, hashing and GitHub
metadata reads/writes (the release-body notice only). See
[`PRESERVATION_AND_ACCOUNTING.md`](PRESERVATION_AND_ACCOUNTING.md).
