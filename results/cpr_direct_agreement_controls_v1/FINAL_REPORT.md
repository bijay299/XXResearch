> **CORRECTION NOTICE (2026-10-19, UTC).** This report is retained in full, unchanged, below the line. Its terminal status `ENTRY_NOT_PASSED/CLOSED`
> stands and no historical failure is converted to a pass. Several descriptions below are corrected by
> [`AMENDMENT_01_SCOPE_AND_PROTOCOL.md`](AMENDMENT_01_SCOPE_AND_PROTOCOL.md) — notably: detector qualification counts (RT-DETR oven 15/20; DETR refrigerator 15/20, couch 14/20),
> GPU work continuing after the failed qualification (strict stopping not achieved), the `min_grad_frac` change in repair 1, the CPU-only index-0 replay (no GPU gradients captured;
> both captured indices preceded the first update), post-hoc numerical tests, incomplete CPU/storage accounting, estimate-vs-measurement repair times, and unverified licensing at publication time.
> Current work continuing this question: [`../cpr_execution_recovery_v1/`](../cpr_execution_recovery_v1/).

---

# FINAL_REPORT — cpr_direct_agreement_controls_v1

> **Terminal status: `ENTRY_NOT_PASSED` — two independent, individually sufficient entry failures, both retained:**
>
> 1. **`ENTRY_QUALIFICATION_FAILED`** — the fixed detectors did not clear the frozen per-class floor on every scored class
>    (RT-DETR: 1 class below floor; DETR: 2 classes below floor).
> 2. **`ENTRY_GATE_FAILED_REPLAY_FP64_GRADIENT_TOLERANCE`** — the direct native loss/gradient replay gate failed on real operands:
>    the native block, the hook-enabled path and the pristine path agree **bit for bit**, but the frozen *elementwise* FP64-reference
>    gradient tolerance (rtol 1e-4 / atol 1e-7) is exceeded by FP32 rounding on 6,461 of 65,536 near-zero gradient elements.
>
> **No scientific decision was reached.** The frozen six-arm comparison **never started**: **0 model-generated images**, no
> M0/development/held-out generation, no arm other than native GEM trained, no CPR probe. `PROMISING_DETECTOR_DEFINED_TRADEOFF`,
> `NO_ADVANTAGE_IN_THIS_SCREEN` and `INCONCLUSIVE` are **not** emitted: those are comparison outcomes, and no comparison data exist.
> This report supports **no** claim about CPR vs native GEM or any control, nor about semantic unlearning, image quality, novelty or SOTA.

## 1. Question and scope (unchanged from the contract)

At comparable target-detection reduction, does CPR retain allowed-object detections better than native GEM and the strongest simple
controls? Detector-defined screening study only. No human reviewers; no Claude judgments or prompts used as presence ground truth.
The study ended at its entry gates (contract §4 "failure ends entry, no substitutions"; §6 "no CPR probe or continuation after a failed gate").

## 2. Design, settings, freeze

* Study dir (work/publish): `cpr_direct_agreement_controls_v1`; release/tag `cpr-direct-agreement-controls-v1`. Earlier studies untouched.
* Prospective freeze commit `a9f6e412b5cb92f861659573df2cbd3050dd90d8` (design, formulas, gate code, predicates, budgets, verification plan); made before any numerical
  loss test, detector inference or FLUX forward pass (`PREFREEZE_LOG.md`, `FREEZE_MANIFEST.json`).
* Metadata-only design (annotation JSON only): triple [6, 9, 10]; **dev target refrigerator** (allowed: panel oven, sink, bus, train); **held-out toilet and bed**
  (allowed: couch, potted plant, bus, train). 9 scored classes; 333 unique qualification photographs (cap 600). Configs under `configs/`.
* Trainer: native GEM `multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd` plus attributed patches (`patches/`), real PEFT `qk_dual` LoRA
  r16 / alpha16 / dropout 0.1 on FLUX.1-dev `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`: 76 LoRA modules, 152 tensors, 7,471,104 trainable parameters, 23 optimizer
  updates per 250 processed states (window of 11).
* Patches P1a (capture hook), P1b (loss-dispatch hook; returning `None` falls through to the original native block), P2 (preview-decode skip, counted), P3 (step hook),
  P4/P5a/P5b (optional `datasets`/`wandb`), P6 (optional `Flux2Pipeline`), P7 (local snapshot path), **P8 (optional `QwenImagePipeline` sentinel) and P9 (optional `RepoFolder`
  sentinel), added pre-freeze after import failures** (5 CPU-only structural runs; 4 failed on import/harness defects, 5th passed; all logged).
* Environment: venv py3.10.12, torch 2.4.0+cu121, diffusers 0.30.2, peft 0.12.0, transformers 4.44.2, huggingface_hub 0.32.4; 4x40 GB GPUs.
* Seeds: dev train 5101; held-out 5201/5202; bootstrap seed 4035604553, N_BOOT=10000 (never reached). Gate trajectory `dev.native_gem.lr_1e-3`; capture indices [0, 7].

## 3. Detector qualification (contract §4) — FAILED

Rule: both detectors must pass **every** scored class: >= 16/20 `pos_boxmatch` and >= 16/20 clean negatives, 20 positives + 20 negatives per class (10 confounder negatives
each, none deficient). Detectors: RT-DETR r50vd (primary) and DETR resnet-50 (secondary) at their frozen revisions, threshold 0.30. No substitution was made or permitted.

| class | RT-DETR pos / neg-clean | pass | DETR pos / neg-clean | pass |
|---|---|---|---|---|
| refrigerator (dev target) | 16 / 20 | yes | **15** / 19 | **NO** |
| oven | **15** / 20 | **NO** | 16 / 20 | yes |
| sink | 18 / 19 | yes | 19 / 18 | yes |
| bus | 20 / 20 | yes | 19 / 20 | yes |
| train | 19 / 20 | yes | 18 / 20 | yes |
| toilet (held-out target) | 19 / 20 | yes | 19 / 20 | yes |
| bed (held-out target) | 17 / 20 | yes | 18 / 20 | yes |
| couch | 17 / 19 | yes | **14** / 19 | **NO** |
| potted plant | 17 / 20 | yes | 17 / 20 | yes |

Raw per-photograph measurements: `artifacts/qualification/qualification_full.json`; summary `qualification_summary.json`. 333 photographs loaded; 0 model-generated images.
Consequence: `entry_qualification_pass = false` (`artifacts/entry_decision.json`). The failed classes include the development target (DETR) and an allowed object (RT-DETR oven).

## 4. Direct native loss / gradient agreement (contract §6)

Tolerances (frozen): loss rtol 1e-5 / atol 1e-6; gradient rtol 1e-4 / atol 1e-7 (elementwise, versus an independent FP64 reference).

| gate | result | scope |
|---|---|---|
| Synthetic native-vs-reference gate | **PASS** (87 checks; `artifacts/gate/synthetic_gate.json`) after repair cycle 1 | d <= 256 fixtures, CPU, FP64 reference; **d = 65,536 is not covered** |
| GPU instrumentation gate (real FLUX/PEFT trajectory, 250 states, 23 updates) | **PASS** on rerun after repair cycle 2 (`gpu_instrumentation.json`); run 1 failed on a harness counting defect, preserved | structural/mode/window checks; includes dropout active during student sampling (native), window invariance with hooks on/off, divisor 11 |
| **Replay gate on real captured operands** | **FAIL** (`artifacts/gate/replay_gate.json`, failed check `native_three_way::idx0`) | real FP32 student velocity / anchor teacher / target teacher at predeclared indices 0 and 7, replayed on CPU; stopped at the first failed required check |

Replay gate, index 0 (the only index evaluated, because the gate raises and stops on the first failure):

* Three-way native agreement **holds exactly**: native block (A) vs hook-enabled path (B), pristine reimplementation (C) and singleton/batched path (D) — loss scalar, per-sample loss and
  gradient **all bitwise equal** (errors 0.0); hook protocol clean (1 call, all return `None`, FP32 args, inputs unchanged).
* FP64 reference comparison: loss within tolerance (absolute 2.88e-3 on a loss of 644.06, relative 4.47e-6 < 1e-5), per-sample ok, **gradient failed**:
  max |FP32 − FP64| gradient difference 1.545e-6 against max |g| 0.146 (1.06e-5 of max). **6,461 / 65,536** elements violate `|diff| <= 1e-7 + 1e-4 |g_ref|`
  (violators are small-magnitude elements; max |g| among violators 0.0127).
* Because required failures stop the gate, **index 7 and all CPR/control-arm replays were not run by the gate.** No CPR probe was run.

**Post-hoc diagnostic (explicitly NOT a gate, run after the failure, `scripts/posthoc_gate_failure_diagnostic.py`, `artifacts/analysis/posthoc_gate_failure_diagnostic.json`):**
the native block executed in FP64 matches the independent FP64 reference to 1.4e-15 (index 0) / 1.4e-14 (index 7) in gradient and ~1e-12 in loss, i.e. the implementations agree;
the residual is FP32 summation error over 65,536 dimensions (relative to max |g|: 1.06e-5 at index 0, 5.7e-6 at index 7; 6,461 and 7,959 violating elements). This explains the failure
but **does not rescue it**: the frozen tolerance was exceeded, the contract allows no tolerance change, and only two repair cycles (spent on unrelated defects) were available.

Coverage limitation (disclosed): at indices 0 and 7 the student equals the target teacher bitwise (adapter B = 0 and the first window precedes any update), so the dev-hinge distance
`D- = eps*sqrt(d)` and the loss/gradient there are a restricted regime; they do not exercise student-teacher gaps that arise after training has moved the adapter. Whether the CPR
arm, or any non-native arm, agrees with its reference on real triplets is **unmeasured**.

## 5. Post-freeze repair cycles (additive, each with documentation)

| cycle | patch commit | defect | change | unchanged |
|---|---|---|---|---|
| 1 | `f4bebf580b33faadfe38ef48f736646f080801a0` | synthetic d=256 fixture: 0/6 rows because optional gradient-conditioning filter (`min_grad_frac=0.01`) rejected 3000/3000 draws | `min_grad_frac=0.0` for that fixture only | seeds, margins, tolerances, objectives, arms, decision rules |
| 2 | `a89ac56c1444cc540626f875b9cf8b8db4ba8444` | GPU-gate sampling-mode assertion counted container + `nn.Dropout` (152 vs 76); orchestrator misreported qualification as pass whenever a summary existed | count only `nn.Dropout`; fix diagnostic flag; qualification reporting; add `gpu_gate` rerun phase | everything frozen |

Engineering minutes (declared estimates in the ledger, no wall timers): **10 and 15 = 25 of 30**; cycles used: **2 of 2**. Each had exactly one patch and one rerun. Repair documents:
`artifacts/repairs/repair_1.md`, `repair_2.md`. The replay-gate failure arose after the second cycle was exhausted and is **not** repaired.

## 6. Outcomes by arm / object / seed

None. **Native GEM** was trained for the gate trajectory only (dev refrigerator, seed 5101, 250 states, two runs; adapter hashes below). **CPR, replay/no-replay, uniform-anchor, shrink and other controls:
0 trajectories. Held-out toilet/bed seeds 5201/5202: 0 trajectories. M0, development generation, techbatch, projection: not run.** The entry record `projection_fits=false` denotes "not computed/not reached".

Gate-trajectory adapters (same seeds): run 1 terminal `d4ae3a67…` (superseded: failed harness assertion), run 2 terminal `f339b141…` (file sha256; tensor-state hash `0c43de15…`). The two
runs, same seed, have different terminal hashes while their captured triplets at indices 0 and 7 are **byte-identical** (`6d65743c…`) — GPU run-to-run nondeterminism after index 7 is evidenced; its cause
(dropout/kernel nondeterminism) is **not** diagnosed here.

## 7. Evaluator limits (carried forward)

Detector-defined presence only (box-match at threshold 0.30), COCO photographs for qualification only; qualification itself failed. Detections are not ground-truth semantic presence; no human or Claude judgement was used.
Nothing about image quality, diversity, novelty, safety or semantic unlearning is inferable.

## 8. Exact counters, retries, unmeasured scope

* Ledger (`artifacts/ledger/{attempts.jsonl, monitor.jsonl, state.json}`): **11 supervised launches** (5 pre-freeze structural CPU runs [4 failed, 1 ok], `gate.synthetic` [failed -> repaired], `gate.synthetic.r1` ok, `qualify` [exit 4 = qualification failure],
  `train.dev.native_gem.lr_1e-3` [exit 3, harness defect], `.r2` ok, `gate.replay` [exit 2 = gate failure]); 0 refusals, 0 overruns, 0 open attempts.
* **GPU charged 725.44 device-s = 0.2015 GPU-h** (entry bucket; main 0, reserve 0) of 24 (entry cap 4). **CPU charged 2,037.79 core-s = 0.566 core-h** of 48.
* **Model-generated images: 0** (primary 0 / technical 0 of 2,200). COCO photographs loaded: 333 of 600. PNG decodes during training: 0 (previews skipped, counted).
* Peaks: RSS 33.97 GiB (cap 128), min free storage 516.7 GiB (floor 50), storage growth 0.22 GiB, thread-count peak **264 (advisory sample, not a worker count; the 8-worker cap is enforced by launch reservation and OMP/MKL limits, not cgroup)**.
* Elapsed: first task action 2026-10-19T17:31:02Z; entry decision 18:55:19Z (about 1.4 h of 7 days; science stop day 6 never approached).
* Unmeasured / accounting gaps: storage polled every 120 s on the task root only (peaks are lower bounds); CPU excludes escaped processes (lower bound); GPU memory per device not monitored; repair engineering minutes are estimates;
  the synthetic gate does not reach d = 65,536; replay indices beyond 0 and all non-native arms were never replayed; the two repair cycles were used on harness defects, not on the final failure.
* The earlier smoke failure of the supervisor (cap/overrun assertion) and failed pre-freeze structural runs are logged under `logs/` and `PREFREEZE_LOG.md`.

## 9. Decision

**`ENTRY_NOT_PASSED`** (`ENTRY_QUALIFICATION_FAILED` + `ENTRY_GATE_FAILED_REPLAY_FP64_GRADIENT_TOLERANCE`). No comparison decision. Per the contract, the frozen matrix, thresholds and decision rules were **not** reopened, no detector or tolerance was substituted,
and no successor task was started.

## 10. Reproduction, assets, licensing, preservation

* Assets (GitHub release `cpr-direct-agreement-controls-v1`): `adapters_gate_run2.tar` (3 checkpoints), `adapters_gate_run1_superseded.tar` (3 checkpoints), `captured_triplets.pt`, `SHA256SUMS`; manifest `RELEASE_ASSETS.json`.
  Size/hash verification in a fresh area and CPU-only adapter restoration results are published in the verification-record commit (`artifacts/verify/`).
* Licensing: backbone `black-forest-labs/FLUX.1-dev@3de623fc` is under the FLUX.1 [dev] **Non-Commercial License**; **no base weights are redistributed**. The LoRA adapters are small derivatives
  published for **non-commercial research verification only**; the licence text was **not present in the local snapshot and was not re-verified here** — consult the upstream licence before any other use.
  Detectors (Apache-2.0) are not redistributed. **COCO photographs are not uploaded** (ids/annotation hashes in `records/coco_annotation_hashes.txt`). GEM is third-party; only attributed diffs are published. No credentials or private responses are published.
* Preservation: only `results/cpr_direct_agreement_controls_v1/` is added plus one appended row in `results/README.md`; old releases, tags, assets and closed paths are checked against `records/` in the verification record.
