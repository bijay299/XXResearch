# AMENDMENT 01 — corrected CPU accounting, withdrawn hardware claim, narrowed fidelity and coverage

**Date:** 2026-10-20
**Amends:** [`cc54d8389cb06223bb9296e7afa620afea2fa6e8`](https://github.com/bijay299/XXResearch/commit/cc54d8389cb06223bb9296e7afa620afea2fa6e8) (`results/llada_emdu_mdu_rwku_v1/`)
**Nature:** additive documentation and accounting correction. **No new experiments.**

> No model was loaded, no GPU allocated, no text generated, no optimizer update performed,
> no correctness test re-run, no literature search repeated, and no frozen scientific code
> or configuration changed. Work was limited to offline inspection, arithmetic, hashing and
> publication checks. **Zero GPU-seconds.**

## Outcome retained

**`PAPER_ALIGNED_RUN_BLOCKED`** — and now stated with its correct scope: **the tested
custom CPU-offload implementation, including its one permitted repair, did not fit the
remaining resource envelope.** No EMDU-versus-MDU behavioral comparison occurred. Neither
this outcome nor Stage A's establishes which method is better.

---

## Correction 1 — CPU accounting was undercounted by an octal parsing bug

**Verified root cause.** The `/usr/bin/time` wrapper wrote attempt `b0003`'s CPU field with
a leading zero — `026063` — into `ATTEMPTS.tsv`. The ledger summed the column with POSIX
shell arithmetic (`$((...))`), which interprets a leading-zero literal as **octal**:
`026063₈ = 11315₁₀`, not `26063₁₀`. Reproduced offline:

```
$((0+27233+1+152+026063+4891)) = 43592      # what the ledger reported
0+27233+1+152+26063+4891       = 58340      # decimal truth
```

### Reconciliation

| | core-seconds | core-hours |
|---|---|---|
| Attempt rows, decimal: `0 + 27233 + 1 + 152 + 26063 + 4891` | **58,340** | **16.2056** |
| Previously reported (octal-corrupted) | 43,592 | 12.1089 |
| **Undercount** | **14,748** | **4.0967** |

Per-attempt rows, preserved as published:

| attempt | label | cpu_core_s (raw cell) | decimal |
|---|---|---|---|
| `b0001` | stageA audit offline_recompute | `0` | 0 |
| `b0002` | feasibility fullparam_probe | `27233` | 27,233 |
| `b0004` | freeze protocol_and_eval_ids | `1` | 1 |
| `b0005` | verify objectives_cpu | `152` | 152 |
| `b0003` | feasibility fullparam_repair | **`026063`** | **26,063** |
| `b0006` | verify runtime_frozen_teacher | `4891` | 4,891 |

**The original rows, the historical counter files and `scripts/ledger.sh` are preserved
unchanged as historical evidence.** The corrected figure is published as a *derived*
ledger (`RESOURCE_LEDGER.md`), not by editing the historical measurements. The original
report described this only as a "cosmetic defect"; it was an arithmetic error of
4.0967 core-hours.

### What this does and does not say about budget compliance

**Historical CPU usage remains unmetered.** The earlier qualification study
(`llada_rwku_baseline_v1`) had no CPU instrumentation, so its consumption is unknown and
no retrospective charge is invented. Therefore:

> **3.7944 core-hours is at most the remaining allowance before unknown earlier usage —
> not a verified remaining whole-envelope budget.** No claim of overall CPU-budget
> compliance is made, in this amendment or anywhere else.

### GPU total — reconciled and unchanged

`323` carried + `548` (`b0002`) + `660` (`b0003`) + `352` (`b0006`) = **1,883 GPU-seconds
≈ 0.5231 GPU-hours**, leaving **19,717 GPU-seconds** against the original six-hour
(21,600 s) ceiling. Each GPU attempt ran on **2 devices** (wall × 2), and the
**maximum-two-GPU** record is preserved. The original clock and envelope are retained; no
clock or budget was restarted.

### Measured versus estimated

The feasibility trade-off table reported "CPU core-h" figures computed as
**thread count × optimizer wall time**. That is a **cost estimate**, not measured core
usage: it assumes every configured thread is fully busy for the whole optimizer phase.
Only the `/usr/bin/time` user+sys figures in `ATTEMPTS.tsv` are measured. The projections
in the original §6 table are relabelled accordingly.

## Correction 2 — the single-80 GB-GPU recommendation is withdrawn

For the recorded **8,015,581,184** parameters:

| Layout | Bytes/param | Size |
|---|---|---|
| FP32 master weights + two FP32 optimizer moments | 12 | **≈ 89.58 GiB** |
| + BF16 student weights + BF16 gradients | 16 | **≈ 119.44 GiB** |
| + BF16 frozen anchor | 18 | **≈ 134.37 GiB** |

These **exclude activations and other buffers.**

**Withdrawn:** the claim that "one 80 GB A100/H100 **would** hold params + grads + fp32
AdamW state on-device". It would not. Even the bare optimizer state (89.58 GiB) exceeds
80 GB *and* 80 GiB; the full stated layout is ~134 GiB, roughly 1.7× an 80 GB card.

**Also withdrawn:** the prediction that requesting such a card would make 240 updates cost
"on the order of minutes". That extrapolated from the measured ~1.0–1.6 s of
forward+backward while assuming the optimizer cost would vanish — an unverified assumption
about hardware never tested here.

**No substitute hardware prescription is offered.** Determining what configuration would
work is outside this amendment's authorization.

## Correction 3 — the blocker's scope, corrected

**Withdrawn:** that the measurements show the slowdown is "**bandwidth-bound**, not an
implementation or parallelism defect", and the implication of universal infeasibility.

**Corrected statement.** What was measured: *this* custom CPU-offload implementation — a
naive per-tensor fp32 AdamW, then one permitted repair using in-place `_foreach` ops with
preallocated buffers — did not fit the remaining envelope at any of the three CPU thread
settings tried (8, 32, 255). Wall time was roughly flat across those settings and the
repair yielded ~1.01×, which is *consistent with* a transfer/bandwidth limit but **does
not establish it**: thread count was the only dimension varied. Untested alternatives
include pinned-memory and overlapped/streamed transfers, fused or chunked optimizer
kernels, parameter-sharded optimizer state across both devices, ZeRO/FSDP sharding, and
8-bit or bf16 optimizer states.

Therefore the measurements **do not prove universal full-parameter infeasibility on two
40 GB GPUs**, and **no optimized offload or sharding solution was qualified** — none was
attempted.

## Correction 4 — the native batch-reduction equivalence claim is withdrawn

The report and the frozen spec state that dividing each microbatch loss by the
accumulation count makes accumulated gradients "equal the mean over the 4 examples —
matching batch-4 mean reduction".

**Withdrawn.** Our accumulation computes

```
mean_i( loss_sum_i / masked_count_i )
```

whereas native pooled masked-token normalization computes

```
sum_i( loss_sum_i ) / sum_i( masked_count_i )
```

These are equal only when all `masked_count_i` are equal. Because `p_mask ~ U(1e-3, 1)` is
drawn **per microbatch**, the counts differ in general. Demonstrated by offline arithmetic
on four microbatches `(loss_sum, masked_count)` = `(10,5) (20,20) (6,2) (30,40)`:

```
ours   = 1.687500
native = 0.985075        -> differ by 0.702425
```

With equal counts the two coincide exactly, confirming unequal counts are the cause.

**This is recorded as an unresolved fidelity issue.** `scripts/objectives.py` and
`frozen/stageb_spec.json` are **not modified**, and nothing was re-tested. Note the check
that would have caught this — `{method}_reduction_is_pooled_mean_over_masked_answer_positions`
(`scripts/verify_stage_b.py:90`) — is itself **unconditional** (`chk(..., True, …)`), so it
asserted the property rather than testing it. Since the comparison never ran, no published
result depends on this; any future run must resolve it first.

## Correction 5 — fidelity, coverage and chronology narrowed

Observed passing counts are retained: **24/24** objective checks, **12/12** runtime checks.
What they establish is narrowed.

### Native-path fidelity

- **Executed:** the extracted native `_null_anchor_kl` helper, pulled by AST from the
  pinned MDU file and run against controlled fixtures. Our MDU arm matched it.
- **NOT executed:** the complete native training path. `MDUTrainer` was never instantiated;
  `dllm` is unavailable, and the shim supplied `scheduler`, `time_epsilon` and
  `_postprocess_outputs`. Executing an extracted helper is **not** verification of the
  native trainer. The original report's framing is corrected accordingly.

### Coverage limits

| Claim | Corrected scope |
|---|---|
| "both objectives verified at runtime" | **Both objective forwards executed** (`runtime_dispatch` recorded for `mdu` and `emdu`). The real optimizer-update check performed **2 EMDU updates only** — no MDU optimizer trajectory was run, and neither method has a trajectory. |
| teacher/student digest invariance | Hashed **40 of 291** parameter tensors (`n_tensors_hashed: 40`). Full-model `requires_grad` inspection (291/291 trainable; anchor all frozen) and **sampled** hash invariance are different checks; the digest result is a sample, not full-model coverage. |
| unconditional assertions | `scripts/verify_stage_b.py:90` (×2, per method) records `passed: True` as a literal. Its coverage is **not established**. |
| checkpoint / reload | **No checkpoint/reload qualification was demonstrated.** No trained state exists. |
| evaluation pipeline | Panels, row IDs and scoring helpers were **configured and frozen**; the **evaluator was never executed** — no generation, scoring or comparison artifact exists. A configured panel is not an executed evaluator. |

### Chronology (existing timestamps, recorded accurately)

| time | attempt | note |
|---|---|---|
| 01:27:01 → 01:31:35 | `b0002` feasibility probe | **3 optimizer updates — before the freeze** |
| 01:33:39 → 01:39:09 | `b0003` feasibility repair | **6 optimizer updates**; started before the freeze, ended after |
| **01:35:39** | `b0004` **protocol + eval-ID freeze** | — |
| 01:37:27 | `b0005` objective checks | after freeze |
| 01:40:19 → 01:43:15 | `b0006` runtime checks | **2 EMDU optimizer updates, 352 GPU-s — after the repaired feasibility blocker** (determined 01:39:09) |

So **engineering model outputs preceded the protocol freeze.** **No comparison outcomes
preceded the freeze, because the comparison never ran** — the freeze governs the unrun
scientific comparison, not the engineering probes.

**"No training occurred" is avoided as inaccurate.** There were **11 engineering optimizer
updates** (3 + 6 + 2) on the real model. What did not occur is the **scientific
comparison**: 0 of the planned 240 trajectory updates (3 subjects × 2 methods × 40), and no
evaluation.

### Memory peaks disambiguated

| Phase | cuda:0 | cuda:1 | Source |
|---|---|---|---|
| Feasibility probe and repair | 34.714 GiB | 15.067 GiB | `outputs/feasibility_probe.json`, `outputs/feasibility_repair.json` |
| **Later runtime checks** | **35.078 GiB** | **16.872 GiB** | `outputs/stageb_runtime_checks.json` |

The original report quoted "peak 34.71/15.07 GiB" as though global. The highest recorded
peaks are the **runtime** figures above; both remain under 40 GiB.

### EMDU code availability

**Replaced:** "EMDU author code does not exist" / "does not exist yet".
**Bounded statement:** *EMDU author code and checkpoints were **not located in the sources
inspected on 2026-10-20*** — the paper (arXiv:2610.25447v1) names no EMDU repository and
says "the checkpoints we will release", and one web search returned only the arXiv entry.
Absence from those sources is not proof of non-existence. **That search was not repeated.**

## Correction 6 — the negative control: explanation versus execution evidence

`evidence/negative_control/README.txt` asserts that the corrected control "exits 1". The
**execution evidence is partial**:

- **Exists:** `broken_tau_exit1.log` — the corrected control's output, ending
  `FAIL tau_multiplies_anchor_logits` / `23/24 passed` / `ALL PASSED: False`.
- **Exists:** `invalid_first_control_exit0.log` — the invalid first control's output,
  `24/24 passed` / `ALL PASSED: True`, confirming the break never loaded.
- **Now published:** the broken variant source itself (`negative_control/objectives_broken_tau.py`),
  recovered from the workspace — it differs from `scripts/objectives.py` in exactly one line
  (`target_logits = target_logits / tau`).
- **Does NOT exist:** a stored record of the **exit codes**. Both logs contain only the
  check verdicts; `0` and `1` were observed in the shell and are not captured in any
  published artifact. The verdict lines are consistent with the claimed codes given
  `sys.exit(0 if all_passed else 1)` in the verifier, but the codes themselves are **not
  independently evidenced**. The controls were **not re-run** to capture them.

---

## Preservation inventory (unchanged by this amendment)

| Path | Bytes | SHA-256 (first 16) |
|---|---|---|
| `outputs/feasibility_probe.json` | 1,414 | `7a526fc84a56934c` |
| `outputs/feasibility_repair.json` | 1,747 | `770e1fcfe4658641` |
| `outputs/stageb_objective_checks.json` | 4,141 | `99ae5610f401e090` |
| `outputs/stageb_runtime_checks.json` | 3,476 | `c6f31ea6cd23eccf` |
| `evidence/ATTEMPTS.tsv` (incl. the raw `026063` cell) | 977 | `fdf2f1599ef4986a` |
| `frozen/stageb_spec.json` | 6,126 | `c38373075a925063` |
| `frozen/stageb_panels.json` | 121,511 | `fb443578ae085650` |
| `scripts/objectives.py` | 6,538 | `5c9b90796e36bbcf` |
| `scripts/verify_stage_b.py` | 9,949 | `218f0e0893f09a3e` |
| `scripts/verify_runtime.py` | 9,598 | `17cca77065f67577` |
| `scripts/ledger.sh` (with the octal bug, as historical evidence) | 1,105 | `5a4e788d9ae6cf41` |
| `scripts/attempt.sh` | 1,856 | `22e8be62b3c9a984` |
| `logs/` (6 original attempt logs, incl. the failed repair) | — | byte-identical to `cc54d83` |

Full current hashes for every published file are in `EVIDENCE_INDEX.md`.

## Measured / estimated / untested

- **Measured:** 1,883 GPU-s (2 devices per attempt); 58,340 new CPU core-s (decimal);
  per-update wall times 43.4–67.8 s; memory peaks 34.714/15.067 (feasibility) and
  35.078/16.872 GiB (runtime); 11 engineering optimizer updates; 24/24 and 12/12 check
  verdicts; 582 pairs per subject; panel counts.
- **Estimated:** the trade-off table's "CPU core-h" (threads × wall), and all 240-update
  projections. Not measured usage.
- **Untested / unestablished:** whether the slowdown is definitively bandwidth-bound; any
  optimized offload or sharding layout; what hardware would suffice; the microbatch
  reduction's fidelity to native pooled normalization; MDU optimizer-trajectory behavior;
  full-model (291-tensor) digest invariance; checkpoint save/reload; the evaluation
  pipeline end-to-end; the negative controls' exit codes as stored evidence; and whether
  EMDU author code exists outside the sources inspected on 2026-10-20.

## Reviewed publications

- This study: [`cc54d8389cb06223bb9296e7afa620afea2fa6e8`](https://github.com/bijay299/XXResearch/commit/cc54d8389cb06223bb9296e7afa620afea2fa6e8)
- Stage A qualification: [`37c31ca`](https://github.com/bijay299/XXResearch/commit/37c31ca17a004b7bf7c44634e395e2eef7a9678f) → [`a349197`](https://github.com/bijay299/XXResearch/commit/a349197c471d8bd74878b6b26f622f3f225c7232) → [`aecefcd`](https://github.com/bijay299/XXResearch/commit/aecefcd662c5c358704a30d5a47bb903ee17e084)
