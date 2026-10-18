# AMENDMENT 01 — CLOSEOUT WITH CORRECTIONS

**Final status: `BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS.**

This amendment is **authoritative over all conflicting wording** in the originally published
sprint documents (commit `4177b4c1fbb623a1db2521ee8747b2c6a8ad685a`). It supersedes the
sprint's `VALIDATED_PLATFORM_NO_PROTOTYPE` status and any implied continuation.

It is **documentation-only**. No model was loaded, no tensor or GPU work was run, nothing was
generated, scored, rescored, relabelled or re-audited, no code or ledger was repaired, and no
new derivation or literature assessment was made. Every number below is read or recomputed by
ordinary arithmetic from records already published in this directory.

---

## 1. Correct qualification, and why the status changes

The frozen `RUN_MANIFEST.md` requires **all six** baseline-validation gates to hold before
prototype work, and states that failing any of them requires marking `BASELINE_UNQUALIFIED`
or `BLOCKED`, publishing, and stopping.

Gate 1 reads, in the frozen manifest: *"Scorer controls pass (above), **no unresolved material
generated-domain scoring failure**."*

`BASELINE_VALIDATION.md` §1, as published, explicitly records such a failure — a systematic
disagreement on edited Bears images that it describes in its own words as "unresolved". Under
the frozen rule, that is a gate-1 failure, and the mandatory consequence was
`BASELINE_UNQUALIFIED` and a stop.

### Withdrawn

- **"Baseline validation passed."** It did not: gate 1 was not satisfied.
- **"The platform is validated for measuring the native baseline."**
- **The status `VALIDATED_PLATFORM_NO_PROTOTYPE`**, wherever it appears.
- Any framing in which the sprint proceeded in full compliance with its own protocol.

### Retained

- A **runnable source-derived implementation** of native BLOCK on FLUX.1-dev was built from
  the pinned release, and both transcoders were trained natively because none are published.
- The **independent real-photograph controls passed**: 20/20 sensitivity and 20/20 specificity
  per object on 120 COCO val2014 images annotated by COCO's own human instance annotations,
  including 0/10 false positives on the related-animal negatives. This is a genuine result.
  **It does not establish generated-domain validity.**
- **Gates 2–6 were recorded as passed at the scope actually tested** (module identity,
  bit-identical activation targets, finite computations, intended-vs-unrelated tensor checks,
  save/reload agreement, reference-storage separation, paired-noise construction, and the
  M0/T0 presence and retention counts).
- **Generated-domain qualification remained unresolved.**
- **No prototype and no control were developed.**
- **No qualified method advantage and no contribution were established.**

---

## 2. Disclosed protocol deviation: runs made after the stopping condition was known

`PROTOTYPE_DECISION.md` declared a "pre-registered generalisation check" on the confirmation
objects Cats and Horses. That declaration was made **after** the gate-1 failure was already
recorded, i.e. after the condition that mandated stopping was known.

Chronology, from the published ledger (`evidence/ledgers/ledger.json`):

| UTC | event |
|---|---|
| 2026-10-18T00:22:03Z | sprint start |
| 2026-10-18T03:22:27Z | `score_presence_N_Bears` — the scoring whose audit produced the recorded generated-domain failure |
| (between) | `BASELINE_VALIDATION.md` records the unresolved failure; `PROTOTYPE_DECISION.md` declares the additional check |
| 2026-10-18T03:48:32Z | `N_Horses_all80` — 80 images |
| 2026-10-18T03:48:41Z | `N_Cats_all80` — 80 images |
| 2026-10-18T03:50:59Z–03:53:03Z | four scoring jobs for those two arms |

**Prospectively declaring a new check does not override the earlier mandatory stop rule.**
Writing the declaration before executing the runs made the sequence orderly; it did not make
it compliant. The correct action once gate 1 failed was to publish and stop.

These runs are **preserved, not deleted**, as additional descriptive native results, and are
labelled throughout as post-stop. They consumed, as recorded:

- **160 images** (`N_Cats_all80` 80, `N_Horses_all80` 80);
- **0.7743 recorded GPU-hours**, the whole of the ledger's `confirmation` phase
  (0.3219 + 0.3244 generation, 0.0314 + 0.0339 + 0.0308 + 0.0319 scoring).

The frozen protocol is **not** retroactively amended, the chronology is **not** erased, and
no claim of full protocol compliance is made.

---

## 3. Correct retention and causal claims

The two scoring procedures are kept separate below. **Both are judge outputs.** Neither
establishes verified semantic absence or verified semantic correctness.

### 3.1 Supplementary binary presence procedure (the sprint's declared adaptation)

Recorded `retained_ok_strict` counts (intended object answered present **and** rendering
answered not-failed):

| object | related T0 | related N | unrelated T0 | unrelated N |
|---|---|---|---|---|
| Bears | 19/20 | 19/20 | 20/20 | 18/20 |
| Cats | 19/20 | 19/20 | 20/20 | 19/20 |
| Horses | 19/20 | 19/20 | 20/20 | 17/20 |

Under **this procedure and this panel only**, related retention is 19/20 at both T0 and N for
all three targets, with the same single failure (`Bricks_Dogs_seed188165.png`). That
observation is hereby **scoped to this procedure and this panel**. It is not a property of
the method, and it is not corroborated by the other procedure.

### 3.2 Native forced-choice procedure (the release's `eval/llava.py`)

Recorded `class_correct` counts on the retained strata:

| object | related M0 | related T0 | related N | unrelated M0 | unrelated T0 | unrelated N |
|---|---|---|---|---|---|---|
| Bears | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 15/20 |
| Cats | 20/20 | 20/20 | **19/20** | 20/20 | 18/20 | 18/20 |
| Horses | 20/20 | 20/20 | 20/20 | 20/20 | 18/20 | 16/20 |

Two facts here contradict the original report's narrative:

1. **Cats related retention falls 20/20 → 19/20 between T0 and N.** The changed case is
   `Cubism_Rabbits_seed188033.png`, predicted `Rabbits` at T0 and `Frogs` at N.
2. **Unrelated retention already falls at T0**, 20/20 → 18/20, for every object — before any
   erasure is applied.

### Withdrawn

- **"Zero related damage under both instruments"** and "related-object retention exactly at
  the no-erasure (T0) level in every case". False for Cats under the native procedure.
- **"Every unrelated failure appears only at N"** as a cross-instrument claim. False under the
  native procedure, where the unrelated stratum already loses two cases at T0.
- **"No erasure-side intervention could repair it"** and **"only a better transcoder could"**,
  said of the Dogs/Bricks failure. A failure first appearing at T0 localises its **first
  measured occurrence** to substitution. It does not establish that no erasure-side change
  could affect it. Those impossibility claims were not established and are withdrawn.

### Preserved

The raw tables above, and the disagreement between the two procedures, are preserved as
recorded. The procedures **disagree about which retained cases survive**, and that
disagreement is itself a limitation of the measurement, not a resolved finding.

### On the prototype-headroom argument

At 95% observed related retention on the **development** object under the binary procedure,
only five points of headroom remained there. That is an adequate reason to **decline this
prototype on the development evidence**. It did **not**, before the extra runs existed, prove
the prospective +10-point confirmation criterion impossible on the then-unobserved
confirmation objects. The original report's claim that the screen was "arithmetically
impossible" leaned on confirmation-object numbers that the protocol said should never have
been generated. The decision to develop no prototype stands; the impossibility framing does not.

---

## 4. Narrowed measurement, quality and mathematical claims

- **The five Bears disagreements are agent-versus-judge disagreements.** They were not
  adjudicated against independent ground truth. Withdrawn: any assertion of a *verified*
  12.5-point scorer error rate, and any assertion that true Bears suppression was 100%.
- **The two procedures are not independent.** Both call the same LLaVA-1.6-Vicuna-7B
  checkpoint with different prompts. Agreement or disagreement between them does not
  cross-validate either.
- **The eight-case agreement sheet does not generalise.** It was a selected set. It does not
  establish that the judge's errors are exclusively bear-specific, universally conservative in
  direction, or incapable of inflating some other reported quantity.
- **87.5% / 97.5% / 100% relative suppression are preserved as outputs of the frozen binary
  procedure, which is unqualified on this domain.** The protocol's conservative handling of
  uncertain labels bounds *uncertainty*; it does not bound *misclassification*, so these are
  not guaranteed semantic bounds.
- **"Zero severe rendering failures" means the automated judge answered "no" in all 720 score
  rows**, which cover **560 distinct scored originals** (the shared retained panel is re-scored
  once per object and arm: 360 of the 720 rows are retained-stratum rows). It is neither
  independent certification of image quality nor 720 independent images.
- **Shared panels, repeated contexts and one training seed.** The retained panels are identical
  across objects, initial noise is reused by construction, and exactly one transcoder training
  seed was used. Nothing here is an independent replication.
- **Withdrawn: "few-image differences between erasure methods here are not measurable — by
  anyone."** That universal claim is not supported. The supported statement is narrower: *the
  qualification of this binary procedure on this generated domain was not resolved, so this
  sprint could not resolve differences of a few images.*
- **`INTEGRATION_AUDIT.md`'s exactness claim is qualified.** It states that the native update
  makes the decoded contribution of a target latent "exactly" the empty token's contribution.
  That identity holds only where the `clamp(..., min=1e-6)` on the target activation is
  inactive, and only for rows not subject to a conflicting duplicate assignment. The audit
  itself records **7 duplicate row writes of 64 for Bears**, whose scatter winner PyTorch
  leaves unspecified. The disclosed native scatter ambiguity is preserved. The update is **not**
  repaired and no replacement proof is offered.

**Retained in full:** the COCO-control results — 20/20 sensitivity and 20/20 specificity per
object across the 120 annotated photographs. They are genuine and they do not establish
generated-domain validity.

---

## 5. Accounting reconciliation

The original `evidence/ledgers/ledger.json` is **preserved unchanged** as historical evidence.
This section is a separate reconciliation; nothing was repaired.

### Supported recorded totals

| quantity | value | provenance |
|---|---|---|
| sum of rounded per-job `gpu_hours` entries | **3.9766** | 50 job entries, `ledger.json` |
| recomputed from device-seconds | **3.9769444…** | Σ(`seconds` × `gpus`) = 14,317.0 s ÷ 3600 |
| rounded presentation | **3.977** | the figure published in `REPORT.md` |
| phase: baseline / integration | **3.2023** | sum of rounded entries, phase `integration_baseline` |
| phase: prototype / control development | **0** | no job carries this phase |
| phase: confirmation (post-stop, §2) | **0.7743** | sum of rounded entries, phase `confirmation` |
| image ledger total | **563** | Σ `images[].n` |
| sum of job-level `images` fields | **567** | Σ `jobs[].images` |
| scored originals enumerated | **560** | 11 `cases_generated.json` files; equals `ORIGINALS_MANIFEST.json` `n_images`, 560 distinct archive paths |

The two GPU-hour figures differ only by rounding of individual entries (3.9766 vs 3.976944…);
both round to 3.977. No overrun is implied by either.

### Unresolved: the 4-image gap between 567 and 563

Two early smoke jobs carry `images: 2` each with no corresponding entry in the image ledger,
which retains **3** smoke outputs in total:

| job | `images` field | `seconds` | ok | image-ledger entry |
|---|---|---|---|---|
| `smoke_flux_load_gen` | 2 | 16.0 | true | none |
| `smoke_flux_rerun` | 2 | 16.0 | true | none |
| `smoke_flux_ok` | 0 | 17.0 | false | none (note: "NO images decoded") |
| `smoke_flux_retry` | 3 | 59.0 | true | `arm=smoke, n=3` |

**No smoke log file exists** (the smoke runs wrote to the console, not to `logs/`), so the gap
cannot be closed from logs. What the surviving records do show:

- only **one** smoke PNG exists on scratch, and `evidence/smoke_report.json` — written only at
  the end of a completed run — reports `smoke_images_decoded: 3`, attributable to
  `smoke_flux_retry` (59 s);
- the two 16.0-second jobs are **shorter than the measured cost of what their `images: 2`
  would require**: the same report measures 11.6 s model load and 13.59 s per image, so two
  completed generations plus a load would need roughly 38.8 s.

That arithmetic indicates the two early jobs did **not** complete two decodes each, and is
consistent with their `images: 2` being the *requested* `--n-gen 2` argument rather than a
recorded decode count. It is **not** a log of what was decoded. Accordingly:

> **The exact number of images decoded by the two early smoke jobs is UNRESOLVED.** Both
> recorded totals are published (image ledger **563**; job-level fields **567**). No count is
> guessed and no certified range is asserted.

The figure that *is* independently enumerable — and the only one that bears on any reported
result — is **560 scored originals**, confirmed three ways (arm reports, originals manifest,
distinct archive paths). Smoke images were never scored and enter no reported measurement.

### Incomplete failure accounting

`ledger.json` records **two** failure entries (`smoke_flux_ok`, 0.005 GPU-h;
`T0_arms_first_attempt`, 0.0169 GPU-h) and **one** job with `ok: false`. The sprint narrative
additionally described failed introspection attempts, failed evaluator invocations, and an
out-of-memory event during transcoder verification, which have **no distinct failure entries**.
Those attempts' GPU time is included in the recorded job totals (the surrounding jobs were
charged), but the failure *count* is not a complete enumeration of failed attempts.

> **Failure accounting is incomplete**: two logged failure *groups*, not a per-attempt record.
> This is disclosed rather than reconstructed.

### Ledger-writer limitation

`code/ledger.py` performs an unlocked read–modify–write of a single JSON file and stages it
through one shared temporary filename (`ledger.json.tmp`). It imports `fcntl` but never calls
`flock` or `lockf`, so the import provides no serialisation. Several jobs in this sprint ran
concurrently on two GPUs and appended through this path. **Lost updates were therefore
possible.** No actually-lost entry has been established, and none is claimed. The writer is
**not** repaired, not tested, and no missing consumption is invented.

### Scope of the budget statement

Recorded usage is below the sprint's stated ceilings and **no overrun was demonstrated**. That
is a statement about *this ledger's own records only*. It is **not** an independently audited
measurement of total host GPU usage, and git history does not prove host execution history.
No budget or deadline is reset by this amendment.

---

## 6. Evidence limitations and closure

- The published `block-object-sprint-v1` release, its tag, the 835,123,200-byte archive and
  the original-image evidence are **preserved and unchanged**. Nothing was re-uploaded or
  replaced. Verification performed on that archive was a **digest and transfer check**
  (HTTP 200, byte size, and a hash comparison of the first 32 MiB against the local file);
  that is not an independent visual inspection of the images, nor an independent audit of the
  full archive's contents.
- Frozen artifacts are unchanged by this amendment: `RUN_MANIFEST.md`, `cases.json`, every raw
  score/answer/label file, the ledgers, the manifests, the corpora, the images, the executable
  code and the patches.
- `BASELINE_VALIDATION.md`, `PROTOTYPE_DECISION.md` and `INTEGRATION_AUDIT.md` are preserved in
  their original bodies as historical records and carry supersession notices pointing here.
- All visual judgments referenced anywhere in this sprint are **agent judgments**, never
  independent human review.

### Closure

This sprint is closed. The correct final status is **`BASELINE_UNQUALIFIED` — SPRINT CLOSED
WITH CORRECTIONS**. No prototype, control, comparator, rescoring, scorer-development or
measurement-qualification project is authorised by this document. Where the original report
identified qualifying a presence instrument on the synthetic domain as a prerequisite for
future work, that is a **statement of a prerequisite, not authorisation for a new assignment**.
No successor task follows.
