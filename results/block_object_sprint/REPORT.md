# BLOCK object erasure on FLUX.1-dev — a small native-method reproduction attempt

> ## ⚠ CORRECTED — read [`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md) first
>
> **Final status: `BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS.**
>
> This report was originally published with the status `VALIDATED_PLATFORM_NO_PROTOTYPE`.
> That status is **withdrawn**. Gate 1 of the frozen baseline-validation set requires "no
> unresolved material generated-domain scoring failure", and the baseline record explicitly
> documents one. Under the frozen manifest a failed gate required `BASELINE_UNQUALIFIED`,
> publication, and a stop.
>
> This report has been corrected in place. Where any wording here still conflicts with the
> amendment, **the amendment governs**. The data tables are unchanged; several narrative
> claims built on them were overstated and are corrected or withdrawn below.

**Status: `BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS.**

A runnable source-derived native BLOCK implementation was built on FLUX.1-dev from the
authors' pinned release, and both transcoders were trained natively because none are
published. Independent real-photograph scorer controls passed. Gates 2–6 were recorded as
passed at the scope actually tested. **Generated-domain scorer qualification remained
unresolved, so the baseline is unqualified.** No prototype and no control were developed, and
**no qualified method advantage or contribution was established.**

This is a **small native-method reproduction attempt**, not a reproduction of the paper's
benchmark numbers. One transcoder training seed, one generator, 80 frozen cases per object,
retained panels shared across objects, reused initial noise. Nothing here is an independent
replication.

---

## 1. Observations actually supported

All figures below are **judge outputs** from an unqualified measurement on this domain.
They are not verified semantic absence or verified semantic correctness. The two procedures
are reported separately and are **not** independent of each other: both call the same
LLaVA-1.6-Vicuna-7B checkpoint with different prompts.

### Supplementary binary presence procedure (the sprint's declared adaptation)

| object | target present at N (/40) | relative suppression vs M0 | related retention | unrelated retention | severe rendering failure |
|---|---|---|---|---|---|
| Bears (development) | 5 | 87.5% | 19/20 (T0: 19/20) | 18/20 (T0: 20/20) | 0 |
| Cats (post-stop, §3) | 1 | 97.5% | 19/20 (T0: 19/20) | 19/20 (T0: 20/20) | 0 |
| Horses (post-stop, §3) | 0 | 100% | 19/20 (T0: 19/20) | 17/20 (T0: 20/20) | 0 |

The suppression percentages are preserved as outputs of the **frozen but unqualified** binary
procedure. The protocol's conservative handling of *uncertain* labels bounds uncertainty; it
does not bound *misclassification*, so these are not guaranteed semantic bounds.

### Native forced-choice procedure (the release's `eval/llava.py`)

| object | UA | IRA related | IRA unrelated | CRA (style) |
|---|---|---|---|---|
| Bears M0 / T0 / N | 2.5% / 10.0% / 95.0% | 20/20 / 20/20 / 20/20 | 20/20 / 18/20 / 15/20 | 30% / 40% / 40% |
| Cats M0 / T0 / N | 0.0% / 5.0% / 97.5% | 20/20 / 20/20 / **19/20** | 20/20 / **18/20** / 18/20 | 30% / 40% / 32.5% |
| Horses M0 / T0 / N | 0.0% / 7.5% / 100% | 20/20 / 20/20 / 20/20 | 20/20 / **18/20** / 16/20 | 30% / 40% / 37.5% |

M0's measured FLUX style accuracy (CRA 30%) lands near the paper's own Table 2 figure for
Flux (31.8%), a mild consistency signal for the harness.

### What the two procedures say about retention — and where they disagree

- Under the **binary** procedure, related retention is 19/20 at both T0 and N for all three
  targets, always the same case (`Bricks_Dogs_seed188165.png`). **This observation is scoped
  to that procedure and that panel.**
- Under the **native** procedure, related retention for **Cats falls 20/20 → 19/20 between T0
  and N** (`Cubism_Rabbits_seed188033.png`: predicted `Rabbits` at T0, `Frogs` at N), and
  **unrelated retention already falls at T0** (20/20 → 18/20) for every object, before any
  erasure.

**Withdrawn:** "zero related damage under both instruments"; "related-object retention exactly
at the no-erasure T0 level in every case"; and the cross-instrument claim that every unrelated
failure first appears at N. The procedures disagree about which retained cases survive, and
that disagreement is a limitation of the measurement, not a resolved finding.

**Withdrawn:** "no erasure-side intervention could repair it" and "only a better transcoder
could", said of the Dogs/Bricks case. A failure first appearing at T0 localises its **first
measured occurrence** to substitution; it does not establish any impossibility.

**"Zero severe rendering failures"** means the automated judge answered "no" in all **720 score
rows**, which cover **560 distinct scored originals** (the shared retained panel is re-scored
once per object and arm; 360 of the 720 rows are retained-stratum rows). It is not independent
certification of image quality, and it is not 720 independent images.

---

## 2. Qualification and fidelity limitations

### Gate 1 — failed; the baseline is unqualified

Real-photograph controls **passed** and are retained as a genuine result: **20/20 sensitivity
and 20/20 specificity per object** across 120 COCO val2014 photographs labelled by COCO's own
human instance annotations, including 0/10 false positives on related-animal negatives.
**They do not establish generated-domain validity.**

On generated images, the agent's audit disagreed with the judge on 5 of 40 native-BLOCK Bears
images. These are **agent-versus-judge disagreements**, not errors adjudicated against
independent ground truth. **Withdrawn:** any assertion of a *verified* 12.5-point scorer error
rate, and any assertion that true Bears suppression was 100%. The selected eight-case
agreement sheet does not establish that the judge's errors are exclusively bear-specific,
universally conservative in direction, or incapable of inflating some other quantity.

Because this disagreement was recorded as unresolved, gate 1 was not satisfied, and the frozen
manifest's consequence is `BASELINE_UNQUALIFIED`.

### Gates 2–6 — recorded as passed at the scope actually tested

Module identity (`context_embedder` `Linear(4096,3072)`; `time_text_embed.text_embedder`
`PixArtAlphaTextProjection(768→3072→3072)`); activation targets **bit-identical** to the live
modules recomputed on a random sample; 321,920 context token-pairs and **330,000 pooled pairs,
exactly Appendix E.1's number**; the native update writing 57/65,536 and 32/12,288 `W_dec`
rows with encoder weight, encoder bias and decoder bias bit-identical afterwards; save/reload
bit-identical; reference-storage separation; paired per-case CPU initial noise. Details in
[`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md) (historical record; superseded framing).

### Fidelity limitations

- **No transcoder checkpoints are released**, so both were trained natively here, with **one
  training seed**. They are *a* native instantiation, not the authors'.
- **EMA discrepancy (unresolved)**: E.1 specifies an EMA of parameters with decay 0.999;
  `trainer.py` carries the constant but never applies EMA. The code was followed.
- **Learning rate**: E.1 states `1e-4`; the code computes `2e-4 / (n_latents/2^14)^0.5`, which
  is 1e-4 for the context transcoder but **2.31e-4** for the pooled one. The code was followed.
- **The pooled corpus is reconstructed**: the release references
  `prompts/coco_train2014_30k_prompts.txt` but does not ship it; it was rebuilt by a documented
  seeded sample of MS-COCO train2014 captions. The authors' exact caption list is unknown.
- **Evaluator prompt**: Figure 8 shows a numbered list; `eval/llava.py` builds a lettered list
  with a different question string. The code was followed.
- **Empty token**: the paper says the empty token; the code uses the literal prompt `"_"` and
  takes only its first T5 token. The code was followed.
- **Native scatter ambiguity, measured not fixed**: for Bears the update writes 64 context rows
  of which only 57 are unique, so **7 writes are order-ambiguous**.
- **`INTEGRATION_AUDIT.md`'s "exactly" is qualified**: the target-to-empty contribution identity
  holds only where the `clamp(..., min=1e-6)` is inactive and only for rows not subject to a
  conflicting duplicate assignment. The update is not repaired and no replacement proof is given.
- **Two release bugs had to be fixed for the evaluator to run at all**, suggesting it was not
  run end-to-end on the released generator's own filenames. Full list in
  [`INTEGRATION_AUDIT.md`](INTEGRATION_AUDIT.md).

---

## 3. No prototype, and the disclosed post-stop deviation

**No prototype and no control were developed.** No comparator was reproduced. No method
advantage was established or claimed.

**Protocol deviation, disclosed.** `PROTOTYPE_DECISION.md` declared a "pre-registered
generalisation check" on Cats and Horses **after** the gate-1 failure was already recorded —
that is, after the condition requiring a stop was known. Declaring a check prospectively does
not override the earlier mandatory stop rule. Those runs consumed **160 images** and
**0.7743 recorded GPU-hours** (the whole `confirmation` phase) and are **preserved as
additional descriptive native results**, labelled post-stop throughout. The frozen protocol is
not retroactively amended and no claim of full protocol compliance is made. Chronology in
[`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md) §2.

**Withdrawn:** the claim that the +10-point related-retention screen was "arithmetically
impossible". At 95% observed related retention on the development object under the binary
procedure, five points of headroom remained there — enough to justify **declining this
prototype on the development evidence**, but not enough to prove the prospective confirmation
criterion unreachable on objects that, per the protocol, should never have been generated.
The decision to develop no prototype stands; the impossibility framing does not.

A mechanism hypothesis about non-selective latent sets is recorded in
[`PROTOTYPE_DECISION.md`](PROTOTYPE_DECISION.md) as **untested speculation**, supported by no
experiment here and authorising nothing.

---

## 4. Resources, as recorded

Full reconciliation with provenance for every figure: [`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md) §5.
The original ledger is preserved unchanged; nothing was repaired.

| quantity | value | cap |
|---|---|---|
| GPU-hours, sum of rounded job entries | **3.9766** | 48 |
| GPU-hours, recomputed from device-seconds (14,317.0 s) | **3.9769444…** → 3.977 | — |
| — baseline / integration | 3.2023 | 20 |
| — prototype / control development | **0** | 8 |
| — confirmation (**post-stop**, §3) | 0.7743 | 20 |
| Image ledger total | **563** | 1,600 |
| Sum of job-level `images` fields | **567** | — |
| Scored originals enumerated (arm reports = manifest = distinct paths) | **560** | — |

**Unresolved:** two early smoke jobs each carry `images: 2` with no image-ledger entry, while
the smoke records retain 3 outputs in total. No smoke log file exists, so the gap cannot be
closed from logs. Both recorded totals are published; **the exact smoke decode count is
unresolved** and is not guessed. Smoke images were never scored and enter no reported
measurement.

**Incomplete:** the ledger records two failure *groups*, not a per-attempt enumeration;
additional failed introspection and evaluator attempts and a verification OOM have no distinct
entries. **Disclosed, not reconstructed.**

**Ledger-writer limitation:** `code/ledger.py` does an unlocked read–modify–write through a
shared temporary filename while jobs ran concurrently on two GPUs, so **lost updates were
possible**. None was established, and the writer is not repaired.

Recorded usage is below the stated ceilings and **no overrun was demonstrated** — a statement
about this ledger's own records only, not an independently audited measurement of host GPU
usage. Git history does not prove host execution history.

**Efficiency, as measured** ([`evidence/efficiency.json`](evidence/efficiency.json)): cold
preparation 0.863 GPU-h (one-time, shared); per-target edit 0.55 s feature identification plus
0.037 s decoder update; inference overhead vs M0 **+3.31%** (T0) and **+2.79%** (N); peak
memory 33.81 GB → 35.69 GB. One A100-40GB per job, batch 1, 1024×1024, 28 steps. No efficiency
claim is derived from author timings or extrapolation, and none is a performance-advantage claim.

---

## 5. Where the evidence is

| what | where |
|---|---|
| **authoritative closeout and corrections** | [`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md) |
| working freeze (cases, seeds, evaluator config, labelling rules) | [`RUN_MANIFEST.md`](RUN_MANIFEST.md), [`cases.json`](cases.json) |
| baseline-validation record (historical; superseded framing) | [`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md) |
| prototype decision (historical; superseded framing) | [`PROTOTYPE_DECISION.md`](PROTOTYPE_DECISION.md) |
| release bugs and paper/code discrepancies (historical; one claim qualified) | [`INTEGRATION_AUDIT.md`](INTEGRATION_AUDIT.md) |
| patched code + exact pins | [`code/`](code/), [`patches/`](patches/), [`env/`](env/) |
| raw evaluator answers, per-case labels | [`evidence/scores/`](evidence/scores/), [`evidence/native_eval/`](evidence/native_eval/) |
| scorer validation (120 COCO photos, raw answers) | [`evidence/scorer_validation/`](evidence/scorer_validation/) |
| agent audit sheets and keys | [`evidence/audit/`](evidence/audit/) |
| fixed-order grids of every scored image | [`grids/`](grids/) |
| ledgers (preserved unchanged) | [`evidence/ledgers/`](evidence/ledgers/) |
| full-resolution originals (560 PNGs) | release `block-object-sprint-v1`; see [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) |

Model weights and credentials are not published. Archive verification was a digest and
transfer check, not an independent visual or full-archive-content inspection.

---

## 6. Closure

The sprint is closed at **`BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS**.

**Withdrawn:** the universal claim that few-image differences between erasure methods on this
benchmark "are not measurable — by anyone." The supported statement is narrower: *the
qualification of this binary presence procedure on this generated domain was not resolved, so
this sprint could not resolve differences of a few images.*

Qualifying a presence instrument on generated, stylised images would be a **prerequisite** for
any future erasure comparison of this kind. That is a statement of a prerequisite and **not
authorisation for a new assignment**. No prototype, control, comparator, rescoring,
relabelling, scorer-development or successor task is authorised by this report.
