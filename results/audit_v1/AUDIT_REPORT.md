# AUDIT_REPORT — AUDIT-01

**Scope.** Audit of the two-seed snapshot
[`a9de625`](https://github.com/bijay299/XXResearch/tree/a9de625), correction of
its narrative claims, an evaluator-audit packet for the researcher, an artifact
inventory, and a proposed matched-effectiveness protocol for central review.

**Authorisation observed.** CPU analysis, documentation and implementation
preparation only. No GPU training, sampling, detector run or sweep was launched;
nothing was downloaded; no image was regenerated; no external party was
contacted. All four A100s were idle and no job was running throughout, so
nothing was interrupted.

**Branch.** `research/audit-and-matched-effectiveness-protocol`, branched from
`pilot/sequential-interference-and-l2` at `a9de625`.

---

## 0. What differs from `a9de625`

**Nothing.** At the start of this audit the working tree was clean and `HEAD`
was exactly `a9de625`. There are no newer commits on the experimental branch and
no uncommitted work. Every claim in this report refers to that snapshot, and the
raw outputs it describes are untouched.

## 1. Headline: the numbers are right, the narrative was not

Every rate and every headline point estimate in the committed summaries
reproduces **exactly** from the raw per-image detector predictions.

| | rate cells | contrast values | mismatches |
|---|---|---|---|
| seed 17 | 126 | 387 | **0** |
| seed 29 | 126 | 387 | **0** |

Completeness and identity are also exact, with no judgement calls required:

- 12 checkpoint slots × 280 rows; every slot has 280 unique
  `(category, prompt_id, gen_seed)` identities and 280 unique image hashes.
- **0** duplicate rows, **0** missing rows, **0** rows absent from the frozen
  manifest. **No de-duplication or row-dropping rule was applied to anything**,
  because none was needed.
- 7 categories × 10 distinct prompt **texts** (70 texts) × 4 generation seeds
  = 280 prompt × generation-seed pairs per slot, 140/140 literal/paraphrase.
- Manifest `frozen_before_any_editing: true`, sha256 `0020c81c4a4d…`, 0 exact
  collisions and 0 near-duplicates against 999 training strings.
- **0** images byte-shared between the two seeds other than the known M0
  reuse, so no file was accidentally shared and the seed-29 table is not a copy
  of seed 17's. **That is a file-level fact, not statistical independence** —
  see the note below.

So the nine claim corrections in
[`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md) are all corrections of
**language and of which quantity was reported**. None is an arithmetic fix.

> **What the two runs do and do not establish.** They differ **only in the
> training RNG seed**. They share one pretrained M0 backbone, byte-identical
> anchor caches (`anchor_seed: 17` for both), the same concepts and anchor
> mappings, the same 70 frozen evaluation prompt texts and four generation
> seeds, and the same detector. They are two draws from **one training pipeline
> on one base model**, bounding *training-seed* variability only — not
> variability over models, concepts, prompts, anchors or detectors.

> **A second round of corrections.** Central review found eight overstatements
> in this audit's own first write-up (commit `f7f4ba8`), now corrected in
> [`CLAIM_CORRECTIONS_V2.md`](CLAIM_CORRECTIONS_V2.md): the independence claim
> above, the direct contrast described as "isolating" the L2-SP effect, the
> non-replication attributed "largely" to the parent, the newest-target result
> stated without its target, "+7.5 pp" called a resolution floor, "280 prompts"
> for 70 prompt texts, upstream stopping described as a certified per-arm
> maximum, and three inference overreaches in the protocol. No measured value
> changed.

## 2. Corrected counts

| quantity | corrected value | as previously stated |
|---|---|---|
| distinct checkpoint evaluations | **11** | 11 ✓ |
| newly generated evaluation images | **3080** | "3360 evaluation images" ✗ |
| table rows across both seeds | 3360 | 3360 ✓ |
| training runs with a saved final delta | 10 | 10 ✓ |

`eval_seed29/M0` is a **symlink** to `eval/M0`; the detection files are
byte-identical and all 280 image hashes are shared. M0 is one evaluation set
appearing under both seed labels. It must be counted once in any generated-image
total, and its rows carry **no** cross-seed replication information.

## 3. The contrast that was missing

The committed `contrasts.json` contained no direct L2-SP-versus-unregularised
comparison at all. Every L2 statement was assembled from two separately
parent-referenced quantities. Adding the direct contrast matters because the
*measured* parent term cancels out of the estimator:

```
new-target suppression  S(X) = D_new(MA) − D_new(X)
    S(no-L2) − S(L2)         = D_new(L2) − D_new(no-L2)

historical cat recovery R(X) = D_cat(X) − D_cat(MA)
    R(L2)   − R(no-L2)       = D_cat(L2) − D_cat(no-L2)
```

So the direct child-vs-child estimator does not inherit the sampling noise of
the measured `D(MA)` term, while the parent-referenced estimands do.

**Two things this does not mean.** It does **not** make the contrast
independent of MA: both children were trained starting from that same parent,
so their states still depend on it. And it is a **different estimand**, not a
parent-free version of recovery — `D_cat(child) − D_cat(MA)` asks how far a
child moved from its own parent; `D_cat(L2) − D_cat(no-L2)` asks how the two
arms differ from each other. Neither substitutes for the other.

Computing it ([`CORRECTED_TABLES.md`](CORRECTED_TABLES.md) §2) changes the
picture in two ways.

**(a) The newest-target residual difference is the most stable quantity in the
snapshot.** `D_new(L2) − D_new(no-L2)` — how much more of the **newest
requested target** (dog in the B branch, sandwich in the C branch) the L2-SP arm
left standing — is **+40.0 to +47.5 pp** in *every* cell: both seeds, both
branches, all three thresholds. Nothing else replicates that tightly.

This is a statement about deletion strength on the newest request. **It carries
no information about the cat history** and must not be read as one.

**(b) The "non-replication" is half parent movement and half child movement.**

| quantity, sandwich branch | seed 17 | seed 29 | agree? |
|---|---|---|---|
| recovery vs MA (parent-referenced) | `+15.0 [+2.5,+27.5]` | `−5.0 [−12.5,0.0]` | **no** |
| `D_cat(MAC_L2) − D_cat(MAC)` (parent-free) | `+7.5 [0.0,+17.5]` | `+2.5 [−7.5,+15.0]` | **yes** |

The child-vs-child quantity agrees in direction and credibility class across
both runs; the parent-referenced one does not. Decomposing the −20.0 pp change
in recovery at t=0.5 shows why, and shows that it is **not** mainly a parent
effect:

| component | seed 17 | seed 29 | contribution to −20.0 pp |
|---|---|---|---|
| child `D_cat(MAC_L2)` | 32.5% | 22.5% | **−10.0 pp** |
| parent `D_cat(MA)` | 17.5% | 27.5% | **−10.0 pp** |

The child's cat presence falls 10 pp while the parent's rises 10 pp — **each
accounts for exactly half**. An earlier version of this report said the gap was
"largely" the parent; that was wrong.

**Separately, the parent's own across-seed gap is a paraphrase effect.**
`D_cat(MA)` by prompt family:

| threshold | literal s17 / s29 | paraphrase s17 / s29 |
|---|---|---|
| 0.3 | 15.0 / 15.0 **identical** | 25.0 / 40.0 |
| 0.5 | 15.0 / 15.0 **identical** | 20.0 / 40.0 |
| 0.7 | 15.0 / 15.0 **identical** | 20.0 / 35.0 |

The two runs produced *identical* literal-prompt parent rates at all three
thresholds, so the parent's entire across-seed gap sits on paraphrases — where
5 prompt texts per family resolve least. This describes the **parent alone**;
per the decomposition above it does not make the non-replication a parent
effect.

## 4. What L2-SP bought, read against what it cost

Direct contrasts at t=0.5, all prompts, L2 minus unregularised:

| branch | new-target residual | bird | horse | chair | bicycle |
|---|---|---|---|---|---|
| dog, s17 | **+40.0** | **+40.0** | +2.5 ns | +5.0 ns | +2.5 ns |
| dog, s29 | **+40.0** | **+35.0** | +2.5 ns | +2.5 ns | +0.0 ns |
| sandwich, s17 | **+42.5** | −2.5 ns | +0.0 ns | +5.0 ns | +0.0 ns |
| sandwich, s29 | **+45.0** | −2.5 ns | +0.0 ns | +0.0 ns | +0.0 ns |

Read per branch:

- **Dog branch** — a genuine, replicated trade: ~40 pp of dog deletion given up
  for ~35–40 pp of bird retention. Both legs exclude zero in both runs.
- **Sandwich branch** — ~42–45 pp of sandwich deletion given up for **nothing
  measurable** on any retained category. Every retention contrast is within
  ±5 pp with an interval including zero.

**The caveat that governs both rows.** Retention is only comparable between arms
that deleted a comparable amount, and these did not. At L2-SP 25000 the sandwich
arm achieved only `+7.5 pp [+0.0, +17.5]` of suppression. Comparing its
retention against an arm that achieved +50 pp is not a preservation result; it
compares an arm that deleted with one that barely did.

That is why the protocol declares the sandwich branch **infeasible for matched
comparison at this coefficient**: the reason is that **L2-SP suppression there
is too weak for the question** — a matched comparison needs a meaningful
deletion level to match *to*, and +7.5 pp does not supply one. It is not that
7.5 pp is unmeasurable. With 10 prompt clusters of 4 images the representable
step is 2.5 pp, so 7.5 pp is a real three-step value; what is true separately is
that its interval reaches zero, so it is imprecise.

## 5. Threshold sensitivity the committed report did not show

Everything headline was reported at t=0.5 only. At 0.3/0.7:

- The direct L2 new-target contrast is stable (+40.0 to +47.5 pp everywhere) —
  the central finding does not depend on the threshold.
- `MAB` recovery, labelled "robust", **changes credibility class at t=0.3**
  (`−17.5 [−37.5, 0.0]` in seed 17).
- `MAC_L2` recovery changes class at all three thresholds between seeds, so that
  non-replication is not a t=0.5 artifact — but its *direction* reading is
  (seed 29 is `+0.0 [−7.5,+7.5]` at t=0.3, unresolved rather than opposite).

Full grid: [`recomputed_rates.csv`](recomputed_rates.csv), 756 rows.

## 6. Interval limitations, stated once and relied on throughout

- The evaluation set is **70 distinct prompt texts** (7 categories × 10 texts)
  × 4 generation seeds = **280 prompt × generation-seed pairs** per checkpoint.
  It is not 280 prompts; an earlier version said so and overstated prompt
  diversity four-fold.
- The resampling unit is the **prompt text** (10 per category), carrying all
  four of its generation seeds in or out of a draw together. One resampled
  prompt list indexes both arms, preserving the pairing the shared manifest
  creates. 10 000 draws, 95% percentile interval.
- Training seeds are **never** pooled or resampled jointly. Every interval is
  sampling uncertainty over prompts **within one training run**.
- With 10 clusters of 4 images the resolution is coarse. **An interval
  including zero is not evidence that the effect is zero**, and a change of
  significance label between two runs is **not** a significant difference
  between them.
- **Lack of significance is not equivalence.** None of these intervals was
  designed to rule out an effect of a stated size; doing that requires the
  interval to lie wholly inside a pre-declared margin, which is a separate and
  stronger claim.
- No multiplicity correction is applied and many contrasts are reported. Each
  interval is descriptive, not a test.
- n=2 training runs supports a **direction check only** — no population
  variance, and no "robust".

## 7. Code defects found and fixed

All CPU-side; verified by a mock test, never against a live GPU job.

| # | file | defect | fix |
|---|---|---|---|
| 1 | `run_seed.sh` | `echo "[done] … exit=$?"` was the function's last statement, so `train()` returned the **echo's** status. Every training failure returned success. | capture `rc=$?` immediately, return it |
| 2 | `run_seed.sh` | `wait $P1 $P2` reports only the **last** pid's status, so a first-child failure vanished. No `set -e`. | `wait` each pid separately, collect into `FAILED[]` |
| 3 | `run_seed.sh` | evaluation guard was `[ -f detections.jsonl ]`. `detect.py` opens with mode `"w"` and appends per batch, so a killed run leaves a **short** file that existence-only logic makes permanent. | require the full 280-row count; quarantine a partial file and re-run |
| 4 | `run_seed.sh` | completion sentinel was `[ -f train_report.json ]` — a report truncated mid-write counted as complete. | require parseable JSON with `finished_utc` and a non-empty `delta.bin` |
| 5 | `run_seed.sh` | only `MA` was guarded; a failed child still reached evaluation and `=== SEED DONE ===` | re-verify every artifact at the end; print per-checkpoint OK/INCOMPLETE and **exit non-zero** |
| 6 | `run_analysis.sh` | `--eval_root` hard-coded to `${SEQ_ROOT}/eval` (**seed 17**) for any seed argument, so `run_analysis.sh 29` would have analysed seed 17's images and labelled them seed 29 | derive the eval root from the seed; also refuse to aggregate an incompletely evaluated seed |
| 7 | `make_report.py` | `--training_seed` defaulted to 17; `"(seed 17)"` and `"One training seed (17)"` were string literals and **leaked into the seed-29 report** | take the seed from the contrasts file and refuse on contradiction; read the anchor seed from `cache_meta.json` |
| 8 | `make_report.py` | finding 3 asserted the L2 narrative as fixed text regardless of input; "**No recovery**" printed for non-resolution | derive direction and magnitude from the data; "no increase was resolved at this precision" |
| 9 | `compare_seeds.py` | replication warning permitted a false-positive reading | enumerate what a credibility-class change does **not** license |

`run_remaining.sh` is left unmodified and marked **SUPERSEDED** in its header:
it is the historical record of how the seed-17 core was finished and carries
defects 1 and 3, so the header says so and points at `run_seed.sh`.

### Second round: the row-count guards were still not enough

Central review showed the AUDIT-01 shell sentinels, which only checked file
existence, parseability and row **count**, accepted three artifacts that are
plainly unusable. All three were reproduced before being fixed:

| accepted by the v1 guards | why it got through |
|---|---|
| a `delta.bin` containing **plain text**, with a valid-looking report and an arbitrary non-empty SHA | the guard checked that the file was non-empty, never that it was a model |
| a `detections.jsonl` of **280 identical rows** | 280 lines satisfied a count check; the rows collapse to one identity |
| a `detections.jsonl` of **280 malformed lines** | never parsed; only counted |

**Fix: contract validation replaces existence and counting.**
`scripts/seq/validate_stage.py` reads the artifacts and checks them against the
frozen contracts **and against the settings of the request being validated**, so
a stale report from a different configuration cannot satisfy a completion check.

*Training:* report schema; run identity (checkpoint, seed, parent, target,
anchor, L2 weight, cross-checked against `effective_hyperparameters`); requested
step count, epoch capacity, and a runtime cross-check; parent lineage by
recomputed hash; the checkpoint's own hash **recomputed**, not trusted from the
report; exact tensor names and shapes against
`configs/checkpoint_contract.json` (32 kv-xattn tensors, 19,169,280 params);
dtype; finiteness; and a CPU `torch.load`.

*Evaluation:* every row parsed; required fields; all three thresholds present;
**exact set equality** of `(category, prompt_id, gen_seed)` identities against
the frozen manifest, with duplicates, unscored and extra identities reported
separately; per-axis coverage; distinct image hashes; checkpoint label; the
companion `image_report`/`detect_report` with consistent counts; manifest
identity; generation settings against `configs/generation_settings.json`; and
the generating checkpoint's hash. **The expected count is derived from the
manifest**, never hard-coded at 280.

*Shared M0 contract:* reuse must be **declared** with `--allow_shared_images`.
Inferring it from the layout is impossible anyway — `eval_seed29/M0` is a
symlink, so resolved paths match and only a literal path comparison plus a
symlink check detects the reuse.

*Other fixes:* completion metadata is written **atomically** (temp + `fsync` +
`os.replace`) and **only after** validation passes; freshly produced invalid
artifacts are quarantined alongside a reason file, not only stale ones found at
entry, and **nothing is ever deleted**; a check that cannot be *performed*
(missing contract, torch unavailable) is reported `UNCHECKED` and **fails** the
stage rather than being skipped; `evaluate()` refuses to evaluate a checkpoint
whose training does not validate; and the GPU selector is a seam
(`GPU_SELECT`) so it can be mocked.

**The seed-17 path split-brain.** `run_seed.sh` wrote `eval_seed17` while
`run_analysis.sh` read `eval` — a seed-17 re-run and its analysis addressed
different directories. Both now call one shared policy, `seq_resolve_eval_root`
in `exp_config.sh`: the legacy `eval/` is **preserved**, used with a printed
note when it is the only layout present, and when **both** layouts exist the
situation is **refused** rather than resolved by existence. `SEQ_EVAL_ROOT_OVERRIDE`
allows a deliberate choice. Nothing is migrated or deleted.

**`run_analysis.sh` masking.** Parameter-movement, grid and required-copy
failures could pass unnoticed. Now every required stage runs through a `req`
wrapper that aborts; products are built in an **isolated staging directory** and
published only after all required stages succeed and every required product is
present and non-empty; required copies fail hard; and PDF/SVG are declared
**optional** and reported as absent rather than silently skipped.

### Tests

| suite | cases | what it covers |
|---|---|---|
| `test_launcher_guards.sh` | **35** | a **regression** check that the weak patterns have not crept back and the validator is wired in — see the note below |
| `test_validate_stage.py` | **78** | malformed / duplicate / missing / extra identities; stale configuration, parent and manifest; wrong checkpoint hash and tensors; NaN; incomplete steps; truncated reports; the shared-M0 contract; manifest-derived counts; marker atomicity — plus revalidation of **all 22 real saved artifacts** |
| `test_launcher_e2e.sh` | 18 groups | clean run, idempotent re-run, failure of the **first** and of the **second** background child, both children of one wave, MA failure **with stale artifacts present**, MA exiting 0 with unvalidatable output, stale configuration, short trajectory, partial detections with quarantine, all-identical rows, settings drift, generation and detection failures, analysis refusing an invalid seed, analysis-stage failure publishing nothing, the seed-17 legacy/new layout policy, a busy GPU |

**Every GPU command is mocked** in the E2E suite — `accelerate`, the three GPU
python entry points, the GPU selector and `nvidia-smi` are shadowed on `PATH`
or through the `GPU_SELECT` seam, each logging to a call log.
`CUDA_VISIBLE_DEVICES` is forced empty throughout. The suite asserts
afterwards that every GPU-bound invocation appears in the call log, that no mock
saw a numeric CUDA device, that no real-sized `delta.bin` was ever produced, and
that it ran against a temporary `SEQ_ROOT` rather than the real `/data` tree.
Analysis output is published to a sandbox via `SEQ_REPO_OUT_OVERRIDE`, and the
committed `results/seq17` and `results/seq29` are verified byte-identical
afterwards.

> **The old guard suite had to be rewritten, not kept.** It worked by
> `sed`-extracting the two shell sentinels from `run_seed.sh`. Once those were
> replaced by validator calls, the extraction found nothing — and because an
> undefined shell function also returns non-zero, **every "expect failure" case
> passed vacuously**: the suite reported 9 passes while testing nothing. That is
> a worse failure mode than a red test, so the file was rewritten as a
> regression check (35 assertions: the validator is called with pinned expected
> settings, the discarded-`$?` and multi-pid-`wait` patterns are absent, the
> hard-coded seed-17 eval root is gone, nothing deletes detection evidence, and
> the validator fails closed). Behavioural coverage now lives in the other two
> suites.

**Real artifacts revalidated on CPU:** all 10 training runs and all 12
evaluation slots pass the new contract validation. **One legacy metadata gap was
found and is reported separately, not worked around:** `l2sp_trace.steps_recorded`
is a *regularisation-loss* trace, populated only when `l2sp_weight > 0`, and is
legitimately `0` on every unregularised arm. So for an unregularised run this
schema records **no direct count of completed optimizer steps** — only
`iterations_requested` (an input, not an outcome) and a runtime-implied
estimate. The validator uses the trace as step evidence only for regularised
arms and says so in its output; the next trajectory should record an explicit
`optimizer_steps_completed` field.

## 8. Evaluator audit packet for the researcher

Built from existing images only; nothing generated. 236 items over cat, bird,
dog and sandwich, both training seeds.

| set | n | construction | permitted use |
|---|---|---|---|
| **R** random | 176 | stratified sample over 88 `(seed, checkpoint, category, family)` cells, `rng_seed=20261009`, per-cell weights recorded | the **only** set for an overall detector error rate, over its strata, with the recorded weights |
| **D** diagnostic | 60 of 494 eligible | **enriched**: hit at 0.3 but not 0.7, and MA→child verdict flips at the same prompt and generation seed | characterising failure modes **only** |

Hidden from the annotator: checkpoint, training seed, detector score, detector
verdict, and set membership. Shown: the image and the category to judge —
without the category there is no question to answer. Order shuffled once under
the fixed seed; blinded ids are `sha256(salt|seed|ckpt|image_sha256)[:10]`.

The sheet records **target presence**, **ambiguity**, and — in separate columns
— **visible quality** and **context** concerns, so a quality concern cannot be
recorded as a presence judgement.

**SET D is biased upward by construction** and must never be reported as an
overall error rate, nor pooled with SET R. M0 is sampled once, not twice, since
it is shared between the seed tables.

Packet: `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/packet/`
(236 images, 26 MB — **not** committed).
Key: `…/annotation_v1/key/KEY_do_not_open_before_annotating.csv`, stored
outside the packet directory.
Committed: [`annotation_packet/`](annotation_packet/) — empty sheet, manifest,
README.

**No human annotation has been performed.** Both this sheet and the
pre-existing seed-17 grid sheet are empty by design. Every visual statement in
the current reports is AI inspection, labelled as such, and is not human review.

## 9. Artifact inventory — the binding constraint

Full detail in [`ARTIFACT_INVENTORY.md`](ARTIFACT_INVENTORY.md).

Reusable: all 10 final `delta.bin` (73 MiB each, distinct hashes, parents
verified at deviation 0.0), 10 complete training reports, all 11 per-image
prediction files with full boxes and scores, the frozen manifest, both anchor
caches (`anchor_seed: 17`, shared byte-identically), the calibration set and
the detector.

**Absent, and not reconstructed:** intermediate checkpoints — **none exist**,
for any of the ten runs (`eval_interval: null`, no periodic dump). Also absent:
optimizer state, scheduler state, RNG state, per-step L2-SP series, any
coefficient other than 25000, human annotation, UnlearnCanvas assets.

- **Mid-trajectory continuation: impossible** for every run.
- **Re-run from scratch: fully specified** (`upstream_argv` and all effective
  hyperparameters recorded) but **bit-exactness unverified** — no run was ever
  repeated under a fixed seed and no determinism flags are set. *Inference*
  determinism **is** verified (no-op reload: 8/8 byte-identical).

So an early-stopping or smaller-update control **cannot** be built from what
exists. It requires a new trajectory. That is the one binding constraint on the
next experiment.

## 10. Proposed next step

[`docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`](../../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md)
— a **proposal for review, not a launched campaign**. It compares unregularised
early stopping at a *matched* deletion level against existing L2-SP, reusing the
seed-specific MA parents and the existing L2 finals, and adding only the
unregularised trajectories with checkpoint dumps.

The run matrix is the bounded one specified for review: dog branch only, two
seeds, a **full fixed 100-step scan** (no bisection), reusing MA17/MA29 and the
existing L2=25000 children. No lower-L2 stage and no sandwich coefficient
search.

Estimated **3.095 GPU-hours** with 25% contingency (subtotal 2.476 = 0.461
training + 1.584 generation/detection + 0.431 setup), from measured throughput
(0.8303 s/optimizer step, 1.0083 s/image generation, 0.0718 s/image detection).
**The 0.431 setup term may be partly double-counted** — the measured
per-image figure already amortises pipeline setup — and is kept as an upper
bound; the protocol discloses this and gives 2.045 GPU-h as the subtotal if
development checkpoints are scored within one process. Storage peak **1.92 GiB**
(20 dumps = 1.43 GiB). Wall time is separate, ≈1.45 h on two idle GPUs, and
depends on shared-GPU availability.

Draft prompt manifests are built, hashed and disjointness-checked on CPU:
**20 dog development prompt texts** (80 pairs/checkpoint) and **140 frozen test
prompt texts** (560 pairs/checkpoint), 0 exact collisions and 0 near-duplicates
across all five pairings against 400 anchor training strings, generation-seed
sets pairwise disjoint and disjoint from the training seeds. Shared template
structure is measured and reported rather than assumed absent.

The protocol claims **no novelty**. Upstream CUIG implements
**threshold-or-patience** early stopping
(`Regularizers/Simultaneous/simultaneous.py:check_early_stopping` — stop when
`ua >= stop_threshold`, default 99.0, **or** when patience is exhausted), wired
into `ConAbl/train_conabl.py` and gated on `--eval_interval`. Three details
matter and are stated in protocol §2 before any distinction is drawn: it is not
a certified per-arm maximum (the weights saved at stop time are the current
ones, not `best_ua`'s); the scored concept pool comes from the **caller** via
`--anchor_target_concepts`, so the helper is no evidence for newest-target-only
scoring; and `--eval_interval` is passed in six `BashScripts/Simultaneous/**`
scripts and in **no** `BashScripts/Sequential/**` script in the pinned tree.
No new regularizer or learned controller is built.

## 11. Unresolved

1. **Training reproducibility is untested.** Needed before any claim rests on a
   re-run trajectory. The protocol gets a cheap check on it, specified as a
   **tensor-level** comparison: a `delta.bin` hash mismatch alone is a
   serialisation observation, not a training difference, and tensor equality is
   in turn a different claim from behavioural equivalence.
2. **Detector validity is unmeasured.** No human labels exist, so detector
   error rate, and whether it differs between literal and paraphrase prompts or
   across checkpoints, is unknown. The packet is built; it needs a human.
3. **Anchor sharing and semantic similarity remain confounded.** dog differs
   from sandwich in both. This audit cannot separate them and the proposed
   diagnostic does not attempt to.
4. **Why the parent diverged only on paraphrases** is unexplained.
5. **n=2.** No population variance over training runs; direction checks only.
6. **One L2-SP coefficient.** Whether any single value serves both branches is
   untested.
7. **No development split exists.** The 280-prompt pilot set has now informed
   hypotheses and is exploratory from here on. Configuration selection needs a
   separate development set and confirmatory claims need a newly frozen one —
   both specified in the protocol.
8. **UnlearnCanvas still blocked**; no UA/IRA/CRA and no baseline reproduction.
9. **No direct completed-step counter exists for unregularised runs** in the
   current report schema (see §7). Step completion is inferred from a runtime
   cross-check only.
10. **The setup term in the cost estimate is not reconciled** against an actual
    implementation and may be partly double-counted; the protocol discloses the
    range (2.045–2.476 GPU-h subtotal) rather than resolving it.
11. **Shared prompt-template structure** means prompt clusters are not fully
    independent, so bootstrap intervals are mildly optimistic. Measured and
    reported for the draft manifests; it applies to the pilot set too and is not
    corrected for.
12. **Monotonicity of suppression in optimizer steps is untested**, which is why
    the proposed scan is a full fixed scan rather than a bisection.

---

## Files

| file | contents |
|---|---|
| [`AUDIT_REPORT.md`](AUDIT_REPORT.md) | this document |
| [`CORRECTED_TABLES.md`](CORRECTED_TABLES.md) | corrected counts, direct contrasts, both estimands, effectiveness context |
| [`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md) | 9 corrections to the pilot narrative, plus what survived |
| [`CLAIM_CORRECTIONS_V2.md`](CLAIM_CORRECTIONS_V2.md) | **8 corrections to this audit's own first write-up** |
| [`ARTIFACT_INVENTORY.md`](ARTIFACT_INVENTORY.md) | checkpoints, logs, caches, what is absent, reproducibility, contracts |
| [`integrity.json`](integrity.json) | counts, identities, hashes, duplicates, M0 reuse, symlink audit |
| [`direct_contrasts.json`](direct_contrasts.json) | all contrasts at 0.3/0.5/0.7 × 3 families × 2 seeds |
| [`crosscheck.json`](crosscheck.json) | re-derived vs committed, 0 mismatches |
| [`recomputed_rates.csv`](recomputed_rates.csv) | 756 rows: seed × checkpoint × category × family × threshold |
| [`protocol_cost_model.json`](protocol_cost_model.json) | the bounded run matrix and its full resource arithmetic |
| [`draft_manifests/`](draft_manifests/) | drafted dev + frozen-test manifests, hashes, disjointness report, `FREEZE.md` |
| [`annotation_packet/`](annotation_packet/) | empty label sheet, manifest, README, and [`ACCESS.md`](annotation_packet/ACCESS.md) for the PI |
| [`preserved_originals/`](preserved_originals/) | pre-audit narrative (`a9de625`) and, in `v1/`, this audit's first round (`f7f4ba8`) |

Code added or changed:

| file | purpose |
|---|---|
| `configs/checkpoint_contract.json`, `configs/generation_settings.json` | the frozen contracts validation checks against |
| `scripts/seq/validate_stage.py` | contract validation for a training or evaluation stage |
| `scripts/seq/compare_checkpoints.py` | CPU tensor-level checkpoint comparison |
| `scripts/seq/build_draft_manifests.py` | builds, hashes and disjointness-checks the draft manifests |
| `scripts/seq/protocol_cost_model.py` | re-derives the resource estimate from measured throughput |
| `scripts/seq/exp_config.sh` | `seq_resolve_eval_root` — the one shared evaluation-path policy |
| `scripts/seq/run_seed.sh`, `run_analysis.sh` | validation-gated, failure-propagating, staged publication |
| `scripts/seq/test_validate_stage.py`, `test_launcher_e2e.sh`, `test_launcher_guards.sh` | the CPU test suites |

Reproduce everything (CPU only; no GPU, no network):

```bash
python3 scripts/seq/audit_evidence.py        # integrity + contrasts + crosscheck
python3 scripts/seq/audit_tables.py          # CORRECTED_TABLES.md
python3 scripts/seq/build_draft_manifests.py # draft manifests + disjointness
python3 scripts/seq/protocol_cost_model.py   # resource estimate
python3 scripts/seq/build_annotation_packet.py --copy_images

# tests
bash    scripts/seq/test_launcher_guards.sh  # 35 regression assertions
python3 scripts/seq/test_validate_stage.py   # 78, incl. all 22 real artifacts
bash    scripts/seq/test_launcher_e2e.sh     # 68 assertions over 18 groups
```
