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
- 7 categories × 10 prompts × 4 generation seeds per slot, 140/140
  literal/paraphrase.
- Manifest `frozen_before_any_editing: true`, sha256 `0020c81c4a4d…`, 0 exact
  collisions and 0 near-duplicates against 999 training strings.
- **0** images byte-shared between the two seeds other than the known M0
  reuse — the two training runs are genuinely independent.

So the nine claim corrections in
[`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md) are all corrections of
**language and of which quantity was reported**. None is an arithmetic fix.

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
parent-referenced quantities. That matters because the parent term cancels:

```
new-target suppression  S(X) = D_new(MA) − D_new(X)
    S(no-L2) − S(L2)         = D_new(L2) − D_new(no-L2)

historical cat recovery R(X) = D_cat(X) − D_cat(MA)
    R(L2)   − R(no-L2)       = D_cat(L2) − D_cat(no-L2)
```

The direct child-vs-child contrasts therefore carry **no** `D_cat(MA)`
uncertainty, while the parent-referenced ones inherit all of it. Computing them
([`CORRECTED_TABLES.md`](CORRECTED_TABLES.md) §2) changes the picture in two
ways.

**(a) The L2 cost is the most stable quantity in the study.**
`D_new(L2) − D_new(no-L2)` is **+40.0 to +47.5 pp** in *every* cell — both
seeds, both branches, all three thresholds. Nothing else in the snapshot
replicates that tightly.

**(b) The "non-replication" was largely in the reference term, not the method.**

| quantity, sandwich branch | seed 17 | seed 29 | agree? |
|---|---|---|---|
| recovery vs MA (parent-referenced) | `+15.0 [+2.5,+27.5]` | `−5.0 [−12.5,0.0]` | **no** |
| `D_cat(MAC_L2) − D_cat(MAC)` (parent-free) | `+7.5 [0.0,+17.5]` | `+2.5 [−7.5,+15.0]` | **yes** |

The quantity that isolates the L2-SP effect agrees in direction and credibility
class across both runs. The quantity that subtracts a moving parent does not.

**Why the parent moves — and it is specifically a paraphrase effect.**
`D_cat(MA)` by prompt family:

| threshold | literal s17 / s29 | paraphrase s17 / s29 |
|---|---|---|
| 0.3 | 15.0 / 15.0 **identical** | 25.0 / 40.0 |
| 0.5 | 15.0 / 15.0 **identical** | 20.0 / 40.0 |
| 0.7 | 15.0 / 15.0 **identical** | 20.0 / 35.0 |

The two runs produced *identical* literal-prompt parent rates at all three
thresholds. The entire cross-seed parent gap sits on paraphrases — exactly where
10 prompts per family is least precise.

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
arm retains only `+7.5 pp [+0.0, +17.5]` of measured suppression — at or below
the resolution floor. Comparing its retention against an arm that achieved
+50 pp is not a preservation result; it compares an arm that deleted with one
that essentially did not. This is precisely the confound the proposed diagnostic
exists to remove, and it is why the protocol declares the sandwich branch
**infeasible for matched comparison at this coefficient** rather than scoring it.

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

- The resampling unit is the **prompt** (10 per category), carrying all four of
  its generation seeds in or out of a draw together. One resampled prompt list
  indexes both arms, preserving the pairing the shared manifest creates.
  10 000 draws, 95% percentile interval.
- Training seeds are **never** pooled or resampled jointly. Every interval is
  sampling uncertainty over prompts **within one training run**.
- With 10 clusters of 4 images the resolution is coarse. **An interval
  including zero is not evidence that the effect is zero**, and a change of
  significance label between two runs is **not** a significant difference
  between them.
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

**Guard regression test.** `scripts/seq/test_launcher_guards.sh` extracts the
two sentinels from `run_seed.sh` and exercises them against fabricated
artifacts — CPU only, no model, no GPU. 11/11 pass. Four of its cases
(truncated report, 137/280 rows, empty file, 281 rows) are cases the pre-audit
guards would have accepted as complete. The new guards were also run against
all 11 real evaluations and all 10 real training runs, and classify every one
as complete — they are strict without being wrong.

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

Estimated **≈2.9 GPU-hours** including contingency, within the 4 GPU-hour
ceiling, from measured throughput (0.831 s/optimizer step, 1.017 s/image
generation, 0.079 s/image detection). Wall time is separate and depends on
shared-GPU availability.

The protocol claims **no novelty**. Upstream CUIG already implements early
stopping (`Regularizers/Simultaneous/simultaneous.py`, wired into
`ConAbl/train_conabl.py`, exposed as `--eval_interval/--patience/
--stop_threshold`); §2 of the protocol states what that code does before
describing how a matched-level stop differs in purpose. No new regularizer or
learned controller is built.

## 11. Unresolved

1. **Training reproducibility is untested.** Needed before any claim rests on a
   re-run trajectory. The protocol gets a free check on it (§8 of the protocol).
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

---

## Files

| file | contents |
|---|---|
| [`AUDIT_REPORT.md`](AUDIT_REPORT.md) | this document |
| [`CORRECTED_TABLES.md`](CORRECTED_TABLES.md) | corrected counts, direct contrasts, both estimands, effectiveness context |
| [`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md) | 9 corrections with evidence links, plus what survived unchanged |
| [`ARTIFACT_INVENTORY.md`](ARTIFACT_INVENTORY.md) | checkpoints, logs, caches, what is absent, reproducibility |
| [`integrity.json`](integrity.json) | counts, identities, hashes, duplicates, M0 reuse, symlink audit |
| [`direct_contrasts.json`](direct_contrasts.json) | all contrasts at 0.3/0.5/0.7 × 3 families × 2 seeds |
| [`crosscheck.json`](crosscheck.json) | re-derived vs committed, 0 mismatches |
| [`recomputed_rates.csv`](recomputed_rates.csv) | 756 rows: seed × checkpoint × category × family × threshold |
| [`annotation_packet/`](annotation_packet/) | empty label sheet, manifest, README |
| [`preserved_originals/`](preserved_originals/) | the five narrative documents as of `a9de625` |

Reproduce the analysis (CPU, ~1 min):

```bash
python3 scripts/seq/audit_evidence.py     # integrity + contrasts + crosscheck
python3 scripts/seq/audit_tables.py       # CORRECTED_TABLES.md
bash    scripts/seq/test_launcher_guards.sh
python3 scripts/seq/build_annotation_packet.py --copy_images
```
