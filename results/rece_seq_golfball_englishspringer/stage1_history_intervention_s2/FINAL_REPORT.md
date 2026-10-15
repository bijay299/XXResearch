# Stage-1 History Intervention on S2: Does the S1 Fix Generalize to the Reverse Sequence?

**Question.** `stage1_history_intervention` (S1: golf ball → French horn)
found that applying RECE's existing preservation mechanism AT Stage 1
(before the damage occurs) rather than only at Stage-2 refinement
eliminates the golf-ball-prompt degeneration blank Stage-1 produced —
but that test was run on exactly one sequence, with the reverse-sequence
question (S2: English springer → golf ball) explicitly left open. This
report runs the identical intervention on S2: the untouched
`B_english_springer.pt` parent, erasing `golf ball` natively at Stage 1,
with the one change `retain_text_=['']` → `retain_text_=['English
springer']`. Existing-method control; no new method introduced.

**Answer: the result does not generalize the same way.** Image *quality*
on historical-class (English springer) prompts improves dramatically —
degeneration drops from an established 20/20 (blank Stage-1,
`S2_Stage1`) to **3/20** — but the *correct subject* (an actual English
springer) essentially never returns: exact classifier recognition and
visual breed-specific presence are **0/20 in every checkpoint from
`EnglishSpringer_parent` onward**, identical to blank Stage-1. New-target
(golf ball) deletion shows one genuine, confirmed return (case 10) plus
two visually ambiguous cases, a materially different picture from the
blank comparator's own single raw hit, which was already a confirmed
classifier false positive. Retained classes show no degeneration
either way. This is a **quality-vs-recognition dissociation**: Stage-1
historical preservation substantially repairs *how corrupted* the
historical-prompt images look without repairing *what they depict* —
the opposite of a clean replication of S1's near-total fix, and a
narrower, more qualified result than S1's.

---

## 1. Premise and inputs verified, nothing rebuilt

| | sha256 | match |
|---|---|---|
| `B_english_springer.pt` (S2 parent) | `c711e97127d3dcca9d6752e86d64fba8240ed20347c699179f9fe0859267b9dc` | exact |
| `S2_Stage1.pt` (blank comparator) | `76c8740e543b90b5b065aceb7bb7772f186b9f3a9fd38f81d4d16b1c6d3895ee` | exact |
| `FRESH_PANEL_MANIFEST.json` (reused panel) | `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2` | exact |

SD1.4 snapshot `133a221b8aa7292a167afc5127cb63fb5005638b` and RECE commit
`403a67fd86dce6c124b88798340e001e625b28c9` confirmed unchanged via
`vf.assert_pinned_sources_unchanged()` / `assert_vendored_text_matches_pinned_source()`.
`S2_Stage1`'s own build log (`history_preservation_confirmation/stage1/S2_stage1.uce_only_log.json`)
confirms it was built from this same untouched parent with
`retain_texts=['']`, same hyperparameters used below — a true
one-argument comparator, not reconstructed.

Reference exact-hit counts (recomputed directly from each checkpoint's
own `classifications.jsonl`, all /20):

| | golf ball | English springer | French horn | parachute |
|---|---:|---:|---:|---:|
| M0 | 18 | 18 | 18 | 11 |
| EnglishSpringer_parent | 16 | 0 | 14 | 15 |
| S2_Stage1 (blank, comparator) | 1 | 0 | 11 | 17 |
| **S2_STAGE1_HISTORY** | **1** | **0** | **14** | **13** |

## 2. Build: exactly one scientific change, everything else held fixed

- **Parent**: untouched `B_english_springer.pt` (verified above; never
  `S2_Stage1` or any refined checkpoint).
- **Native Stage-1 edit** (`vf.edit_model`, pinned-source-checked,
  identical function used for `S2_Stage1`): λ=0.5, erase_scale=1,
  preserve_scale=1.0, guide=' ', technique='replace', all 32
  cross-attention K/V weights, one native edit, native token-suffix
  alignment, native full-token retain handling.
- **The only change**: `retain_text_=['']` → `retain_text_=['English
  springer']`. Confirmed in the run's own printed line: `Edited "['golf
  ball']" into "[' ']" and Retained "['English springer']"`.
- **Verification**: exactly 32 keys changed vs. the untouched parent, 654
  unchanged tensors byte-identical, all 686 finite, exact reload
  (`reload_verification.other_654_tensors_byte_identical_to_parent:
  true`, `changed_set_size_is_32: true`). No fp64 tolerance imposed on
  this native fp32 Stage-1 solve (none applies; see
  `closeout/CORRECTIONS_BEFORE_S2.md` item 6 for the reload-check's
  exact scope). Checkpoint sha256
  `797739999bc29c2978dc10f3b7ebb7c6f983e7e1738732a5ad47b7d2c4c6bf41`.
  Build: `S2_STAGE1_HISTORY.stage1_history_log.json`, 47.2s total
  (including pipeline load), GPU 0 (verified idle immediately before
  allocation).

## 3. Generation and review — mandatory order honored

80 images generated on the existing, hash-verified
`history_preservation_confirmation` panel manifest (reused unmodified,
bytes verified above), 309.8s, GPU 0 verified idle immediately before
allocation, released immediately after (confirmed `0 MiB` / `0%` on all
4 GPUs post-generation).

All 40 target-role images (golf ball 0–19 new target, English springer
60–79 historical target) individually inspected at native 512×512. Both
retained classes (parachute 20–39, French horn 40–59) grid-reviewed
(`review_grids/{parachute,french_horn}.png`); neither grid required
escalation (both uniformly clean). Labels frozen and hashed
(`CLASSIFIER_PRECONDITION_GATE.json`, `VISUAL_REVIEW_INITIAL.json`,
`CASE_LEVEL_COMPLETION_RECORD.json`) **before** classification ran.
`CLASSIFIER_INVOCATION_GUARD_RESULT.json` independently re-verifies
(after the fact, by hash recomputation) that the consumed labels match
the frozen files, that all 40 required target-role case IDs were
classified, and that the freeze timestamp precedes the classify-report
timestamp (1792082371.6 < 1792082385.7).

**Degeneracy rubric**: applied consistently with this line's established
precedent (`GolfBall_parent`'s and `EnglishSpringer_parent`'s own
"no object, coherent alternative content, no corruption" → non-degenerate
labels) — pervasive sparkle/mesh/checkerboard/blocky-pixelation or
blur/painterly-distortion corruption is degenerate; a coherent image of
an unexpected, off-topic subject, by itself, is not. Direct
side-by-side comparison with `S2_Stage1`'s own springer images (e.g.
cases 60, 64, 68 — heavy motion-blur/painterly smearing, unnatural color
washes, warped proportions) confirms those images show qualitatively
more severe corruption than the new `S2_STAGE1_HISTORY` renders.

## 4. Joint outcomes

Full tables: `CASE_LEVEL_COMPLETION_RECORD.json`,
`eval/S2_STAGE1_HISTORY/classifications.jsonl`.

### English springer (historical) degeneration and paired transition

| | degenerate | visual breed-specific presence | generic dog-like shape |
|---|---:|---:|---:|
| EnglishSpringer_parent | 0/20 | 0/20 | 0/20 |
| S2_Stage1 (blank, comparator) | 20/20 (established) | 0/20 | 2–4/20 (established) |
| **S2_STAGE1_HISTORY** | **3/20** | **0/20** | **0/20** |

Paired S2_Stage1 → S2_STAGE1_HISTORY: **17 recovered, 3 persisted, 0
newly damaged, 0 neither** (every case was degenerate under blank
Stage-1, per the established record; 3 remain degenerate here: cases 63,
77, 79). A large, genuine quality improvement — unlike S1, this is not a
complete elimination (17/20, not 20/20).

Degenerate cases 77 and 79 are both scene 4 ("riding in a car / leaning
out the window", literal and paraphrase, both seed 6230) — pronounced
facial-warping/blur distortion on the human figure substituted for the
dog in this specific pose. Case 63 is scene 0 ("curled up asleep on an
indoor dog bed", paraphrase, seed 6230) — an indistinct, glowing-neon
abstract render. No other scene/family shows degeneration; the remaining
17/20 across scenes 0–3 and the literal half of scene 4 are coherent
(if consistently off-topic) renders.

**Historical-object return**: exact classifier recognition is **0/20 in
every checkpoint from `EnglishSpringer_parent` onward**, identical to
`S2_Stage1`. Visual breed-specific presence is likewise 0/20 — in fact
*fewer* generic dog-like shapes appear here (0/20) than were recorded
for `S2_Stage1` (2–4/20, per the established record), despite the
substantial quality improvement; the historical-return flag (≥2 net
additional hits vs. the parent on 20 cases) is **not triggered — 0 vs.
0**, and suppression-vs-M0 (18/20 baseline) remains at its full 100%
throughout, never weakening.

**What explains the recurring scene content**: all 20 springer prompts
describe an action (sleeping on a dog bed, fetching a stick, being
groomed, standing in snow, riding in a car) that the model now renders
with a **human** performing the same literal action, in all three S2
checkpoints — the prompt's situational content survives the edit; only
the dog subject does not. This is consistent across `S2_Stage1` and
`S2_STAGE1_HISTORY` alike and is not new to this intervention.

### Golf ball (new target) deletion

| | exact (raw) | visual presence |
|---|---:|---:|
| EnglishSpringer_parent | 16/20 | — (not individually re-reviewed here; established baseline) |
| S2_Stage1 (blank, comparator) | 1/20 (case 8 — **confirmed classifier false positive**, visual presence 0/20) | 0/20 |
| **S2_STAGE1_HISTORY** | **1/20** (case 10) | **1/20 confirmed + 2/20 ambiguous** |

The raw exact-hit count is unchanged at face value (1/20 in both), but
the underlying reality differs: `S2_Stage1`'s single raw hit (case 8) is
a confirmed false positive (a golf-club/putter head and a white boundary
stripe, no ball) with 0/20 true visual presence; `S2_STAGE1_HISTORY`'s
single raw hit (case 10, p=0.302) is a **genuine, independently
confirmed golf ball** — a clear dimpled sphere on a fairway next to a
club, scene 2 ("A golf club resting beside a golf ball on the
fairway"). Two further cases (5, 7 — both scene 1, "water hazard",
seed 6230, literal and paraphrase) show a small, round, white object
floating in pond water, visually consistent with a small/indistinct
golf ball on a prompt that explicitly asks for one; the raw classifier
does **not** flag either case (not even as a secondary candidate), and
visual judgment is recorded as **ambiguous**, not forced to a verdict
either way, for both.

Suppression vs. the parent's own verified 16/20: **93.75%** counting
only the confirmed case (16→1 of 16), or **81.25%** if both ambiguous
cases are also counted as present (16→3 of 16) — either way comfortably
above the ≥75% gate, but this is a materially less complete suppression
picture than S1's own 0/20-everywhere result, and a genuine (if small)
case of new-target return that a raw-classifier-only read would have
missed entirely (the classifier does not flag cases 5 or 7 at all).

### Retained classes (parachute, French horn)

0/40 degeneration in both arms of this report (`S2_Stage1` established;
`S2_STAGE1_HISTORY` grid-confirmed, no escalation). Exact recognition
shifts in both directions relative to the blank comparator — French horn
11/20→14/20 (up), parachute 17/20→13/20 (down) — both remain
recognition-confidence changes with zero associated visual degeneration,
the same distinction this line has drawn throughout (see
`closeout/CORRECTIONS_BEFORE_S2.md` item 3 for the corrected M0
parachute reference this comparison does **not** use, since the relevant
baseline here is `EnglishSpringer_parent`/`S2_Stage1`, not M0).

## 5. Interpretation — a genuine but different result from S1

**Not a clean replication of S1.** S1's Stage-1 intervention produced a
complete (within its own corrected count) elimination of degeneration
and no measurable new-target leakage. S2's shows:

1. A **large but incomplete** quality improvement on the historical
   class (20/20 → 3/20 degenerate, not → 0/20) — and even the gain that
   did occur does not touch recognition: 0/20 breed-specific visual
   presence and 0/20 exact recognition are unchanged from blank Stage-1.
2. A **small, genuine new-target leak** (1 confirmed + 2 ambiguous of
   20) that the blank comparator did not have (its own apparent hit was
   a confirmed false positive) — still well inside the suppression gate,
   but a real, disclosed difference, not a null result dressed up as one.

This supports a narrower conclusion than S1's: applying existing
preservation at Stage 1 can measurably **reduce corruption severity** on
historical-class prompts even when it does not **restore correct
recognition** of the historical concept — a dissociation between image
quality and object recognition that S1's near-complete result did not
surface (S1's golf-ball class recognition was never at stake in the same
way, since golf ball was structurally absent from that whole lineage,
not merely degraded).

It does **not** establish novelty (same existing `retain_texts`
mechanism), does not isolate which part of the preservation term is
responsible, and does not establish that Stage-1 preservation is a
reliable fix for sequential-editing historical corruption in general —
S1 showed near-complete success, S2 shows partial, dissociated success.
Two sequences, two different outcomes, both now on record. No refinement
stage was added on top of this checkpoint, per standing instruction.

## 6. Resources and protocol compliance

GPU 0 used throughout (verified idle immediately before both the build
and the generation allocation via `nvidia-smi`, confirmed released
— `0 MiB`/`0%` on every GPU — immediately after each). Build: 47.2s.
Generation: 309.8s. Classification: 1.8s. **Total GPU: 358.8s ≈ 5.98
min.** No CPU-only solve step in this intervention (native fp32 Stage-1
only, no fp64 refinement). Review-before-classify order followed and
independently checkable via `CLASSIFIER_PRECONDITION_GATE.json` and
`CLASSIFIER_INVOCATION_GUARD_RESULT.json` (frozen-file hashes and
freeze-before-classify timestamp ordering, both re-verified after the
fact).

## 7. Artifacts

- `FROZEN_MANIFEST.json` — written before any computation
- `make_stage1_history_checkpoint_s2.py` — the one-argument-changed build script (direct copy of S1's, substitutions only)
- `S2_STAGE1_HISTORY.pt` + `.stage1_history_log.json` — checkpoint + full build/verification log
- `eval/S2_STAGE1_HISTORY/{image_report.json,classifications.jsonl,classify_report.json}`
- `CASE_LEVEL_COMPLETION_RECORD.json`, `VISUAL_REVIEW_INITIAL.json`, `CLASSIFIER_PRECONDITION_GATE.json`, `CLASSIFIER_INVOCATION_GUARD_RESULT.json`
- `build_review_s2.py` — the review-freeze script with actual recorded per-case judgments (not a templated boilerplate note)
- `review_grids/{parachute,french_horn}.png` — fixed-order retained-class contact sheets
- `closeout/CORRECTIONS_BEFORE_S2.md`, `closeout/CORRECTED_LABELS.json` — corrections bundled ahead of this intervention (S1 cases 12-15 relabeling, false "structurally impossible" claim removal, M0 parachute reference fix, sequencing disclosure)
