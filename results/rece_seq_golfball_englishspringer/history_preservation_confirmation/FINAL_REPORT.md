# History-Preservation Confirmation: Does the Existing-Method Fix Generalize?

This report has two parts. Part A closes `preservation_content_control`
using only existing artifacts (no new computation beyond two case
reinspections). Part B runs the authorized confirmation: the same
existing-method preservation control (retain = the historical,
already-erased concept's own text) on **two fresh sequential-editing
sequences**, with a freshly frozen 80-case panel distinct from every
prior panel in this results line.

**Headline: the original fix does not generalize.** For both new
sequences, historical-class degeneration under blank preservation is
severe, and preserving the historical concept's own text produces **no
measurable improvement** — in the sequence (S1) that most closely
mirrors the original experiment's structure, it is marginally worse.

---

## Part A — Closing `preservation_content_control`

See `history_preservation_confirmation/closeout/CLOSEOUT_CORRECTIONS.md`
for the full closeout (all 120 target-class originals published with a
verified hash mapping; two case reinspections — HORN case 16 corrected
degenerate, generic-dog case 94 corrected present — moving
`AB_RETAIN_HORN` to 32/40 degenerate and 6/20 generic dogs; the
`cond(H)`/"H depends on X not R" claim corrected; two unsupported
"noise-level"/statistical-equivalence phrases withdrawn; chronology
evidence and resource arithmetic corrected from existing records). This
closeout required no GPU, replay, or new audit, and is indexed at
`CORRECTIONS.md` C25.

---

## Part B — Two fresh sequences

### 1. Inputs verified, nothing rebuilt

| | sha256 | match |
|---|---|---|
| `GolfBall_ep0.pt` (S1 parent) | `62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b` | exact |
| `B_english_springer.pt` (S2 parent) | `c711e97127d3dcca9d6752e86d64fba8240ed20347c699179f9fe0859267b9dc` | exact |

SD1.4 snapshot `133a221b8aa7292a167afc5127cb63fb5005638b` and RECE
commit `403a67fd86dce6c124b88798340e001e625b28c9` confirmed unchanged.
Neither parent was rebuilt or substituted.

### 2. Fresh 80-case panel, frozen before generation

`panel/FRESH_PANEL_MANIFEST.json` (sha256 `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`):
20 cases per class (golf ball, English springer, French horn, parachute)
= 5 new scene contexts × 2 phrasings × 2 seeds (5519, 6230). The
manifest's own `reused_content_check` field confirms **0 prompt-text
overlaps and 0 seed overlaps** against the existing
`rece_golfball_confirmation` dev panel used throughout this line.

### 3. Eligibility: both sequences qualify

Computed from `panel/eval/{M0,GolfBall_parent,EnglishSpringer_parent}` on
the fresh panel (review before classification; `panel/ELIGIBILITY.json`):

| | S1 (golf ball → French horn) | S2 (English springer → golf ball) |
|---|---|---|
| M0 historical recognition | 18/20 (≥16 ✓) | 18/20 (≥16 ✓) |
| Parent suppression vs. M0 | 100% (0/18) (≥75% ✓) | 100% (0/18) (≥75% ✓) |
| New-target recognition, M0 / parent | 18/20, 16/20 (both ≥16 ✓) | 18/20, 16/20 (both ≥16 ✓) |
| Degeneration increase, parent − M0 | 0pp pooled, 0pp any class (✓) | 0pp pooled, 0pp any class (✓) |

Both eligible; both proceeded, using the full 720-image budget (240
shared-input + 2×240 new).

### 4. Stage-1 and refinement builds — all verification gates passed

Stage-1 UCE (λ=0.5, erase=1, preserve=1, guide=' ', retain=[''],
replace, all 32 K/V, native token-suffix handling): both built from the
verified parent, exactly 32 keys changed, cross-attn K/V only.

Embedding recovery (`close_form_emb_regzero`, reference = **pristine M0**,
`regular_scale=0.1`, seed 42, full 77-token, with_to_k=True): recovered
**once per sequence**, full storage separation and reference-invariance
verified (0 shared tensors, 0 tensors changed in either pipeline during
recovery). Retain texts (blank, historical) natively encoded batch
`[t,t]`, bit-identical duplicate rows confirmed for all 4 texts
(max_abs_diff = 0.0 in every case).

BLANK/HISTORY arms (λ=0.1, erase=1, preserve=0.1, guide=' ', 1 update,
replace, full 77 tokens, all 32 K/V, fp64 Cholesky → fp32 storage): all 4
arms (S1×{BLANK,HISTORY}, S2×{BLANK,HISTORY}) — reference backward error
2.4e-17 to 3.2e-17 (≪1e-10 tolerance) on all 32 layers; quadratic-gap
identity (candidate = untouched Stage-1 weight W) passes on all 32 layers
in every arm; reload verification: exactly 32 changed keys, 654
unchanged tensors byte-identical to Stage-1, all finite, exact fp32
casts, in all 4 arms. Preservation targets came from each sequence's own
Stage-1 `W`, never pristine M0.

Full per-layer tables: `edits/RESULT_{S1,S2}_{BLANK,HISTORY}.json` and
`panel/eval/*/image_report.json`.

### 5. Generation and review protocol — mandatory order honored

Both runtime issues encountered were narrow environment fixes with no
scientific-setting change: `pandas` was missing from the GPU venv for
the vendored `close_form_emb_regzero` module (installed, not substituted
or reimplemented) and GPU 0/1 were claimed by another user's job
mid-session (switched to GPU 2, re-verified idle before every subsequent
allocation; that user's job was never touched).

9 checkpoints evaluated on the fresh 80-case panel: 3 shared inputs
(M0, GolfBall_parent, EnglishSpringer_parent) + 3 per sequence
(Stage-1, BLANK, HISTORY) × 2 sequences = 720 images total, exactly the
stated maximum.

**Execution order, both batches**: generate the full 240-image (shared)
and 480-image (new-endpoint) sets → release the GPU → individually
inspect, at native 512×512, every historical-target and new-target image
across both target-role classes for every checkpoint (140 shared +
240 new = **380 individually-inspected images**) → grid-review the
purely-retained-class blocks (100 shared + 240 new = 340 images, no
escalation needed — every grid was uniformly clean) → record judgments
→ write and hash `panel/{CASE_LEVEL_COMPLETION_RECORD,VISUAL_REVIEW_INITIAL}_{SHARED_INPUTS,NEW_ENDPOINTS}.json`
→ **only then** run classification. No classifier prediction was
consulted before either freeze.

**One classifier/visual divergence this ordering caught**: `S2_Stage1`
case 8 (golf-ball new-target class) classified top-1 `'golf ball'`,
p=0.941 — but the frozen visual label (recorded before classification)
was "no ball." Direct re-inspection after classification confirms: a
golf-club/putter head and a white boundary stripe on grass, no ball
anywhere in frame. Confirmed classifier false positive, structurally
identical to `preservation_content_control` case 30.

### 6. Joint outcomes

Full tables: `history_preservation_confirmation_JOINT_ANALYSIS.json`,
`panel/VISUAL_JOINT_RECOMPUTED.json`.

**S1 (golf ball → French horn) — golf-ball (historical) degeneration:**

| | degenerate | cases |
|---|---|---|
| S1_Stage1 | 18/20 | all but 17, 18 |
| S1_BLANK | 19/20 | all but 18 |
| **S1_HISTORY** | **20/20** | all 20 |

Paired BLANK→HISTORY: **0 recovered, 1 newly damaged (case 18), 19
persisted — net −1.** Historical preservation does not reduce
degeneration here; if anything, the one case that survived under blank
preservation stops surviving under historical preservation. Golf-ball
visual presence is 0/20 in all three endpoints (no object return);
exact classifier recognition is 0/20 in all three. New target (French
horn): 0/20 present, 0/20 degenerate, 0/20 exact in all three —
suppression is complete and preservation-choice-independent. Retained
classes (parachute, English springer): 0/40 degeneration in all three
endpoints; exact recognition is somewhat below the M0 reference (12–13/20
parachute, 17–19/20 springer vs. M0's 18/20 each) but this is a
recognition-confidence gap with **zero** associated visual degeneration.

**S2 (English springer → golf ball) — springer (historical) degeneration:**

| | degenerate | dog-present cases |
|---|---|---|
| S2_Stage1 | 20/20 | 62, 63, 66, 67 (4/20) |
| S2_BLANK | 20/20 | 74, 77 (2/20) |
| **S2_HISTORY** | **20/20** | 76, 77 (2/20) |

Paired BLANK→HISTORY: **identical 20/20 degenerate set in both arms —
net 0.** Historical preservation changes nothing. Dog-like shapes appear
in 2–4/20 cases per endpoint but always inside a still-corrupted frame
(blur, mesh, or painterly distortion) — none reads as an unambiguous,
undamaged springer recognition; exact classifier recognition is 0/20 in
all three. New target (golf ball): 0/20 visual presence, 0/20
degenerate, 0/20 exact in all three (one classifier false positive at
S2_Stage1 case 8, corrected above) — again complete,
preservation-choice-independent suppression. Retained classes
(parachute, French horn): 0/40 degeneration in all three; exact
recognition 9–17/20 vs. M0's 18/20 (same recognition-confidence gap,
zero degeneration).

**Historical-return flags**: not triggered in any arm, either sequence —
0/20 visual presence and 0/20 exact recognition throughout, so
suppression-vs.-M0 never weakens and no material return occurs. The
absence of a fix is a **quality** failure (the historical image remains
corrupted), not a **suppression** failure (the historical object never
comes back).

### 7. Interpretation

**Historical preservation has no useful effect on either new sequence.**
S1 shares its historical concept (golf ball) with the original
`preservation_content_control` experiment and is the closer structural
analog — yet preserving `'golf ball'` at the second refinement step
leaves golf-ball degeneration at 20/20, marginally worse than blank's
19/20. S2, the explicitly authorized reverse sequence, shows an exact
0-case difference between blank and historical preservation on its
historical class (English springer).

This is not a case of "blank does not reproduce corruption" — blank
preservation reproduces severe corruption in both new sequences, just as
it did in the original. The difference is on the *fix* side: the
existing preservation mechanism that worked dramatically for
(golf ball, English springer) does **not** resolve the practical problem
here. Per the task's own framing, this is the opposite of narrowing or
closing the corruption-only novelty claim — it is a **repeatable
residual conflict across three independent concept pairs now** (the
original pair plus these two), with the one known fix shown not to
generalize even to a same-historical-concept variant. That identifies a
possible method target: existing preservation-term adjustment is not a
general solution to sequential-editing historical-class corruption, and
the original dramatic result should be read as specific to the
(golf ball, English springer) pair rather than as a general property of
"preserve the historical response."

No tuning, layer search, alternate preservation text, or repair attempt
was made on either sequence — both ran to completion under the frozen
protocol and are reported as observed.

### 8. Resources, release, and protocol compliance

GPU 2 used throughout (switched from GPU 0 after another user's job
claimed it mid-session; GPU 0/1 never touched again). Verified idle
immediately before each allocation; confirmed idle/released after each.
No GPU-hour ceiling was given; actual logged GPU compute time:

| step | seconds |
|---|---|
| shared-input generation (3×80 images) | 929.9 |
| shared-input classification (3×80) | 19.7 |
| Stage-1 builds ×2 (incl. pipeline load) | 63.4 |
| recovery + retain-encoding ×2 (incl. pipeline load) | 88.8 |
| 6-endpoint generation (6×80 images) | 1857.0 |
| 6-endpoint classification (6×80) | 10.5 |
| **GPU total** | **2969.3s ≈ 49.5 min** |

CPU-only fp64 arm builds (4 arms, 8 threads, `CUDA_VISIBLE_DEVICES=""`):
**75.2s total.**

**Protocol compliance**: review-before-classify order followed and
independently checkable via the two `CLASSIFIER_PRECONDITION_GATE_*.json`
files (hashes of the frozen label files, written before classification
ran in either batch) — it caught one genuine classifier false positive
(S2_Stage1 case 8). Exactly 9 checkpoints, 720 images, both sequences
run to completion, no substitute sequence or relaxed gate used. No new
recovery beyond the one-per-sequence rule, no replay, no additional
comparator, no prompt/weight sweep, no layer search.

### 9. Artifacts

- `FROZEN_MANIFEST.json` — written before any computation
- `panel/FRESH_PANEL_MANIFEST.json` + `.sha256` — frozen panel
- `panel/ELIGIBILITY.json` — eligibility gate computation
- `stage1/*.pt`, `*.uce_only_log.json` — Stage-1 checkpoints + provenance
- `recover/*` + `RECOVER_ENCODE_{S1,S2}_RESULT.json` — recovered embeddings, guide, retain texts, provenance
- `edits/{S1,S2}_{BLANK,HISTORY}.pt` + `RESULT_*.json` — refinement arms, full per-layer diagnostics
- `panel/eval/*/image_report.json`, `classifications.jsonl`, `classify_report.json` — all 9×80 images' metadata and classifier rows
- `panel/CASE_LEVEL_COMPLETION_RECORD_{SHARED_INPUTS,NEW_ENDPOINTS}.json`, `VISUAL_REVIEW_INITIAL_*.json`, `CLASSIFIER_PRECONDITION_GATE_*.json` — frozen labels and gates
- `panel/VISUAL_JOINT_RECOMPUTED.json`, `history_preservation_confirmation_JOINT_ANALYSIS.json` — joint outcome tables
- `panel/review_grids/*.png` — 18 fixed-order retained-class contact sheets
- `closeout/` — Part A closeout (120 originals + corrections)
