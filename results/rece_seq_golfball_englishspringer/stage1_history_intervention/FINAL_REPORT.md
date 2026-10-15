# Stage-1 History Intervention: Does Preserving Earlier Prevent the Damage?

> **Correction note (2026-10-15, see `CORRECTIONS.md` C29 and
> `closeout/CORRECTIONS_BEFORE_S2.md`).** Re-inspection of golf-ball
> cases 12-15 found the `S1_Stage1` labels for those four cases did not
> match the frozen corruption rubric (coherent-but-unexpected macro
> textures, not sparkle/mesh/checkerboard corruption) and corrected them
> to non-degenerate. This moves `S1_Stage1`'s own golf-ball degeneration
> from ~~18/20~~ to **14/20**, and the paired recovery below from
> ~~18 recovered / 2 neither~~ to **14 recovered / 6 neither**.
> `S1_STAGE1_HISTORY` itself is unaffected (still exactly 0/20
> degenerate) — the intervention's result is unchanged in kind (complete
> recovery, zero newly damaged), only in magnitude. A separate sentence
> below overclaimed that golf-ball return is "structurally impossible";
> that claim is removed (a later edit can restore an earlier-erased
> concept in principle — the 0/20 return result here is an empirical
> observation, not a structural guarantee).

**Question.** Can RECE's existing preservation mechanism (`retain_texts`)
prevent historical-prompt corruption when applied **at Stage 1, before
the damage occurs**, rather than only at Stage-2 refinement, after it?
Existing-method control; no new method introduced.

**Answer: yes, for this one fixed Stage-1 preservation choice, on this
one sequence.** `S1_STAGE1_HISTORY` (same native Stage-1 edit as the
existing `S1_Stage1` comparator, with the one change
`retain_texts=['']` → `retain_texts=['golf ball']`) eliminates the
golf-ball-prompt degeneration that `S1_Stage1` showed: **14/20 → 0/20**
(corrected from a previously reported 18/20 — see correction note
above), an exact, complete paired recovery of every correctly-labeled
degenerate case, zero newly damaged. French-horn suppression is unchanged (0/20, ≥75%
suppression against the parent's own 16/20 — denominator verified, not
assumed) and retained-class recognition is, if anything, slightly higher
than the blank comparator (parachute 13→16/20, springer 18→20/20), with
zero degeneration on either retained class in either endpoint.

This is a **narrow, qualified** result: one sequence, one fixed
preservation text, one 80-case panel. It supports applying existing
preservation earlier in this specific setting. It does not establish
novelty, does not isolate which part of the Stage-1 mechanism is doing
the work, and was not tested on S2.

## 1. Premise verification (see `closeout/VERIFICATION_AND_CORRECTIONS_2.md`)

Recomputed directly from frozen labels, not assumed: `GolfBall_parent`
0/20 degenerate → `S1_Stage1` **14/20** (corrected from 18/20; see
correction note above) → confirming the damage is introduced at the
native Stage-1 edit itself, materially, before any refinement step runs. All
four proceed-conditions (eligibility, parent-vs-Stage1 contrast,
widespread Stage-1 corruption, provenance availability) were verified
and held; this intervention was authorized to proceed automatically.

## 2. Build: exactly one scientific change, everything else held fixed

- **Parent**: untouched `GolfBall_ep0.pt`, sha256
  `62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b`
  (verified, never `S1_Stage1` or any refined checkpoint).
- **Native Stage-1 edit** (`vf.edit_model`, pinned-source-checked,
  identical function used for `S1_Stage1`): λ=0.5, erase_scale=1,
  preserve_scale=1.0, guide=' ', technique='replace', all 32
  cross-attention K/V weights, one native edit, native token-suffix
  alignment, native `[t,t]`-equivalent full-token retain handling (no
  erase slicing applied to retain).
- **The only change**: `retain_text_=['']` → `retain_text_=['golf ball']`.
  Confirmed in the run's own printed line: `Edited "['French horn']"
  into "[' ']" and Retained "['golf ball']"`.
- **Verification**: exactly 32 keys changed vs. the untouched parent, 654
  unchanged tensors byte-identical, all 686 finite, exact reload. No
  fp64 tolerance was imposed on this native fp32 Stage-1 solve (none
  applies). Checkpoint sha256 `e9595e105c718e5207ada59ad9b8100f93dd94523f832a70fdd4589d161030e9`.
  Build: `stage1/S1_STAGE1_HISTORY.stage1_history_log.json` (58.3s
  total, including pipeline load).

## 3. Generation and review — mandatory order honored

80 images generated on the **existing, hash-verified**
`history_preservation_confirmation` panel manifest (sha256
`cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`,
reused unmodified, not reconstructed), 310.3s, GPU verified idle
immediately before allocation and released after.

All 40 target-role images (golf ball 0–19, French horn 40–59)
individually inspected at native 512×512. Both retained classes
(parachute 20–39, English springer 60–79) grid-reviewed; neither grid
required escalation to full resolution (both were uniformly clean).
Labels frozen and hashed (`CLASSIFIER_PRECONDITION_GATE.json`,
`VISUAL_REVIEW_INITIAL.json`, `CASE_LEVEL_COMPLETION_RECORD.json`)
**before** classification ran.

## 4. Joint outcomes

Full tables: `CASE_LEVEL_COMPLETION_RECORD.json`,
`eval/S1_STAGE1_HISTORY/classifications.jsonl`.

**Golf-ball (historical) degeneration and paired transition:**

| | degenerate |
|---|---|
| GolfBall_parent | 0/20 |
| S1_Stage1 (blank, comparator) | 14/20 (corrected from 18/20) |
| **S1_STAGE1_HISTORY** | **0/20** |

Paired S1_Stage1 → S1_STAGE1_HISTORY: **14 recovered, 0 newly damaged, 0
persisted, 6 neither** (cases 12, 13, 14, 15, 17, 18, already
non-degenerate in both — corrected from a previously reported 2; see
correction note above) — a complete, exact recovery of every case that
was (correctly) degenerate under blank Stage-1 preservation.

**Golf-ball return**: exact classifier recognition 0/20 in every
checkpoint from `GolfBall_parent` onward. Visual presence likewise 0/20.
The historical-return flag (≥2 net additional hits vs. the parent on 20
cases) is **not triggered** — 0 vs. 0. (A prior version of this sentence
claimed golf-ball return was "structurally impossible" because the
untouched parent already erased it; that claim is removed — a later
edit can in principle restore an earlier-erased concept, and the 0/20
result here is an empirical observation on this panel, not a structural
guarantee.)

**French-horn (new target) deletion**: `GolfBall_parent`'s own panel
count is **16/20** (verified directly, not assumed as 16/20 by default).
`S1_Stage1` and `S1_STAGE1_HISTORY` both show 0/20 exact, 0/20 visual
presence, 0/20 degenerate — **100% suppression**, well above the ≥75%
gate, unchanged by the preservation-text choice.

**Retained classes** (parachute, English springer): recognition is
slightly *higher* under `S1_STAGE1_HISTORY` than under the blank
`S1_Stage1` comparator (parachute 13/20→16/20; springer 18/20→20/20),
and degeneration is **0/40 in both endpoints** — no retained-class
quality cost from this preservation choice, on this panel.

**Scene/family breakdown**: no single scene or phrasing family drove the
golf-ball recovery. Corrected from a prior "all 20/20 cases flipped"
overclaim: **14/20 cases flipped** from degenerate to clean (4 of 5
scene contexts, both phrasings, both seeds represented among them,
including the golf-cart interior scenes that showed heavy corruption
under blank Stage-1 preservation); the remaining **6/20 were already
non-degenerate under blank Stage-1** (cases 12, 13, 14, 15 — the macro
dimple-texture scene, scene_index 3, correctly non-degenerate in both
blank and historical Stage-1 once the rubric is applied consistently —
plus cases 17, 18). Final state is 0/20 degenerate either way.

## 5. Interpretation — qualified, not generalized

**A successful joint outcome, within the scope tested.** Applying the
existing preservation mechanism at the point of injury (Stage 1, before
the damage occurs) rather than only afterward (Stage-2 refinement, after
the damage is already largely baked in) eliminates the golf-ball
degeneration this report's sibling experiment found refinement-only
preservation could not fix. This directly supports applying existing
preservation earlier as a practical mitigation in this setting.

It does **not** establish novelty (this is RECE's own existing
`retain_texts` mechanism, applied at a stage the original paper already
supports), does not isolate which specific part of the preservation term
is responsible (no ablation of λ/erase_scale/preserve_scale was run),
and was tested on exactly one sequence (S1) with exactly one
preservation text. S2 was not re-tested at Stage 1; the reverse-sequence
question remains open. No refinement stage was added on top of this
checkpoint, per the task's explicit instruction not to advance
automatically even though Stage 1 improved.

## 6. Resources and protocol compliance

GPU 0 used throughout this intervention (verified idle before both the
build and the generation allocation, released after each). Build: 58.3s.
Generation: 310.3s. Classification: 6.5s. **Total GPU: 375.1s ≈ 6.25
min.** Review-before-classify order followed and independently checkable
via `CLASSIFIER_PRECONDITION_GATE.json` (frozen-file hashes written
before classification ran).

## 7. Artifacts

- `FROZEN_MANIFEST.json` — written before any computation
- `make_stage1_history_checkpoint.py` — the one-argument-changed build script
- `S1_STAGE1_HISTORY.pt` + `.stage1_history_log.json` — checkpoint + full build/verification log
- `eval/S1_STAGE1_HISTORY/{image_report.json,classifications.jsonl,classify_report.json}`
- `CASE_LEVEL_COMPLETION_RECORD.json`, `VISUAL_REVIEW_INITIAL.json`, `CLASSIFIER_PRECONDITION_GATE.json`
- `review_grids/{parachute,springer}.png` — fixed-order retained-class contact sheets
- `closeout/VERIFICATION_AND_CORRECTIONS_2.md` — premise verification and corrections to the prior report
- `closeout/CORRECTIONS_BEFORE_S2.md` — cases 12-15 relabeling, "structurally impossible" claim removal, and other corrections bundled ahead of the S2 intervention
