# Human review report — S2 springer quality question, unblinded (2026-10-16)

**One selected human response, independently verified, unblinded, and
analyzed.** The earlier survey submission was explicitly withdrawn by
the user from this analysis and is **not** published, pooled, or
compared here.

## 1. Provenance and integrity

| | |
|---|---|
| Selected file | `review_export_final_2026-10-16T14-06-23-096Z.json` |
| Raw bytes | 4,705 (verified independently, matches) |
| Raw SHA256 | `7c4c509f470abbdbc79b8bef3aacc2f065bca35cc5f792cdf59fe957fc3a57b8` (verified independently, matches) |
| Preserved verbatim at | `responses/review_export_final_2026-10-16T14-06-23-096Z.json` |
| Receipt | `responses/RESPONSE_RECEIPT.json` |
| Package / protocol version | `blinded_quality_review_s2-v1` / `FROZEN_PROTOCOL-v1` (both match) |
| Completeness | 20/20 pairs, 0 missing fields, 0 "uncertain" answers |

**Factual exposure and protocol-deviation disclosure** (reported as
stated, not inferred, and not used to alter any raw answer): this is
one selected updated assessment by reviewer alias "Bijay," submitted
after a rubric clarification and after an earlier survey submission.
Before this submission, the reviewer had already inspected the images,
was aware of the assistant's own aggregate claim for this comparison
(2/20 → 6/20 historical-class degeneration), and explicitly asked
another assistant (Codex) about study image Q10_B; Codex described that
image and gave case-specific feedback (localized defects: yes;
whole-frame degeneration: leaning no) before this submission was
frozen. **This departs from the original no-calibration-on-study-images
rule** in `FROZEN_PROTOCOL.md`. The endpoint mapping itself was **not**
revealed by Codex before this submission was frozen — Codex's own
preservation record states the response bytes were fixed at
2026-10-16T14:07:42.219527+00:00, before obtaining or deriving the
mapping. **This review is reported as blinded to case/endpoint
identity, with disclosed prior exposure to the study's general claim
and one piece of case-specific guidance on a study image — it is not a
fully naive or independent external validation**, and is not described
as one anywhere in this report. The exposure field (`read_summaries_or_labels`)
and session-start timestamp (`2026-10-16T01:52:43.826Z`) are reported
as submitted; no review duration or proof of viewing is inferred from
them.

## 2. Mapping verification (performed before unblinding)

| check | result |
|---|---|
| Private mapping SHA256 vs. required value | `bbf8a9eb8b6459bdb2e4ff33ed8638c5708fe4ecb10e58c09a02ecbdcb47689a` — **match** |
| Private mapping SHA256 vs. `MAPPING_COMMITMENT.json` at commit `f6fe138` | **match** |
| Panel manifest SHA256 | `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2` — **match** |
| Mapping's 40 recorded image hashes vs. independent source verification | **0 mismatches** |
| Exactly one image per endpoint per pair, all 20 pairs | **confirmed** |
| 20 unique case IDs, range 60–79 | **confirmed** |
| A-position balance | `S2_STAGE1_HISTORY`: 10, `S2_HISTORY_BOTH_STAGES`: 10 — **confirmed** |

No mismatch, no missing file, no substitution. The mapping and seed are
published alongside this report (`UNBLINDED_MAPPING.json`), unchanged
from the committed file. Full machine-readable verification:
`UNBLINDING_VERIFICATION.json`. Full joined table: `UNBLINDED_JOINED_TABLE.json`.

## 3. Recomputed counts (independently derived from the raw response + verified mapping)

**Reminder on the `pair_preference` field**: it records which image has
**more widespread rendering corruption** — not which the reviewer
preferred. "B" means B is *worse*. A/B labels are per-pair and random;
they do not correspond to "before"/"after" or to a consistent model
stage across pairs.

| | whole-frame degeneration (yes) | local defects (yes) |
|---|---:|---:|
| `S2_STAGE1_HISTORY` | **2/20** (cases 63, 72) | **19/20** (all but case 72) |
| `S2_HISTORY_BOTH_STAGES` | **1/20** (case 63) | **20/20** (all cases) |

**Paired transition, `S2_STAGE1_HISTORY` → `S2_HISTORY_BOTH_STAGES`,
whole-frame degeneration** (the previously-unresolved discordant-pair
direction is now resolved):

| | count | case(s) |
|---|---:|---|
| persisted (degenerate in both) | 1 | 63 |
| **recovered** (degenerate → clean) | **1** | **72** |
| newly damaged (clean → degenerate) | 0 | — |
| neither (clean in both) | 18 | 60–62, 64–71, 73–79 (minus 63, 72) |

**No unresolved pairs remain** — every one of the 20 pairs now has a
determinate, mapped direction.

**Severity comparison ("which image has more widespread rendering
corruption")**, resolved to endpoints:

| more-corrupted endpoint | n pairs | case(s) |
|---|---:|---|
| `S2_STAGE1_HISTORY` | 2 | 72, 78 |
| `S2_HISTORY_BOTH_STAGES` | 2 | 68, 79 |
| no clear difference | 16 | all remaining |
| uncertain | 0 | — |

Balanced 2-vs-2 at the severity-comparison level; no systematic
direction either way in this measure. (Per protocol, a "no clear
difference" judgment at cases 68/78/79 is not inconsistent with the
corresponding individual whole-frame-degeneration answers both being
"no" — the paired question measures a finer-grained difference below
that binary threshold, exactly as anticipated going in.)

## 4. The selected response does not reproduce the assistant's claimed direction

The assistant's own published claim (`history_both_stages_s2/FINAL_REPORT.md`,
C32; corrected baseline per C31) is a **regression**: `S2_STAGE1_HISTORY`
2/20 → `S2_HISTORY_BOTH_STAGES` 6/20 (0 recovered, 2 persisted, 4 newly
damaged, 14 neither). This selected human response, independently
recomputed from verified raw data, gives **2/20 → 1/20** — a net
**improvement of one case** (1 recovered, 1 persisted, 0 newly damaged,
18 neither), the opposite direction from the assistant's claim on this
specific whole-frame-degeneration measure, for this one reviewer, on
this one panel.

**Case-level agreement is low even where totals happen to be close.**
Independently cross-tabulating this response against the assistant's
own current (corrected) per-case labels:

| endpoint | assistant-flagged cases | human-flagged cases | overlap | case-level agreement |
|---|---|---|---|---:|
| `S2_STAGE1_HISTORY` | 77, 79 | 63, 72 | **none** | 16/20 |
| `S2_HISTORY_BOTH_STAGES` | 61, 63, 68, 71, 77, 79 | 63 | **{63} only** | 15/20 |

The *only* case where human and assistant agree a specific image is
degenerate is `S2_HISTORY_BOTH_STAGES` case 63. Every other
assistant-flagged case (61, 68, 71, 77, 79 in `S2_HISTORY_BOTH_STAGES`;
77, 79 in `S2_STAGE1_HISTORY`) is judged clean by this reviewer, and
this reviewer additionally flags two cases (63, 72 in
`S2_STAGE1_HISTORY`) the assistant does not — including case 63 in
`S2_STAGE1_HISTORY` specifically, which the assistant's own C31
correction concluded was an *inherited, non-degenerate* pattern; this
reviewer's judgment disagrees with that specific correction. Neither
label is adjusted to match the other; both stand as independently
recorded.

**Local defects diverge sharply and are not comparable one-to-one.**
This reviewer flags localized structural/facial/anatomical defects on
39 of 40 images (effectively universal) — a far more liberal standard
than the assistant's own free-text notes, which flagged specific local
concerns on only a handful of cases (e.g. `S2_HISTORY_BOTH_STAGES`
cases 76 and 78, neither counted as whole-frame-degenerate). The
assistant's review schema for this results line does not include a
structured, directly-comparable local-defects field, so no numeric
agreement rate is computed for this dimension — the divergence is
reported qualitatively, as a difference in sensitivity/threshold
between reviewers, not adjudicated.

## 5. What this does and does not establish

- **Does not** reproduce the assistant's 2/20→6/20 regression claim
  under this one reviewer's whole-frame-degeneration judgments; the
  measured direction is a smaller improvement (2/20→1/20), not a
  regression, on this panel, from this one reviewer.
- **Does not** establish broad quality improvement, acceptable absolute
  quality, equivalence between the two checkpoints, or a new method —
  these are 20-case counts from one reviewer on one reused panel.
- **Does not** establish that local defects on 39/40 images were caused
  by the refinement edit, or by anything else — presence alone does not
  establish cause.
- **Does** establish that the specific case-level identities behind the
  assistant's and this reviewer's degeneration counts are almost
  entirely disjoint, which is itself informative: close or matching
  totals should not be read as confirmation of the same underlying
  cases.
- No hypothesis test, significance claim, or new success threshold is
  computed or needed for this finite, descriptive closeout.
- This is one reviewer. Per the frozen protocol, further reviewers'
  responses (if any are returned) are reported separately, not pooled
  into this result.

## 6. Textual errata corrected alongside this closeout

See `closeout/CORRECTIONS_BEFORE_HUMAN_REVIEW.md` (updated) and the
corresponding fixes applied directly to
`history_both_stages_s2/FINAL_REPORT.md` and `CORRECTIONS.md`:

- A mis-cited `S2_HISTORY` parachute comparator (9/20, which is
  French horn's figure) is corrected to 11/20 in both the report's
  headline paragraph and the `CORRECTIONS.md` C32 entry. (French horn's
  own 9/20 comparator was already correctly cited elsewhere and is
  unchanged.)
- The claim that the `S2_HISTORY_BOTH_STAGES` 6/20 count was "the first
  non-saturated count in this comparison" is corrected — `S2_STAGE1_HISTORY`'s
  own corrected 2/20 (C31) was already non-saturated and precedes it.
- The retained-class object-presence table in
  `history_both_stages_s2/FINAL_REPORT.md` §5 is corrected to stop
  implying `S2_STAGE1_HISTORY`'s full 19/20 and 16/2/2 breakdown was
  individually verified — only cases 32 and 54 were; this is now
  explicit in the table and its surrounding text.

None of these three items changes any disputed quality count; they are
documentation fixes, verified directly against existing classification
and correction records before being applied.

## 7. One next scientific recommendation

Given case-level agreement between this reviewer and the assistant's
own labels is low (16/20 and 15/20, with only one specific case in
common across both endpoints) while the *aggregate* direction also
flips (improvement vs. the assistant's claimed regression), the
single most informative next step is **not** another model experiment:
it is recruiting **one or two additional independent human reviewers**
on this exact same frozen package (same 20 pairs, same protocol,
already published and ready to reuse) to determine whether this
reviewer's direction, the assistant's direction, or neither is the more
consistent human read — before any further refinement-quality claim,
in either direction, is treated as established. Per the frozen
protocol, any additional reviewer's responses are reported separately,
not pooled into a manufactured consensus, so this does not require
reopening or modifying the existing package.
