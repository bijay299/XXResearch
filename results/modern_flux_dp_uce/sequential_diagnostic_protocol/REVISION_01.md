# Revision 01 — correction mapping for the coordinator's NEEDS_CHANGES review

Reviews the first published protocol (commit `7e166534e2fdced7fa0cf034d159d46aae38a925`)
against the coordinator's NEEDS_CHANGES verdict and maps every numbered
review issue to its specific correction in this revision. **This document
is itself protocol-only: no model, encoder, checkpoint tensor, GPU, or
experiment was run to produce it** — every correction below is a document
edit grounded in source re-inspection (already-cloned, unmodified) or CPU
arithmetic on existing numbers/hypothetical examples. Two existing
checkpoint files were additionally hashed at the file level (no tensor
loading) as part of this revision.

| # | Review issue (paraphrased) | Correction | Where |
|---|---|---|---|
| 1 | Primary question needed restoring to the exact given wording; design should not claim to isolate a "save/reload"/"serialization" effect | Primary question replaced verbatim; explicit statement added that comparators compare workflows, cannot isolate serialization, and that exact parent loading/reload remain separate technical gates | `PROTOCOL.md ## 1` |
| 2 | DP loops through *all* guide concepts for every erase token — pooled, not per-target assignment | New subsection with exact citation (`Double_proxy.py:174-189`) explaining the pooling behavior; corrected throughout (table, appendix commands, endpoint descriptions) | `PROTOCOL.md ## 5`, `## 6`, `## 7` appendix; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 6 |
| 3 | Batch call changes preservation sets (neither target preserved in a joint call) | New explicit statement that a batch call's preserve list excludes both erase targets, differing from SEQ's second edit (which preserves chain saw); reflected in the appendix's `DP_BATCH`/`UCE_BATCH` preserve lists | `PROTOCOL.md ## 5`, `## 6`, `## 7`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 9 |
| 4 | One-call DP casts to bf16 once; two-call sequence has an intermediate bf16 checkpoint/cast | New precision-boundary analysis: DP's native edit dtype is bf16 (one extra rounding step in SEQ vs. BATCH), UCE's is fp32 (lossless reload, no analogous step) — stated as DP-specific | `PROTOCOL.md ## 5`, `## 6`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 10 |
| 5 | UCE's batch call is a simultaneous solve in *ideal* arithmetic only; order invariance is not a bitwise-floating-point claim; requires reordering corresponding erase/guide pairs together | Explicit "ideal arithmetic" qualifier added; floating-point non-associativity noted; UCE's correct positional `zip`-based guide pairing (vs. DP's pooling) now stated as the actual mechanism | `PROTOCOL.md ## 5`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 8 |
| 6 | Rename JOINT endpoints to "native batch-edit comparator" throughout | `DP_JOINT`/`UCE_JOINT` renamed `DP_BATCH`/`UCE_BATCH` everywhere (protocol, JSON, ledger, appendix, README pointer) | all four published files |
| 7 | Describe DP as ordered-with-pooled-guides, UCE as simultaneous-paired-target-solve; include a changed-objectives/precision table; do not implement new per-target guide semantics or a replacement exact-joint algorithm | New `## 6` summary table added; explicit statement that no new guide semantics or replacement algorithm were implemented — native behavior used as released | `PROTOCOL.md ## 5`, `## 6` |
| 8 | Comparators cannot establish serialization caused a difference; parent loading/reload remain technical gates; do not add a serialization experiment | Explicit statement added in `## 1`; no new experiment proposed; technical-integrity gate (`## 9`) now explicitly includes a loaded-parent file-hash check (not previously present) | `PROTOCOL.md ## 1`, `## 9` |
| 9 | Keep A in SEQ's B preservation list; load cumulative parent before copying reference modules; references from the edited parent | Unchanged in substance from the first version (already correct); now explicitly tied to the fact that chain saw is *already* present in the released parachute preserve list, requiring no reordering | `PROTOCOL.md ## 4`, `## 7` |
| 10 | Separate the ideal identity `ΔW·C_pres=0` from native computation: SVD cutoff `1e-6`, fp32+SVD, skip threshold `1e-10`, final bf16 cast (`Double_proxy.py:215-238`) | All four points added explicitly with exact line citations; stated that a large null space does not establish a usable nonzero projection for the *current* erase vector | `PROTOCOL.md ## 4`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 4 |
| 11 | Illustrative bf16 rounding example (parent `[1,1]`, update `[3/1024,-3/1024]`, input `[1,1]`) — label as illustrative, not measured; no re-execution | Example reproduced verbatim and independently verified by hand (bf16 spacing arithmetic); explicitly labeled "illustrative arithmetic example, not a measured FLUX effect, no re-execution performed" | `PROTOCOL.md ## 4`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 4 |
| 12 | Protection concerns isolated content-token vectors and CLIP pooled input, not full-sequence/whole-generator invariance; actual-prompt mismatch concerns both conditioning streams, not just T5 | New paragraph explicitly covering both `context_embedder` (T5) and `time_text_embed.text_embedder.linear_1` (CLIP pooled, itself contextual) | `PROTOCOL.md ## 4`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 11 |
| 13 | Keep DP-vs-UCE relapse ranking explicitly PREDICTION; note token-selection/precision confounds | New paragraph stating the ranking prediction is confounded by native token-selection and precision-pathway differences, not attributable to hard/soft preservation alone | `PROTOCOL.md ## 4`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 22 |
| 14 | Specify preservation-residual logging plan for a future run (frozen prompts/computation/cost); add no image endpoints now; collect no logs now | New subsection specifying what a future implementation should log and the frozen evaluation prompts to use; explicitly states nothing was collected | `PROTOCOL.md ## 4`, final subsection |
| 15 | Remove all noise-band/floor language; M0 100/100 -> parent 0/100 is the erasure effect, not noise | "Noise band" language removed everywhere (rejection criterion, outcome branches, prediction) and replaced with the three explicit quantities in item 16 below | `PROTOCOL.md ## 1`, `## 10`, `## 14`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 22 |
| 16 | Define separately: raw parent->SEQ transitions; confirmed detector-discovered witnesses; matched SEQ-vs-batch transitions and net differences; specify whether "beyond batch" means SEQ-only-discordant or net difference | New `## 10` defines all three quantities explicitly, and specifies "beyond batch" means SEQ-only-discordant count, not net difference | `PROTOCOL.md ## 10` |
| 17 | One confirmed witness supports existence only; choose and justify a separate practical meaningful-aggregate-deficit criterion prospectively | New prospective threshold fixed: >=10/100 confirmed witnesses = meaningful aggregate deficit; explicitly labeled a judgment call, not a significance calculation | `PROTOCOL.md ## 10`; `ELIGIBILITY_AND_ENDPOINTS.json` |
| 18 | Rewrite rejection/outcome branches: zero relapse can end diagnostic without comparator; shared relapse changes comparative conclusion, does not erase observation; specify objective per-method comparator triggers; define ambiguous-witness treatment; do not leave "unexplained deficit" undefined | `## 14` fully rewritten with 9 explicit branches, including the objective per-method trigger rule and an ambiguous-witness branch; "unexplained deficit" language removed, replaced with the operationalized threshold + comparator-qualification language | `PROTOCOL.md ## 14` |
| 19 | Preserve direct/sequential B suppression gates and <=10/100 comparability band; explain practical rationale; out-of-band result is a confound, not proof of causation | Gate table now includes an explicit rationale sentence for the 10-case band and a sentence stating an out-of-band result is a confound for interpretation, not proof of causation | `PROTOCOL.md ## 9` |
| 20 | Choose and justify explicit fixed-panel retention acceptance bounds, per-class and aggregate, naming the reference endpoint; exclude A and B from the 8-class aggregate (already done); label descriptive quantities as descriptive; prohibit "all quality gates clean" from a descriptive quantity alone | Two new retention gate rows added (vs. M0, cumulative; vs. A-parent, incremental), each with explicit numeric bounds, a named reference endpoint, and an explicit "descriptive, not proof" / "does not establish all quality gates clean" statement | `PROTOCOL.md ## 9` |
| 21 | Reuse the original fixed all-class descriptive panel (var 0, first two CSV rows per class, 20 cases), showing M0/parent/direct-B/SEQ/batch | Rendering section now explicitly reuses the same 20 cases from the completed reproduction, across all five applicable endpoints | `PROTOCOL.md ## 12` |
| 22 | Define operational rule for quality-confounded/ambiguous results: target substitution and ordinary M0 defects are not automatically degeneration; small panel cannot establish full-panel rate or human validation | New operational rule added distinguishing guide substitution and pre-existing M0 defects from genuinely new rendering damage; limitation restated explicitly | `PROTOCOL.md ## 12` |
| 23 | Qualify each batch comparator on both erased targets, B-deletion comparability, and retention/rendering requirements before calling it successful; report failed comparators honestly | New "batch comparator qualification" gate added; a comparator that fails is explicitly to be reported as failed, not silently used | `PROTOCOL.md ## 9` |
| 24 | Add outcome: weaker baseline fails, stronger applicable baseline succeeds (e.g., UCE relapse + qualified DP success != need for new method) | New outcome branch 7 added with this exact distinction | `PROTOCOL.md ## 14` |
| 25 | Every outcome stops the diagnostic; no automatic follow-up | Restated explicitly at the end of `## 14` (unchanged in substance, kept) | `PROTOCOL.md ## 14` |
| 26 | Freeze paired witness verification: inspect M0/A-parent/SEQ together (plus batch image for comparator claims) for every candidate relapse; use the union of candidate positives so verification is not asymmetric; freeze 4 categories (confirmed relapse / residual parent content / final classifier false positive / ambiguous); require evidence of both parent absence and final presence; a bare chain/stick is not a chain saw | New `## 11` fully specifies this rule, replacing the prior, less precise historical-outcome-only framing. Explicitly: no image has been inspected to write this rule | `PROTOCOL.md ## 11` |
| 27 | Preserve raw classifier results, report verification separately; detector-discovered witnesses are not a comprehensive true relapse-rate estimate or absence certificate | Explicit sentence added in `## 11` | `PROTOCOL.md ## 11` |
| 28 | Describe as assistant visual checks, not independent human validation; document rules only, do not inspect/label images now | Explicit sentence added in `## 11`; confirmed no image was inspected in this revision (see below) | `PROTOCOL.md ## 11`, this document |
| 29 | Exactly one literal prompt per target, 100 distinct seeds across 10 CSV rows; do not imply ten distinct contexts | Verified directly against `SD/data/small_imagenet_prompts.csv` (all 10 rows per class share identical prompt text); language corrected throughout | `PROTOCOL.md ## 8`, `## 13`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 16 |
| 30 | Keep parachute; reframe SD1.x-exclusion as a selection preference, not proof of statistical independence or a requirement from the brief; no new outcomes informed selection; do not repeat the old-line search | Selection-rule step 3 rewritten explicitly as "a documented preference, not a proof"; confirms no new search was performed (same existing grep as before, not repeated or expanded) | `PROTOCOL.md ## 3`; `ELIGIBILITY_AND_ENDPOINTS.json` |
| 31 | Use the exact released DP parachute preservation-list order for direct B; freeze the intended sequential order; document necessary deviations | Verified the exact released order from `Double_Parachute_flux.sh`; used unchanged (and unreordered) for `DP_B_direct`, `UCE_B_direct`, `DP_SEQ`, `UCE_SEQ`; the only documented deviation is reusing this DP-sourced order for UCE (no independent UCE order exists) | `PROTOCOL.md ## 3`, `## 7`, appendix; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 15 |
| 32 | Pin reused parent checkpoint hashes from existing records/files; matching two keys alone does not establish correct parent identity | Both A-parent `.safetensors` files hashed at the file level (sha256, no tensor loading) and recorded; explicit statement that key-name matching is a separate, insufficient check | `PROTOCOL.md ## 2`, `## 9`; `ELIGIBILITY_AND_ENDPOINTS.json`; `SOURCE_AND_ASSUMPTION_LEDGER.md` item 14 |
| 33 | Freeze edit-time sequence length separately from generation-time length | Both explicitly stated as `256` (source-confirmed, same code path's `'schnell' in model_id` branch for edit-time; native default for generation-time), recorded as two separate facts rather than assumed identical | `PROTOCOL.md ## 8` |
| 34 | Preserve native method precision and token behavior | Restated explicitly; no changes made to either method's native token handling anywhere in this revision | `PROTOCOL.md ## 8` |
| 35 | Mark B candidates provisionally recognition-qualified in JSON | `candidate_B_table` statuses updated to "provisionally_recognition_qualified..." with an explanatory note field | `ELIGIBILITY_AND_ENDPOINTS.json` |
| 36 | Synchronize endpoint names, conditions, gates and budgets across all documents; fix broken section references | `DP_BATCH`/`UCE_BATCH` naming, gate definitions, and the 6-endpoint/6,000-image budget are now identical across `PROTOCOL.md`, `ELIGIBILITY_AND_ENDPOINTS.json`, and `SOURCE_AND_ASSUMPTION_LEDGER.md`; `PROTOCOL.md`'s internal `## N` references were checked against the renumbered section list after the rewrite | all three documents |

## Unresolved limitations (carried forward honestly, not resolved by this revision)

- Items 18, 19, and 21 in `SOURCE_AND_ASSUMPTION_LEDGER.md` remain
  UNESTABLISHED: whether DP's exact (ideal-arithmetic) preservation
  identity transfers to the contextual, in-prompt token embeddings of
  either conditioning stream; whether UCE's soft preservation term stays
  close enough to the parent's response in practice; whether parachute's
  own erase-token vectors clear DP's `1e-10` skip threshold. None of these
  were, or could be, resolved without running the edits — which this
  revision does not do.
- The meaningful-aggregate-deficit threshold (`>=10/100` confirmed
  witnesses) and the retention acceptance bounds (`## 9`) are judgment
  calls, stated as such, not derived from any statistical procedure.
- The DP-vs-UCE relapse-magnitude ranking remains a confounded prediction,
  not a claim this protocol can cleanly attribute to hard-vs-soft
  preservation even after execution, because of the native token-selection
  and precision-pathway differences noted above.
- No experiment was run to confirm this revision's technical description
  (e.g., that parachute's tokens actually survive DP's skip threshold, or
  that the batch comparators behave as the source-level analysis predicts)
  — this remains a documentation-only correction pass, per the authorized
  scope.

## Confirmation

No model was loaded, no checkpoint tensor was loaded or deserialized
(file-level sha256 hashing only), no encoder was run, no feature was
computed, no checkpoint was edited, no image was generated, classified, or
annotated, and no GPU was allocated or polled to produce this revision.
