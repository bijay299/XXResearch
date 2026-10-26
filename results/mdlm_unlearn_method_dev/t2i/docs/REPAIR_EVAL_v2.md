# REPAIR_EVAL_v2 — corrected re-scoring of the FINAL split (defect D-013)

Status: BUILT AND CPU-TESTED. **No GPU work has been launched** (`FREEZE_ACTIVE`; the
implementation lead executes the bounded run). Date 2026-10-26.

Owned files, all new except the last:

| file | role |
| --- | --- |
| `scripts/clip_scorer.py` | shared scorer: one source of truth for classifier, label text, identity mapping |
| `scripts/rescore_final_v2.py` | the corrective re-evaluation pipeline |
| `scripts/test_rescore.py` | 46 CPU-only tests |
| `scripts/gen_eval_final.py` | repaired: refuses overwrite, writes atomically, language corrected |
| `docs/REPAIR_EVAL_v2.md` | this document |

`scripts/gen_eval.py` and `scripts/attack_relearn.py` are **unmodified**. `clip_scorer.py` imports
*from* `gen_eval.py`, never the reverse, so `gen_eval.py` stays self-contained and every one of the
~800 historical records reproduces unchanged (test `T7f`).

---

## 1. The defect, stated at the strength the evidence supports

`gen_eval.py` carries two twelve-element lists: `LABELS` (line 48), the **CLIP text**, and
`LAB_KEY` (line 50), the **record keys**. `gen_eval_final.py` passed `list(ge.CONCEPTS)`
(element-wise equal to `LAB_KEY`) as the CLIP text. Verified mechanically by
`clip_scorer.label_set_diff()`:

* **nine of twelve strings differ** — `garbage_truck`, `van_gogh`, `cassette_player`, `chain_saw`,
  `english_springer`, `french_horn`, `gas_pump`, `golf_ball`, `monet`;
* **three are byte-identical** — `church`, `parachute`, `tench`;
* of the nine, **six are underscore-only** edits (`cassette_player`, `chain_saw`, `french_horn`,
  `garbage_truck`, `gas_pump`, `golf_ball`) and **three are genuine rewordings**
  (`english_springer` → `english springer spaniel`, `monet` → `monet painting`, `van_gogh` →
  `van gogh painting`). An underscore is not cosmetic: it is a different token sequence and so a
  different text embedding. All nine are counted as differing (test `T8d`).

**The direction and size of the consequence are NOT established.** The only sanctioned reading
until the GPU run produces numbers is, verbatim and carried in every emitted record:

> The evaluator used inconsistent label text. The effect on scores and conclusions is unresolved
> pending corrected evaluation.

Specifically not assumed anywhere in the code or this document: that the legacy wording weakened
competitors; that any historical count was inflated or deflated; that `church`, `parachute` or
`tench` arms are a priori unaffected because their own text did not change — the decision is a
closed-set argmax over all twelve, so a change to any competitor's text can move any arm's decision
in either direction. An earlier comment in `gen_eval_final.py` asserted an UPWARD bias; that
assertion was unsupported and is explicitly marked **withdrawn** (tests `T8f`, `T8f2`).

### Exposure in the historical records — corrected 2026-10-26

**37 of the 180 images** in `outputs/final_*.json` were classified as one of the nine differing
labels. Recomputed independently by the lead, the reviewer, and
`rescore_final_v2.recompute_exposure()`:

| arm | on a differing label | median winning margin |
| --- | --- | --- |
| `utrap_sd15_church_l0.15k8` | 12/15 | 0.0063 |
| `utrap_church_l0.15k8` | 10/15 | 0.0044 |
| `frp_church_r3_l05` | 7/15 | 0.0051 |
| `uce_sd15_church_lam1000` | 5/15 | 0.0027 |
| `BASE_sd14_anchor` | 2/15 | 0.0110 |
| `ATK_uce_sd15_church_lam1000` | 1/15 | 0.0081 |
| the other six arms | 0/15 | — |

Median winning margins across the four defence arms span **0.0027–0.0063**.

**Small margins mean the outcomes are sensitive to label text; they do not indicate which way.**
This count is **not** a bound on how many decisions could move: the decision is a closed-set argmax
over all twelve labels, so an image predicted as an *unchanged* label (`church`, `parachute`,
`tench`) can also flip when a competitor's text changes. Images predicted on a differing label are
where a change is most directly visible, not where it is confined.

> **Withdrawn.** An earlier version of this document and of `gen_eval_final.py` stated
> "26 of their 180 images ... concentrated in the four pre-attack defence arms (9/15, 7/15, 5/15,
> 4/15) at median winning margins 0.0065–0.0131 — a count of affected decisions, which bounds how
> many could move". Every part of that is wrong: the total (37, not 26), the per-arm counts (which
> summed to 25, not even to their own stated total), the margin range, the claim that exposure is
> confined to the defence arms, and the bound claim — which also contradicted the closed-set
> coupling argued two paragraphs above it. It is withdrawn in full. `DOCUMENTED_EXPOSURE` in
> `rescore_final_v2.py` is now asserted equal to the recomputation by test `T11`, because a test
> that greps for *absent* strings (`T8f`) can only catch a claim that was removed, never a wrong
> one that is present.

---

## 2. What the run is, and what it is not

`rescore_final_v2.py` labels itself, in the manifest and in every arm record:

* `evaluation_role: "CORRECTIVE_RE_EVALUATION"`,
  `split_status: "INSPECTED_BEFORE_THIS_RUN"`. The FINAL split was inspected in **FINDING65** and
  **FINDING66**. These numbers are **development measurements** repairing a scoring defect. They are
  **not** a fresh held-out confirmation and must never be reported as one.
* `method: "REGENERATION_PLUS_CORRECTED_SCORING"`. The original images were **never saved**, so the
  original pixels cannot be re-scored. Images are regenerated from the recorded prompts and seeds.
  **Matching seeds do not establish byte-identical regeneration** — the original run recorded no
  library, driver or device versions, and fp16 generation depends on all three. Determinism is an
  **assumption, tested**, never a premise (§4).
* **Original** arm precommitment is kept strictly separate from **this** evaluation's status.
  `original_arm_precommitted` is a fact about when an arm entered `ARMS` (7 pre-committed,
  5 post-hoc); `this_evaluation_status` is the same for every arm. Precommitment describes arm
  *selection*; it does not restore the split's held-out character (tests `T9a`, `T9b`).

---

## 3. Design

**Versioned run directory, overwrite impossible.** Everything lands in
`outputs/rescore/<run_id>/`, `run_id = rsc_<UTC timestamp>Z_<16 hex>[_tag]`. `make_run_dir` refuses
an existing directory outright; `_guard_path` refuses any write resolving into
`outputs/final_*`, `outputs/gen_*`, `adapters/`, `attacked/`, or anywhere outside the run root.

Three reviewer findings are fixed here:

* **BLOCKING 2 — traversal through directory creation.** `make_run_dir` called `os.makedirs`
  directly, so the guard covered *file* writes only and
  `--run-tag '../../../attacked/PWNED'` created `<P>/attacked/PWNED/{images,embeddings,arms}`.
  Reproduced before fixing. Now every mkdir goes through `guarded_makedirs`, the whole tree is
  guard-checked *before* anything is created (so a refusal creates nothing at all, not a partial
  tree), and `--run-tag` is validated against `^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$` — no separators,
  no dots — at the top of `main`, before any work (tests `T1f`, `T1g`).
* **Defect 3 — string-prefix containment.** `_guard_path` used `str.startswith`, so
  `outputs/rescore_evil/x.json` was *allowed*: it has `outputs/rescore` as a string prefix but is
  not inside it. Containment is now compared **component by component** via `_components` /
  `_is_within`, with `realpath` normalising `..` before the comparison (test `T1h`).
* **Defect 6 — flaky run-id uniqueness.** The salt was 6 hex = 24 bits, at which 2000 draws
  collide ~11 % of the time; the previous write-up pasted `2000/2000` as if deterministic. The salt
  is now 16 hex = 64 bits, and within-process uniqueness is **guaranteed by construction** through
  an issued-id set with retry, so `T3a` is deterministic rather than probabilistic; `T3a2` asserts
  the width (tests `T3a`, `T3a2`).

**Atomic writes.** `atomic_write_json` / `atomic_write_bytes` create a temp file in the *destination
directory*, `fsync`, then `os.replace` — a same-filesystem rename, so a reader sees the old file or
the complete new one, never a truncation. A failure mid-write leaves the prior content and no temp
litter (test `T2`).

**Everything needed for future re-scoring is persisted.** Per arm: every generated PNG
(`images/<label>/p<pi>_k<k>.png`); L2-normalised CLIP image embeddings plus both full 12×N
similarity matrices (`embeddings/<label>.npz`); and in `arms/<label>.json`, per image, the
**complete twelve-label score vector under both label sets** — not just the argmax — with
`pred_label`, `margin`, `runner_up_label`, `target_sim`, `target_rank`, prompt, image index and
seed. Any future label set can be scored from the saved embeddings with no regeneration
(tests `T6a`, `T6b`).

**Paired scoring.** `ClipScorer.score_all_sets` computes image features **once** and scores both
the corrected set (`gen_eval.LABELS`) and the legacy set (`gen_eval.LAB_KEY`) against them. The
image side is bit-identical across the pair, so any difference in the decision is attributable to
the label **text** alone, *within this run*. `pair_records` refuses a ragged or single-set pair
rather than silently truncating the comparison (tests `T6c`–`T6f`). Per arm the record reports
`corrected_minus_legacy_target_hits/_acc`, `n_images_whose_prediction_changed` and
`n_images_whose_target_decision_changed`.

**One source of truth.** `clip_scorer.py` reads `LABELS`/`LAB_KEY`/`CONCEPTS` from the unmodified
`gen_eval.py` and asserts `CONCEPTS` ordering still matches `LAB_KEY`, so prediction indices cannot
silently mean the wrong concept. `verify_numerics_contract()` re-derives gen_eval's prompt list from
`PROMPT_TEMPLATE` and greps gen_eval's source for its normalisation, failing loudly if either has
drifted; both `rescore_final_v2.py` and the repaired `gen_eval_final.py` abort if the contract is
broken (tests `T7e`, `T7f`).

**Unmodified-file baseline** (reviewer defect 7). The claim that `gen_eval.py` and
`attack_relearn.py` are unmodified previously rested on **mtime alone** — the only prior hash record
(`outputs/provenance_manifest.json`) predates this repair and truncates its per-file hashes to 16
hex characters. Full sha256 values are now recorded in `UNMODIFIED_BASELINE_SHA256`
(`gen_eval.py` `6f978e60...d131f51a`, `attack_relearn.py` `ec5fe224...44fb40e2`, taken 2026-10-26)
and verified by `verify_unmodified_baseline()` on every run and in test `T12`. **Scope:**
byte-identity is proven only from the baseline date onward; for the period before it the evidence
is still mtime only, and nothing stronger is claimed.

**Provenance** (`run.json → provenance`, test `T7`): file sha256 of
`rescore_final_v2.py`/`clip_scorer.py`/`gen_eval.py`/`gen_eval_final.py`/`split_registry.py` —
the source tree is **not** a git repository, and that is recorded rather than papered over
(a git rev is captured if one ever exists); per-arm checkpoint sha256 with an explicit hash **scope**
(`FULL` or `FIRST_<n>_BYTES`); per-arm base model, **revision and variant** (the plan spans two
distinct base revisions); classifier id with its **unpinned-revision** gap restated; the numerics
recipe; scheduler class and full config, read off the instantiated pipeline per arm; steps,
guidance, `n_per`, resolution, safety-checker state; the FINAL prompts with split version and freeze
timestamp; all **15 seeds enumerated**, with the `seed_for('final_eval', pi*100+k)` formula; the
label-set diff; `rescore_version` and `scoring_version`; full environment (python, platform, torch
2.4.1+cu124, diffusers 0.31.0, transformers 4.46.3, numpy, accelerate, safetensors, PIL, CUDA,
cuDNN, GPU name, TF32 state) — carrying the caveat that these are *this* run's versions and the
original recorded none, which is precisely why bit-identical regeneration cannot be asserted.

**Explicit status, never silence.** Each arm ends `complete`, `missing_checkpoint` or `failed`, and
in the latter two cases a record is still written naming the cause and its consequence — no arm is
silently skipped. The manifest is written `incomplete` *before any arm runs*, so a crashed run can
never read as a finished one; exit code is non-zero unless the run is exactly `complete`
(tests `T4`, `T5`).

`run_status` is one of `complete`, `complete_comparability_mixed`,
`complete_not_comparable_to_history`, `partial`, `incomplete`, `failed`. **`complete` requires both
that every requested arm completed and that every completed arm reproduced its historical record
exactly** (`derive_history_comparability`). This was reviewer defect 5 and is the one that matters
most scientifically: previously an all-`DISCREPANT` run — a run whose numbers cannot be placed
against the historical ones at all — was reported as `run_status: "complete"`, i.e. as a successful
correction. It is now `complete_not_comparable_to_history`, and the manifest states that the
*within-run* corrected-vs-legacy comparison remains valid while any claim about what the original
records would have become is lost. A missing probe is aggregated as `unknown`, never as fine
(tests `T4a`, `T4d`, `T4e`).

---

## 4. Determinism: tested, not assumed

For every arm, the **legacy-label** reproduction is compared image-by-image against the historical
`outputs/final_<label>.json` (`compare_with_history`): per-image `pred_label` agreement and the
disagreeing indices, max/mean |Δtarget_sim|, max |Δmargin|, hits and accuracy both ways, and whether
the historical record carries the `label_text_set` field (its **absence** is the pre-repair marker).
Verdict is `EXACT_REPRODUCTION` or `DISCREPANT` — and `DISCREPANT` is reported as a finding, not
smoothed away:

> the original images were never saved, so the cause cannot be separated here into (a)
> nondeterministic regeneration under a different library/driver/device, or (b) any other change.
> The corrected-vs-legacy comparison WITHIN this run remains valid regardless, because both label
> sets score the SAME regenerated images; what a discrepancy limits is the comparability of this
> run to the historical numbers.

Tests `T10a`–`T10d` exercise absent history, exact match against a real historical record, and a
single perturbed prediction (correctly reported as `DISCREPANT`, index `[3]`, max |Δtarget_sim|
0.0100).

---

## 5. What this cannot measure

1. **Six of the twelve historical records cannot be corrected at all.** Every attacked checkpoint
   behind the `ATK_*` arms (`attacked/E6B_*`, `E6C_*`, `E6D_*`) is **absent from disk**. Those arms
   emit `missing_checkpoint` and force the run to `partial`. Their historical FINAL numbers are not
   recoverable from the retained artefacts by this or any pipeline. Regenerable: the four defence
   arms (`utrap_church_l0.15k8`, `frp_church_r3_l05`, `utrap_sd15_church_l0.15k8`,
   `uce_sd15_church_lam1000`) and the two base anchors (`BASE_sd14_anchor`, `BASE_sd15_anchor`).
   Confirmed by test `T5d`.
2. **It cannot re-score the original pixels** — only regenerate and score. See §4.
3. **It is not a held-out confirmation** and cannot be made into one: FINAL was inspected in
   FINDING65/66.
4. **It does not pin the CLIP classifier revision.** Unpinned in every historical run; left
   unpinned deliberately so the re-scoring uses the same cached weights rather than introducing a
   second classifier identity into the comparison. Disclosed, not patched.
5. **It measures the label-text effect for the arms it runs, on one target (`church`) and 15 images
   per arm.** Quantisation is 1/15 ≈ 0.0667. Results for one arm are not evidence about another, and
   the paired difference is a measurement, not a prediction that was made in advance in either
   direction.

---

## 6. Repairs to `gen_eval_final.py`

Kept runnable for reproducing the historical procedure, with these changes:

* **Refuses to overwrite** `outputs/final_{label}.json`. The pre-flight check runs *before* any
  model import or GPU touch, names every colliding path, points at `rescore_final_v2.py`, and exits
  non-zero. A second guard catches a record appearing mid-run. `--allow-overwrite` is the only way
  past it and is documented as destroying historical evidence (test `T1d`).
* **Atomic write** (same-directory temp + `fsync` + `os.replace`) instead of `open(..., "w")`.
* **`first_and_only_legitimate_use` is gone.** It was emitted unconditionally `True`, including for
  arms the file itself documents as post-hoc. Replaced by `original_arm_precommitted` /
  `original_arm_precommitment_note` (about arm selection) alongside
  `this_evaluation_status: "DEVELOPMENT_MEASUREMENT_ON_INSPECTED_SPLIT"` (about this measurement).
* **"never used" / "FIRST and ONLY legitimate use" language removed** from the header, the record
  notes and the summary banner. The FINAL prompts were never used for *selection*, but they have
  been *inspected*, so they are no longer held-out — the output says exactly that.
* **Directional claims withdrawn** in the D-013 comment and the `label_text_note`, replaced by the
  sanctioned interpretation.
* **Routed through the shared scorer** (`clip_scorer.ClipScorer`), with the numerics contract
  asserted at startup. Emitted field names and semantics are unchanged, so new records stay
  comparable field-by-field with the pre-repair ones.
* **Missing checkpoints are recorded, not silently skipped**: named in the end-of-run summary as
  `ARMS NOT EVALUATED`, with a non-zero exit.
* `images_saved: false` is stated in the record, with a pointer to the pipeline that does save them.

---

## 7. Test output

`python scripts/test_rescore.py` — CPU only, no CUDA, no diffusers pipeline, no CLIP weights, no
multi-GB checkpoint reads. Run on 2026-10-26 with `CUDA_VISIBLE_DEVICES=""`. **69 pass, 0 fail**
(46 before the review; 23 added for the seven defects).

```
==============================================================================
CPU tests: corrected FINAL re-scoring pipeline (D-013). No GPU, no model weights.
==============================================================================

--- t1_overwrite_refusal ---
[PASS] T1a_run_dir_write_once: refused: REFUSING to reuse run directory /data/bijaypandey/tmp/tmpkgk...
[PASS] T1b_guard_refuses_historical_paths: 6/6 protected paths refused
[PASS] T1c_atomic_write_refuses_and_leaves_history_intact: refused: REFUSING to write /data/bijaypandey/t2i_unlearn/outputs/final_BASE_sd1... historical record intact=True
[PASS] T1d_gen_eval_final_refuses_overwrite: exit=1, refusal message present=True, points at rescore_final_v2=True
[PASS] T1e_all_12_historical_records_present: 12 outputs/final_*.json present
[PASS] T1f_run_tag_charset_validated: 13/13 malicious tags rejected; plain labels accepted
[PASS] T1g_directory_creation_cannot_traverse: refused: REFUSING to create run directory /data/bijaypandey/tmp/tmp_4mcbz6g/out... nothing created outside the run root=True
[PASS] T1h_component_wise_containment: 4/4 sibling-prefix paths refused; legitimate in-root path still allowed

--- t2_atomicity ---
[PASS] T2a_atomic_write_replaces_in_place: v 1 -> 2 at one path
[PASS] T2b_failed_write_leaves_prior_content_and_no_temp: raised=True content v=2 (expected 2) temp_files=[]
[PASS] T2c_atomic_write_bytes: binary write atomic, no temp litter

--- t3_run_id_uniqueness ---
[PASS] T3a_run_ids_unique_by_construction: 5000/5000 distinct over 5000 draws; guaranteed by the issued-id set, not by salt luck
[PASS] T3a2_salt_width_sufficient: salt=16 hex = 64 bits; P(collision in 2000 draws) = 1.08e-13 (was 6 hex = 24 bits, P = 1.1e-01)
[PASS] T3b_run_id_shape: prefixed+salted; tagged example rsc_20261026T191004Z_ba95bfebc02a020e_mytag
[PASS] T3c_versioned_run_root: run root = /data/bijaypandey/t2i_unlearn/outputs/rescore

--- t4_status_handling ---
[PASS] T4a_status_derivation: 11/11 status cases correct
[PASS] T4b_complete_requires_every_arm: 'complete' unreachable with any missing/failed/unattempted arm
[PASS] T4d_all_discrepant_run_is_not_complete: run_status='complete_not_comparable_to_history', comparability='none_exact'; the within-run label-text comparison is still stated as valid
[PASS] T4e_comparability_aggregation: 6/6 comparability cases correct
[PASS] T4f_exit_code_gated_on_complete: process exits non-zero unless run_status is exactly 'complete'
[PASS] T4c_manifest_written_incomplete_before_any_arm: run.json is written with run_status=incomplete before the scorer is even constructed

--- t5_missing_checkpoint ---
[rescore] ATK_ghost: MISSING_CHECKPOINT (attacked/DOES_NOT_EXIST/unet.pt)
[PASS] T5a_missing_checkpoint_is_explicit: status='missing_checkpoint', detail+consequence recorded
[PASS] T5b_missing_arm_record_is_written_not_skipped: arms/ATK_ghost.json written; no scores fabricated
[PASS] T5c_missing_arm_forces_partial: run_status='partial'
[PASS] T5d_observed_checkpoint_inventory: 12 arms planned; regenerable=['BASE_sd14_anchor', 'BASE_sd15_anchor', 'frp_church_r3_l05', 'uce_sd15_church_lam1000', 'utrap_church_l0.15k8', 'utrap_sd15_church_l0.15k8']; NOT regenerable (checkpoint gone)=['ATK_frp_church_r3_l05', 'ATK_uce_sd15_church_lam1000', 'ATK_uce_sd15_church_lam1000_sd1', 'ATK_uce_sd15_church_lam1000_sd6', 'ATK_utrap_church_l0.15k8', 'ATK_utrap_sd15_church_l0.15k8']

--- t6_paired_scoring ---
[PASS] T6a_all_twelve_scores_retained: every record carries all 12 label scores keyed by identity key, not just the argmax
[PASS] T6b_argmax_margin_rank_target_sim: pred/margin/target_rank/target_sim correct for a target win and a target loss
[PASS] T6c_pairing_keeps_both_sets_per_image: each paired row carries both label sets' full records plus prompt/seed provenance
[PASS] T6d_label_text_change_flags: pred_changed_by_label_text flags the images whose decision differs between sets
[PASS] T6e_ragged_or_single_set_refused: refused: paired scoring bookkeeping broken: label set 'legacy' has 1 ... and a lone set is refused
[PASS] T6f_both_sets_score_the_same_features: score_all_sets computes image features ONCE and scores every label set against them

--- t7_provenance ---
[PASS] T7a_top_level_provenance_keys: all 14 present
[PASS] T7b_provenance_fields_populated: 18 nested fields populated
[PASS] T7c_per_arm_base_revision_and_variant: 12 arms, 2 distinct base revisions, variant recorded for all
[PASS] T7d_seeds_enumerated: 15 distinct final_eval seeds recorded, first=4000000
[PASS] T7e_numerics_contract_with_gen_eval: issues=none
[PASS] T7f_gen_eval_unchanged_and_self_contained: gen_eval.py keeps its own clip_scorer/LABELS/LAB_KEY and does not import clip_scorer.py

--- t8_label_sets ---
[PASS] T8a_twelve_labels: n_labels=12
[PASS] T8b_exactly_nine_differ: n_differ=9; differing=['garbage_truck', 'van_gogh', 'cassette_player', 'chain_saw', 'english_springer', 'french_horn', 'gas_pump', 'golf_ball', 'monet']
[PASS] T8c_exactly_three_identical: identical=['church', 'parachute', 'tench']
[PASS] T8d_underscore_only_differences_counted: 6 underscore-only ['cassette_player', 'chain_saw', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball'] + 3 reworded ['english_springer', 'monet', 'van_gogh'] = 9 differing
[PASS] T8d2_prose_split_matches_computation: gen_eval_final.py states 6 underscore-only (garbage_truck included) + 3 reworded, matching the computed split
[PASS] T8e_identity_mapping_shared: LAB_KEY is the identity mapping under both label sets; indices mean the same concept
[PASS] T8f_no_directional_claim_encoded: all 4 pre-repair directional assertions removed from the 3 owned files
[PASS] T8f2_direction_disclaimer_present: all 3 owned files state that direction and size are unestablished; the earlier UPWARD assertion is explicitly marked withdrawn
[PASS] T8g_sanctioned_interpretation_recorded: every record carries: 'The evaluator used inconsistent label text. The effect ... is unresolved pending corrected evaluation.'

--- t9_role_labelling ---
[PASS] T9a_this_eval_role_uniform_and_corrective: every arm labelled CORRECTIVE_RE_EVALUATION_OF_INSPECTED_SPLIT regardless of original precommitment
[PASS] T9b_original_precommitment_kept_separate: original pre-committed=7, post-hoc=5 (['BASE_sd15_anchor', 'ATK_frp_church_r3_l05', 'ATK_uce_sd15_church_lam1000', 'ATK_uce_sd15_church_lam1000_sd1', 'ATK_uce_sd15_church_lam1000_sd6'])
[PASS] T9c_gen_eval_final_language_repaired: no unconditional first_and_only_legitimate_use field; no 'never used'/'FIRST and ONLY' header claim
[PASS] T9d_rescore_self_describes_correctly: run labels itself INSPECTED_BEFORE_THIS_RUN and method=REGENERATION_PLUS_CORRECTED_SCORING

--- t10_history_comparison ---
[PASS] T10a_absent_history_reported: status='NO_HISTORICAL_RECORD'
[PASS] T10b_exact_reproduction_detected: verdict=EXACT_REPRODUCTION, agreement=1.0, hits 1 vs history 1
[PASS] T10c_discrepancy_reported_not_assumed: verdict=DISCREPANT, disagreeing_indices=[3], max|dtarget_sim|=0.0100
[PASS] T10d_prerepair_marker_detected: historical record carries no label_text_set field, the pre-repair marker

--- t11_exposure_figure ---
[PASS] T11a_recomputed_over_all_twelve_records: 12 records, 180 images scanned
[PASS] T11b_documented_total_equals_recomputed: documented 37/180 == recomputed 37/180
[PASS] T11c_documented_per_arm_equals_recomputed: per-arm {'ATK_uce_sd15_church_lam1000': 1, 'BASE_sd14_anchor': 2, 'frp_church_r3_l05': 7, 'uce_sd15_church_lam1000': 5, 'utrap_church_l0.15k8': 10, 'utrap_sd15_church_l0.15k8': 12}
[PASS] T11d_per_arm_counts_sum_to_total: sum(per_arm) = 37 == total 37
[PASS] T11e_documented_margin_range_equals_recomputed: defence-arm medians {'utrap_church_l0.15k8': 0.0044, 'frp_church_r3_l05': 0.0051, 'utrap_sd15_church_l0.15k8': 0.0063, 'uce_sd15_church_lam1000': 0.0027} span [0.0027, 0.0063] == documented [0.0027, 0.0063]
[PASS] T11f_exposure_not_confined_to_defence_arms: non-defence arms also carry exposure: {'ATK_uce_sd15_church_lam1000': 1, 'BASE_sd14_anchor': 2}; 'concentrated in the four defence arms' is false and is withdrawn
[PASS] T11g_wrong_figure_absent_except_as_withdrawal: the withdrawn figures appear only inside explicit withdrawals
[PASS] T11h_correct_figure_documented_everywhere: 37/180 and the 0.0027-0.0063 margin range present in: ['scripts/gen_eval_final.py', 'scripts/rescore_final_v2.py', 'docs/REPAIR_EVAL_v2.md']
[PASS] T11i_bound_claim_removed: the 'bounds how many could move' claim is removed; EXPOSURE_FRAMING states it is NOT a bound, because an unchanged-label prediction can also flip
[PASS] T11j_register_framing_matched: framing matches the audit register and FINDING65 verbatim

--- t12_unmodified_baseline ---
[PASS] T12a_baseline_covers_both_protected_files: ['attack_relearn.py', 'gen_eval.py']
[PASS] T12b_both_files_match_recorded_sha256: gen_eval.py=6f978e60870486b4... match=True; attack_relearn.py=ec5fe2249b19df3b... match=True
[PASS] T12c_full_sha256_not_truncated: 64-hex full digests recorded (the prior provenance_manifest.json truncates to 16)
[PASS] T12d_baseline_scope_not_overclaimed: baseline taken 2026-10-26; scope_note says byte-identity is proven only from that date onward, mtime only before it
[PASS] T12e_mismatch_is_detected: a wrong expected hash makes verify_unmodified_baseline() report ok=False

==============================================================================
PASSED 69   FAILED 0
==============================================================================
```

Also verified after the test run: `outputs/rescore/` does not exist (nothing was created outside the
tests' temp dirs); all 12 `outputs/final_*.json` retain their original mtimes (2026-10-25T23:55 –
2026-10-26T01:07, all predating this work); no `.tmp_*` litter in `outputs/`; `gen_eval.py` still
imports and exposes `LABELS[2] == "garbage truck"`, `LAB_KEY[2] == "garbage_truck"`.

A no-GPU dry run of the plan is available and was exercised:

```bash
source /data/bijaypandey/t2i_unlearn/env.sh
CUDA_VISIBLE_DEVICES="" python /data/bijaypandey/t2i_unlearn/scripts/rescore_final_v2.py --plan-only
```

It prints the 12 planned arms, which checkpoints are present, the label-set diff, the numerics
contract and the provenance keys, and **creates nothing**.

---

## 8. Bounded GPU execution — for the coordinator

**GPU count: 1.** The pipeline is single-device and loads one SD pipeline at a time.

`FREEZE_ACTIVE` is in force, so the reservation needs both `ALLOW_FROZEN_RUN=1` and
`--allow-frozen`. Exact command:

```bash
cd /data/bijaypandey/t2i_unlearn && \
source env.sh && \
source scripts/gpu_reserve.sh && \
ALLOW_FROZEN_RUN=1 gpu_reserve_hold "rescore_final_v2" 120 --allow-frozen \
  || { echo "ABORT: no GPU reservation" >&2; exit 1; } && \
python scripts/rescore_final_v2.py --run-tag d013 2>&1 \
  | tee logs/rescore_final_v2_$(date -u +%Y%m%dT%H%M%SZ).log
```

`gpu_reserve_hold` exports `CUDA_VISIBLE_DEVICES` itself and releases on exit.

To re-score only the six arms whose checkpoints still exist (the other six are recorded as
`missing_checkpoint` either way; this just skips their pipeline loads):

```bash
python scripts/rescore_final_v2.py --run-tag d013 --only \
  utrap_church_l0.15k8,frp_church_r3_l05,utrap_sd15_church_l0.15k8,uce_sd15_church_lam1000,BASE_sd14_anchor,BASE_sd15_anchor
```

**Expected duration: 10–25 minutes.** 6 regenerable arms × 15 images × 25 steps at 512×512 fp16 on
one A100. The historical E6 records show ~0.26 min per arm including pipeline load; this run adds a
capped (64 MiB) checkpoint hash, PNG writes and a second label-set scoring pass, and loads CLIP once
(~15 s). The six `missing_checkpoint` arms resolve instantly. Budget 30 minutes.

**Exit code is non-zero**: the run will be `partial`, because six arms have no checkpoint. That is
the expected and correct outcome, not a failure — `run.json → summary` names them.

**Output paths** (all under a fresh `run_id`, nothing pre-existing is touched):

```
outputs/rescore/<run_id>/run.json                     manifest: status, summary, provenance,
                                                      label_mismatch, paired-scoring design
outputs/rescore/<run_id>/arms/<label>.json            per-arm: status, per-image all-12-label
                                                      scores under BOTH label sets,
                                                      paired_label_text_effect, headline_number,
                                                      determinism_probe
outputs/rescore/<run_id>/images/<label>/p<pi>_k<k>.png   all 15 images per completed arm (90 total)
outputs/rescore/<run_id>/embeddings/<label>.npz       normalised image embeddings +
                                                      sims_corrected + sims_legacy
logs/rescore_final_v2_<ts>.log
```

Disk: ~90 PNGs at ~400 KB plus small `.npz` ≈ 40 MB. `/data` had 330 GB free at 98 % use.

**What to read first when it finishes**, in order:

1. `run.json → run_status` and `summary` — confirm `partial` with exactly the six expected
   `missing_checkpoint` arms and zero `failed`.
2. Per arm, `determinism_probe.verdict` — `EXACT_REPRODUCTION` or `DISCREPANT`. This decides whether
   this run's numbers may be compared to the historical ones at all. Do not skip it, and do not
   assume the answer.
3. Per arm, `paired_label_text_effect` and `results_by_label_set` — the measured corrected-vs-legacy
   difference on identical images. This is the first evidence about the **direction and size** of the
   D-013 effect. Until it is read, the standing statement remains: *the evaluator used inconsistent
   label text; the effect on scores and conclusions is unresolved pending corrected evaluation.*
4. `headline_number` per arm — `target_acc` under the corrected label text, to be reported as a
   development measurement on an already-inspected split.


---

## 9. Reviewer defect register (2026-10-26)

Seven defects were raised by an independent reviewer. **All seven are fixed.** None could not be
fixed.

| # | defect | status | fix and evidence |
| --- | --- | --- | --- |
| **1** | BLOCKING — exposure figure wrong (26/180, per-arm 9/7/5/4, margins 0.0065–0.0131) and carried an unsupported "bounds how many could move" claim | **fixed** | Recomputed independently: **37/180**, per-arm 12/10/7/5/2/1, defence-arm median margins **0.0027–0.0063**, and exposure is **not** confined to the defence arms. Figures corrected in `gen_eval_final.py` and §1 of this document, with the old ones explicitly withdrawn. The bound claim is **removed** — it contradicted the closed-set coupling argued two paragraphs earlier, since an image predicted on an *unchanged* label can also flip. `EXPOSURE_FRAMING` now matches the register and FINDING65 verbatim: *"Small margins mean the outcomes are sensitive to label text; they do not indicate which way."* New test `T11` **recomputes** the figure from `outputs/final_*.json` and asserts `DOCUMENTED_EXPOSURE` equals it (`T11b`, `T11c`), that per-arm counts sum to the total (`T11d` — the withdrawn figure failed this on its face: 9+7+5+4 = 25, documented as 26), that the margin range matches (`T11e`), that non-defence arms carry exposure (`T11f`), and that the wrong figures appear nowhere except inside explicit withdrawals (`T11g`, `T11h`). The G14 lesson is taken: a grep for *absent* strings catches a removed claim, never a wrong one that is present. |
| **2** | BLOCKING — path traversal: `make_run_dir` never called `_guard_path`, so `--run-tag '../../../attacked/PWNED'` created `<P>/attacked/PWNED/` | **fixed** | Reproduced first (in a sandbox root) to confirm. Every directory creation now goes through `guarded_makedirs`; the whole tree is guard-checked *before* anything is created, so a refusal creates nothing at all. `--run-tag` is validated against `^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$` — no separators, no dots — at the top of `main`. Retried end-to-end via the CLI: the exploit is refused with a usage error and neither `attacked/PWNED` nor `adapters/PWNED` exists. Tests `T1f` (13 malicious tags rejected), `T1g` (traversal refused even with the tag check bypassed, nothing created). |
| **3** | `_guard_path` used a string-prefix test, so `outputs/rescore_evil/x.json` was allowed | **fixed** | Containment is now compared **component by component** (`_components` / `_is_within`), with `realpath` normalising `..` first. Verified: `outputs/rescore_evil/x.json` refused, `outputs/rescore/ok/x.json` still allowed. Test `T1h` covers four sibling-prefix variants plus the legitimate in-root path. |
| **4** | `gen_eval_final.py` still split the nine as 4/5 and misclassified `garbage_truck` as a wording change | **fixed** | Corrected to **6 underscore-only** (`garbage_truck`, `cassette_player`, `chain_saw`, `french_horn`, `gas_pump`, `golf_ball`) and **3 reworded** (`van gogh painting`, `english springer spaniel`, `monet painting`). The test no longer *documents* the wrong split in a comment: new `T8d2` asserts the stale 4/5 prose is absent and that the correct-split prose is present and lists exactly the computed underscore-only keys. |
| **5** | Determinism verdicts never aggregated — an all-`DISCREPANT` run still recorded `run_status: "complete"` | **fixed** | New `derive_history_comparability` aggregates per-arm verdicts into `all_exact` / `mixed` / `none_exact` / `unknown`, and `derive_run_status` makes `complete` require **both** every arm complete **and** `all_exact`. New statuses `complete_comparability_mixed` and `complete_not_comparable_to_history`. A missing probe aggregates as `unknown`, never as fine. The manifest and console print the verdict and its meaning, stating that the within-run corrected-vs-legacy comparison stays valid while any claim about the original records is lost. Tests `T4a` (11 cases), `T4d` (the all-discrepant case specifically), `T4e` (6 aggregation cases), `T4f` (exit code). |
| **6** | `T3a_run_ids_unique` flaky — 24-bit salt collides ~11 % over 2000 draws, while the write-up pasted 2000/2000 as deterministic | **fixed** | Salt widened to **16 hex = 64 bits**, and within-process uniqueness is now **guaranteed by construction** via an issued-id set with retry, so `T3a` is deterministic rather than probabilistic (now 5000/5000). New `T3a2` asserts the width and prints the birthday bound: P(collision in 2000 draws) < 1e-9, versus 1.1e-01 before. |
| **7** | "`gen_eval.py` unmodified" rested on mtime alone; the only hash baseline predates the repair | **fixed, with the scope stated** | Full sha256 recorded now in `UNMODIFIED_BASELINE_SHA256` — `gen_eval.py` `6f978e60870486b4e844580c3b480ac62ce4a59037eb2d5534756e26d131f51a`, `attack_relearn.py` `ec5fe2249b19df3b60e551f8800ed3fb7c7474a814e312578f1e88dd44fb40e2`, taken 2026-10-26 — and verified by `verify_unmodified_baseline()` on every run plus test `T12`. **The claim is deliberately not overstated:** byte-identity is proven only from the baseline date onward; for the period before it the evidence is mtime only, and `scope_note` says so in the manifest. |

Post-fix integrity re-verified: 12 `outputs/final_*.json` present with original mtimes
(2026-10-25T23:55 – 2026-10-26T01:07) and `final_utrap_church_l0.15k8.json` md5 unchanged at
`dbfec4e429d404eb653dba4af61cbe90`; `outputs/rescore/` still does not exist; no `.tmp_*` litter;
`gen_eval.py` and `attack_relearn.py` hash-identical to the recorded baseline.
