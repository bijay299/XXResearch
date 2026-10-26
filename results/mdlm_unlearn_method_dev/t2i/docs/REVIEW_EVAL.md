# REVIEW_EVAL — independent review of the EVALUATION re-scoring repairs (D-013)

Reviewer: independent; did not build this code. CPU only, no GPU, no git, nothing modified except
this file. Date 2026-10-26.

Reviewed: `scripts/rescore_final_v2.py`, `scripts/clip_scorer.py`, `scripts/test_rescore.py`,
`scripts/gen_eval_final.py`. Builder's write-up: `docs/REPAIR_EVAL_v2.md`.

## Observed test output

Run as instructed with the venv (`venv_t2i/bin/python scripts/test_rescore.py`), 7 times:

```
PASSED 46   FAILED 0      (5 of 7 runs)
PASSED 45   FAILED 1      (2 of 7 runs)
failures: T3a_run_ids_unique
  [FAIL] T3a_run_ids_unique: 1999/2000 distinct over 2000 draws
```

**T3a is flaky**, not passing. `new_run_id` (rescore_final_v2.py:171) truncates the salt to 6 hex
= 24 bits, so 2000 draws collide with ~11 % probability by the birthday bound. The write-up §7
(docs/REPAIR_EVAL_v2.md:230) pastes `T3a ... 2000/2000` as if deterministic; it is not reproducible.
Not a safety hole — `make_run_dir` refuses an existing directory — but the pasted evidence overstates.

---

## 1. HISTORICAL PRESERVATION — **DEFECT FOUND**

**D-R1 (reachable, blocking).** `make_run_dir` (rescore_final_v2.py:175-184) calls `os.makedirs`
**without ever calling `_guard_path`**. `--run-tag` is user-supplied and concatenated straight into
the run id (`new_run_id`, :172). Demonstrated in a sandbox mirroring the real tree layout:

```
--run-tag '../../../adapters/utrap_church_l0.15k8/PWNED'
  -> CREATED <P>/adapters/utrap_church_l0.15k8/PWNED/{images,embeddings,arms}
--run-tag '../../../attacked/PWNED'
  -> CREATED <P>/attacked/PWNED/{images,embeddings,arms}
```

The guard sits only on *file* writes, so the run then dies on its first `atomic_write_json` — but
directories have already been created inside `adapters/` and `attacked/`. This directly contradicts
"No write path can touch ... `adapters/`, `attacked/`" and docs:80-84. Low damage (empty dirs,
needs a deliberate flag), but it is precisely the class of hole the guard claims to close.

**D-R2 (latent).** `_guard_path` (:120) tests `rp.startswith(os.path.realpath(RESCORE_ROOT))` —
a bare string prefix with **no separator**. Observed ALLOWED:

```
/data/bijaypandey/t2i_unlearn/outputs/rescore_evil/x.json   ALLOWED
/data/bijaypandey/t2i_unlearn/outputs/rescoreX.json         ALLOWED
```

Not reachable through the current CLI (`os.path.join` always inserts `/`), so this is latent — but
it is the G14 point: **T1b passing proves six specific paths are refused, not that the guard is
sound.** The same loose prefix is used for `PROTECTED_PREFIXES` (:116), there in the safe direction.

**Correctly refused** (verified by direct probe): symlink targets and `..` (realpath is resolved
first, :114), absolute paths outside the tree, relative paths, the empty path, and
`outputs/rescore/<rid>/../../gen_x.json`. Empty `run_id` → `make_run_dir("")` returns the root
itself; refused if the root exists, otherwise it creates the root and writes `run.json` there.
Unreachable today (run ids are never empty) but unguarded.

**`gen_eval.py` / `attack_relearn.py` unmodified — VERIFIED by mtime, CLAIM UNSUPPORTED by hash.**
Both are `2026-10-25 17:38:24`, predating all repair work (`2026-10-26 14:11-14:23`). All 12
`outputs/final_*.json` are `2026-10-25 23:55` – `2026-10-26 01:07`, also predating it, and 12 are
present. However the only recorded baseline, `outputs/provenance_manifest.json` (generated
`2026-10-23T11:28`), lists `gen_eval.py = acedf6f6983b652c` and
`attack_relearn.py = 354e6964c0be77c9`, while the files now hash to `6f978e60...` and `ec5fe224...`.
That is explained by legitimate edits between 10-23 and 10-25, **but no post-10-25 hash baseline
exists**, so "genuinely unmodified" rests on mtime alone, which is not tamper-evidence. Code
reading confirms neither file is imported-and-mutated: `clip_scorer` only reads from `gen_eval`
(clip_scorer.py:70-75, 28-33) and `attack_relearn` is never touched.

## 2. ATOMICITY — **VERIFIED**, one gap

Every write goes through `atomic_write_json` (:126) or `atomic_write_bytes` (:149), both using a
same-directory `mkstemp`, `flush`, `fsync`, `os.replace`, with temp cleanup on any `BaseException`.
Call sites audited — **all** of them, not just the JSON: PNGs :503, embeddings `.npz` :517, arm
records :451/:562/:577, manifest :738/:757/:770. **A partially written PNG is not possible**: the
PIL `save` runs into the temp file and only a complete file is renamed into place.

Manifest/disk consistency: the manifest is written `incomplete` (:704) before the scorer is built,
rewritten after every arm (:757), and finalised in a `finally` (:758-770). A kill mid-arm leaves
PNGs on disk with no arm entry and `run_status: "incomplete"` — detectable, never readable as
finished. Residual: nothing verifies the PNG count on disk against `rec["n_total"]`.

**Gap:** neither writer fsyncs the containing **directory** after `os.replace`. The rename is
atomic but not durable across a power loss, so docs:86-89 ("`fsync`, then `os.replace`") slightly
overstates. Non-blocking.

## 3. NO DIRECTIONAL CLAIM — **DEFECT FOUND** (the most serious item)

The sanctioned sentence is present and carried in every emitted record via
`label_set_diff()["interpretation"]`: clip_scorer.py:20-21 and :121-122, rescore_final_v2.py:23-24,
gen_eval_final.py:272-274, docs:41-42.

No surviving assertion that degraded wording weakens competitors, that counts were inflated, that
unchanged-label arms are unaffected, that the finding survives, or that base accuracy 1.000 proves
scorer validity. A broad grep (`inflat|deflat|upward|weaken|degrad|survives|unaffected|sanity|
validat|proves|bias`) over all four scripts and the doc returns **only negations and disclaimers**.
Good.

**But a quantitative magnitude claim survives, and it is wrong.** gen_eval_final.py:176-179 and
docs/REPAIR_EVAL_v2.md:51-54 both assert:

> 26 of their 180 images were classified as one of the differing labels, concentrated in the four
> pre-attack defence arms (9/15, 7/15, 5/15, 4/15) at median winning margins of 0.0065-0.0131 —
> a count of affected decisions, which bounds how many could move.

Recomputed from all 12 `outputs/final_*.json` against `label_set_diff()["differing"]`:

| arm | images predicted as a differing label |
| --- | --- |
| utrap_sd15_church_l0.15k8 | 12/15 |
| utrap_church_l0.15k8 | 10/15 |
| frp_church_r3_l05 | 7/15 |
| uce_sd15_church_lam1000 | 5/15 |
| BASE_sd14_anchor | 2/15 |
| ATK_uce_sd15_church_lam1000 | 1/15 |
| all others (6 arms) | 0/15 |
| **total** | **37 / 180** |

(a) The count is **37, not 26**. (b) The per-arm figures `(9, 7, 5, 4)` match no arm set; the actual
four defence arms are `(10, 12, 7, 5)`, and 3 further images sit outside them, so "concentrated in
the four pre-attack defence arms" is also wrong. The quoted figures do not even self-sum
(9+7+5+4 = 25 ≠ 26). (c) Median winning margins for those four arms are **0.0027–0.0063**
(0.00439, 0.00629, 0.00513, 0.00269), not 0.0065–0.0131. (d) Worst, **"bounds how many could move"
is false, and contradicts the closed-set-coupling argument asserted two sentences earlier in the
same comment** (gen_eval_final.py:168-171, clip_scorer.py:24-26, docs:46-48): an image currently
predicted `church` — an *unchanged* label — can flip when a competitor's text changes, so the count
of images predicted as a *differing* label bounds nothing. This is a size-of-effect claim smuggled
in under a no-direction disclaimer, in the two most-read places.

**G14 on T8f** (test_rescore.py:428-442): `banned_claims` is four verbatim pre-repair strings.
Passing proves those four strings are absent — not that no directional or magnitude claim exists.
The 26/180 bound claim sails straight past it. `T8f2` (:445-454) likewise only greps three
disclaimer substrings, and `T8g` (:456-461) only checks the interpretation string is present. No
test asserts the *absence* of an unsupported magnitude claim, and none checks the 26/180 figure
against the records — which is why a wrong number survived in both the code and the write-up.

## 4. THE NINE LABELS — **VERIFIED**, with a code-comment defect

Confirmed element-wise from `gen_eval.py:48` (`LABELS`) and `:50` (`LAB_KEY`): exactly **9 of 12
differ**; identical are **church, parachute, tench**. The split is **6 underscore-only**
(garbage_truck, cassette_player, chain_saw, french_horn, gas_pump, golf_ball) + **3 reworded**
(van_gogh → "van gogh painting", english_springer → "english springer spaniel", monet → "monet
painting"). `T8d` (test_rescore.py:409-420) encodes this correctly and I reproduce its output.

No surviving "4 of 12" *count*. **But** gen_eval_final.py:148-152 still splits the nine as "the
four whose wording changes (garbage_truck, van_gogh, english_springer, monet)" plus "the five that
differ by an underscore alone" — a **4/5 split**, which misclassifies `garbage_truck`
(underscore-only). The correct split is 6/3; the nine total is unaffected.
test_rescore.py:406-408 *acknowledges this in a comment* yet the source comment was left wrong —
the test documents the defect instead of failing on it.

The internally inconsistent line ~176-179 flagged by the prior reviewer is **still present** and is
now shown to be numerically wrong (item 3).

## 5. PAIRED SCORING — **VERIFIED**

`ClipScorer.score_all_sets` (clip_scorer.py:210-218) computes `f = self.image_features(images)`
**once** and returns `{name: self.score_features(f, name) for name in self._sets}`. Text features
are cached per set (:183-193). `rescore_final_v2.py:512` is the only call site; the persisted
embeddings (:517-522) come from that same `feats` tensor, and both `sims_corrected` and
`sims_legacy` are saved alongside. The image side is bit-identical across the pair, so the measured
difference is attributable to label text alone *within the run*. `pair_records` (:595-620) refuses
a ragged or single-set pair rather than truncating.

## 6. DETERMINISM TESTED — **VERIFIED**, with a reporting gap

`compare_with_history` (:347-401) returns `EXACT_REPRODUCTION` only if `n` matches on both sides
**and** every per-image `pred_label` agrees **and** `max|Δtarget_sim| == 0.0` (:387-388) — strict;
it cannot silently assume equivalence. Absent history yields `NO_HISTORICAL_RECORD` (:356), not a
match. No byte-identical claim anywhere: the opposite is stated at :16-21, :334-336, :427-430 and
docs:66-70.

**Gap:** the verdict is recorded per arm but **never aggregated**. `derive_run_status` (:623-643)
ignores it, so a run in which every arm is `DISCREPANT` still records and prints
`run_status: "complete"`, and the end-of-run console summary (:771-773) never mentions determinism.
A reader of `run.json → summary` alone would not see it; only the per-arm `determinism_probe` has
it. Also, for a `NO_HISTORICAL_RECORD` arm the probe dict has **no `verdict` key**, so :566 prints
`None` and the write-up's instruction to read `determinism_probe.verdict` (docs:362) silently
yields nothing there.

## 7. MISSING ARMS — **VERIFIED**

Checkpoint presence is checked **before** any pipeline construction or GPU touch (:435-453). An
absent checkpoint yields `status: "missing_checkpoint"` with `status_detail`, an explicit
`status_consequence` ("Do not treat this arm as unchanged, and do not treat the run as covering
it"), and a written arm record — no scores fabricated. `derive_run_status` makes `complete`
unreachable while any requested arm is missing, failed or unattempted (`T4b` confirms); the
manifest starts `incomplete` before the scorer exists; exit code is non-zero unless complete
(:775). No arm can be silently skipped: the loop writes a record on every path and
`not_attempted` (:632-633) catches any requested label with no entry at all.

Observed inventory (`T5d`): 12 arms planned, **6 absent, all `ATK_*`** —
`ATK_frp_church_r3_l05`, `ATK_uce_sd15_church_lam1000`, `ATK_uce_sd15_church_lam1000_sd1`,
`ATK_uce_sd15_church_lam1000_sd6`, `ATK_utrap_church_l0.15k8`, `ATK_utrap_sd15_church_l0.15k8`
(the E6B/E6C/E6D checkpoints). Matches the claim exactly.

## 8. PROVENANCE — **VERIFIED**

`collect_provenance` (:235-343) records all twelve required items: per-file sha256 code identity
with the git-absence note (:257-266); environment — python, platform, 7 dependency versions, CUDA,
cuDNN, GPU name, TF32 state — plus an explicit caveat that the original run recorded none
(:268-292); `models.pin_file`, and per arm `base_model` + `revision` + **`variant`** +
`torch_dtype` (:299-305), with a `variant_note` that a repo lacking the variant FAILS loudly rather
than falling back; classifier id with its unpinned-revision gap restated; numerics recipe;
inference steps/guidance/`n_per`/resolution/safety-checker (:311-320); **scheduler class and full
config read off the instantiated pipeline per arm** (:478-481); FINAL prompts with `split_version`
and `FINAL_FROZEN_AT` (:321-326); **all 15 seeds enumerated** plus the formula (:327-337); the
label-set diff; `rescore_version` and `scoring_version`; per-arm checkpoint sha256 with explicit
scope (:454-460).

**The variant is recorded**, in provenance (:302) and per arm as `base_variant_used` (:477) — this
is the load-bearing item given that main vs fp16 SD-1.4 differs in 686/686 tensors at rel
2.074e-04. Minor note: checkpoint hashing is capped at 64 MiB by default (:657), so identity is a
prefix hash. It is honestly labelled `FIRST_<n>_BYTES` (:457) and the docstring (:94-96) says a
capped hash is weaker evidence — acceptable, but it will not distinguish two checkpoints differing
only past 64 MiB.

## 9. ROLE LABELLING — **VERIFIED**

`evaluation_role: "CORRECTIVE_RE_EVALUATION"` (:419, :708) with a note that the split was inspected
in FINDING65/66 and these are development measurements; `split_status: "INSPECTED_BEFORE_THIS_RUN"`
(:422-426); `method: "REGENERATION_PLUS_CORRECTED_SCORING"` (:714-720). Original precommitment is
kept strictly separate from this evaluation's status in `build_arm_plan` (:215-229):
`original_arm_precommitted` / `original_arm_status` describe the original E6 run and are
non-uniform (7 pre-committed / 5 post-hoc, observed), while `this_evaluation_status` is identical
for every arm and `this_evaluation_note` states that precommitment "confers nothing on this one".
`first_and_only_legitimate_use` is **gone** from every emitted record; it survives only as
historical commentary about its own removal (gen_eval_final.py:88, :239-241).

## 10. gen_eval_final.py — **VERIFIED**

The overwrite pre-flight (:124-137) runs before `from diffusers import StableDiffusionPipeline` and
`from transformers import ...` (:139-140) and before any GPU touch; the module-level `import torch`
(:45) is not a model load. It names every colliding path, points at `rescore_final_v2.py`, and
exits non-zero — observed `T1d: exit=1`. A second guard (:294-298) catches a record appearing
mid-run. The header no longer claims FINAL is untouched: :3-10 and :28-37 state the opposite
explicitly, and no "FIRST and ONLY" / "never used" language remains.

---

## Gate recommendation

**Block on:**

1. **The 26/180 claim** — gen_eval_final.py:176-179 and docs/REPAIR_EVAL_v2.md:51-54. The count is
   wrong (37/180), the per-arm figures are wrong (actual 10/12/7/5, plus 3 images outside the four
   defence arms), the margin range is wrong (actual 0.0027–0.0063), and "bounds how many could
   move" is logically false and self-contradictory with the closed-set-coupling argument in the
   same comment. This is an unsupported magnitude claim in the two most-read places, which is
   exactly what item 3 forbids. Delete the sentence or replace it with the sanctioned reading.
2. **D-R1 — `make_run_dir` is unguarded** (rescore_final_v2.py:175-184). `--run-tag` traversal
   creates directories inside `adapters/` and `attacked/`, demonstrated above. Route the run
   directory through `_guard_path`, and reject separators and `..` in `--run-tag`.

**Fix before the GPU run, not necessarily gate-blocking:**

3. `_guard_path` separator-less prefix test (:120) — compare with `os.path.commonpath` or append
   `os.sep`. Add the `outputs/rescore_evil` case to T1b; its current pass is not proof.
4. The 4/5 vs 6/3 underscore split in gen_eval_final.py:148-152 (`garbage_truck` is
   underscore-only). test_rescore.py:406-408 documents this rather than failing on it.
5. T3a is flaky (observed 2 failures in 7 runs). Widen the salt or drop the draw count; and
   re-paste a real run into docs §7 — the current paste is not reproducible.
6. Aggregate `determinism_probe.verdict` into `run.json → summary` and the console line, and give
   `NO_HISTORICAL_RECORD` an explicit `verdict` so docs:362 is actionable.
7. `fsync` the destination directory after `os.replace` in both atomic writers, or soften
   docs:86-89.

**Untouched and intact, confirmed:** `scripts/gen_eval.py` and `scripts/attack_relearn.py`
(mtime `2026-10-25 17:38:24`, by mtime only — no post-10-25 hash baseline exists); all 12
`outputs/final_*.json` present with pre-repair mtimes; `outputs/rescore/` does not exist; no
`.tmp_*` litter in `outputs/`. Nothing was modified by this review except this file.
