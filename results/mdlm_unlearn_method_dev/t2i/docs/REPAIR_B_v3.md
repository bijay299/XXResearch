# REPAIR B v3 — meta reload guard, split coverage enforcement, UCE comparison

AGENT B, bounded repair cycle, 2026-10-26. Project is under `FREEZE_ACTIVE`.

**No GPU work was launched.** Everything below is implemented and CPU-verified. The three items
that need a GPU (R2 smoke, R4c/R4d re-run) are *prepared, not executed*; exact commands in §4.

### Status vocabulary, used strictly throughout

| Term | Meaning |
|---|---|
| **implemented** | code written; not run |
| **CPU-verified** | run to completion on CPU, output quoted below |
| **independently verified** | a claim someone else made, re-derived here from scratch |
| **prepared, not executed** | command written and argument-checked; needs the lead's GPU launch |

**A passing unit-test suite is not an executed run.** Nothing in this document is a measurement of
unlearning, of method effectiveness, or of a reproduction of any paper.

---

## 1. TASK 1 — the meta reload guard replaced by a direct evidence chain

### 1.1 The four criteria, each verified unsound from source

All four lived in `run_r2_meta_official.sh` stage 9 (pre-repair lines 553–681; header comment
553–579, checks 648–668). Counterexamples computed on CPU
(`venv_t2i`, output verbatim):

```
== (1) CHECK B is FALSE: 'trained ckpt is closer to its own init than to the other variant'
   init A=1.0  other B=1.01  trained T=1.007 (started from A)
   rel_vs_same(A) = 6.999969e-03   rel_vs_other(B) = 2.970318e-03
   guard requires rel_same < rel_other  ->  False  == CHECK B FAILS A GENUINE RUN
   no-op scored vs wrong ref: rel_same=0.000e+00 rel_other=9.901e-03 -> B 'passes' = True

== (2) the Pythagorean step assumes ORTHOGONALITY of the update to (same-other)
   claim: ||T-B|| == sqrt(||T-A||^2 + ||A-B||^2) = 1.220655e-02
   actual ||T-B||                                = 3.000021e-03

== (3) AdamW does NOT bound the per-coordinate update by lr
   step 1: g=1e+00    |w|=9.999999e-06  cumulative/(lr*S)=1.000x
   step 5: g=1e-12    |w|=2.973097e-05  cumulative/(lr*S)=0.595x

== (4) there is no justified positive MINIMUM change inferable from lr
   one EXECUTED AdamW step with a zero gradient: |dw| = 0.000000e+00
```

* **(1)** `rel_vs_same < rel_vs_other` — pre-repair line 657. Defect: the inequality's sign is a
  fact about the *direction* of the update, which nothing constrains. A genuine update toward the
  other variant fails it.
* **(2)** The Pythagorean derivation at pre-repair line 570 (`rel_vs_B = sqrt(d^2 + floor^2)`)
  requires `(T−A) ⟂ (A−B)`. Nothing makes a gradient update orthogonal to an fp16 quantisation
  offset; measured gap above is 4×.
* **(3)** `rel <= lr*S*sqrt(N)/||w||` — pre-repair lines 573–577 and 662. `m̂/(√v̂+ε)` is bounded
  by 1 only in the `β₁² ≤ β₂` steady state; decoupled weight decay adds `−lr·wd·w` on top. The
  measured ratio reaches 1.000× at S=1 — i.e. the bound is *attained*, so it is a boundary, not a
  margin; any `wd>0` term pushes past it. Note the project's own config runs `weight_decay=1e-4`
  (`esd_meta.py:132` default), which the bound ignores entirely.
* **(4)** `rel >= bound(S=1)/100` — pre-repair line 667. An executed step with zero gradient moves
  nothing; so does an executed step at `lr=0` during warmup.

All four are **removed as pass/fail criteria. No substitute norm threshold was added.** Norms are
still reported and are labelled descriptive. Asserted structurally by
`selftest::four_criteria_removed_no_substitute`.

### 1.2 A fifth, independent BLOCKING defect found in stage 7

`esd_meta.py:1070` nests the periodic save **inside the validation gate** (verified from source):

```
1070        if (step % args.eval_every == 0) and (step >= args.eval_after) and (len(args.validation_prompts) > 0):
1084            if step % args.save_every == 0:
1086                save_checkpoint(..., step=step)
```

Pre-repair stage 7 set `--eval_every $((SM_STEPS + 1))` "to skip validation sampling". With
`eval_every = SM_STEPS+1`, `step % eval_every == 0` is false for every step in `1..SM_STEPS`, so
**no `step=*/` directory is ever written** and stage 8's
`die "no step=*/ checkpoint was written"` fires unconditionally. `R2_MODE=smoke` was structurally
incapable of reaching stage 9 at all — independently of the threshold problem.

**Fix:** `--eval_every` is set to `SM_STEPS`, so the gate fires exactly once, on the final step.
Guarded at run time by `selftest::shim_binds_to_upstream_symbols`, which fails if the nesting,
the two skip paths, or the absence of a save-time dtype cast ever change upstream.

### 1.3 What replaces it: six links

Implemented in a **generated** module `upstream_runs/meta/out/r2_evidence.py`, emitted by
`run_r2_meta_official.sh` stage 4c and bound into the run record by sha256.

| Link | Records | How |
|---|---|---|
| 1 | state loaded **before** optimization: repo, revision, variant, dtype, subfolder; trainable parameter **names**; per-tensor sha256 fingerprints | wraps `diffusers.*.from_pretrained` and upstream's own `gather_parameters` (`esd_meta.py:269-306`) |
| 2 | **EXECUTED** updates, lrs actually in effect, grad diagnostics (None count, global L2, non-finite), observed per-tensor change | global torch optimizer step pre/post hooks |
| 3 | **in-memory** UNet state at each save point + executed-update count there | wraps `esd_meta.save_checkpoint` |
| 4 | reload: keys, shapes, finiteness, **bit equality** at the intended precision | `torch.equal`-grade sha256 comparison vs Link 3 |
| 5 | fp16 export vs `src.half()`, element for element | stage 10 |
| 6 | binding: run config, upstream commit, source sha256, module sha256 | — |

**Executed / configured / skipped are three separate fields.** The two real upstream skip paths are
named in the record: `esd_meta.py:928-929` (`if args.use_fp16: pass` — skips loss, backward *and*
the step for every step) and `esd_meta.py:1053` (`if step % gradient_accumulation_steps == 0:`).
A link that cannot be evaluated is `INDETERMINATE`, which **fails**.

**Upstream source identity preserved.** No byte of `upstream/Meta-Unlearning` is edited.
Instrumentation is external (global hooks + attribute wrapping). Stage 4c writes
`upstream_runs/meta/out/instrumentation.patch`, whose recorded diff of upstream against itself is
empty, plus the shim as a new-file hunk:

```
    upstream sha256      6c006964da950aef5d4ef357bc97621079eb29272f0718bd5d0c44c59c3f5992  (UNMODIFIED: self-diff is empty)
    instrumentation sha  b93d2872d11f8b2c68768357ab03a1d8ea5fac609068e5d4d6eddf51c54037bb
```

### 1.4 A real bug the unit tests caught

The first implementation wrapped `torch.optim.Optimizer.step`. **`AdamW` defines its own `step`**,
so the wrapper never fired and the evidence record would have been *silently empty* —
`executed_updates == 0` on a genuinely training run. `selftest::recorder_counts_executed_not_configured`
caught it; Link 2 now uses `register_optimizer_step_pre_hook` / `..._post_hook` (present in
torch 2.4.1 in both venvs), which fire for every optimizer class and modify none.

### 1.5 CPU test output (unit tests — **not** an executed run)

Required cases plus six more. Run under `venv_imma`:

```
  PASS shim_binds_to_upstream_symbols
  PASS recorder_counts_executed_not_configured
  PASS no_op_training                           <- required
  PASS mismatched_initialization                <- required
  PASS stale_checkpoint                         <- required
  PASS unrecorded_save_point
  PASS genuine_update                           <- required
  PASS corrupt_fp16_export
  PASS four_criteria_removed_no_substitute
  PASS zero_grad_executed_update
10 passed, 0 failed
```

Representative failure messages (verbatim), showing each verdict rests on a recorded fact:

```
no_op_training:      FAIL LINK2: ZERO optimizer updates were EXECUTED. The trainer ran but did not
                     train. configured_steps=2; recorded skip paths=[]
                     FAIL LINK4: every one of the 3 trainable tensors is bit-identical to its
                     pre-optimization state: the saved checkpoint contains no parameter change
mismatched_init:     FAIL LINK1: a UNet was loaded at revision 'SOME-OTHER-REVISION', expected 'REV'
                     (and LINK4 still PASSES — it fails on IDENTITY, not on a magnitude)
stale_checkpoint:    FAIL LINK4: 206 reloaded tensor(s) are NOT BIT-EQUAL to the in-memory state at
                     the save point, at the intended serialization precision
zero_grad:           FAIL LINK2: no executed update changed any tracked tensor; the recorded
                     gradient L2 was [0.0]           (cites the recorded gradient, not a bound)
```

`zero_grad_executed_update` additionally asserts no failure string contains the word "bound".

---

## 2. TASK 2 — split coverage and provenance enforcement

### 2.1 Defects verified from source, with the pre-repair output

`scripts/split_registry.py` v2.

| # | Defect | Source | Pre-repair evidence |
|---|---|---|---|
| S-1 | newer builders unregistered | v2 line 43-44 `DEFENCE_SCRIPTS` (10 entries) | v2 output: `defence_scripts_on_disk_not_registered: ["erase_utrap_v2","erase_wdr_v2"]` |
| S-1b | `meta_unlearn_v2` not even **detected** | v2 lines 131-133: sweep matched only `erase_*` plus hardcoded `("meta_unlearn.py","undo_t2i.py")` | absent from both lists |
| S-2 | advertised failure never enforced | v2 docstring line 11-12 "fails loudly"; `__main__` 249-255 has no exit code | **v2 exited 0 while reporting two unregistered builders** |
| S-3 | coverage claimed from regex over source alone | v2 lines 98-99, four regexes | cannot see `erase_utrap_v2.py:127` `DEF_TRAIN_TARGET`, which is **function-local** — the 16 prompts that built all four FINDING76 arms |
| S-4 | one verdict for two questions | v2 line 227 `FINAL_usable` from string disjointness only | v2 output: `FINAL_usable: true` while `splits.FINAL.status` in the **same record** was `INSPECTED_DEVELOPMENT_DATA` |

### 2.2 Fixes

* **Registration.** `DEFENCE_SCRIPTS` += `erase_utrap_v2`, `erase_wdr_v2`, `meta_unlearn_v2`.
  New `UPSTREAM_WRAPPERS` (`r2_build_meta_pools`, `r4b_uce_upstream_vs_port`,
  `r4c_uce_checkpoints_and_images`, `r4_uce_official_vs_port`, `r1_eval_imma`); `EVAL_SCRIPTS` +=
  `gen_eval_final`, `rescore_final_v2`. Discovery is by `BUILDER_PATTERNS` (incl. `_v\d$`) plus an
  explicit `NOT_PROMPT_AUTHORITIES` roster where every exemption carries a reason; an exemption
  naming an absent file is itself a failure.
* **Static extraction by AST** (`_ast_prompt_literals`), reaching function-local pools and
  f-string templates — and **explicitly no longer treated as coverage**.
* **Resolved runtime records** (`RUNTIME_POOL_RECORDS`): meta `pools.json`, UCE port
  `fingerprint.json`, defence build records. Missing kinds ⇒ `coverage_complete: False`.
* **Enforcement**: `enforce()` returns failures; `require_clean()` raises; `__main__` exits
  non-zero. Structural failures are always fatal; eligibility is fatal when a new claim is declared.
* **Two verdicts separated**: `string_disjoint` vs `eligible_for_confirmatory_evaluation`
  (+ `eligibility_blockers`). `FINAL_usable` retained as a **deprecated, labelled alias** of
  `string_disjoint`.
* **FINAL preserved** as `INSPECTED_DEVELOPMENT_DATA` exactly as the lead set it. The regression
  guard `test_audit_repairs.py::t_dev_is_labelled_as_used` is **unchanged and passing**.

### 2.3 A genuine NEW LEAK found by the repaired registry

Registering `r2_build_meta_pools` and inspecting its pools immediately surfaced this. The
`coherent` scheme — the one the launcher recommends for "any run intended to support a
concept-erasure claim" — had:

```python
"hrm": ["a photo of a church", "a photo of a church building",
        "a photograph of a church exterior"],
```

and `gen_eval.CONCEPTS["church"] == attack_relearn.SPLIT["church"]["H"] ==`
`["a photo of a church", "a church building", "a photograph of a church exterior"]`.

**Two of the three FORGET prompts were exactly evaluation target prompts.** Unseen because
`r2_build_meta_pools` was outside v2's roster. Fixed (`scripts/r2_build_meta_pools.py` SCHEMES):
replaced with three verified-disjoint church paraphrases. `legacy` is left **untouched** — its
`hrm` is an eval prompt by construction; that is the defect it reproduces, and it stays recorded.

New `check_split_disjointness()` runs on every build *and* verify, checks the **resolved** pools
(including `irt`, sampled from `coco_object_retain.csv` at run time and present in no source
listing), and hard-fails a non-historical scheme. CPU-verified both ways:

```
--- non-historical scheme with the OLD hrm pool:
[r2pools] SPLIT OVERLAP vs_EVAL_TARGET: 2 prompt(s)
    hrm: 'a photo of a church' <- attack_relearn.SPLIT[church]['H']
    hrm: 'a photograph of a church exterior' <- attack_relearn.SPLIT[church]['H']
refused as expected
--- the FIXED coherent hrm pool, non-historical:
accepted; overlaps = {'vs_EVAL_TARGET': 0, 'vs_ATK_TRAIN': 0, 'vs_FINAL': 0} n_checked = 12
```

### 2.4 A second defect fixed in passing

`scripts/r1_eval_imma.py` parsed argv **at module level**, so the moment the registry added it to
its roster the whole audit died with `error: the following arguments are required: --esd, --out`
and exit 2 — misattributed to `split_registry`. Two fixes: an explicit `ImportError` guard in
`r1_eval_imma.py` naming the cause, and `_module_pools` now catches `BaseException` (v2 caught
`Exception`, and `SystemExit` is not one), so a non-import-safe script is a recorded note rather
than a crash of the registry.

### 2.5 Tested through a REAL caller, and the runner ENFORCES

New **stage 4b** in `run_r2_meta_official.sh` consumes and enforces the verdict. Full
`R2_MODE=cpu` run, CPU-verified, exit 0:

```
--- stage 4b: split registry enforcement
    split_version=v3 new_claim=False
    registered: 13 defence, 5 upstream wrappers, 4 eval
    unregistered builders on disk: []
    runtime pool record kinds present: ['meta_image_pools', 'uce_port_fingerprints']
    VERDICT string_disjoint=True  eligible_for_confirmatory_evaluation=False  coverage_complete=False
    blocker: FINAL has status 'INSPECTED_DEVELOPMENT_DATA': it was SCORED AND INTERPRETED by
             FINDING65, FINDING66 ... needs a freshly frozen set
    blocker: coverage is incomplete: no RESOLVED RUNTIME pool records of kind(s) ['defence_build_records']
    structural checks PASS. META_NEW_CLAIM is not set, so this run may NOT be reported
    as a clean/confirmatory evaluation.
...
R2 CPU VERIFICATION COMPLETE -- stages 1,2,3,4,4b,4c ALL RAN and passed.
```

`scripts/split_registry.py` standalone now **exits 1** (was 0). `test_audit_repairs.py`:
**24 passed, 0 failed**. No new confirmatory results were generated.

---

## 3. TASK 3 — the UCE comparison

### 3.1 Review of the lead's two changes

**`--no-preserve` in `erase_uce.py` (lines 111-114, 148-151): CORRECT, and the claim is
independently verified — and stronger than stated.** Re-derived from scratch on CPU:

```
   torch.float64: numerator torch.equal=True  denominator torch.equal=True  W_new torch.equal=True max|A-B|=0.0e+00
   torch.float32: numerator torch.equal=True  denominator torch.equal=True  W_new torch.equal=True max|A-B|=0.0e+00
   the zero blocks contain no -0.0/NaN: True, signbit set on 0 of 245760 elements
```

With `c_p` of shape `[0,768]`, `(W @ c_p.T) @ c_p` is a `[320,768]` exact zero and `c_p.T @ c_p` a
`[768,768]` exact zero, so both preserve terms drop out. The lead proved float64; it also holds
bit-exactly in **float32** (the dtype the arms actually run in), and the numerator and denominator
are *separately* bit-equal, not only the final `W_new`. Default-off, so historical arms reproduce.

**The `n_preserve_port` / `preserve_asymmetric` fix (lines ~450-452): CORRECT but INCOMPLETE —
two problems.**

1. **A stale caveat the fix missed (now fixed by me).** The same artifact still appended, whenever
   `--no-preserve` was passed:
   `"ASYMMETRIC: --no-preserve empties UPSTREAM's preserve pool only; erase_uce.py always uses its
   hard-coded 10-concept PRESERVE"`. That is now **false**, and it is the very claim the config-field
   fix corrected. A repaired, symmetric run would have shipped an artifact declaring itself
   asymmetric. The caveat now derives from the **measured** `len(preserve) != len(port_preserve)`.
2. **The artifacts on disk are still wrong and were not regenerated.** mtimes:
   `scripts/erase_uce.py` 16:09, artifacts `r4d_uce_church_lam0.5_...json` **16:13** and
   `lam0.1` **16:14**, `scripts/r4c_...py` **16:18**. So both artifacts **predate** the field fix and
   still read `n_preserve_port: 10, preserve_asymmetric: true` — while the fingerprints beside the
   checkpoints read `port_n_preserve: 0, port_no_preserve_flag_passed: true`. The code is fixed; the
   **artifacts are not**. I did not alter or relabel them (freeze rule); they must be re-run (§4.2).

### 3.2 Why the last run reported `complete: False`

`complete` was `len(score_keys) > 1`, and `--retain-concepts` defaulted to `""`, so
`score_keys == [concept]` and `complete` was **always** False. `run_r4c_queue.sh` passes no
override, so no run could ever report complete. Confirmed in all four 16:13–16:14 artifacts:
`complete: False, scored_concepts: ['church']`.

**Fixed:** `--retain-concepts` now defaults to `DEFAULT_RETAIN = (cassette_player, chain_saw,
french_horn, gas_pump)` — `gen_eval.LAB_KEY` entries that are *not* `attack_relearn.SPLIT` targets
and *are* in `erase_uce.PRESERVE`, i.e. the neighbours the preserve term is meant to protect. So
BOTH target and retain are measured by default, which is what PREREG_H38 §6 requires and what
AGENT1 D-105 needs to separate erasure from collateral damage.

`complete` is no longer one opaque boolean: a `completeness` block records `target_measured`,
`retain_measured`, `images_saved`, `per_image_scores_recorded`, `upstream_executed`,
`port_executed_end_to_end`, `preserve_pools_matched`, `accumulation_order_measured`,
`upstream_cli_path_exercised`, and `complete_criteria` states which of these `complete` requires.

### 3.3 Accumulation order — kept, because the experiment does vary it correctly

The `--order-probe` block (`r4c:~302-346`) solves the same matrix twice from **identical** `W`,
`c_e`, `c_t`, `c_p`, `lam`, `preserve_scale`, dtype and device, varying **only** the accumulation
order (`lam*W` last + fused GEMMs vs `lam*W` first + per-concept rank-1). That is the single-factor
condition required, so the claim is admissible and is retained. Each column now records `varies`
and `held_fixed` explicitly; the fp64 column is relabelled as a **precision** comparison (varies
dtype at fixed order), not an order one.

Added `NOT_A_CAUSAL_DECOMPOSITION`: the two scalar error norms come from two different
single-factor comparisons, both amplified by the *same* ill-conditioned inverse, so their
difference/ratio/sum is **not** an attribution to separable causes and must not be presented as one.
Each column licenses only "changing this one factor alone moves `W_new` by this relative amount."

Because `--order-probe` is default-off and no run has passed it, a caveat is now emitted when
`order is None` stating that accumulation order was **not measured** and that no port-vs-upstream
difference may be attributed to it from that artifact.

### 3.4 Execution paths — one honest gap, now recorded in the artifact

Upstream's edit code **is** executed (`r4b.load_upstream_uce` + `up.UCE(...)`, pinned
`7c724d9`), and the port **is** executed end-to-end as a subprocess with a checked return code and
a full-config fingerprint. The gap: `uce_sd_erase.py` applies `expand_prompts` in `__main__`
*before* calling `UCE()`, so a harness importing `UCE()` can never exercise it — and
`expand_prompts` is the only upstream lever measured to change the edit size. This is now a
first-class field, `completeness.upstream_cli_path_exercised: false`, with the note that the
upstream arm must be described as *"UCE() at erase_scale=1 with unexpanded prompts"*, not as
*"upstream UCE as its authors run it"*.

Already present and verified by reading: config-specific checkpoint paths and fingerprints
(`cfg_tag` covers concept/lamb/erase_scale/preserve_scale/guide/n_preserve; fingerprint covers
erase+preserve prompts, lambda, base model/revision/variant, embedding selection); BASE vs
UPSTREAM vs OUR_PORT at matched prompts, seeds (`9000 + pi*100 + k`), steps and guidance; saved
images under `outputs/img_r4{c,d}_<tag>/`; per-image `pred`/`is_target` records.

Also fixed: `--save-images` was `action="store_true", default=True`, i.e. unsettable, making the
`images_saved_to ... else None` branch dead. Now `--no-save-images`.

---

## 4. Commands for the lead

All require the lead's own GPU launch. **Agent B launched none of these.**

### 4.1 R2 — SHORT REAL training smoke test (prepared, not executed)

CPU half first (no GPU, ~45 s; this is what was run above):

```bash
cd /data/bijaypandey/t2i_unlearn
META_POOL_SCHEME=legacy META_HISTORICAL_RECONSTRUCTION=1 META_SIZING=total_per_pool \
  R2_MODE=cpu bash run_r2_meta_official.sh
```

Then the smoke test, through the real launcher and the real evaluator:

```bash
cd /data/bijaypandey/t2i_unlearn
ALLOW_FROZEN_RUN=1 \
META_POOL_SCHEME=legacy META_HISTORICAL_RECONSTRUCTION=1 META_SIZING=total_per_pool \
R2_MODE=smoke META_SMOKE_STEPS=2 META_NIPP=1 META_BATCH=2 \
  bash run_r2_meta_official.sh
```

`META_SIZING=total_per_pool` is required: the pools on disk were built that way and stage 4 refuses
a definition mismatch. `META_NIPP=1 META_BATCH=2` keeps the one validation pass (now unavoidable —
see §1.2) cheap. `ALLOW_FROZEN_RUN=1` is the lead's freeze override; the launcher also needs
`gpu_reserve_hold`'s `--allow-frozen`, which it passes automatically from that variable.

Expect on success: `upstream_runs/meta/out/train_evidence.json`,
`.../step=000002/reload_evidence.json`, `adapters/R2_*/evidence_chain.json` with
`verdict: PASS` and all six links PASS, and a final `R2 DONE`.

**Read `train_evidence.json` first on any failure** — `link2_executed_updates.executed_updates` vs
`configured_steps` distinguishes "did not run" from "ran but did not train".

### 4.2 R4c / R4d — re-run so the artifacts match the repaired code

Needed because the 16:13–16:14 artifacts predate the field fix (§3.1.2) and all report
`complete: False` (§3.2). Preserves the old artifacts; the config-specific paths mean no run can
shadow another.

```bash
cd /data/bijaypandey/t2i_unlearn
# R4c (preserve pools ON both sides), with retain axis + order probe:
for lam in 0.5 0.1; do
  venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py \
    --concept church --lamb "$lam" --erase-scale 1.0 --preserve-scale 1.0 \
    --order-probe --refresh-port
done
# R4d (matched EMPTY preserve pools, both sides):
for lam in 0.5 0.1; do
  venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py \
    --concept church --lamb "$lam" --erase-scale 1.0 --preserve-scale 1.0 \
    --no-preserve --order-probe --refresh-port
done
```

`--retain-concepts` now defaults to the four-concept retain panel, so it need not be passed.
`--refresh-port` forces a port rebuild rather than reusing a fingerprint-matched checkpoint.
Note `run_r4c_queue.sh` (**Agent A's / the lead's file, not edited by me**) still passes neither
`--order-probe` nor `--refresh-port`; it needs those two flags added, or these commands used directly.

### 4.3 Registry, standalone

```bash
cd /data/bijaypandey/t2i_unlearn
venv_t2i/bin/python scripts/split_registry.py --debugging   # structural only; exits 1 today
venv_t2i/bin/python scripts/split_registry.py               # + eligibility; exits 1 today
venv_t2i/bin/python scripts/test_audit_repairs.py           # 24 passed, 0 failed
```

Both non-zero exits are **correct**: `coverage_complete` is False (no `defence_build_records`
carry resolved prompt pools) and FINAL is inspected development data. To make
`eligible_for_confirmatory_evaluation` true, a fresh FINAL set must be frozen under a new version
and defence build records must record their resolved pools.

---

## 5. Files changed

| File | Change |
|---|---|
| `run_r2_meta_official.sh` | stage 7 smoke gating fix + instrumented launch; **stage 9 replaced** by the evidence chain; stage 10 verifies the fp16 export; **new stages 4b** (registry enforcement) **and 4c** (evidence module, patch record, unit tests) |
| `scripts/split_registry.py` | v2 → v3: registration, AST extraction, runtime records, enforcement, two separated verdicts, `BaseException` on import |
| `scripts/r2_build_meta_pools.py` | `coherent` hrm leak fixed; `check_split_disjointness()` added and wired into build + verify; recorded in `pools.json` |
| `scripts/r1_eval_imma.py` | import-safety guard (module-level `parse_args` broke every importer) |
| `scripts/r4c_uce_checkpoints_and_images.py` | stale ASYMMETRIC caveat fixed; retain panel default; structured `completeness`; order-probe decomposition refusal; `--no-save-images` |
| `docs/REPAIR_B_v3.md` | this file |

Not touched: `scripts/gpu_reserve.sh`, `run_r1*.sh`, `run_v1/v2*`, `scripts/gen_eval.py`,
`scripts/attack_relearn.py`, `run_r4c_queue.sh`, GPUs 2–3,
`/data/bijaypandey/mdlm_unlearn`. No historical artifact was deleted or relabelled.
