# INDEPENDENT REVIEW — IMMA launcher, five preregistered requirements

Reviewer: independent (no authorship of the reviewed changes). CPU only, no GPU work, no commits.
Date 2026-10-26.

**Versions reviewed** (the files were edited three times during this review; these are the sha256s
the verdicts below apply to):

| file | sha256 | mtime |
|---|---|---|
| `run_r1b_imma_train.sh` | `b0038d1e8045b6f3dbe31513ee7cdafd3cdff3501823e972a39e246048540dc9` | 20:38:04 |
| `instrumentation/imma_evidence/sitecustomize.py` | `667d1a076866be3461ba1ea64a83b4af39066ad4b53cc889bace6367e8f098e1` | 20:38:04 |
| `docs/PREREG_R1_IMMA_AUTHOR_BUDGET.md` | `ebf644b729714bf01714bccb2d62458dae6a70e3b24eb16d3f4ebfdf6466ddba` | 20:28:20 |

Two defects I found in the 20:33 revision — the stage-5 verifier importing `sitecustomize` and
overwriting the training evidence, and `bud.get("mine", bud)` reading a key the budget JSON does not
have (it is `ours`), which made "EXPECTED ascent" `None` and printed the false line
`executed == expected` — are **fixed in the reviewed revision** (`run_r1b_imma_train.sh:551-553`,
`:622`). They are recorded here only because they bound what the earlier evidence can support.

---

## Requirement 1 — EXECUTED update counts: **VERIFIED** (final count NEEDS-REAL-GPU)

* Upstream is genuinely byte-unmodified: `git -C upstream/IMMA status --porcelain` is **empty** (so
  no modified *and* no untracked files, i.e. no injected `sitecustomize.py` inside the tree),
  `HEAD = 2f55041141705031d503134dec22f674f09e21bf`, `git diff --stat HEAD` empty. The only
  mechanism is `PYTHONPATH` + `IMMA_EVIDENCE_OUT`, both scoped to the training subshell
  (`run_r1b_imma_train.sh:551-553`).
* **PYTHONPATH does survive `accelerate launch` — verified on CPU, no GPU needed.** I ran
  `accelerate.commands.launch --num_processes=1 --mixed_precision=no --cpu` on a no-op script with
  the launcher's exact env and got two per-pid evidence files, both with the hook installed:

  ```
  ev2.2380104.json pid 2380104 steps 0 hook True argv ['-m', '--num_processes=1']   <- accelerate launcher
  ev2.2380208.json pid 2380208 steps 0 hook True argv ['noop.py']                   <- trainer
  ```

  This also confirms the two-process design note at `sitecustomize.py:45-53` is real, and that the
  per-pid file naming (`:54-56`) plus the clobber guard (`:210-230`) are the right fix rather than a
  story. No `sitecustomize.py` exists in either venv, so nothing is shadowed.
* **No double counting through accelerate.** `defend_text_to_image_lora.py:740-741` passes both
  optimizers through `accelerator.prepare`, and `AcceleratedOptimizer` *is* a `torch.optim.Optimizer`
  subclass (verified `isinstance == True`), which could have hooked the wrapper and the inner
  optimizer. Measured with accelerate 1.1.1 / torch 2.4.1: the global post-step hook fires **once**
  per logical step, on the inner `AdamW` — `[('AdamW', 1e-05), ('AdamW', 0.0001)]`. Counts are not
  inflated 2x.
* lr-role classification (`sitecustomize.py:73-85`) reads the live `param_groups` lr. `--scale_lr` is
  commented out upstream (`:575-578`) and `lr_scheduler=constant`, so no drift. Every mislabel route
  I could construct is **fail-closed**, not silent: colliding lrs or a drifted lr yield
  `role=unclassified`, and stage 5a exits (`:628-629`); an lr equal to the LoRA lr yields
  `asc == 0` and exits (`:630-631`).
* Skipped steps: measurable only as executed-vs-expected. Confirmed loud and non-fatal with a
  synthetic 413/500 record — `*** EXECUTED != EXPECTED: executed=413 expected=500 delta=-87 ***`.
* The hook does not alter training: post-step, returns `None`, catches its own exceptions; the
  `torch.save` wrapper passes `obj`/`f` through and returns the real return value.

Residual weaknesses (none blocking, all worth a line of code):

1. The role thresholds `1e-5`/`1e-4` are hardcoded in `sitecustomize.py:79-82` while the launcher
   exposes `IMMA_LR` / `IMMA_LR_LORA` overrides (`:104-105`). A non-default lr fails closed with the
   misleading message "could not be attributed".
2. `n_param_tensors` is recorded (`sitecustomize.py:100-101`) but never checked. Asserting the
   ascent optimizer holds **160** tensors would corroborate the role with a signal independent of
   lr; today lr is the only evidence used.
3. `ev = max(..., total_steps_observed)` (`:607`) picks one record instead of summing. Correct for
   `--num_processes=1` (hardcoded at `:553`), but with >1 process it would silently report **one
   worker's** count as the run's, and `unclassified` steps in the other records are ignored.
4. The glob `"$EVID_BASE".*.json` (`:582`, `:591`) is never cleared before training, so evidence
   files from an aborted earlier attempt at the same artifact name can be picked up. Fail-closed in
   practice (a stale record has no matching save), but add `rm -f` plus a start-time check.

The real count (is it 500?) cannot be established without the GPU run: **NEEDS-REAL-GPU** for that
one number only; the mechanism is verified.

## Requirement 2 — enforced split/provenance gate: **DEFECT FOUND** (2, both small)

Enforcement itself is real: `run_r1b_imma_train.sh:389` calls `sr.require_clean(for_new_claim=False,
rep=rep)`, which **raises SystemExit** (`scripts/split_registry.py:596-610`), un-wrapped, so the
failure propagates and `die` fires. Absent verdict keys now fail instead of printing `None`
(`:398-400`), and I confirmed the registry's block is `verdict` (singular) and that `verdicts` does
not exist. My own re-run reproduces the quoted output exactly: `string_disjoint True`,
`eligible_for_confirmatory_evaluation False`, 22 scripts inspected, this run's resolved dataset at
**20 files / 1 distinct prompt / sha256 `c4061ca4634a2ac6`**, the 33-record named exclusion, and
`SCRIPT_EXIT=0`.

Requiring `eligible_for_confirmatory_evaluation is False` (`:406-408`) rather than tolerating it is
**right**: for a developmental run a `True` here would mean the registry had forgotten that FINAL was
inspected, so the assertion is a regression sentinel, not a formality (prereg §5 "no confirmatory
protocol has been frozen").

**DEFECT 2a — the archived gate record carries no verdict.** `:484` still writes
`"verdicts": rep.get("verdicts")`, and the registry key is `verdict`. The file written by my re-run
proves it:

```
upstream_runs/imma/split_gate_r1b.json -> {... 'enforced': True, 'problems': [], 'verdicts': None ...}
```

This is the same bug the comment at `:394-396` says was fixed — it was fixed for *printing* only.
The record a later reader would cite says "enforced, no problems" and contains neither
`string_disjoint`, nor the eligibility verdict, nor `coverage_complete`.

**DEFECT 2b — the header overstates what is enforced.** `:367` says "incomplete coverage FAILS. It
does not pass with a note." The audit returns `coverage_complete: False` with blocker
`"coverage is incomplete: no RESOLVED RUNTIME pool records of kind(s) ['defence_build_records']"`,
and `enforce(for_new_claim=False)` returns `[]` (I measured both; `for_new_claim=True` returns 2
failures). So the gate **does** pass with a note — the note at `:466-471`.

On the substance: the NAMED EXCLUSION is **defensible**, not silent passing. The 33 records are
historical arms this run does not consume; the run writes and validates its own resolved pool
(`:421-459`, fails on a missing caption row or an empty record), and the exclusion is printed with
"Not treated as clean". But its defensibility rests on a record that, because of 2a, does not
contain `coverage_complete: False`. Fix 2a and 2b together and this requirement is clean.

Gap worth noting (not a false claim — the gate never claims it): nothing checks **this run's**
resolved prompt against the evaluation panel. I checked by hand: training caption
`"An artwork by Van Gogh"` vs `gen_eval.py:33` `"a painting in the style of Van Gogh"`, … — disjoint,
so there is no leak here, by luck rather than by check.

## Requirement 3 — checkpoint verification: **VERIFIED for correspondence; fp16 clause CLAIM UNSUPPORTED**

The save-point fingerprint is now a sha256 of canonicalised bytes (`sitecustomize.py:157-175`,
compared at `run_r1b_imma_train.sh:661-670`), with the float summaries demoted to diagnostics. I
tried to defeat it using the **real 160-tensor historical checkpoint** (374 MB,
`REF_imma_vangogh/imma_unet_xatten_layer.pt`) and the stage-5a block run verbatim:

| scenario | stage 5a |
|---|---|
| A unmodified file | PASS (160/160 checked, 0 mismatch) — correct |
| B **60 of 160 tensors silently dropped from the file** | **PASS**, "reloaded tensors are EXACTLY the in-memory state", `fingerprints 160 / checked 100 / missing 0 / MISMATCHES 0` |
| C one tensor's elements permuted (shape, dtype, `absmax` identical) | exit 1, names the tensor — correct |
| D executed 413 vs expected 500 | reported loudly, non-fatal — correct |
| E whole file downcast to fp16 | exit 1, 160 mismatches |

**DEFECT 3a — scenario B.** There is no `checked == len(fps)` guard, so a file missing tensors passes
5a with an "EXACTLY" claim. Stage 5b's `assert len(imma) == 160` (`:690`) catches it afterwards, so
the run still fails — but the acceptance check that the prereg names is the one that is wrong, and it
prints a true-sounding sentence about a file it has only partly checked.

The fingerprint is otherwise **strong enough**: a permutation (which leaves shape, dtype, `sum_f64`
*and* `absmax_f64` intact) is caught by the byte hash. Note the earlier float64-sum version was not
merely weaker but **flaky**: I reproduced that class of failure independently — a `float64` sum of
fp32 values is not thread-count invariant at the real scope sizes (identical at numel 102400/245760/
409600/491520; **two distinct values** across 1/8/64/128 threads at numel 983040 and 1638400, which
is **48 of the 160** scope tensors), and `accelerate launch` sets 128 CPU threads for the trainer
while the verifier runs at the default. The move to a byte hash removes both that and the ULP issue
the author's own note cites. Good change.

The wrapper cannot miss the relevant save: upstream writes the artifact with a bare
`torch.save(...)` at `defend_text_to_image_lora.py:977`, resolved through the module attribute at
call time, so the wrapper sees it; there is no `accelerator.save`, `save_pretrained` or
`safetensors` on this path (grep: none), and the save happens in the same (worker) process that
steps, so steps and fingerprints land in one record. End-to-end on CUDA-origin strided tensors is
**NEEDS-REAL-GPU**, but it is sound by construction: `.to("cpu").contiguous()` on both sides is an
exact copy, no arithmetic, so the bytes must agree.

**CLAIM UNSUPPORTED 3b — the fp16 evaluation export is still not verified.** Prereg §4 criterion 2
requires "any fp16 export checked against the explicitly converted source tensors". Nothing in the
launcher does this: the checkpoint is fp32 and stage 5a/5b compare fp32 only. The conversion does
happen — `scripts/gen_eval.py:56-62` builds the pipeline with `torch_dtype=torch.float16`,
`variant=BASE_VARIANT` (default `"fp16"`, `:22`) and then `pipe.unet.load_state_dict(sd)` casts the
fp32 IMMA tensors to fp16 **implicitly and unverified**. Scenario E shows stage 5a would *reject* an
fp16 file rather than verify one. Update magnitudes are correctly treated as diagnostics
(`:572-575`, `:694-701`).

## Requirement 4 — Amendments 1 and 2: **DEFECT FOUND** (1, minor, documentation accuracy)

Checked line by line against the pinned trainer:

* "no `variant` argument" — correct, and stage 1 re-asserts it with a whole-file substring test
  (`:186`).
* "loads the **main** weights at the pinned revision, then replaces the UNet entirely … strict full
  `load_state_dict` (`:452-454`)" — correct: `UNet2DConditionModel.from_pretrained(..., revision=...)`
  at `:447-449`, then `unet.load_state_dict(torch.load(args.delta_ckpt))` at `:452-454`, strict by
  default.
* evaluation precision — correct (`gen_eval.py:22`, `:56-62`).
* **DEFECT 4a:** Amendment 1's auxiliary row claims VAE, text encoder, tokenizer **and scheduler**
  are "loaded from the pinned base revision by the trainer's own calls". The scheduler is not:
  `DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")` at
  `defend_text_to_image_lora.py:439` passes **no `revision=`**, unlike `:440-449`. Low impact (a JSON
  config, and the local cache holds one snapshot) — but an amendment whose entire purpose is to say
  precisely which artifact each stage loads should not get a row wrong.
* No upstream behaviour is changed by either amendment — confirmed by the clean `git status`.
* Amendment 2 is **accurate**: my re-run shows this run at `n_subset=20` via
  `shuffle(seed=42).select([20,40))` against the historical 100 rows with no subsetting, so sample
  count and update count move together by 5x and the comparison is between configurations. The
  "controlled ablation not part of this experiment" framing matches the launcher.

## Requirement 5 — reservation output: **VERIFIED**

`run_r1b_imma_train.sh:513-523` checks `GPU_RESERVED` and `CUDA_VISIBLE_DEVICES` are present and
logs them; the old `export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` is gone. `scripts/gpu_reserve.sh:1779`
is what exports it. Those five lines are the only mentions of either variable in the launcher
(grep), so nothing downstream re-derives it, and stage 4's `env VAR=… $V …` wrapper (`:551-553`)
adds variables without replacing the environment, so the reservation's mapping reaches the trainer
unchanged.

---

## verdict: **NOT READY**

Nothing found is deep — the instrumentation design is sound, the gate really does raise, the
fingerprint survived every defeat I could build, and the two defects I found in the earlier revision
are already fixed. But three of the five requirements still have an acceptance check or a declared
claim that is wrong on paper, and all five fixes below are a few lines each and need no GPU.

Minimal list to reach READY:

1. `run_r1b_imma_train.sh:484` — write `rep.get("verdict")` (not `verdicts`) into
   `split_gate_r1b.json`, and include `coverage_complete`. Today the archived record says
   `"verdicts": null` beside `"enforced": true, "problems": []`.
2. `run_r1b_imma_train.sh:367` — reword to what the gate does ("coverage is incomplete; this
   developmental gate passes it as a NAMED EXCLUSION"), or pass `for_new_claim=True` and actually
   fail. As written the header claims an enforcement the code skips.
3. `run_r1b_imma_train.sh:673-679` — add `if checked != len(fps): fail`. Scenario B above passes a
   file missing 60 of 160 tensors with the message "EXACTLY the in-memory state".
4. `docs/PREREG_R1_IMMA_AUTHOR_BUDGET.md` Amendment 1 — correct the auxiliary-components row: the
   scheduler is loaded with no `revision=` (`defend_text_to_image_lora.py:439`).
5. Either implement prereg §4.2's fp16 clause (verify the eval-time fp32→fp16 cast at
   `scripts/gen_eval.py:56-62` against explicitly converted source tensors) or add a dated amendment
   stating it is deferred to the evaluation launcher. Right now the criterion is declared and
   unimplemented, which is the same posture these five requirements were created to end.

Recommended with the fixes, not required: assert the ascent optimizer holds 160 param tensors
(requirement 1 residual 2), clear stale `*.executed_updates.*.json` before training (residual 4),
and sum rather than `max` over per-process records if `--num_processes` ever moves off 1
(residual 3).
