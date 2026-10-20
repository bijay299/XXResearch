# AMENDMENT 01 — publish missing evidence, and narrow eight overstated claims

**Date:** 2026-10-20
**Amends:** `37c31ca17a004b7bf7c44634e395e2eef7a9678f` (`results/llada_rwku_baseline_v1/`)
**Nature:** evidence publication + documentation correction. **No model was executed.**

> Zero GPU-seconds were consumed by this amendment. No alias was altered, no frozen
> evaluator was revised, no frozen script or raw record was modified, and the study's
> outcome is **unchanged**.

## Outcome retained, unchanged

| | |
|---|---|
| Pristine recall | 🔴 **`INSUFFICIENT_BASE_RECALL`** — 0 of 10 entities at the frozen ≥6/10 threshold; max 5/10 |
| Training steps | **zero** |
| Qualified baseline | **none** |

The frozen threshold is **not** revised and the study is **not** retrospectively
qualified. The offline audit re-derived the outcome at the original threshold and
confirms it (13/13 checks, 200/200 per-item scores reproduce exactly).

---

## Part 1 — the missing evidence is now published

**The entire `outputs/` directory was absent from `37c31ca`.** Five artifacts the report
relied on were never published:

| Artifact | Bytes | SHA-256 | Hash verification |
|---|---|---|---|
| `outputs/pristine_screen_primary.json` | 204,662 | `13419c52c6cdd8fe9f52f9489254d873cd0a931418df1d795f2084708d76e660` | ✅ **matches** the hash recorded in published log `a001` |
| `outputs/pristine_screen_secondary.json` | 214,176 | `263f7f78784ef715aaca21558c51c2e16633e9caf898346c4dac25e7b2b6958a` | ✅ **matches** the hash recorded in published log `a002` |
| `outputs/objective_verification.json` | 3,415 | `c8235873706f9fd1be4ebadb5525a9deea3b08ca35c3696eba31977aaa8c231b` | no hash was recorded in any log; actual hash published |
| `outputs/evaluator_verification.json` | 8,821 | `e22de66b324234dbd14122e11c924032cd69c20423f2e4739ce1c01463474150` | no hash was recorded in any log; actual hash published |
| `outputs/teacher_wiring.json` | 560 | `dfb255f885e8a11848c070ef30ff6dc7516108ef842afa7d36e43029580efa1c` | no hash was recorded in any log; actual hash published |

**Root cause (verified, not inferred).** `.gitignore` line 67 is `outputs/`, so
`git add -A results/llada_rwku_baseline_v1` silently skipped the whole subtree — the same
class of failure as the `logs/` rule in the previous study's amendment. The originals were
preserved in the task workspace and are published here **verbatim**; nothing was
regenerated or reconstructed.

Staged with a per-path `git add -f` so only these five intended files bypass the rule;
**`.gitignore` is unchanged**.

**The index was honest; the README was not.** `EVIDENCE_INDEX.md` at `37c31ca` was
generated from `git ls-files`, so it contained **zero** rows referencing `outputs/` — it
never claimed them. But `README.md` line 39 listed
`| outputs/ | Every generation and score, both decoders; … |` as published content. That
claim was false and is corrected.

### Artifacts that were overwritten — disclosed, not reconstructed

`objective_verification.json` and `evaluator_verification.json` were written to a fixed
path by each successive attempt, so **only the final passing version of each survives**:

- `objective_verification.json` — written by `a003` (which **failed**: `kl_direction`), then
  **overwritten** by `a004` (all-pass). The `a003` JSON no longer exists.
- `evaluator_verification.json` — written by `a006` (which **failed**: 42/44), then
  **overwritten** by `a007` (48/48). The `a006` JSON no longer exists.

The *logs* of both failed attempts are preserved and published (`a002`, `a003`, `a006`),
so the failures remain auditable; their JSON outputs do not exist and were **not**
manufactured.

---

## Part 2 — eight corrected claims

### 1. The old result is about ten frozen panels, not LLaDA in general

The report's framing invited a broader reading than the evidence supports. Corrected scope:
the result concerns **the first ten canonical RWKU subjects**, **10 frozen qualification
facts each**, **one frozen conservative alias policy**, **two frozen prompt templates**
(answer-only and cloze), and **two frozen decoding schedules** (64 tok/64 steps and
64 tok/32 steps, block 32, temperature 0).

It establishes that *pristine LLaDA-8B-Instruct did not clear a 6/10 exact-match bar on
those specific panels under those specific settings.* It does **not** establish LLaDA's
general ability — or inability — to support RWKU unlearning. Different subjects, prompts,
alias policies, decoders or metrics could yield materially different recall.

### 2. MMLU success does not validate the open-answer scoring choices

The report used 75% MMLU as evidence that "the pipeline, prompting and parsing are sound."
That is **too strong**. MMLU and the forget panels run through **disjoint scorers**:

- MMLU → `parse_mmlu_letter` → one of four letters (25% chance baseline).
- Forget panels → `parse_short_answer` → `normalize` → `exact_match` vs frozen aliases.

So 75% MMLU evidences that the model loads, generates, and follows a constrained
single-letter instruction — and that the *letter* parser works. It exercises **none** of
the alias policy, the normalization rules, or the open-answer parser that actually produced
the 0/10 outcome. Those were validated separately (48 assertions), and that is the correct
support for them — not the MMLU number.

### 3. `span_recall` measures reference-token overlap, not answer-span detection

The report called this "token-boundary answer-span recall", which implies detection of a
contiguous span. It does not do that. It computes the fraction of reference tokens present
anywhere in the prediction as whole tokens — **set membership, order- and
adjacency-insensitive**. Demonstrated on the frozen metric itself:

```
span_recall("Hathaway and also Berkshire", "Berkshire Hathaway") == 1.0
span_recall("Berkshire Hathaway",          "Berkshire Hathaway") == 1.0
```

A scrambled, non-adjacent prediction scores identically to the exact string. The metric is
**reference-token overlap**; it is renamed as such here. (ROUGE-L recall, reported
alongside, *is* order-sensitive via LCS.) This does not change any number — only what the
number means.

### 4. Objective verification was helper equivalence + source-traced dispatch, not native trainer execution

The report's phrasing ("verified against the pinned native functions", "the real training
path executes this objective") overstated the runtime scope. What was actually done:

- **Genuinely executed:** the extracted source text of two methods — `_null_anchor_kl` and
  `_random_sft_null_anchor_loss` — run inside a minimal shim against **stub models** on
  controlled fixtures. This is real execution of native *helper* code and establishes
  equivalence of our port on those inputs.
- **NOT executed:** `MDUTrainer` was never instantiated and the complete native trainer
  never ran. It cannot, without `dllm` (unavailable, revision still unverified). The shim
  supplied `scheduler`, `time_epsilon` and `_postprocess_outputs` itself.
- **Established only by source-string inspection, not runtime:** the `compute_loss`
  dispatch claim (that `match_mode=random` + `null_anchor` routes to
  `_random_sft_null_anchor_loss`, that the loss is added not negated, and that `alpha=0`
  skips the retain branch). The check searched for literal substrings in the file. **That
  is static inspection and must not be called runtime verification.**

### 5. Native KL arithmetic is FP32, despite float64 fixture inputs

The report said comparisons were made "in float64 where the native code allows" and
reported max-abs-diff `0.0`. More precisely: `_null_anchor_kl` casts via `.float()`
unconditionally, so **the KL and its gradient are computed in float32 even when the
fixtures are float64**. The reported `0.0` agreement is therefore an **FP32-precision**
agreement obtained because both sides perform the identical cast — not a float64 result.

This is exactly what the first verification attempt exposed: it failed a 1e-9 tolerance
with a residual of 2.6e-07, which is float32 epsilon scale. Corrected reading: the
objective port is verified to FP32 precision, which is the precision the native code
actually uses.

### 6. Gate weakness: two failing checks returned exit code zero

Of eight attempts, there was **one execution exception** and **two failed verification
attempts that nevertheless exited 0**:

| attempt | what happened | printed verdict | **exit code** |
|---|---|---|---|
| `a002` objective | `RecursionError` (a `torch.rand` monkeypatch called itself) | traceback | **1** ✅ correct |
| `a003` objective | `kl_direction` check **FAILED** | `ALL OBJECTIVE CHECKS PASSED: False` | **0** ❌ |
| `a006` evaluator | parser + no_ar_likelihood **FAILED**, 42/44 | `EVALUATOR CHECKS: 42/44 … all=False` | **0** ❌ |

Neither `verify_objective.py` nor `verify_evaluator.py` contains any `sys.exit`
(`grep -c` → 0 for both), so **a failing correctness gate was indistinguishable from
success to any automated caller.** The failures were caught only because the logs were read
by eye. Had they not been, the study could have proceeded on unverified checks.

The final passing results (`a004` 10/10, `a007` 48/48) are preserved and remain valid —
they were re-run after real fixes. The frozen scripts are **not** modified here. Stage B's
checks exit nonzero on failure, and that behavior is proven by a negative control.

### 7. Chronology: the supplementary model checks ran after the outcome was known

The report presented the verification suite without timing, which could imply these were
pre-registered gates. They were not. The primary recall screen (`a001`) ended at
**00:45:10**, at which point 0/10 qualifying was already determinable. **All seven
subsequent attempts ran after that:**

| time | attempt | note |
|---|---|---|
| 00:45:10 | — | **0/10 qualify becomes known** |
| 00:46:36 | `a002` recall secondary + MMLU (114 s, GPU) | supplementary; cannot change selection |
| 00:46:38 | `a002` objective (3 s) | failed, rc=1 |
| 00:47:05 | `a003` objective (2 s) | failed, rc=0 |
| 00:48:08 | `a004` objective (2 s) | pass |
| 00:49:06 | `a006` evaluator (2 s) | failed, rc=0 |
| 00:49:55 | `a007` evaluator (2 s) | pass |
| **00:50:44** | **`a008` teacher wiring (9 s, GPU)** | **~5.5 min after the stop condition** |

So the objective, evaluator and teacher checks are **post-hoc infrastructure
verification**, performed knowing the study would terminate — not gates that informed a
decision. This does not invalidate them, but they should not be read as pre-registered.

Additionally, on the teacher check specifically: the report already stated that invariance
after updates was unverified. Restated plainly here — **a freshly zero-initialized adapter
agreeing with the base model does not establish teacher invariance after updates.** A
zero-init LoRA is a no-op by construction, so that agreement is arithmetically guaranteed
and carries no information about whether the frozen teacher stays frozen once the student
changes. That property remains **unverified** in this study.

### 8. Historical CPU usage remains unmetered

The original report already declined to give a CPU number. Restated and preserved: no
`getrusage` instrumentation was wired into that study's attempt runner, so its CPU
consumption is **unmetered and unknown**. No retrospective charge is invented, and no
claim of comprehensive CPU accounting is made for it. CPU metering begins with the work
that follows this amendment.

---

## Offline audit record

Deterministic, model-never-executed, using only stored outputs
(`outputs/stage_a_audit.json`, 13/13 passed):

- **200/200 per-item scores reproduce exactly** — EM, reference-token overlap and ROUGE-L
  recomputed from the stored `pred_parsed` strings with the frozen scorers; zero mismatches.
- Per-entity summaries reproduce; 10 entities × 10 facts × 2 decoders = 200 generations.
- **0 entities qualify at the frozen threshold of 6**; max 5/10 primary, 4/10 secondary →
  outcome `INSUFFICIENT_BASE_RECALL` **retained**.
- Zero mechanical failures across all 200 generations, confirmed by recomputation.
- MMLU panel is 40 items, accuracy 0.75 reproduces, every item parsed.
- The audit gate's exit code was negative-control tested: breaking one assertion yields
  exit **1**.

## Files changed

| File | Change |
|---|---|
| `AMENDMENT_01.md` | **new** — this document |
| `outputs/*.json` (5 files) | **newly published**, original contents verbatim |
| `outputs/stage_a_audit.json` | **new** — offline audit record |
| `README.md` | corrected the false `outputs/` publication claim; amendment pointer |
| `FINAL_REPORT.md` | amendment banner; corrections 1–8 applied at the affected claims |
| `EVIDENCE_INDEX.md` | regenerated from tracked files |
| `FREEZE_NOTE.md`, `frozen/*`, `scripts/*`, `logs/*`, `evidence/ATTEMPTS.tsv` | **unchanged** |
