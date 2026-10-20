# AMENDMENT 02 — audit-contradiction resolution and stale-claim corrections

**Date:** 2026-10-20
**Amends:** [`a349197c471d8bd74878b6b26f622f3f225c7232`](https://github.com/bijay299/XXResearch/commit/a349197c471d8bd74878b6b26f622f3f225c7232) (which itself amended [`37c31ca17a004b7bf7c44634e395e2eef7a9678f`](https://github.com/bijay299/XXResearch/commit/37c31ca17a004b7bf7c44634e395e2eef7a9678f))
**Nature:** additive documentation correction. **No new experiments.**

> No model was loaded, no GPU allocated, no text generated, no optimizer update
> performed, no correctness test re-run, no literature search repeated, and no frozen
> scientific code or configuration changed. Work was limited to offline inspection,
> arithmetic, hashing and publication checks. **Zero GPU-seconds.**

## Outcome retained

**`INSUFFICIENT_BASE_RECALL`** — 0 of 10 entities at the frozen ≥6/10 threshold, max 5/10,
three required. Zero *scientific* training steps in this study. No qualified baseline.
The frozen threshold is not revised and the study is not retrospectively qualified.
The independently confirmed **200/200 per-item score agreement** stands, as do the
original answers, thresholds and panels.

---

## Correction 1 — the audit contradiction, resolved

Amendment 01 claimed the published audit was "13/13 passed". **The published artifact says
otherwise.** `outputs/stage_a_audit.json` (sha256 `fdb663da5952866f…`, preserved unchanged)
records:

```
all_passed    : false
n_fail        : 1
failed_checks : ["ten_entities_screened"]
```

So the unqualified "13/13 PASS" claim in Amendment 01 (lines 21 and 208) **was not
supported by the artifact it cited**. That is corrected here.

### What the existing records establish

The normal-run log (`logs/b0001_stageA_audit_offline_recompute.log`, published in Stage B's
directory as the attempt log) ends:

```
recomputed 200 per-item scores from stored predictions
AUDIT: ALL PASS
```

The discrepancy is explained — and the explanation is **established by existing records,
not merely possible**, on three independent grounds:

1. **Logically unreachable verdict.** The audit script asserts
   `chk("ten_entities_screened", len(pri["entities"]) == 10, len(pri["entities"]))` —
   the comparison target and the recorded `detail` are the same quantity. The published
   JSON records `passed: false` **with `detail: 10`**. In the normal script `detail == 10`
   forces `passed == true`; the published combination cannot be produced by it.
2. **The negative-control variant survives and matches.** `audit_negative.py` (retained at
   `/data/bijaypandey/llada_emdu/tmp/audit_negative.py`, published here as
   `evidence/audit_negative_control/audit_negative.py`) differs from the audit script in
   exactly one line — `len(pri["entities"]) == 999` — which produces precisely
   `passed: false, detail: 10`. It writes to the **same hard-coded output path** as the
   normal run.
3. **Timestamps.** The JSON's mtime (`01:17:26.279`) is **0.100 s after** the normal run's
   log mtime (`01:17:26.179`), consistent with the negative control executing immediately
   afterwards within the same shell invocation.

**Conclusion:** the normal run reported 13/13 and the negative control then **overwrote**
its JSON 0.1 s later. The 12 other checks in the published file still record `passed:
true`, including `per_item_scores_reproduce_exactly` and
`zero_entities_qualify_at_frozen_threshold` — so the two load-bearing results (200/200
agreement; 0 entities qualifying) are intact in the artifact itself.

**A matching normal-run JSON was not recovered and has not been manufactured.** The audit
was **not** re-run. What is published is: the overwritten artifact as-is, the negative
control that overwrote it, and this explanation.

**Residual uncertainty.** The "13/13" verdict now rests on the *log* plus the 12 passing
checks inside the overwritten JSON — not on a recovered passing JSON. Anyone requiring a
machine-readable all-pass artifact must re-run the audit themselves; that is explicitly
out of scope here.

## Correction 2 — MMLU does not establish a sound evaluation harness

Amendment 01 already narrowed this once. It is narrowed again, because residual wording
still over-claimed. `README.md` said the 75% MMLU result shows "the model loads, generates
and parses correctly", and `FINAL_REPORT.md` §8 said "MMLU at 75% shows the harness is
sound".

**Corrected:** MMLU exercises a 4-way single-letter path (`parse_mmlu_letter`, 25% chance
baseline) that is **disjoint** from the open-answer path which produced the 0/10 outcome.
A 75% MMLU score is evidence that the checkpoint loads and that *that one* constrained
decode-and-parse path works. It does **not** establish that the whole evaluation harness
is sound, and in particular says nothing about the alias policy, the normalization rules
or the open-answer parser.

## Correction 3 — decoding effects are not ruled out

`FINAL_REPORT.md` §9 said the shortfall is "not decoder-specific (secondary agrees), so it
will not be fixed by tuning decoding."

**Corrected and withdrawn.** Two decoding schedules were tested — 64 tokens/64 steps and
64 tokens/32 steps, both block 32, temperature 0, low-confidence remasking. Agreement
between **two** points in a large decoding space does not rule out decoding effects and
does **not** prove that decoding changes cannot affect recall. Untested dimensions include
block length, generation length, temperature, remasking strategy, CFG and prompt
formatting. The honest statement is: *recall was low at both tested schedules; other
decoding configurations were not evaluated.*

## Correction 4 — the 200-entity screen recommendation is withdrawn

`FINAL_REPORT.md` §9 recommended screening all 200 RWKU entities, projecting "roughly
1.1 GPU-h" from the 10-entity screen's 200 GPU-s.

**Withdrawn.** That recommendation is stale: the shared resource envelope has since been
consumed by subsequent work (see Stage B's corrected ledger), and this amendment is
authorized to perform **no new experiments**. The projection is also a linear
extrapolation from a single measurement, not a measured cost. No replacement experimental
recommendation is offered.

## Correction 5 — "two failures" disambiguated

`FINAL_REPORT.md` §6 used the heading "**Two failures, both preserved and neither
overwritten**", conflating two different things.

**Corrected.** There were **three** distinct abnormal events of two different kinds:

| Kind | Attempt | What | Exit code |
|---|---|---|---|
| **Execution exception** (1) | `a002` objective | `RecursionError` — the process crashed | **1** (correct) |
| **Logical check failure returning zero** (2) | `a003` objective | `kl_direction` FAILED; printed `ALL OBJECTIVE CHECKS PASSED: False` | **0** ❌ |
| | `a006` evaluator | 42/44; printed `all=False` | **0** ❌ |

One crash (correctly signalled) and **two logical failures that exited 0** — neither
`verify_objective.py` nor `verify_evaluator.py` contains any `sys.exit`, so a failing gate
was indistinguishable from success to an automated caller.

Also noted here, relevant to Correction 1: the §6 phrase "neither overwritten" was true of
the *logs* but **not** of the JSON outputs. Amendment 01 disclosed that the objective and
evaluator JSONs were overwritten by successive attempts; Correction 1 above shows the audit
JSON was overwritten too.

## Correction 6 — unconditional assertions in the published checks

Retained: the observed passing counts (`a004` 10/10, `a007` 48/48, audit 13/13 per its
log). **Narrowed:** some of those checks cannot fail, so their coverage is **not
established**:

| Check | File | Why it cannot fail |
|---|---|---|
| `reduction_is_mean_over_masked_answer_positions` | `scripts/verify_objective.py:179` | records `"passed": True` as a literal; no comparison is performed |
| `mmlu_and_open_answer_use_disjoint_scorers` | `scripts/audit_stage_a.py:125` | `chk(..., True, …)` — a documentation statement, not a test |

These are descriptive entries, not verifications. The frozen scripts are **not modified**;
this amendment only states what they do and do not establish.

---

## Preservation inventory (unchanged by this amendment)

| Path | Bytes | SHA-256 (first 16) |
|---|---|---|
| `outputs/pristine_screen_primary.json` | 204,662 | `13419c52c6cdd8fe` |
| `outputs/pristine_screen_secondary.json` | 214,176 | `263f7f78784ef715` |
| `outputs/stage_a_audit.json` | 5,030 | `fdb663da5952866f` |
| `outputs/objective_verification.json` | 3,415 | `c8235873706f9fd1` |
| `outputs/evaluator_verification.json` | 8,821 | `e22de66b324234db` |
| `outputs/teacher_wiring.json` | 560 | `dfb255f885e8a118` |
| `evidence/ATTEMPTS.tsv` | 1,002 | `f435c6bf61c3aed5` |
| `frozen/frozen_spec.json` | 2,363 | `84554f932e075dab` |
| `frozen/frozen_panels.json` | 445,126 | `7c7ce60149b3945f` |
| `scripts/freeze_eval.py` | 17,440 | `6bb748af1c9792ed` |
| `scripts/verify_objective.py` | 13,491 | `a6a85c1e4e346667` |
| `scripts/verify_evaluator.py` | 7,515 | `f133d2632dc614e0` |
| `scripts/audit_stage_a.py` | 8,494 | `ab5cbcb2b6b9099f` |
| `logs/` (8 original attempt logs) | — | byte-identical to `37c31ca` |

Full current hashes for every published file are in `EVIDENCE_INDEX.md`.

## Measured / estimated / untested

- **Measured:** the 200 qualification generations and their scores; the 200/200 offline
  recomputation; the 40-item MMLU result; the eight attempt rows and their exit codes.
- **Estimated:** nothing new in this amendment. The withdrawn 200-entity projection was an
  extrapolation, now removed.
- **Untested / unestablished:** decoding configurations beyond the two frozen schedules;
  the coverage of the two unconditional checks above; whether a passing audit JSON would
  reproduce (not re-run).
