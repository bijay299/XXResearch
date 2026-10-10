# Claim corrections, V4 — two interpretations withdrawn

Both corrections were raised by the PI on 2026-10-10. Both are about
**interpretation**, not measurement: every number stands, and nothing was
re-measured, re-run or re-hashed to produce this file.

Earlier rounds: [`CLAIM_CORRECTIONS_V2.md`](CLAIM_CORRECTIONS_V2.md),
[`CLAIM_CORRECTIONS_V3.md`](CLAIM_CORRECTIONS_V3.md).

---

## C4.1 — Different endpoints do not isolate nondeterminism as their cause

**What was claimed**

> "Because configuration and parent are identical, the divergence is **run-to-run
> nondeterminism** on this stack."
> — `DIAGNOSTIC_01_FINDINGS.md` §3, and the `conclusion` field of
> `diag_v1_evidence/u_vs_mab_comparison.json`

**Why it is wrong.** The comparison established two premises — the 32 *recorded*
effective hyperparameters are equal, and both runs loaded the same saved `MA` by
digest — and then named a mechanism. The premises do not identify a mechanism.
Everything *not* in those 32 fields was free to differ between a pilot run and a
run made weeks later, including:

* nondeterministic GPU kernels (attention/atomics/autotune) — the hypothesis, not
  the conclusion;
* library, driver and upstream-code state at the two run times;
* the **code path**: the new run trained with periodic checkpointing switched on
  (`--turn_on_checkpointing`, cadence 100); the saved `MAB` did not;
* dataloader worker count and the resulting batch ordering;
* any upstream default outside the recorded set.

No controlled repeat — same host, same commit, same cadence, twice — was run.

**What stands.**

| claim | status |
|---|---|
| 32/32 recorded hyperparameters identical; same saved `MA` by digest | **ESTABLISHED** |
| 0 of 32 tensors bitwise identical; max abs 2.045e-03 (s17), 1.778e-03 (s29); relative Frobenius 5.652e-03 / 5.717e-03 | **ESTABLISHED** |
| a re-run does not reproduce the original endpoint bitwise, so checkpoints from a new run are **not interchangeable** with the original run's | **ESTABLISHED** — and this is the only part the early-grid design relies on |
| the cause is run-to-run nondeterminism | **WITHDRAWN. NOT ESTABLISHED.** |
| training equivalence, or any behavioural claim | **NOT ASSESSED**, as before |

**No determinism study is being proposed.** The design simply stops asserting a
cause and measures the difference where it matters — at step 100, by the bridge
check.

---

## C4.2 — Straddling the target at steps 0 and 100 does not guarantee a matching checkpoint

**What was claimed**

> "The match point therefore lies **between step 0 and step 100**."
> — `DIAGNOSTIC_01_FINDINGS.md` §1 item 4, and `AMENDMENT_01_EARLY_GRID.md` §1

**Why it is wrong.** It treats a discrete, possibly non-monotone sequence as if
it were a continuous monotone function in which an intermediate-value argument
applies. What was measured is: suppression 0 pp at step 0 (the parent, by
definition) and 38.75 / 42.50 pp at step 100, against L2 levels of 30.00 / 33.75
pp. Measurement granularity is 1.25 pp (one detection in 80 pairs). Nothing was
observed at steps 1…99, and suppression is not known to be monotone in steps.
So the sequence may step **over** the ±5 pp band between two consecutive grid
points, or enter and leave it, and there may be **no** step whose suppression
lies within tolerance.

**What stands.**

| claim | status |
|---|---|
| suppression is 0 pp at step 0 and above the L2 level by step 100, on both seeds | **ESTABLISHED** |
| the measured points **straddle** the L2 level | **ESTABLISHED** |
| the frozen 100-step grid offers no point inside that interval | **ESTABLISHED** |
| a matching discrete checkpoint **exists** between steps 1 and 99 | **WITHDRAWN. NOT ESTABLISHED.** |
| the early grid **tests** whether such a point exists on 10…90 | the corrected framing; the grid may legitimately find none, and that is the result |

Also unchanged from V3 and restated: **when L2 first attained its endpoint
suppression level is unknown** (no L2 intermediates exist), whether U attained
that level strictly before step 100 is unknown, no matched bird-retention result
exists, the frozen test set has never been evaluated, and detector output is
**provisional** until the blinded human annotation is complete.

---

## Where the superseded wording still appears, and why

`results/audit_v1/diag_v1_evidence/` is the evidence bundle the coordinator
**independently verified**, including its per-file `MANIFEST.json` digests. It is
therefore left **byte-identical**: editing a verified artifact to improve its
prose would invalidate the verification it exists to support.

The superseded sentence survives inside
`diag_v1_evidence/u_vs_mab_comparison.json` → `conclusion`. **This file governs
instead**, and a correction notice pointing here has been appended to
`diag_v1_evidence/README.md` — the one file in the bundle that `MANIFEST.json`
does not cover, so the notice disturbs no verified digest. The
live documents and the generating code have been corrected, so nothing new will
repeat it:

| corrected | what changed |
|---|---|
| `results/audit_v1/DIAGNOSTIC_01_FINDINGS.md` | §1 item 4 and §3 rewritten to the statements above |
| `docs/research/AMENDMENT_01_EARLY_GRID.md` | §1 and §3 rewritten; the grid is framed as a hypothesis test |
| `scripts/seq/compare_u_vs_mab.py` | the `conclusion` it emits no longer names a cause; "NOT ASSESSED" now covers causation |
| `scripts/seq/bridge_check.py` | records, in every output, that a weight difference's cause is not isolated |
