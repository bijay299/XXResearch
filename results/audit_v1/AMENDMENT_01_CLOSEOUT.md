# Amendment 01 — closeout

**Status: CLOSED.** The pipeline completed, the diagnostic is finished, and no
further work on it is proposed. All outputs and both selected step-70
checkpoints are preserved. No GPU work was done for this closeout and none is
authorised by it.

---

## The outcome, in one paragraph

Development matches were found **at step 70 on both seeds**. On the frozen test
set, **test eligibility failed on both seeds** — all four arms fall below the
≥ 30 pp dog-suppression gate (17.50 / 16.25 pp on seed 17; 22.50 / 28.75 pp on
seed 29) while clearing the ≤ 60% residue gate — and on seed 29 the ≤ 5 pp match
tolerance **additionally** failed at 6.25 pp. **Neither seed supplies a valid
matched comparison** in the prespecified regime, so the **matched-retention
question remains inconclusive in the intended regime**. The bird contrasts and
intervals are retained as **descriptive** results. **Human labels remain
pending**; every detector number stays provisional.

---

## The three test conditions, kept separate

Protocol §5 requires "the **same gates and the same match**" on TEST. That is
three questions, and the analysis now reports three:

| | | seed 17 | seed 29 |
|---|---|---|---|
| **(a) tolerance** | arms agree within 5 pp | **pass** (1.25 pp) | **FAIL** (6.25 pp) |
| **(b) eligibility** | each arm ≥ 30 pp suppression **and** ≤ 60% residue | **FAIL** (both arms) | **FAIL** (both arms) |
| **(c) validity** | (a) **and** (b) | **NOT VALID** | **NOT VALID** |

Seed 17's arms agree with each other to 1.25 pp — but **outside** the authorised
regime, both having deleted about half of what the gates require. Agreement
between two under-deleting arms is agreement, not a matched comparison. The
comparison is therefore **rejected, not the hypothesis** — §7's last row says so
for a failed match, and §10 makes a reference-gate failure a stop-and-report
condition in its own right.

## What was withdrawn, and what stands

**Withdrawn — for seed 17 as well as seed 29:** practical equivalence at the
declared ±10 pp margin; exclusion of a ≥ 10 pp material bird benefit; and any
statement of the form "at matched dog deletion, L2-SP shows no retention
advantage".

**Stands, as description:** bird contrast **+1.25 pp [−2.50, +5.00]** (seed 17)
and **−1.25 pp [−6.25, +3.75]** (seed 29), and every per-category,
per-threshold and per-family interval, unchanged in value. These are
measurements of what two particular checkpoints do.

**Withheld is not disproved.** Nothing here shows L2-SP retains more bird, and
nothing shows it does not. "Inconclusive in the intended regime" is a third
answer, not a disguised negative one. The seeds' directions also disagree
(+1.25 vs −1.25 pp), which §7 reads as unresolved on its own terms.

**Not claimed.** This is **not** a general defect of matched-effectiveness
designs — it is one selected pair failing declared conditions in one pilot.
Causes and general reliability are **unresolved**; why the development level did
not transfer to the test prompts is **not established**, and no determinism or
repeatability study was requested or run.

## What was repaired, CPU only

| | |
|---|---|
| `scripts/seq/diag_analysis.py` | re-applies the §5 **gates** on TEST alongside the match, reports tolerance / eligibility / validity as three separate fields, and withholds every protocol-level claim where validity fails while keeping the interval |
| `scripts/seq/test_diag_test_conditions.py` | **59 checks, 0 failures** — including arms-close-but-both-under-suppressed, a valid positive case, the gate boundaries, and source-bound assertions against the committed analysis record |
| `analysis_test.json` | **v3**. v1 and v2 preserved verbatim beside it; nothing overwritten |
| annotation | the prescribed **180-item paired** packet built from existing images; blinding **attacked rather than asserted** — its first build was de-blinded 180/180 through the published sampling seed and row order, now fixed with a secret salt *and* a separate secret-seeded row order ([`BLINDING_AUDIT.md`](annotation_packet_paired180/BLINDING_AUDIT.md)). The **228-item** packet is preserved as a separate supplementary audit and is **BROKEN** on the identifier channel — reproducible from files committed at `7e01fbd`, so not caused by this archive; 0 labels filled in any packet, 13 images overlap, no key disclosed, no samples mixed |
| evidence archive | `unattended.log` supplied and verified against the identity the published manifest recorded; raw development and frozen-test detector rows added; **access limitations: none** |
| compute | **2.9013889** device-hours total, **1.6991667** this amendment, 27 earlier entries intact — recomputed, not re-engineered |

Nothing was reselected, no tolerance was widened, no grid was refined, and the
frozen test set was not evaluated again. **All 378 estimates per seed are
bit-identical between the published analysis and the corrected one** — only the
interpretation fields changed, which is the whole of the repair.

## Where to read it

- [`AMENDMENT_01_RESULTS.md`](AMENDMENT_01_RESULTS.md) — the corrected results, §7.1–§7.2 for the failure, §9 for the evidence index
- [`CLAIM_CORRECTIONS_V5.md`](CLAIM_CORRECTIONS_V5.md) — every withdrawn claim, with what replaced it
- [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) — **exact paths and sha256 of every published file**, key-free, verifiable row by row with `git show <commit>:<path> | sha256sum`
- [`amendment01_evidence/`](amendment01_evidence/) — 15 hash-listed payloads; selection and all intervals re-derivable without images or weights
- [`annotation_packet_paired180/`](annotation_packet_paired180/) — the prescribed audit, key-free metadata
- [`../../docs/research/NEXT_SCREEN_PROPOSAL.md`](../../docs/research/NEXT_SCREEN_PROPOSAL.md) — the next scientific screen, CPU-only, **not authorised to run**
