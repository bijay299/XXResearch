# Diagnostic 01 — findings, and corrections to how I first stated them

**Outcome: a GRID-MATCH FAILURE.** The matched bird-retention question is
**inconclusive** — it was never measured. The frozen test set was not evaluated.

Original run: commit `3a5a69c`, output `/data/bijaypandey/cuig_pilot/seq_pilot/diag_v1`,
1.2022 device-hours recorded, 0 failed stages, the frozen test set never
evaluated. Evidence bundle: [`diag_v1_evidence/`](diag_v1_evidence/).

Two interpretations stated here were corrected on 2026-10-10; the corrections are
marked in place below and set out in
[`CLAIM_CORRECTIONS_V4.md`](CLAIM_CORRECTIONS_V4.md). The GPU-hour ceiling that
framed the original run was retired on the same day
([`RESOURCE_POLICY_2026-10-10.md`](../../docs/research/RESOURCE_POLICY_2026-10-10.md));
the run's recorded usage is unchanged.

Detector-based and **PROVISIONAL**: the blinded human annotation is a
prerequisite for any scientific conclusion and no labels exist yet.

---

## 1. What is established

Measured on the **development set only** (dog-only, 20 prompt texts × 4
generation seeds = 80 pairs per checkpoint, threshold t=0.5, against each
seed's **own** MA), by this COCO detector:

1. **No dump on the frozen 100-step grid matched**, on either seed. All ten
   dumps passed both reference gates; none landed within the 5 pp tolerance of
   its seed's L2 suppression. The closest mismatch was **8.8 pp on both seeds**.
2. **The L2 references were themselves eligible**, under the gate now applied to
   the reference as well as to the candidates: seed 17 at 30.0 pp (exactly on
   the ≥30 pp gate) and 52.5% residue; seed 29 at 33.8 pp and 48.8% residue.
3. **By step 100, U's measured dog suppression exceeded the saved L2 endpoint's
   measured suppression** — 38.75 pp vs 30.0 pp (seed 17), 42.50 pp vs 33.75 pp
   (seed 29).
4. The measured points therefore **straddle** the L2 level: 0 pp at step 0,
   where step 0 is the parent MA by definition, and above it by step 100. The
   frozen grid provides no point in between. **Straddling is all that is
   established** — see §2.

| seed | MA residue | L2 residue → suppression | step-100 residue → suppression | closest mismatch |
|---|---|---|---|---|
| 17 | 82.5% (66/80) | 52.5% → **30.00 pp** | 43.75% → **38.75 pp** | **8.75 pp** |
| 29 | 82.5% (66/80) | 48.75% → **33.75 pp** | 40.00% → **42.50 pp** | **8.75 pp** |

Granularity is 1.25 pp (one detection in 80 pairs).

---

## 2. What is NOT established — and corrections to what I wrote

I previously reported:

> "L2-SP took 1,000 steps to reach ~30 pp of dog suppression; the unregularised
> arm passed that level in under 100."

**Both halves of that overstate the evidence. Withdrawn.** Corrected:

| claim | status |
|---|---|
| "L2-SP took 1,000 steps to reach ~30 pp" | **NOT ESTABLISHED.** No L2 intermediate checkpoints exist — only its 1000-step endpoint was ever measured. **When L2 first attained its endpoint suppression level is unknown.** It may have reached it far earlier. |
| "U passed that level in under 100 steps" | **NOT ESTABLISHED.** Only step 100 was measured, where U is already above the L2 level. Whether U attained it **strictly before** step 100 is unknown; the measurements are discrete and nothing between steps 0 and 100 was observed. |
| an implied speed comparison between the arms | **NOT SUPPORTED.** It compares a measured endpoint against an unmeasured history. |

Also not established:

* **That a matching checkpoint exists between steps 1 and 99.** An earlier
  version of §1 said "the match point therefore lies between step 0 and step
  100". **Withdrawn** ([`CLAIM_CORRECTIONS_V4.md`](CLAIM_CORRECTIONS_V4.md)
  §C4.2): that applies an intermediate-value argument to a **discrete**,
  possibly **non-monotone** sequence measured at 1.25 pp granularity. Steps
  1–99 were never evaluated; the sequence may step over the ±5 pp band between
  consecutive grid points, or enter and leave it. There may be no step whose
  suppression falls within 5 pp of the L2 level, and the grid cannot distinguish
  "no such step exists" from "the grid is too coarse to find it".
* **Any matched bird-retention result.** No retention contrast was computed on
  any set. The frozen test set has never been evaluated — its output directory
  is **empty, 0 files**, documented in the bundle inventory.
* **Any mechanism.** Nothing here speaks to why either arm behaves as it does.
* **Symmetry of completion evidence.** The saved L2 endpoints' training
  completion is **UNVERIFIED** legacy evidence (their reports predate the
  completed-step counter and the check then in use was circular). The new
  unregularised trajectories' completion is counter-verified at 1000 steps. The
  asymmetry is disclosed, not corrected, and nothing was backfilled.

---

## 3. A related finding: training is not bitwise reproducible here

The planned CPU comparison of each new 1000-step `U` endpoint against its saved
`MAB` endpoint — the **same** configuration, parent, target, anchor,
`l2sp_weight 0`, step count and training seed — found them numerically
different ([`u_vs_mab_comparison.json`](diag_v1_evidence/u_vs_mab_comparison.json)):

| seed | configuration | parent | bitwise-identical tensors | max abs diff | relative Frobenius |
|---|---|---|---|---|---|
| 17 | identical, 32/32 hyperparameters | same saved MA `3cb72ee162b9…` | **0 of 32** | 2.045e-03 | 5.652e-03 |
| 29 | identical, 32/32 hyperparameters | same saved MA `fbf49199a306…` | **0 of 32** | 1.778e-03 | 5.717e-03 |

Serialisation hashes alone could not have shown this, and would not have been
sufficient: a hash mismatch can arise from serialisation order or metadata
alone. The four questions — configuration, parent identity, serialisation,
numerical agreement — are answered separately, and **training equivalence is
refused**: it would need matched generation and scoring.

**The CAUSE is not established.** This section previously concluded "because
configuration and parent are identical, the divergence is **run-to-run
nondeterminism** on this stack". **Withdrawn**
([`CLAIM_CORRECTIONS_V4.md`](CLAIM_CORRECTIONS_V4.md) §C4.1): equal *recorded*
configuration and an equal parent do not identify a mechanism. Everything
outside those 32 fields was free to differ between the pilot run and this one —
library and driver state, the upstream code path (this run trained with periodic
checkpointing switched on; the saved MAB did not), dataloader worker ordering,
and nondeterministic kernels. Separating them needs a controlled repeat — same
host, same commit, same cadence, twice — which was not run and is not proposed.

What **is** established is the part the design depends on: a re-run does not
reproduce the original endpoint bitwise, so checkpoints from a new run are
**not** interchangeable with the original run's, and a design that mixes them
must measure the difference rather than assume it away. This shapes
[Amendment 01](../../docs/research/AMENDMENT_01_EARLY_GRID.md), which bridges the
two realisations at step 100 — in settings, in weights and in development
behaviour — and declares a stop condition on that bridge in advance.

---

## 4. Preserved records

Both selection records are retained, and they agree on every decision-relevant
value:

| record | decided | note |
|---|---|---|
| original, inside the `all` run | 2026-10-09T21:50:20Z | survives as printed output in `run_all.log`; **the JSON was overwritten** by the CPU re-run before non-destructive writing was implemented |
| CPU-revised (bracketing) | 2026-10-09T21:59:25Z | preserved as `selection.20261009T215925.superseded.json` |
| CPU-revised (bracketing + L2 reference eligibility) | 2026-10-10 | current `selection.json` |

The overwrite cost the **file**, not the evidence: a field-by-field comparison
of the original printed output against the revised record found **every**
decision-relevant value identical — decision, MA residue, L2 residue and
suppression, all ten per-dump rows, gate outcomes, mismatches, and
`selected: null`. The only additions are the bracketing and eligibility fields.
Selection now **refuses to overwrite** a prior record and preserves it with its
own digest.
