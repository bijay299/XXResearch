# Amendment 01 — a fixed early dump grid at steps 10…90

**Status: PREPARED, NOT EXECUTED. No GPU work has been done for this amendment
and none is authorised by it.** It is submitted for the PI's separate amendment
decision. The frozen test set remains untouched.

Parent protocol: [`MATCHED_EFFECTIVENESS_PROTOCOL.md`](MATCHED_EFFECTIVENESS_PROTOCOL.md).
Original run: commit `3a5a69c`, output `/data/bijaypandey/cuig_pilot/seq_pilot/diag_v1`.

---

## 1. Why, and the disclosure that comes with it

The approved run completed training and the full development scan, and **stopped
at selection by the pre-declared infeasibility rule**. All ten dumps passed both
reference gates on both seeds, and the L2 reference was itself eligible on both
seeds, but no dump landed within 5 pp of its seed's L2 suppression:

| seed | L2 suppression | step-100 dump | closest mismatch | lower bracket |
|---|---|---|---|---|
| 17 | 30.0 pp (eligible, exactly on the gate) | 38.8 pp | 8.8 pp | MA, step 0, 0 pp |
| 29 | 33.8 pp (eligible) | 42.5 pp | 8.8 pp | MA, step 0, 0 pp |

The match point for each seed lies **between step 0 and step 100** — inside the
frozen 100-step grid, which cannot resolve it.

> **Disclosure, stated plainly because it matters for how any result from this
> amendment must be read.** This early grid is **motivated by the development
> results**. It was not pre-registered independently of the data: we chose
> 10…90 *because* the completed scan showed the match point falls inside the
> first 100 steps. The grid is therefore **post-hoc with respect to the
> development set**.
>
> What that does and does not cost:
> * The **development set is a selection instrument**, and selecting a grid on
>   it is the use it was frozen for. No confirmatory claim rests on it.
> * The **frozen test set is unaffected**: it has never been evaluated, its
>   manifest identity `5dd87dbb5a77c0f4…` is unchanged, and nothing in this
>   amendment inspects it.
> * But the *grid* no longer has the "fixed before looking" property the
>   original 100-step grid had, and any eventual report must say so in the same
>   sentence as the result.

---

## 2. What changes, and what does not

**Changes — exactly one thing.** A second, finer dump grid at optimizer steps
**10, 20, 30, 40, 50, 60, 70, 80, 90** for both training seeds, obtained by
retraining each trajectory for **100 steps** with `--checkpoint_every 10`.

**Unchanged, explicitly:**

| | |
|---|---|
| training horizon of the arms being compared | 1000 steps for the L2 endpoint; the new dumps are *earlier points on the same trajectory shape*, not a shortened arm |
| training configuration | all 32 effective hyperparameters identical to the original run — verified byte-for-byte equal between the original U and the saved MAB reports |
| RNG | same `--seed` per trajectory (17, 29); evaluation generation seeds untouched and never reused as training seeds |
| parents | the same saved `MA` per seed, bound by digest: `3cb72ee162b9…` (s17), `fbf49199a306…` (s29) |
| L2 references | the same saved `MAB_L2` endpoints: `8ee7d56f62b0…` (s17), `c4f3ecdcb945…` (s29). **No L2 retraining.** |
| development selection | dog-only, 80 pairs per checkpoint, t=0.5, against that seed's own MA |
| eligibility gates | ≥30 pp suppression **and** ≤60% residue, applied to candidate dumps **and to the L2 reference itself** |
| match tolerance | **≤5 pp**, not widened |
| tie-break | earliest step |
| both-seed requirement | both seeds must match or the paired test does not run |
| frozen test manifest | `5dd87dbb5a77c0f4…`, 140 texts × 4 seeds = 560 pairs; evaluated once, no reselection |
| reporting | every category separately and never averaged; literal/paraphrase separately; thresholds 0.3/0.5/0.7 with 0.5 primary |
| uncertainty | per-training-seed intervals, never pooled; frozen grouping `2ba0e8df5034d011…`, scene-cluster unit, category strata, 10 000 draws, RNG 2026100901 |

**Excluded, as before:** iterative refinement (this is **one** grid, and if it
fails that is the end of this line), tolerance widening, L2 retraining,
additional training seeds, any test-based reselection, any sandwich-branch work,
any new M0 evaluation, any unmatched reference arm on the test set.

---

## 3. The rerun, and why checkpoints from two runs are still comparable

**A rerun is required.** Dumps can only be written while training runs; there is
no way to recover step-10…90 states from a finished trajectory. So each seed's
trajectory is retrained from its saved `MA` for 100 steps.

**The honest problem with that.** The planned CPU comparison of the new
1000-step `U` endpoints against the saved `MAB` endpoints — same configuration,
same parent, same seed — found that **training on this stack is not bitwise
reproducible** ([`u_vs_mab_comparison.json`](../../results/audit_v1/diag_v1_evidence/u_vs_mab_comparison.json)):

| seed | configuration identical | same saved parent | bitwise-identical tensors | max abs weight diff | overall relative Frobenius |
|---|---|---|---|---|---|
| 17 | yes, 32/32 hyperparameters | yes | **0 of 32** | 2.045e-03 | 5.652e-03 |
| 29 | yes, 32/32 hyperparameters | yes | **0 of 32** | 1.778e-03 | 5.717e-03 |

So the retrained early dumps are a **second realisation** of the same
configuration, not points on the original run's path. They are not
interchangeable with the original run's dumps and this amendment does not
pretend otherwise.

**How comparability is established rather than assumed.**

1. **All dumps used for the match come from ONE run per seed.** Selection under
   this amendment uses only the new run's early grid (10…90) plus the already
   scored `MA` and `MAB_L2`, which are **unchanged saved artifacts** and need no
   rescoring. The original run's 100…1000 dumps are *not* mixed into the match.
2. **A measured bridge between the two realisations.** The new run also writes a
   step-100 dump, which is scored on the development set and compared with the
   original run's step-100 suppression (38.75 pp s17, 42.50 pp s29). This is the
   one extra development slot per seed in the costing.
3. **A declared stop condition on that bridge.** If the two realisations'
   step-100 dog suppression differs by **more than 5 pp** — the same tolerance
   the match itself uses — then run-to-run variation is of the same order as the
   quantity being matched, the early grid cannot be trusted to locate the match
   point, and **this amendment stops and reports that instead**. Declared now,
   before the measurement.
4. **The original run is retained in full.** Its trajectory, ten dumps, 24
   development slots, selection records and ledger are preserved; nothing is
   overwritten. The new run goes to its own directory.

**What this still does not establish.** That either realisation's step-*k* state
is "the" state of that configuration at step *k*. Run-to-run variation is
measured at the endpoint and bridged at step 100; it is not characterised across
the early grid, and no claim is made that it is small there.

---

## 4. Cost, against the same 4.000 GPU-hour ceiling

Estimates are from the **measured, conservative** model
([`cost_model_measured.json`](../../results/audit_v1/diag_v1_evidence/cost_model_measured.json)):
maximum observed per-image generation time, maximum observed per-slot overhead,
plus a 25% margin. Each estimate is **both** an admission amount **and** a
runtime bound.

| line | GPU-hours |
|---|---|
| **cumulative past spend** (27 completed stages, all charged) | **1.2022** |
| retrain 100 steps with dumps every 10, both seeds (2 × 0.0442) | 0.0883 |
| development scoring, 20 new slots (9 early + 1 bridge, × 2 seeds, × 0.0397) | 0.7938 |
| **frozen-test reserve** (6 slots × 0.2246, withheld from the start) | 1.3474 |
| explicit contingency | 0.3000 |
| **total** | **3.7317** |
| **headroom under the ceiling** | **0.2683 (6.7%)** |

Runtime bounds: 159 s per 100-step trajectory, 143 s per development slot, 808 s
per frozen-test slot.

**The reserve is reconciled.** It is now *defined* as the sum of the frozen-test
stage's own admitted estimates (6 × 0.2246), so the two cannot disagree. The
earlier hand-written **1.008** could not cover the stage it existed to protect:
the runner admitted six slots at 0.18 = **1.08 GPU-h** against it. That shortfall
would have surfaced only after selection had already been paid for.

**Prior spend is carried forward, not reset.** The ledger now lives at
`${SEQ_ROOT}/matched_effectiveness_gpu_budget.json` — the experiment's, not an
output directory's — with all 1.2022 GPU-h of the original run imported and
attributed. A new output directory cannot restore a zero balance.

---

## 5. What a result from this amendment could and could not support

* **If both seeds match** within 5 pp on the early grid, and the step-100 bridge
  holds, the frozen test set is evaluated **once** and the primary paired bird
  contrast is reported per seed with its interval, every category separately,
  and the matching check re-reported on TEST without reselection.
* **If either seed does not match**, the paired test does not run. That is a
  result and the end of this line of work; no further grid refinement follows.
* **In either case** the report must carry: that the grid was chosen after
  seeing development results; that the arms' completion evidence is asymmetric
  (the L2 endpoints' training completion is **UNVERIFIED** legacy evidence,
  the new unregularised arms' is counter-verified); that run-to-run
  nondeterminism was measured and bridged, not eliminated; and that detector
  results are **provisional** until the blinded human annotation is complete.
* **No mechanism claim** follows from any outcome, and none is sought.

---

## 6. Decision requested

Approve, modify, or decline **this single amendment**. Nothing runs until a
separate decision is recorded. If approved, execution is:

```bash
# all CPU-gated, admitted against the carried-forward ledger
bash scripts/seq/run_diagnostic.sh preflight     # verifies identities + capacity
# then the amendment stage, once it exists and has been CPU-tested
```

The runner stage that performs the early grid is **not yet written**: writing it
is cheap and CPU-only, but it is deliberately deferred until the amendment is
approved, so no implementation choice pre-empts the decision.
