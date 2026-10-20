# Finding 2 — the trade-off axis with headroom is composition, not disclosure

> ## ⚠ CORRECTED SAME DAY — the IGNORANT rate below is WRONG
>
> The partition defined REPLACE by membership in the dataset's 32-answer pool. That is too
> narrow. Re-scoring our exclusion arm's 64 forget generations by **what value is actually
> asserted** (`scripts/repartition_ignorant.py`, `outputs/repartition_ignorant.json`):
>
> | cell | n | rate | as reported below |
> |---|---|---|---|
> | DISCLOSE | 2 | 0.031 | 0.062 |
> | asserts an **in-pool** alternative | 28 | 0.438 | REPLACE 0.500 |
> | asserts an **off-pool** alternative | 34 | **0.531** | **wrongly counted IGNORANT** |
> | **non-committal** | **0** | **0.000** | — |
>
> The model always asserts something — `Montréal`, `Tashkent`, `Accra`, `tuba`, `oboes` — and
> all 16 `field` probes assert a value the pool test cannot see (`seismoology` ×8,
> `mycological` ×4). **True IGNORANT ≈ 0.000, not 0.438. Headroom ≈ 1.000, not 0.562.**
>
> The argument of this finding is therefore *stronger* than written and its headline number
> is wrong. Three definitional fixes are now mandatory and have been issued to the matched
> re-run before its evaluator was finished:
> 1. **REPLACE is defined by type-correctness**, not pool membership; pool membership becomes
>    a sub-split.
> 2. **Fluent non-words** (`seismoology`, `harpa`, `clarion`, `tympani` — ~8-12 of 64) belong
>    in DEGENERATE.
> 3. **The refusal rule must be reference-relative.** The current 9-phrase list contains
>    "i do not know" and "no information", so a model that correctly expresses ignorance would
>    be scored up to **1.000 DEGENERATE** and discarded for the wrong reason. Refusal-like
>    output is now reported as its own `ABSTAIN` cell.
>
> Revised partition: `DISCLOSE > DEGENERATE(incl. non-words) > ABSTAIN > REPLACE(in-pool |
> off-pool) > IGNORANT`, plus a Modal Answer Rate per relation.
>
> **A second instrument is also now required.** Retain accuracy is a *rank* statistic: it
> reads exactly 1.000 until the answer margin flips, so it can report "no collateral damage"
> whether or not there is any. The margin
> `log p_θ(v_true|s) − log max_{v≠v_true} p_θ(v|s)` must be reported alongside it. On the
> historical shared-token slice the accuracy-only reading showed no damage while the
> underlying rate was 0.832 versus 0.869 independent.
>
> Everything below is retained as the original record.


**Development evidence, 2026-10-20.** Zero new compute: a re-partition of generations
already on disk from our own arms and from four reproduced baselines.

## Why disclosure is the wrong primary

Our exclusion-target control reaches **0.062** forget disclosure. The never-memorized floor
on the comparable probe set is **0.023-0.046**. So the control sits *inside* the band of a
model that never knew the facts: further disclosure reduction is not forgetting, it is
over-suppression. All remaining headroom is in **what the model says instead**.

## The partition

Mutually exclusive, priority `disclose > degenerate > replace > ignorant`:

- **DISCLOSE** — the reference answer appears as a whole-token span (diacritic-insensitive).
- **DEGENERATE** — empty, repetitive, refusal, or malformed.
- **REPLACE** — non-disclosing and non-degenerate, but asserts a *different valid answer from
  the same relation's pool*. The fact was overwritten, not removed, and the model now states
  a falsehood fluently.
- **IGNORANT** — none of the above. The only genuinely desirable cell.

| arm | n | DISCLOSE | DEGEN | REPLACE | **IGNORANT** | top replacement |
|---|---|---|---|---|---|---|
| memorized reference | 8 | 1.000 | 0.000 | 0.000 | 0.000 | — |
| **EMDU, 150 steps** | 8 | **0.000** | **1.000** | 0.000 | **0.000** | — |
| EMDU, 50 steps | 8 | 0.000 | 0.875 | 0.000 | 0.125 | — |
| **MDU τ=1, 150 steps** | 8 | **0.000** | **1.000** | 0.000 | **0.000** | — |
| MDU τ=1, 50 steps | 8 | 0.750 | 0.000 | 0.125 | 0.125 | Oslo ×1 |
| NPO, 150 steps | 8 | 0.500 | 0.250 | 0.000 | 0.250 | — |
| NPO, 50 steps | 8 | 1.000 | 0.000 | 0.000 | 0.000 | — |
| SC-TF, 150 steps | 8 | 0.125 | 0.000 | **0.500** | 0.375 | Tbilisi ×1 |
| SC-TF, 50 steps | 8 | 0.250 | 0.000 | 0.500 | 0.250 | Tbilisi ×1 |
| ours, exclusion target | 64 | 0.062 | 0.000 | **0.500** | **0.438** | Tbilisi ×19 |
| ours, ARC K=512 | 64 | 0.188 | 0.000 | 0.594 | 0.219 | Tbilisi ×19 |

## What this shows

**No method exceeds 0.438 IGNORANT, against a ceiling of 1.000.**

Every method fails, and they fail in *different directions*:

- **The two methods that achieve perfect 0.000 disclosure do it with 100% degeneracy and
  0.000 ignorant.** They score perfectly on the metric the MDLM unlearning literature
  reports and are useless as models. This is the quantitative form of a qualitative point
  our closed ARC study already made about MDU/EMDU emitting nothing.
- **The two methods that avoid degeneracy pay 50-59% replacement.** They state a confident
  falsehood: *"Alira Birchwood lives in Tbilisi. Tbilisi is the capital city of Georgia,
  and…"* for a fact whose answer is Lisbon. **19 of 64** forget probes converge on the same
  substitute, which is the exclusion target's analytically predicted fixed point
  `argmax_{v∉F} p₀` — a deterministic functional of the reference, so many facts collapse
  onto one token.
- NPO at 150 steps is the only arm that is bad at both (0.500 disclose, 0.250 degenerate).

The axis therefore has **0.562 of absolute headroom** and is, as far as every primary source
we have read goes, unreported. It was derived analytically by our formalization (the
exclusion target's fixed point is degeneracy class (d), replacement) *before* it was
measured here.

## Caveats, stated because they are load-bearing

- **Different datasets and different n.** The four baselines were trained and evaluated on
  the 24-fact PoC set (8 forget probes per arm); our two arms are the 128-fact set (64
  probes). The comparison is **indicative, not matched**. An 8-probe rate has a quantisation
  unit of 0.125 and a Wilson 95% interval roughly ±0.33 at mid-range. **No claim is made
  from this table beyond "every method fails, in different directions."**
- **The required fix is scheduled, not assumed:** re-run all four baselines plus our control
  on the 128-fact set from the new qualified 3-seed backbone, giving matched data, matched n,
  and matched selection. Until then this is a framing result, not a comparison.
- **REPLACE is defined relative to a closed answer pool**, which exists only because the
  synthetic set was built with 32 shared answers. On real pretrained knowledge (RWKU-RQ) the
  pool is open and replacement must be detected differently — most likely by checking whether
  the emitted answer is a *type-correct* alternative, which needs a judge and is therefore a
  harder, separately-validated measurement. Flagged to the evaluation lead.
- Priority ordering matters: a generation that both discloses and repeats is counted as
  DISCLOSE. Reordering would move rates between DEGEN and DISCLOSE but cannot create
  IGNORANT, so the headline is ordering-robust.
- Single seed per baseline arm; our arms are single-seed too. No intervals reported, by
  design — see the n caveat.

## Consequence for the method

This is the trade-off the project should target, and it reframes the objective: not *"reduce
disclosure"* (saturated) but *"reduce disclosure without substituting and without
degenerating."* It also supplies the primary separating prediction for the candidate design —
the modal-answer histogram over forget probes should match an ignorance reference's, rather
than collapsing onto a single substitute as the control's does (19/64 on one token).

It does **not** address relearning robustness, which remains a separate axis and is not
claimed here.
