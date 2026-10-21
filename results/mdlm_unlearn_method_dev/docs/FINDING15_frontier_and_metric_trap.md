# Finding 15 — localization does not move the frontier, and a two-axis evaluation ranks a garbage model first

**Date:** 2026-10-21
**Status:** H15 closed as dominated. Two results here: the measured frontier, and a metric
artifact I reproduced on myself.
**Outputs:** `outputs/arm_ll_*.json`, `outputs/arm_cv_mdu_l2.json`, `outputs/arm_mdu_pub.json`,
`outputs/arm_emdu_pub.json`, `outputs/arm_h15_lammdu_l8.json`

---

## 1. H15 is dominated, not merely short of its threshold

Finding 14 recorded that H15 failed R1 (retain ASCR 0.199 < 0.40) while passing R2 and R3, and
noted a 2.2× retain gain at matched forget. The low-λ sweep now shows that gain does not
generalise. Matched `λ_forget`, generation-level, 120 forget / 80 retain:

| λ_forget | MDU DISC / retain | H15 DISC / retain |
|---|---|---|
| 1 | 0.317 / **0.524** | 0.092 / 0.085 |
| 2 | 0.225 / **0.403** | 0.150 / 0.094 |
| 3 | 0.233 / **0.404** | 0.133 / 0.013 |
| 8 | 0.067 / 0.091 | 0.067 / **0.199** |

The curves **cross**. H15 wins only at DISCLOSE 0.067, where both models are unusable
(retain 0.199 and 0.091 against θ₀'s 0.762). In the usable region — retain above 0.4 — MDU is
strictly better and H15 never arrives: at DISCLOSE ≈ 0.15 H15 gives retain 0.094 while MDU
interpolates to ≈ 0.25.

Localization makes the forget loss **more aggressive per unit λ** on the same or a worse
frontier. It is a reparameterisation of strength, not an improvement in trade-off. The selector
itself remains validated (Finding 14 §3a) — this is a negative about what the selector buys,
not about whether it works.

H15's retain ASCR is also non-monotonic in λ_forget (0.085, 0.094, 0.013, 0.199 at λ = 1, 2, 3,
8), single seed. The dominance conclusion is robust across all four points; no individual point
should be read.

## 2. The frontier, under this project's own standard

Degeneracy is part of the objective here — the pristine reference has DEGENERATE 0.000, and the
six-cell partition exists because a fluent wrong answer and `'........'` are not the same
outcome. Applying R2's threshold (DEGENERATE ≤ 0.25):

| arm | DISCLOSE | DEGENERATE | retain ASCR | TV→pristine |
|---|---|---|---|---|
| H15 λ=8 | 0.067 | 0.200 | 0.199 | 0.241 |
| MDU λ=2 | 0.225 | 0.092 | 0.403 | 0.349 |
| MDU λ=3 | 0.233 | 0.050 | 0.404 | 0.282 |
| **MDU λ=1** | 0.317 | 0.075 | **0.524** | 0.407 |
| H15 anti-control λ=2 | 0.733 | 0.000 | 0.700 | 0.704 |
| *pristine target* | *0.030* | *0.000* | — | *0.000* |

**MDU at λ_forget = 1 is the point to beat: DISCLOSE 0.317, DEGENERATE 0.075, retain 0.524.**
H15 λ=8 holds one frontier point — it dominates `mdu_pub` at identical DISCLOSE 0.067 — but the
usable region belongs to the baseline.

Excluded for degeneracy: `emdu_pub` (DEG 1.000), `ll_h15_f1` (0.425), `ll_h15_f3` (0.325).

## 3. The metric trap, reproduced on myself

My first frontier computation ranked arms on (DISCLOSE ↓, retain ↑) only. It returned:

```
Pareto-optimal:  emdu_pub   DISCLOSE 0.000   retain 0.580
```

**EMDU came out optimal while emitting pure garbage** — DEGENERATE 1.000, TV→pristine 1.000,
forget ASCR 0.000. On a two-axis forget-versus-retain view it is the best arm in the study,
because suppressing the answer completely *is* perfect forgetting and its retain answers survive
at 0.580.

This is exactly the artifact Finding 13 §5 raised about MDU's and EMDU's own reported TOFU
metrics — ROUGE-L and token probability cannot distinguish a fluent wrong answer from an empty
one — and I reproduced it in my own analysis one step after writing that warning down.

Two consequences, stated rather than implied:

1. **It validates the six-cell partition as a design choice**, not as decoration. Any evaluation
   of MDLM unlearning that reports only forget quality and retain utility will rank a
   degenerate model first. That is a reviewable claim about the literature's standard metrics
   and it is supported by a measured example.
2. **It is a caution on our own analysis**, recorded because the same mistake would have
   produced a false headline. Every frontier in this project is computed under the DEGENERATE
   constraint from here on, and the constraint is stated with the number.

## 4. What this licenses, and what it does not

Licensed: that position-level reweighting of an anchor-matching loss does not improve the
forgetting–utility frontier on this backbone and dataset; that the frontier is currently held by
the published baseline at a weak setting; and that a two-axis evaluation is unsafe for this
problem.

Not licensed: any claim about other backbones, datasets, or seeds; any claim that MDU is a good
method (its best point still sits at DISCLOSE 0.317 against a pristine 0.030, with retain 0.524
against θ₀'s 0.762 — the whole field is far from the target); any robustness claim.

## 5. Where the work goes

The diagnosis from Finding 14 stands and is now better supported: the damage transfers across
**inputs** through shared parameters, and nothing applied to the forget set's *support* reaches
it. H16 (`docs/PREREG_H16_conflict_projection.md`) tests the parameter-space route, and its
first measurement is already informative — at the start of training the forget and retain
gradients are **negatively aligned, cos = −0.357**, and a rank-8 projection removes 92% of that
(`−0.357 → −0.028`). So the conflict is real, large, and mechanically removable. Whether
removing it buys frontier movement is what the arms decide, against Q5's strict-dominance
requirement over MDU λ=1.
