# Training seeds 17 and 29, reported separately

Two training runs cannot establish population variance. Each seed's intervals below are **sampling** uncertainty over prompts within that single run. The only cross-seed claim made here is whether the **direction** of an effect agrees.

## Initial cat suppression

| seed 17 | seed 29 |
|---|---|
| +82.5 [+67.5,+95.0] | +72.5 [+50.0,+90.0] |

## Historical cat recovery  D_cat(child) − D_cat(MA)

| child | seed 17 | seed 29 | direction agrees? |
|---|---|---|---|
| MAB | -17.5 [-32.5,-5.0] | -27.5 [-50.0,-10.0] | yes (decrease) |
| MAB_L2 | -12.5 [-22.5,-2.5] | -22.5 [-42.5,-7.5] | yes (decrease) |
| MAC | +7.5 [+0.0,+15.0] | -7.5 [-25.0,+5.0] | both ns |
| MAC_L2 | +15.0 [+2.5,+27.5] | -5.0 [-12.5,+0.0] | **NO** (increase vs ns) |

## New-request suppression  D_new(MA) − D_new(child)

| child | target | seed 17 | seed 29 | direction agrees? |
|---|---|---|---|---|
| MAB | dog | +82.5 [+65.0,+97.5] | +82.5 [+67.5,+95.0] | yes (increase) |
| MAB_L2 | dog | +42.5 [+25.0,+62.5] | +42.5 [+20.0,+67.5] | yes (increase) |
| MAC | sandwich | +50.0 [+35.0,+62.5] | +52.5 [+30.0,+72.5] | yes (increase) |
| MAC_L2 | sandwich | +7.5 [+0.0,+17.5] | +7.5 [+0.0,+15.0] | both ns |

## Raw cat detection rate D (%) at t=0.5

| checkpoint | seed 17 | seed 29 |
|---|---|---|
| M0 | 100.0 | 100.0 |
| MA | 17.5 | 27.5 |
| MAB | 0.0 | 0.0 |
| MAB_L2 | 5.0 | 5.0 |
| MAC | 25.0 | 20.0 |
| MAC_L2 | 32.5 | 22.5 |

**Credible, same-direction agreement on historical recovery: 2/4 children.** 'both ns' is not agreement on an effect -- it is two runs each finding nothing. With n=2 training runs this is a consistency check, not an estimate of variability.


> **Replication warning.** MAC_L2 changed credibility class between the two training seeds. Any claim resting on that child is **not** supported by this pair of runs and must not be presented as an established effect.

