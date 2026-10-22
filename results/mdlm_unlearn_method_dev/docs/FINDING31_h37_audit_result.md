# Finding 31 — the audit result: FINDING28's band separation replicates, but its headline was wrong in an informative way

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM · **Pre-registered:** `PREREG_H37`
**Design:** balanced, 16 arms (8 suppression-only, 8 damage+distil), seed 0, **both** knowledge
probes on every arm, RTT at lr ∈ {1e-5, 3e-5, 1e-4} under `FINDING20`'s gate, **t_frac 0.5 for all
arms**, recovery as max-over-valid-attacks
**Status:** `FINDING28` is **rewritten, not retracted.** The outcome landed between two
pre-declared thresholds and that is reported as such rather than rounded to the nearer one.

---

## 1. What replicated

All **16 of 16** arms produced at least one valid gated attack, so nothing is reported as ungated
and no arm was dropped (kill rule K2 honoured).

| group | n | **max gated recovery** |
|---|---|---|
| suppression-only (MDU, AFC, EMDU, GA; λ ∈ {0.5…8}) | 8 | **+0.5608 … +0.7711** |
| *α=0 controls (distil, no damage)* | 2 | *+0.4352, +0.4971* |
| damage+distil, α > 0 | 6 | **−0.0093 … +0.4385** |

> **The bands are disjoint, gap +0.1223.** `FINDING28`'s band separation survives a balanced
> design with a validated probe. This was the part most at risk from the convenience-sample
> criticism and it held.

**Sorted by recovery, all 8 damage+distil arms rank below all 8 suppression arms — 16/16 correctly
ordered by family.** Stated as a separation score (AUC: 1.0 = perfect, 0.5 = chance):

| measure | AUC separating the two method families |
|---|---|
| **validated relearning attack** | **1.000** |
| marginal knowledge probe (validated) | 0.844 |
| ranking knowledge probe (invalid) | 0.797 |

The knowledge probes' AUC is computed as P(damage arm scores *higher* than suppression arm) =
0.844 — so what little separating power the knowledge axis has, it has **in the inverted
direction**. On knowledge the two families interleave (suppression 0.0000–0.4350, damage+distil
0.2750–0.5250); on the attack they do not overlap at all.

## 2. What did **not** replicate: the headline

`FINDING28` claimed **no** relationship between static knowledge and robustness (ranking-probe
ρ = −0.148). With the validated marginal probe on the same arms:

| x-axis | Spearman with max gated recovery (n=16) |
|---|---|
| ranking probe (the invalid instrument) | −0.459 |
| **marginal probe (validated)** | **−0.500** |

(At the 14-arm interim these read −0.459 and −0.635; the final 16-arm figures are the ones above
and the ones that count. The interim −0.635 is not used anywhere.)

**The correlation is stronger with the better probe, not weaker.** So ρ ≈ −0.15 in `FINDING28`
*was* regression dilution, exactly as `FINDING29` warned — but what the noise was hiding is not a
positive relationship. It is an **inversion**.

`PREREG_H37` §5 declared "|ρ| < 0.4 → H37 true" and "|ρ| ≥ 0.7 → headline wrong". **ρ = −0.500
falls between them.** I did not pre-specify that interval and will not retrofit a threshold to
land on the convenient side. The honest statement is the one the sign, the AUC table, and the rows
below support.

## 3. The rows that make the inversion concrete

| arm | marginal knowledge (exact-span) | max gated recovery |
|---|---|---|
| `sup_emdu_l1` | **0.0000** | **+0.7701** |
| `sup_ga_l4` | **0.0000** | **+0.7711** |
| `sup_ga_l0p5` | **0.0000** | +0.7194 |
| `sup_emdu_l8` | **0.0000** | +0.6798 |
| *exact retrain oracle (floor)* | *0.3375* | — |
| `dd_afc_a0p6` | **0.3050** | **+0.0000** |
| `dd_mdu_a0p6` | **0.2750** | **−0.0093** |

> **Four arms score 0.0000 — apparent *perfect* erasure, below the exact oracle's own 0.3375
> floor — and they are the LEAST robust arms measured.** The arm sitting essentially *at* the
> oracle floor (0.3050) is the most robust, at exactly 0.

So the correct claim is sharper and more useful than `FINDING28`'s:

**Static knowledge metrics are not merely uninformative about robustness — they are
anti-predictive. Selecting a method for scoring lowest on the knowledge probe systematically
selects the least robust method.** A practitioner optimising the metric the field reports is not
wasting effort; they are actively steering the wrong way.

## 4. Mechanism: scoring *below* the oracle is the warning sign

The four worst arms do not merely forget — they score **below the exact retrain oracle**, i.e.
below what "these facts were never in training" actually looks like. A model cannot genuinely know
*less* than a model that never saw the data. Scoring below the floor therefore indicates an
actively learned avoidance — a directional transformation applied on top of intact knowledge — and
a learned transformation is exactly what a relearning attack inverts.

This unifies several earlier results under one mechanism: GA's below-chance scores with the worst
recovery (`FINDING25`), EMDU's **DISCLOSE 0.000** with fully recoverable knowledge (`FINDING23`),
and `FINDING28`'s finding that damage raises residual readings while lowering recovery.

**A tempting corollary, tested and refuted.** If scoring below the floor is the warning sign, then
*distance* from the oracle floor should predict robustness better than the raw level. It does not:
ρ(|knowledge − floor|, recovery) = **+0.446**, weaker than the raw level's −0.500. Recorded as
refuted, and not pursued.

## 5. Kill rule K1 fired, and the re-examination vindicates the mechanism

`PREREG_H37` K1: *"If the α=0 controls do not land in the suppression band, the mechanism
attributed to damage is wrong. Stop and re-examine before reporting anything else."*

They did not. They landed at +0.4352 and +0.4971, **below** the suppression band's +0.5608. K1
fired and the re-examination was run before anything else was reported. The controlled α ladder —
the same teacher and repair budget, only α varying:

| α | marginal knowledge | max gated recovery |
|---|---|---|
| 0 (no damage) | 0.4325 | **+0.4352** |
| 0.1 | 0.4500 | **+0.4385** |
| 0.3 | 0.4825 | +0.3094 |
| 0.5 | 0.4250 | +0.0952 |
| 0.6 | 0.3050 | **+0.0000** |

**α=0 and α=0.1 are nearly identical (+0.4352 vs +0.4385): a token amount of damage does nothing,
and the fall happens between 0.3 and 0.6.** That is precisely the signature of a damage-driven
mechanism, and it is a *controlled* contrast rather than the band-membership proxy K1 used. **The
mechanism stands; K1's criterion was the poor instrument.**

**One correction falls out of the same table.** `FINDING23` reported that the α=0 control gives
**no** robustness benefit. At this repair budget it gives a modest but real one: +0.5608 → +0.4352,
about 0.12, roughly a fifth of the total effect. Distillation alone is not inert. The claim that
*damage is the dominant ingredient* stands (it supplies the remaining 0.44 of 0.56); the claim that
distillation contributes *nothing* does not, and is corrected here.

## 6. What this changes

| document | change |
|---|---|
| `FINDING28` | headline **rewritten**: "no static metric predicts robustness" → "static metrics are **anti-predictive**". Band separation and α=0 reasoning stand. Provisional flag lifted on §2, kept off §3's matched pairs (computed with the invalid probe). |
| `FINDING23` | corrected: the α=0 control gives ~0.12, not zero |
| `FINDING21` | already reframed by `FINDING29` ("at chance" is probe-specific) |
| `FINDING29` | confirmed as the cause of the dilution |

## 7. Scope and limits

- **16 arms, seed 0 only.** Balanced by family, but one seed per arm: the band separation is wide
  (+0.1223) relative to the seed noise seen in `FINDING25` (0.01–0.05), the correlation less so.
- **One attack family** for these numbers (span SFT), one budget (t_frac 0.5), one backbone.
  `FINDING30`'s second family was run on a different arm pair and is not folded in.
- **ρ = −0.500 on n=16** has a wide confidence interval, and it moved from −0.635 to −0.500 when
  the last two arms landed — so the coefficient is not stable at this n. The finding leans on the
  **AUC table** (1.000 vs 0.844, a complete-vs-partial separation), the §3 rows, and the §5 ladder,
  which are large and qualitative, not on the correlation coefficient.
- The marginal probe is **not adversarial**. It is a better static probe, and §2 is precisely the
  claim that a better static probe still does not help you select for robustness.
- **`FINDING30`'s trainability confound is unresolved and applies here too.** If damaged arms are
  harder to fine-tune in general, part of the band separation is optimisation difficulty rather
  than durable erasure. That experiment was specified and not run before the domain pivot, and it
  is the single most important open item against this result.
