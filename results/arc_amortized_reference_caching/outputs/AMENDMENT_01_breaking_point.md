# AMENDMENT 01 — ARC's validity boundary, measured; and my per-fact prediction refuted

**Run at 2:05 into the sprint**, executing the single "next step" named in §10 of
`FINAL_REPORT.md`. It changes the headline claim's scope materially, so it is an
amendment rather than an addendum.

## What was run

A harder probe: **64 forget + 64 retain** synthetic facts about fictional entities (vs 24
facts before), with **4 entities sharing every answer** (2 forget + 2 retain), 32 distinct
answers, 256 questions per checkpoint, identical frozen scoring and greedy decoder. Three
arms, 300 steps, seed 0 — the pool is reused ~hundreds of times, which the 24-fact probe
never forced:

| arm | pool size | pool coverage of the 64 forget facts |
|---|---|---|
| uncached (fresh masked states every step) | — | n/a |
| **ARC K=16** (2 pooled batches × 8 states) | 16 | **15/64 — 49 facts never in the pool** |
| ARC K=512 (64 pooled batches) | 512 | **64/64 — full coverage** |

Both ARC arms passed the Proposition 6 exactness gate (`max|Δ| = 0.000e+00`).

Memorization of the 128-fact set **narrowly missed its own 0.80 qualification gate at
0.797** (51/64 forget, 51/64 retain recalled), and no retry fit in the remaining clock.
All headline numbers below are therefore reported on the **51 forget facts θ₀ actually
recalled** — the only subset on which "did this arm remove a memorized fact?" is a
meaningful question. The all-64 column is shown alongside for completeness.

## Result: ARC fails outright when its pool is too small

| arm | forget ALL ↓ | **forget RECALLED ↓** | recalled IN-pool | recalled ABSENT | retain ↑ | degenerate |
|---|---|---|---|---|---|---|
| memorized reference θ₀ | 0.781 | **0.980** | 0.950 (n=20) | 0.988 (n=82) | 0.773 | 6 |
| **uncached control** | 0.047 | **0.049** | 0.000 (n=20) | 0.061 (n=82) | **1.000** | 0 |
| **ARC K=16** | 0.945 | **0.931** | 0.850 (n=20) | 0.951 (n=82) | 0.953 | 0 |
| ARC K=512 | 0.172 | **0.216** | 0.350 (n=20) | 0.183 (n=82) | **1.000** | 0 |

**ARC K=16 does not forget.** Disclosure stays at 0.931 against θ₀'s 0.980 — it removed
roughly 5% of what the uncached control removed (0.049), at no gain in retention. This is
the **pool-overfitting failure mode I flagged as a risk in the main report and could not
produce at 24 facts**. It is now produced, unambiguously.

## My pre-registered per-fact prediction is refuted as stated

`scripts/build_big.py` pre-registered a sharp claim: ARC K=16 should fail *specifically on
facts absent from its pool*, "predictably per fact." Measured on the recalled subset:

```
ARC K=16   recalled, IN pool  (n=20 probes):  disclosure 0.850   (theta0: 0.950)
ARC K=16   recalled, ABSENT   (n=82 probes):  disclosure 0.951   (theta0: 0.988)
                                      gap:   +0.101
```

The gap is in the predicted direction, but it is nowhere near large enough to carry the
failure: in-pool facts moved only 0.950 → 0.850 while the uncached control drove the same
facts to **0.000**. Pool membership bought a fact almost nothing. **The prediction is
refuted as stated.** The correct mechanism is not "unpooled facts keep their answers" but
*a pool too small to span the state space destroys the gradient signal globally* — 300
steps against 2 fixed batches drives the forget loss to near zero on those batches while
generalising to nothing, in-pool facts included.

## A second limitation the small probe concealed

Even at **full coverage**, ARC K=512 reaches 0.216 disclosure against the uncached
control's 0.049 — **4.4× more residual disclosure** — while matching it on retention
(1.000). The 24-fact probe showed exact parity (0.125 vs 0.125). **That parity does not
generalise**: a fixed pool costs real efficacy at 128 facts even when every fact is
represented in it.

## Corrected headline

The main report's claim — *"1.26× faster per step and 18.8–75× fewer reference forwards at
bit-identical targets and matched efficacy"* — now carries a scope condition:

> The **cost** results are unaffected: exactness (`max|Δ| = 0` on every run, these
> included), the forward-count reductions, and the 1.257× per-step speedup are properties
> of the mechanism, not of the data. **The efficacy-parity claim holds at 24 facts and
> does not hold at 128 facts.** With full pool coverage ARC loses accuracy relative to
> fresh states (0.216 vs 0.049); with under-coverage it fails almost completely (0.931).
> ARC is an efficiency-at-parity method only in a pool-coverage regime this sprint did not
> locate — it is bounded below by "full coverage at 24 facts works" and above by "full
> coverage at 128 facts does not."

What stands unchanged: **Proposition 5** (the reference pass is not removable by algebra),
**Proposition 6** (caching is exact), the measured forward counts and speedup, and the
**clean-forgetting** metric together with its finding that MDU τ=1 and EMDU's apparent
perfect 0.000 disclosure is output collapse.

## Caveats

- The reference θ₀ **failed its own qualification gate** (0.797 < 0.80). Absolute numbers
  are indicative; the relative comparison is still informative because all three arms
  start from the *identical* reference and the uncached control forgets cleanly (0.049)
  from it.
- One seed, 300 steps, one model, synthetic facts. Not a benchmark result.
- Retention above θ₀ (1.000 vs 0.773) reflects continued retain-loss training on facts
  the reference had not fully learned, not a repair of forgetting.
- The four evaluations were launched **directly rather than through `attempt.sh`** after
  repeated backgrounding failures, so their cost is **absent from the attempt ledger**:
  4 × ~7.8 min on 1 GPU each ≈ **0.52 GPU-device-h unrecorded**. The ledger's
  1.35 GPU-device-h therefore undercounts total usage by roughly that much. Disclosed
  rather than retro-fitted.

## Revised next step

Sweep `K ∈ {16, 64, 128, 256, 512, 1024}` at a fixed 128 facts and plot clean forgetting
against coverage (`distinct pooled states / (facts × masks needed per fact)`). One curve
converts ARC from "works at small scale" into a method with a stated applicability
condition, which is what it needs before the efficiency claim is usable by anyone.
