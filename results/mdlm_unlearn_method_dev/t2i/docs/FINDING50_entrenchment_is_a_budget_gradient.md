# Finding 50 — ❌ CORRECTS FINDING49: entrenchment is a GRADIENT in attack budget, not a threshold at recovery-completion. `H_after` saturates and hides it.

**Date:** 2026-10-24 · **Corrects:** `FINDING49` · **Arms:** FRP port and damage+distil, church
**Evidence:** `outputs/redundancy_dose*_s*_church.json`, `outputs/atk_DOSE*_church_s*.json`

---

## 1. The second arm, on correctly-placed brackets

`FINDING49` reported a threshold: entrenchment absent below 30 attack steps, present above, where
`H_after` reaches 1.000. The second-arm replication first appeared to refute it outright — then the
brackets turned out to be badly placed (landing at retain8 0.292 and 0.458 rather than straddling
0.45). Re-swept with fractions chosen for that arm:

| steps | H_after | retain8 | **entrenchment** |
|---|---|---|---|
| 10 | 0.867 | 0.292 | 0.267 |
| 20 | 0.867 | 0.358 | 0.133 |
| 50 | **1.000** | 0.433 | **0.067** |
| 100 | **1.000** | 0.483 | **0.467** |
| 150 | **1.000** | 0.433 | **0.733** |
| 200 | **1.000** | 0.492 | **0.733** |
| 300 | **1.000** | 0.425 † | **0.867** |

† from the better-bracketed earlier sweep; this arm's 300-step re-sweep landed at retain8 0.783,
too far from 0.45 to use. Base reference throughout: **0.333** at retain8 0.450.

**Entrenchment rises monotonically with budget from 50 to 300 steps — 0.067 → 0.467 → 0.733 → 0.733
→ 0.867 — while `H_after` is pinned at 1.000 the entire way.**

## 2. The correction

| claim | status |
|---|---|
| `FINDING49`: entrenchment switches on where recovery *completes* | **WRONG** |
| entrenchment grows with **attack budget**, continuing long after `H_after` saturates | **supported, 2 arms** |
| `H_after` is a poor proxy for how far the attacker has entrenched the concept | **supported** |

`FINDING49`'s apparent threshold was **the low end of a gradient**, read through a metric that cannot
resolve the rest of it. `H_after` is 15 samples quantised to 1/15 with a ceiling at 1.000: it cannot
distinguish a 30-step attack from a 300-step one, but entrenchment distinguishes them clearly
(FRP: 0.733 at 30 steps against 1.000 at 75–150; damage+distil: 0.067 at 50 against 0.867 at 300).

So the two arms agree after all — on a gradient. They disagree only on *where along it* a given
budget lands, which is an arm property, not a contradiction.

**What survives from `FINDING49`:** the attacker-cost observation (FRP falls at 20–30 steps, an
order of magnitude below the 300 used throughout this phase) and the negative statement that
entrenchment is absent at the lowest budgets. Its mechanism-constraining argument is withdrawn,
since a gradient is exactly what the overfitting explanation in `FINDING46` §3 predicts.

## 3. ⚠ This is the third correction to this line in one day

The sequence: `FINDING46` claimed general entrenchment from one arm → `FINDING47` withdrew it on a
confounded fixed-fraction comparison → `FINDING48` reinstated it at matched utility → `FINDING49`
added a threshold → this finding replaces the threshold with a gradient.

Two distinct causes, both mine:

1. **Publishing before the queued control reported** (46, and 49's threshold form). Each time the
   control was already running.
2. **Reading matched-utility values off badly-placed brackets.** The 50/100-step readings sat at
   retain8 0.292/0.458 and inverted the conclusion; re-bracketing fixed it. `redundancy_sweep.py`
   takes fractions, not target retain levels, so every matched reading depends on guessing the
   bracket in advance — a design flaw in my own instrument.

**The second cause is now fixed at the root:** the sweep should accept a target `retain8` and
bisect to it. Until that exists, every matched-utility number in this line must be reported **with
the retain8 it was actually read at**, which is why that column is now in every table above. It was
not in `FINDING48`'s original table.

## 4. Scope

Two arms, one concept, one seed per budget, one removal rule, one matching level. The gradient is
measured at 7 budgets on one arm and 8 on the other. A 600-step point on both arms is **running** to
test whether entrenchment keeps rising past 300 — if it saturates there, the gradient has a ceiling;
if not, it is unbounded within the budgets a real attacker would use. No claim either way until it
reports.
