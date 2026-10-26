> ### ⚠ §3's MECHANISM CHECK IS OVERSTATED — see §7 (added 2026-10-26)
>
> §3 claimed the probe delta "grows with K, roughly in proportion" and that `PREREG_H90`'s second
> falsifier therefore did not fire. The underlying samples are **n = 7 / 3 / 1** for K = 1 / 8 / 32,
> and K=1's own deltas span **−0.00011 to −0.00457, a 40× spread**, which **overlaps** K=32's single
> value of −0.00638. The ordering is **suggestive, not established**, and the depth conclusion in
> §1 inherits that weakness. §7 states what is and is not supported.

# FINDING 76 — With the inner attacker reconciled to the evaluation attacker **and** compute genuinely matched, **unroll depth buys nothing**: K ∈ {0, 1, 8, 32}, 12/12 gate-valid, **every seed at H = 1.0000**. The mechanism check passed, so the null is interpretable.

**Date:** 2026-10-26 · **Experiment:** E4m builds + E4mg grid · **Preregistered:** `PREREG_H90`
(§1–§3), with the §8 compute-accounting amendment applied **before** these arms were built
**Evidence:** `outputs/atk_E4MG_*.json`, `outputs/utrap2_utrap2m_*.json`, `logs/e4m_matched.log`,
`logs/e4mg_grid.log`

---

## 1. The result

`erase_utrap_v2.py`, λ=0.15, lr 1e-5, 300-step attacks, **n = 3 seeds per arm**:

| arm | K | outer | **samples** | as-is target | retain8 | gate-valid | **max-over-valid H** | per-seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `utrap2_church_l0.15k0` | **0** | 400 | 1600 | **0.0000** | 0.8667 | 3/3 | **1.0000** | 1.0, 1.0, 1.0 |
| `utrap2m_church_l0.15k1` | 1 | 320 | 1600 | 0.1333 | 0.9667 | 3/3 | **1.0000** | 1.0, 1.0, 1.0 |
| `utrap2m_church_l0.15k8` | 8 | 133 | 1596 | 0.3333 | 0.9667 | 3/3 | **1.0000** | 1.0, 1.0, 1.0 |
| `utrap2m_church_l0.15k32` | 32 | 44 | 1584 | 0.2667 | 0.9500 | 3/3 | **1.0000** | 1.0, 1.0, 1.0 |

**All twelve attacks gate-valid. All twelve at exactly 1.0000.** `PREREG_H90`'s prediction —
*"max-over-valid H is non-increasing in K, and at least one K reaches max-over-valid H < 1.000"* —
is **falsified by its own declared falsifier**.

## 2. Why this verdict is worth more than `FINDING63`'s

`FINDING63` §3a(a) announced the same conclusion and it was **withdrawn**, because the sweep varied
the horizon of a surrogate that differed from the evaluation attacker on six other axes
(`ERRATUM_utrap_inner_attacker.md`). This run fixes both defects that invalidated it:

| defect | `FINDING63` | here |
|---|---|---|
| inner attacker | SGD, rebuilt per step, batch 2, no clipping, teacher-generated images, one **evaluation** prompt | **Adam, persistent, batch 1, clip 1.0, base-model images, DEF_TRAIN prompts** — the evaluator's own configuration, via `attack_core.py` |
| compute matching | claimed matched at K+1; actually **1600 → 468 samples**, decreasing with K | **sample-matched: 1600 / 1600 / 1596 / 1584** |
| seeds | single build, attacks at n=1–3 | **n = 3 at every cell** |
| K=0 control | absent | **present** |

## 3. The mechanism check passed — which is what makes the null interpretable

`PREREG_H90` §1 declared a **second falsifier**: *"if the inner-attacker probe delta does not grow
with K … no conclusion about depth may be drawn from the run at all."* The fixed-noise target probe,
measured immediately before and after each unroll:

| K | mean probe delta |
|---:|---:|
| 1 | **−0.00124** |
| 8 | **−0.00269** |
| 32 | **−0.00638** |

**The delta grows with K**, roughly in proportion. So the unroll *is* doing what the design assumes —
deeper unrolls adapt further toward the target — and the second falsifier did **not** fire. The null
is therefore about depth, not about a broken inner loop. This is the check `FINDING63` never had.

## 4. The K=0 control is the sharpest part

With **K = 0** the objective degenerates to `L_retain(θ) − λ·L_target(θ)` — erasure with **no
adversarial anticipation whatever**. It achieves the **best as-is suppression of the four**
(target **0.0000**, vs 0.3333 at K=8) and is reversed identically.

So at matched compute the unrolling does not merely fail to buy robustness — **it costs
suppression**. Spending the same budget on plain erasure produced a better pre-attack operating
point than spending it on anticipating an adversary.

## 5. Limits

- **One concept (church), one teacher, one λ, one defence build per K.** Attack-seed replication
  (n=3) is **not** defence-build replication; a second build per K is unrun.
- **A saturated ceiling cannot rank depths.** All four land at 1.0000, so this says *depth did not
  prevent reaching the ceiling at this budget* — not that depth has no value at any budget
  (`ERRATUM_interpretations_r5.md` §1). Recovery-versus-budget curves per K are unmeasured.
- **The trajectory is still detached.** A fully differentiated unroll remains untested and is not
  claimed either way.
- **Probe overhead is not matched** across arms (8 probes at K=0 vs 1 at K=32); it is forward-only
  and excluded from the sample counts, and that residual is declared, not argued away.
- **The λ=1.2 arm** (`PREREG_H90` §7 amendment, from the measured parity of 1.23) is built but not
  yet in this grid.
- **DEV prompts** throughout; H quantised to 1/15.

## 6. Standing of UTRAP

UTRAP is **our proposed method**. Authorship establishes neither novelty nor effectiveness, and the
record now shows: reversed at every setting on SD-1.4 (`FINDING63`), on SD-1.5 and BK-SDM
(`FINDING64`), on held-out prompts (`FINDING66`), and — with the attacker and the compute budget
both corrected — **at every unroll depth including none at all**.

---

## 7. Correction (2026-10-26) — the mechanism check is weakly supported, not passed

§3 presented three mean probe deltas as evidence that the unroll does what the design assumes. The
raw per-probe values, which §3 did not show:

| arm | K | outer | probes **n** | per-probe deltas | mean |
|---|---:|---:|---:|---|---:|
| `utrap2_church_l0.15k0` | 0 | 400 | 8 | 0.0 ×8 | **0.00000** |
| `utrap2m_church_l0.15k1` | 1 | 320 | **7** | −0.00093, −0.00020, −0.00011, −0.00044, −0.00018, −0.00222, **−0.00457** | −0.00124 |
| `utrap2m_church_l0.15k8` | 8 | 133 | **3** | −0.00320, −0.00219, −0.00268 | −0.00269 |
| `utrap2m_church_l0.15k32` | 32 | 44 | **1** | **−0.00638** | −0.00638 |

**Three problems with §3's reading:**

1. **n is 7 / 3 / 1.** The K=32 figure is a **single measurement**. `--probe-every 50` against
   44 outer steps yields exactly one probe; the counts differ because the matched-compute design
   gives each arm a different number of outer steps.
2. **The within-arm spread swamps the between-arm trend.** K=1's deltas range over **40×**
   (−0.00011 to −0.00457). Its own maximum is the same order as K=32's single value. No ordering
   survives that.
3. **Even a clean trend would prove less than claimed.** A deeper unroll takes more gradient steps,
   so a larger probe movement is close to tautological. The probe shows the inner loop **moves**;
   it does not show the inner loop **approximates the evaluation attacker**, which is what the
   design assumes.

**What is still supported:** the **K=0 control is exactly 0.00000 on all 8 probes**, confirming
`ABLATION_K0` is a true no-op — that part of §3 stands and is verified.

**What is withdrawn:** "the delta grows with K, roughly in proportion", and the inference that
`PREREG_H90`'s second falsifier "did not fire". **It is undetermined.** Per the prereg's own
wording, a conclusion about depth requires the probe to grow with K, so **§1's null result is
weaker than stated**: it is *"K ∈ {0,1,8,32} all reach H = 1.000 at matched samples"* — an
empirical fact — without the mechanism check that would let it be read as a statement about depth.

**The proper measurement is cheap and is queued:** the probe can be run post-hoc at any K on a
fixed arm (load, run K inner steps, measure the fixed-noise target loss before/after), giving
n ≫ 1 at every K without rebuilding anything. Until that runs, §1 is an outcome, not a mechanism.
