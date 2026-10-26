# FINDING 78 — Restoring Meta-Unlearning's two missing Eq. 8 terms buys **+0.27 retention at identical suppression**, and produces the **first arm in this project to satisfy both halves of the mandate at any attack budget** — at 25 steps. It is fully reversed by 300.

**Date:** 2026-10-26 · **Experiments:** E2 (builds) + E2g (frozen grid, 18/18 cells, n=3)
**Implementation:** `meta_unlearn_v2.py` — **ours, corrected**, explicitly **not** an official
reproduction (`UPSTREAM_FIDELITY_MANIFEST.md` §2)
**Evidence:** `outputs/gen_meta2_church_*.json`, `outputs/atk_E2G_*.json`, `logs/e2g_grid.log`

---

## 1. The objective repair buys large retention

`ERRATUM_meta_objective.md` predicted that variant A's benign-capability collapse was an artifact
of the two **missing** Eq. 8 terms (`−L_F(θ′)` and `+ζL_R(θ)`), not the published method misfiring.
Restoring both:

| objective | λ | target | **retain8** |
|---|---:|---:|---:|
| variant A | 0.6 | 0.0000 | 0.6250 |
| variant A | 0.7 | 0.0000 | 0.7000 |
| variant A | 0.9 | 0.0000 | **0.0000** (destroyed) |
| **v2, full Eq. 8** | **0.5** | **0.0000** | **0.9000** |
| v2, full Eq. 8 | 1.0 | 0.1333 | 0.9250 |

**At identical target suppression (0.0000), the full objective holds retain8 0.9000 where variant A
managed 0.6250–0.7000** — **+0.20 to +0.27**, and it clears the mandate's 0.85 retention bar. At
λ=1.0, where variant A is destroyed (0.1167), v2 holds 0.9250. The prediction is confirmed.

Build telemetry: target pool **64/64 = 1.0000** church-classified (base-generated, content-checked,
unlike variant A's unverified erased-teacher pool); scopes separated as the reference does
(meta `xattn` 44.0M / attack `full` 859.5M).

## 2. The frozen grid — all 18 cells, gated **and** ungated

`meta2_church_lt0.5` (as-is target 0.0000, **retain8 0.9000**):

| steps | gate-valid | **max-over-valid H** | max **ungated** | utility | per-seed (gate, H) |
|---:|---:|---:|---:|---|---|
| **25** | **1/3** | **0.0667** | **0.2000** | 0.8750 → 0.9306 | (0,f,0.2000) (1,f,0.0667) (6,**P**,0.0667) |
| 100 | 3/3 | 0.6667 | 0.6667 | 0.8750 → 0.9583 | (0,P,0.6667) (1,P,0.1333) (6,P,0.5333) |
| 300 | 3/3 | **1.0000** | 1.0000 | 0.8750 → 0.9583 | (0,P,1.0) (1,P,0.9333) (6,P,1.0) |

`meta2_church_lt1.0` (target 0.1333, retain8 0.9250): 25 → 0.6000 (1/3 valid), 100 → 1.0000,
300 → 1.0000.

## 3. **The first arm to meet both mandate halves — at one budget**

The bar is **retain8 ≥ 0.85 AND max-over-valid H ≤ 0.40**. For `meta2_church_lt0.5` at **25 steps**:

- retain8 **0.9000** ≥ 0.85 ✅
- max-over-valid H **0.0667** ≤ 0.40 ✅ — and even the **ungated** maximum (0.2000) is below the bar,
  so this does not depend on discarding the two inconclusive runs.

**No previous arm has satisfied both at any budget.** The closest, Meta *variant A* at 25 steps
(`FINDING74`), failed the retention half outright at 0.7000.

**Four reasons this is not a solved mandate, stated plainly:**

1. **It holds only at 25 steps.** At 100 it is 0.6667 and at 300 it is **1.0000**. The project's
   mandate and every historical comparison are evaluated at 300. Against the cheapest *reproducible*
   full-recovery attacker measured anywhere here — lr 5e-5, 25 steps, all seeds at 1.0000
   (`FINDING70`) — this arm has not been tested at all.
2. **Only 1 of 3 attacks was gate-valid.** That is exactly `FINDING72` §2's bias: heavily-erased
   arms (this one is at target 0.0000) show depressed validity. Under the `FINDING35` §4 ratchet a
   sub-ceiling number **can only rise** with more seeds.
3. **n = 1 build.** Attack-seed replication is not defence-build replication.
4. **One concept, one λ pair, DEV prompts, H quantised to 1/15** — 0.0667 is **one image in fifteen**.

## 4. What it is evidence *about*

`meta_unlearn_v2.py` is **our corrected implementation**, not the authors' code. Two structural
departures from the reference remain (`UPSTREAM_FIDELITY_MANIFEST.md` §2.1): **two data pools
instead of three** (the reference separates `irt` unrelated-retain from `tgt` related-retain with
independent γ), and no `--fix_timesteps` — though the README confirms the paper itself uses
`fix_timesteps=False`, so **that one is not a departure from the published configuration**.

So this is evidence that **Eq. 8's full signed structure matters a great deal for retention** — a
claim about the objective, which is what `ERRATUM_meta_objective.md` set out to test. Whether the
authors' own three-pool implementation does better is **R2**, and R2 has not run.
