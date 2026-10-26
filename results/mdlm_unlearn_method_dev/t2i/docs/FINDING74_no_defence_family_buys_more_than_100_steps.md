# FINDING 74 — Across **six defence families**, not one buys the attacker more than **100 gradient steps**. The ≤100-step correction is not method-specific. One cell — Meta variant A at 25 steps — is the only sub-0.40 number in the project, and it fails the mandate anyway.

**Date:** 2026-10-26 · **Experiment:** E12, 24 cells, **n = 3 per cell**, held until all returned
**Closes** the "two families only" limit implicit in `FINDING68`–`FINDING73`, which curved FRP and
UTRAP alone.
**Found by** a registry query, not new builds — the `FINDING73` §9c lesson applied: 36 church arms
across seven families were already on disk.
**Prediction recorded before the run** (`run_e12_method_axis.sh`): *"if the correction generalises,
every family reaches max-over-valid H = 1.000 by 100 steps, and none holds below 0.40 at 25."*
**Evidence:** `outputs/atk_E12_*.json`, `logs/e12_method_axis.log`

---

## 1. The method axis, church, lr 1e-5, n = 3

| family | arm | as-is retain8 | **25 steps** valid / max H | **100 steps** valid / max H | util @100 |
|---|---|---:|---|---|---:|
| **GIFT** port | `gift_church_b1.35` | 0.6000 | 3/3 · **1.0000** | 3/3 · **1.0000** | 1.0000 |
| **WSR-anchor** (ours) | `wsr2_church_lr100` | 0.9500 | 3/3 · **1.0000** | 3/3 · **1.0000** | 0.9583 |
| **IMMA** port | `imma_church_frp_a1e-7_s1500` | 0.9417 | 3/3 · 0.9333 | 3/3 · **1.0000** | 0.9583 |
| **FRP** port | `frp_church_r3_l05` | 0.9417 | 4/4 · 0.9333 | 1/1 · **1.0000** | 0.9792 |
| **UTRAP** (ours) | `utrap_church_l0.15k8` | 0.8667 | 3/3 · 0.9333 | 1/1 · **1.0000** | 0.9167 |
| **Meta variant A** | `h80_meta_church_lam0.7` | 0.7000 | **1/3 · 0.1333** | 3/3 · **1.0000** | 0.9375 |

**Every one of six families reaches max-over-valid H = 1.000 by 100 steps.** The first clause of the
prediction is confirmed: the ≤100-step cost correction of `FINDING68`/`FINDING70` is **not
method-specific**. It now rests on six families rather than two.

Two of them — **GIFT and our own WSR-anchor — are already at the ceiling at 25 steps**, faster than
FRP, IMMA or UTRAP. GIFT's arm here is the best-suppressed operating point its own β sweep produced
(target 0.2000 at retain8 0.6000), so this is not a weak-arm artifact.

## 2. The one exception, and why it is not a mandate result

**`h80_meta_church_lam0.7` at 25 steps is the only cell in this entire project with a gate-valid
max-over-valid H below 0.40** — **0.1333**. The second clause of my prediction therefore failed, and
it is worth saying so plainly rather than rounding it into the headline.

It is nonetheless **not** a mandate result, for three independent reasons:

1. **It fails the retain half outright.** Its retain8 is **0.7000**, against the mandate's ≥ 0.85.
   The bar is conjunctive; this arm never qualified.
2. **Two of its three attacks were gate-INVALID.** The 0.1333 is a maximum over **one** valid run.
   That is exactly the bias `FINDING72` §2 measured: heavily-erased arms (this one is at
   `H_before` = 0.0000) show depressed validity, so their max-over-valid is taken over a filtered
   and very small sample. Under the `FINDING35` §4 ratchet the figure can only **rise** with more
   seeds.
3. **It is fully reversed at 100 steps** — 3/3 valid, max 1.000, utility 0.9375. Whatever the
   25-step number reflects, it buys the attacker 75 extra steps at most.

So the honest statement is: **Meta variant A resists a 25-step attacker at a utility level the
mandate already rejects, and collapses by 100.** It is the most interesting cell in the table and it
changes no verdict.

## 3. Methods do differ at low budget — the first such ordering with n = 3 throughout

At 25 steps, max-over-valid: GIFT 1.0000 = WSR-anchor 1.0000 > FRP 0.9333 = IMMA 0.9333 =
UTRAP 0.9333 ≫ Meta-A 0.1333.

This is a **real** spread and the first method ordering in this project measured at n = 3 in every
cell. But `FINDING72` §1's caution applies in full: these arms differ in `H_before` (0.0000 to
0.2000) and retain8 (0.6000 to 0.9500), so the ordering **conflates method with operating point**.
No claim that one method is intrinsically weaker is made here.

## 4. Utility: no damage anywhere, consistent with `FINDING71`

Post-attack utility across all 24 cells sits between **0.8889 and 1.0000**, squarely in the band
`FINDING71` established for lr 1e-5 (corpus mean 0.9284, n = 169). Every family's concept is fully
restored with the measured non-target panel intact. The damage regime remains a property of **high
learning rate**, not of budget and not of method.

## 5. Limits

- **One concept (church), one learning rate, three seeds, one arm per family.** The within-concept
  replication of `FINDING73` §9 was possible for garbage_truck; it has **not** been done per family,
  so "family" and "that family's chosen arm" remain inseparable here.
- Two cells carry **n = 1** at 100 steps (FRP, UTRAP), inherited from E11. They sit at the ceiling
  where variance is minimal (`FINDING70` §6c) but are not held to the n ≥ 3 standard.
- **UCE is absent**: no SD-1.4 church UCE arm exists on disk. Six families, not seven.
- **Meta variant A is variant A** — two of Eq. 8's three terms are missing
  (`ERRATUM_meta_objective.md`). This says nothing about published Meta-Unlearning. **E2 is building
  the full Eq. 8 objective now**, and that arm is the one that could speak to the paper.
- **DEV prompts** throughout; H quantised to 1/15.

## 6. Where this leaves the mandate

**Unmet, and now across six families.** No arm reaches max-over-valid H ≤ 0.40 while holding
retain8 ≥ 0.85, at any budget, learning rate, concept or defence family measured. The single
sub-0.40 cell fails the retention half by 0.15 and survives only to 25 steps.
