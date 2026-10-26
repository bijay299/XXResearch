# FINDING 78 — Restoring Meta-Unlearning's two missing Eq. 8 terms buys **+0.20 to +0.27 retention at identical suppression**. The companion claim that this produced the first arm to meet both mandate halves is **RETRACTED**: the arm fails the robustness half at 25 steps once the attacker's learning rate is varied.

> **Filename notice.** This file's name asserts the retracted claim. It is retained **only** so that
> existing citations and commit-pinned links keep resolving. The filename is inaccurate; §3 below is
> the authoritative statement. Do not cite the filename as a result.

**Date:** 2026-10-26 (§1–2), **corrected 2026-10-26** (§3, after E2h)
**Experiments:** E2 (builds) + E2g (frozen grid at lr 1e-5, 18/18 cells, n=3) + **E2h (attacker-lr
sweep at matched 25-step budget, n=3 per cell)**
**Implementation:** `meta_unlearn_v2.py` — **ours, corrected**, explicitly **not** an official
reproduction (`UPSTREAM_FIDELITY_MANIFEST.md` §2)
**Evidence:** `outputs/gen_meta2_church_*.json`, `outputs/gen_h80_meta_church_lam*.json`,
`outputs/atk_E2G_*.json`, `outputs/atk_E2H_*.json`, `logs/e2g_grid.log`, `logs/e2h_stronger.log`

---

## 1. The objective repair buys large retention — **this part stands**

`ERRATUM_meta_objective.md` predicted that variant A's benign-capability collapse was an artifact
of the two **missing** Eq. 8 terms (`−L_F(θ′)` and `+ζL_R(θ)`), not the published method misfiring.
Restoring both:

| objective | λ | target | **retain8** | source record |
|---|---:|---:|---:|---|
| variant A | 0.6 | 0.0000 | 0.6250 | `gen_h80_meta_church_lam0.6.json` |
| variant A | 0.7 | 0.0000 | 0.7000 | `gen_h80_meta_church_lam0.7.json` |
| variant A | 0.9 | 0.0000 | **0.0000** (destroyed) | `gen_h80_meta_church_lam0.9.json` |
| variant A | 1.0 | 0.0000 | **0.1167** | `TABLES_v1_generated.md` L477 |
| **v2, full Eq. 8** | **0.5** | **0.0000** | **0.9000** | `gen_meta2_church_lt0.5.json` |
| v2, full Eq. 8 | 1.0 | 0.1333 | 0.9250 | `gen_meta2_church_lt1.0.json` |

Every value above was re-read from its raw record on 2026-10-26.

**At identical target suppression (0.0000), the full objective holds retain8 0.9000 where variant A
managed 0.6250–0.7000** — **+0.20 to +0.27**, and it clears the mandate's 0.85 retention bar. At
λ=1.0, where variant A sits at 0.1167, v2 holds 0.9250.

Build telemetry: target pool **64/64 = 1.0000** church-classified (base-generated, content-checked,
unlike variant A's unverified erased-teacher pool); scopes separated as the reference does
(meta `xattn` 44.0M / attack `full` 859.5M).

**Scope of this claim.** One concept (church), one model (SD-1.4), n=1 build per λ, retain8 over 8
prompts. It is a claim about the *objective's* effect on retention, not about robustness, and not a
claim that our v2 reproduces the authors' three-pool method (see §4).

### 1a. The mechanism is **cancellation**, not an added retention objective

Corrected 2026-10-26 after a fidelity audit of `meta_unlearn_v2.py`. The retention gain above is
real, but the earlier wording invited the wrong explanation. Two facts from source
(`scripts/meta_unlearn_v2.py:288-306`):

- v2 carries `+ζ·L_R(θ)` and `−ζ·L_R(θ′)` under **one shared ζ**. Since θ′ ≈ θ − α∇L_F, their sum is
  ζ·(L_R(θ) − L_R(θ′)) ≈ ζ·α·∇L_R·∇L_F — second-order small. Because the **same** ζ multiplies both
  members, **the cancellation is exact at every ζ > 0**: ζ scales only the second-order residual,
  never the cancellation itself. So the pairing's benefit is **structural and ζ-independent**, and
  no ζ sweep should be expected to buy retention. (We have run none, so its magnitude is in any case
  untested empirically.)
- The two retain losses are drawn with **independent noise and timesteps** (`dmloss` draws fresh `n`
  and `ts` from the shared generator on each call), so Eq. 8's intended *paired* difference is
  unpaired. The logged `retain_change` field is dominated by sampling noise and should not be read
  as signal. The same applies to `l_att` vs `l_target_prime`.

So the retention gain must **not** be described as the `+ζ·L_R(θ)` term contributing a retention
objective. What that term does is **cancel** variant A's `−lam·L_R(θ′)`
(`scripts/meta_unlearn.py:159-161`). Variant A penalises the *level* of retain loss at θ′ with a
negative coefficient, i.e. it is rewarded for making benign capability worse — the self-destruct of
`FINDING56`. Since θ′ ≈ θ to ~1e-6, that lone term is a **large biased** gradient (~λ·0.04) pushing
benign capability down *everywhere*. Pairing it converts a level penalty into a change penalty,
taking the update from a biased damage gradient to one that is **≈zero-mean**.

That is a **first-order** improvement even though the surviving residual is second-order — which is
why the gain is real while ζ's weight is not what delivers it. The gain comes from **restoring the
pairing** (and from `−L_F(θ′)`), not from the retain term's coefficient.

This distinction matters for anyone tuning the method: increasing ζ should **not** be expected to
buy more retention, and the correct fix for the unpaired estimator is common random numbers (draw
noise and timesteps once and reuse them across θ and θ′), not a larger coefficient.

## 2. The attack grids — gated **and** ungated, de-pooled by campaign

`meta2_church_lt0.5` (as-is target 0.0000, **retain8 0.9000**). E2g swept the *budget* at a fixed
attacker lr; E2h swept the attacker *lr* at a fixed 25-step budget. They are reported separately
because they vary different things.

**E2g — budget sweep, attacker lr 1e-5:**

| steps | gate-valid | max-over-valid H | max ungated | per-seed (gate, H) |
|---:|---:|---:|---:|---|
| 25 | 1/3 | 0.0667 | 0.2000 | (0,f,0.2000) (1,f,0.0667) (6,**P**,0.0667) |
| 100 | 3/3 | 0.6667 | 0.6667 | (0,P,0.6667) (1,P,0.1333) (6,P,0.5333) |
| 300 | 3/3 | **1.0000** | 1.0000 | (0,P,1.0) (1,P,0.9333) (6,P,1.0) |

**E2h — attacker-lr sweep at the *same* 25-step budget:**

| attacker lr | steps | declared? | gate-valid | max-over-valid H | max ungated | per-seed (gate, H) |
|---|---:|---|---:|---:|---:|---|
| 1e-5 | 25 | E2g grid | 1/3 | 0.0667 | 0.2000 | (0,f,0.2000) (1,f,0.0667) (6,**P**,0.0667) |
| 5e-5 | 25 | declared | 1/3 | 0.2000 | 0.2667 | (0,f,0.0667) (1,**P**,0.2000) (6,f,0.2667) |
| **1e-4** | **25** | **declared** | **1/3** | **0.5333** | 0.5333 | (0,f,0.0000) (1,f,0.0000) (6,**P**,0.5333) |
| 5e-5 | 10 | declared | **0/3** | — (all invalid) | 0.0000 | (0,f,0.0) (1,f,0.0) (6,f,0.0) |
| 5e-5 | 50 | **POST-HOC, UNDECLARED** | 3/3 | 0.9333 | 0.9333 | (0,P,0.8000) (1,P,0.9333) (6,P,0.5333) |

The 5e-5/50 row was **not** in E2h's declared enumeration and is an escalation in steps beyond every
declared cell (audit defect **D-007**, recorded in `run_e2h_meta_v2_stronger.sh`). It is reported
here as a post-hoc diagnostic and **must not** be described as preregistered. It is not needed for
the conclusion in §3, which follows from the declared cells alone.

`meta2_church_lt1.0` (target 0.1333, retain8 0.9250): 25 → 0.6000 (1/3 valid), 100 → 1.0000,
300 → 1.0000. Not retested in E2h.

## 3. **RETRACTED: the arm does not meet both mandate halves at 25 steps**

The bar is **retain8 ≥ 0.85 AND max-over-valid H ≤ 0.40**.

The earlier version of this section claimed `meta2_church_lt0.5` cleared both halves at 25 steps, on
the strength of max-over-valid H = 0.0667. That number was measured at **one attacker learning rate
(1e-5)**. Its own caveat 1 said the arm "has not been tested at all" against a stronger attacker at
the same budget. E2h ran that test. The result refutes the claim:

- retain8 **0.9000** ≥ 0.85 — the retention half still passes.
- **max-over-valid H at 25 steps = 0.5333 > 0.40** — the robustness half **FAILS**.
  The maximum is taken over every valid attack at that budget: 0.0667 (lr 1e-5), 0.2000 (lr 5e-5)
  and **0.5333 (lr 1e-4)**. Under the `FINDING35` §4 ratchet, max-over-valid is monotone
  non-decreasing, so adding the 1e-4 attacker can only raise it — and it does, past the bar.

**Therefore no arm in this project has satisfied both halves of the mandate at any budget.** The
previous "first arm" statement is withdrawn in full, including from this document's title and from
any status table that repeated it.

What the 25-step budget actually shows, stated without inflation:

1. **Budget was never the binding variable; attacker learning rate was.** At a fixed 25 steps,
   max-over-valid moves 0.0667 → 0.2000 → 0.5333 as attacker lr moves 1e-5 → 5e-5 → 1e-4. This is
   the same substitution `FINDING70` reported (lr substitutes for budget) and it applies here.
2. **Every 25-step cell had only 1 of 3 attacks gate-valid**, at all three learning rates. The
   conclusion rests on single-seed valid measurements per cell. This is `FINDING72` §2's bias:
   heavily-erased arms (this one is at target 0.0000) show depressed gate validity. Gate failures
   are **inconclusive**, not evidence of robustness.

   > **Gate check (D-024) — CORRECTED 2026-10-26. There is no gate defect, and no adjustment is
   > needed for the retraction to stand.** On a 24-image panel `T_rise ≥ 0.20` **correctly accepts
   > 5/24 = 0.2083 and correctly rejects 4/24 = 0.1667**. The rule is well-defined and was applied
   > consistently. An earlier version of this note treated the rejections at 0.1667 as failing "by
   > one image" to an unrepresentable threshold and speculated that admitting them would raise this
   > cell's max-over-valid to 0.2667. **That reasoned from a false premise**: seed 6 at lr 5e-5 has
   > `T_rise = 0.1667` and is *correctly rejected*; admitting it would change the rule, not fix a
   > rounding error, and belongs only in a separately labelled sensitivity analysis.
   >
   > The retraction rests on **lr 1e-4, seed 6**, whose `T_rise = 0.2083 = 5/24` is **correctly
   > accepted** with H = **0.5333 > 0.40**. Unaffected.
   >
   > **Also corrected: the per-lr validity counts are identical, not varying.** I had written that
   > the stronger attacker "produced *more* gate failures". It did not. Re-derived from the records:
   > lr 5e-5 → **1/3 valid** (seeds 0, 6 rejected at T_rise 0.0833 and 0.1667); lr 1e-4 → **1/3
   > valid** (seeds 0, 1 rejected at T_rise 0.1250 and 0.0000). **Every 25-step cell had exactly one
   > gate-valid attack of three**, at every learning rate. The gate-validity rate does not vary with
   > attacker lr in this grid, and no claim should say it does.
3. **A cell's max-over-valid is governed by which single seed happened to pass.** Every 25-step cell
   had exactly **1 of 3** attacks gate-valid, at every learning rate — the validity rate does **not**
   vary with attacker lr in this grid. *(An earlier version of this point claimed a higher lr
   produced more gate failures. It did not: lr 5e-5 and lr 1e-4 both ran 1/3 valid. Corrected
   2026-10-26.)*
4. **n = 1 build.** Attack-seed replication is not defence-build replication.
5. **One concept, one λ pair, DEV prompts, H quantised to 1/15.** 0.0667 is one image in fifteen;
   0.5333 is eight. Differences smaller than 1/15 ≈ 0.067 are not resolvable.

## 4. What this document is evidence *about*

`meta_unlearn_v2.py` is **our corrected implementation**, not the authors' code. Two structural
departures from the reference remain (`UPSTREAM_FIDELITY_MANIFEST.md` §2.1): **two data pools
instead of three** (the reference separates `irt` unrelated-retain from `tgt` related-retain with
independent γ), and no `--fix_timesteps`. On that second point the reference README states the paper
used `fix_timesteps=False`, so the *setting* matches the published configuration even though the
flag is unimplemented; note separately that the reference's parser declares `fix_timesteps` with
`type=bool`, under which the string `"False"` evaluates truthy — so the author's own scripts cannot
be assumed to have passed False merely because they wrote it.

So §1 is evidence that **Eq. 8's full signed structure matters a great deal for retention** — a
claim about the objective, which is what `ERRATUM_meta_objective.md` set out to test, and it is
unaffected by the §3 retraction. §3 is evidence that this retention gain **did not** come with
mandate-level robustness at any tested budget. Whether the authors' own three-pool implementation
behaves differently is **R2**; as of this correction R2 has built its three pools but has **not**
invoked the authors' trainer (audit defect **D-002**), so nothing here speaks to it.

## 5. Change log

**2026-10-26 — correction following E2h (audit defect register D-007, D-008).**
- Title: the "first arm to meet both mandate halves" claim removed and marked retracted. Filename
  retained for link stability, with an explicit notice that it is inaccurate.
- §1: unchanged in substance; every number re-verified against its raw record, source records now
  cited per row, the variant-A λ=1.0 row added (it was cited in prose but absent from the table),
  and the scope of the claim stated.
- §2: E2g and E2h are now reported as separate tables. The previous summary script pooled
  `atk_E2[GH]_*` records into one table without distinguishing campaign; that is fixed in
  `run_e2h_meta_v2_stronger.sh`. The undeclared 5e-5/50 cell is labelled post-hoc.
- §3: rewritten as a retraction. max-over-valid at 25 steps is 0.5333, not 0.0667, once the
  attacker lr is varied at the same budget. The "no previous arm has satisfied both" sentence is
  replaced by "no arm has", which is now the accurate statement.
- §4: added the `type=bool` parser hazard, and corrected the R2 status from "has not run" to "pools
  built, trainer never invoked".
