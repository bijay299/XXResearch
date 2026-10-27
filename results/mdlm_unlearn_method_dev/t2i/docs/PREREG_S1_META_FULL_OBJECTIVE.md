# PREREG S1 — the full three-term meta objective: does the trap fire only under attack?

**Declared 2026-10-27, BEFORE execution.** Stage S1 of `MANDATE_v2_2026-10-27.md` §6. Frozen on
first result; corrections go in a dated amendment. Development data; no confirmatory claim.

> **AMENDMENT 1 (2026-10-27) — read `PREREG_S1_AMENDMENT_1_2026-10-27.md` before using this file.**
> Two corrections, both recorded before any S1 result existed. (a) §3's description of arm
> `s1_metaV2_z0.0_g0.0` as "the erasure base alone" is **wrong**: both base terms distil to the
> teacher and the model is initialised at the teacher, so that arm is a **bit-exact no-op** and the
> port has no erasure objective of its own. The v1 `zeta=0.0` row below (0.9417) is likewise the
> teacher's own retain8, not evidence that zeta=0 preserves utility. (b) §7's cost estimate is
> confirmed correct at 1.11 s/step steady. Arms 2 and 3, the measurements and the §5 decision rule
> are unaffected. Original text below is left verbatim.

## 1. Why this experiment and not another

MANDATE v2 §3 established that **no defence which merely displaces theta can win** in this threat
model: every method tested reaches H = 1.000 under a calibrated 25-step attack, and erasing harder
does not help — even lambda = 0.60, which drops retain8 to 0.5417, is fully reversed. The only
remaining class with a concrete mechanism and a published instance is **a trap**: make fine-tuning
on the target destroy retained capability, so the attacker gains the concept but loses the model.

Our existing port of that mechanism implements **one of the three components** of Eq. 8 of
arXiv:2410.12777 (`docs/ERRATUM_meta_objective.md`), and it failed in the predicted way.

## 2. The hypothesis, and why the previous attempt failed

The paper's meta term is `-L_F(theta') - zeta * [L_R(theta') - L_R(theta)]`. The bracket is a
**difference**, not a level. Our v1 port kept only `-zeta * L_R(theta')`, so the cheapest way to
satisfy it was to destroy benign capability unconditionally. The measured consequence is monotone:

| v1 arm | zeta | retain8 (no attack) |
|---|---:|---:|
| `h80_meta_church_lam0.0` | 0.0 | 0.9417 |
| `h80_meta_church_lam0.5` | 0.5 | 0.6333 |
| `h80_meta_church_lam0.8` | 0.8 | 0.4750 |
| `h80_meta_church_lam0.9` | 0.9 | **0.0000** |

**H1:** restoring the missing `+zeta * L_R(theta)` term makes the objective penalise the *rise* in
benign loss across the simulated attack rather than its level, so benign capability survives normal
use at zeta values where v1 collapsed. Restoring `-gamma_f * L_F(theta')` additionally keeps the
target erased *after* the attack.

## 3. Arms, and the single variable

`scripts/meta_unlearn_v2.py` (`config_version meta_v2`). Data construction is replicated line for
line from the v1 port: same pools, same generation seeds (30,000 / 40,000), same 64 images per pool,
same sampler settings. **The objective is the only difference.**

| arm | zeta | gamma_f | purpose |
|---|---:|---:|---|
| `s1_metaV2_z0.0_g0.0` | 0.0 | 0.0 | control: meta term off, i.e. the erasure base alone |
| `s1_metaV2_z0.8_g0.0` | 0.8 | 0.0 | **the direct test of the fix** — v1 at this zeta gave retain8 0.4750 |
| `s1_metaV2_z0.8_g1.0` | 0.8 | 1.0 | full objective, both restored terms |

Teacher: `adapters/frp_church_r3_l05/unet.pt`, concept church, 1500 steps, lr 1e-6, seed 0,
inner-alpha 1e-5. One build per arm. **n=1 is scouting**; nothing from S1 is established (five of
this campaign's retractions were n<=2 claims that died at n=3).

## 4. Measurements

1. **Does the trap fire on normal use?** `gen_eval.py --target church`: `retain8`,
   `utility_all_nontarget`, `target_suppression_acc`, with no attack.
2. **Does a successful attack collapse utility?** `attack_relearn_v2.py`, 25 steps, attacker lr
   swept {1e-5, 5e-5, 1e-4}, 3 seeds. Primary quantity: **change in retained capability from before
   to after a gate-valid attack**.
3. Cost: wall-clock minutes per build, recorded by the script.

## 5. Decision rule, declared before any number exists

| outcome | verdict |
|---|---|
| `retain8 >= 0.85` un-attacked **and** post-attack retained capability drops **>= 0.25** on gate-valid attacks | **H1 supported** — proceed to S2 (n=3 builds, full attacker sweep, measured cost) |
| `retain8 < 0.85` un-attacked at zeta=0.8 | **trap still fires on normal use.** The three-term fix did not work; class A retired per the mandate's stopping rule |
| post-attack retain drop `< 0.10` | **no trap effect.** Class A retired |
| drop between 0.10 and 0.25 | inconclusive; one bounded zeta sweep only, then decide |

**Baseline for the drop:** across 45 evidence-complete cells in this project, post-attack retained
capability *rises*, mean **+0.0231**, median 0.0000. Any reliable negative drop is a departure from
every method measured here.

## 6. What S1 cannot establish

- **Not a reproduction.** This is our implementation of the paper's objective on our pools and
  prompts, first-order, with one benign pool where the reference carries two with separate gammas.
  It is evidence about the *mechanism*, not about the authors' reported results.
- n=1 build per arm. No claim survives S1 alone.
- One concept, one teacher, one attack budget.
- A trap that works would still need S3 to show it is not concept-specific, and an independent
  check that it does not simply make the model fragile to *any* fine-tuning, which is the obvious
  failure mode and is **not** tested by S1.

## 7. Budget

3 builds x ~30 min (v1 measured 22.2 min at 1500 steps; v2 adds one backward per step) + 3
`gen_eval` panels + up to 27 attack cells ~= **2.5 GPU-hours**, inside S1's 3-hour allowance.
