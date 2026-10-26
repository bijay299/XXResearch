# FINDING 71 — I tested my own newest claim on a second arm and **half of it failed.** The *damage* at lr 1e-4 is real and corpus-wide (mean post-attack utility **0.93 → 0.75 → 0.50** by learning rate, n = 229). The *non-monotone H* I explained with it **did not reproduce** and is specific to one arm.

**Date:** 2026-10-26 · **Experiments:** E11h (replication) + a zero-GPU corpus analysis
**Prediction recorded before running**, in `run_e11h_damage_replication.sh`: *"if the mechanism is
general, the 25-step cell should show **both** a lower mean H than the 10-step cell **and** a
collapse in post-attack utility."*
**Evidence:** `outputs/atk_E11H_*.json`, `outputs/h_vs_utility_corpus.json`, `logs/e11h_damage.log`

---

## 1. What I was testing, and why

`FINDING70` §6b claimed that at lr 1e-4 past ~10 steps the attack stops relearning and starts
destroying the model — explaining a **non-monotone** H (0.9555 at 10 steps falling to 0.6222 at 25)
by a utility collapse (0.8472 → 0.5278). **That rested on one arm.** Single-arm mechanism claims are
exactly what this project has retracted four times, so it was replicated on
`frp_church_r3_l05` before being relied on.

## 2. The replication: the first half of my prediction **failed**

`frp_church_r3_l05` @ lr 1e-4, n = 3:

| steps | valid | mean H | max H | mean utility after | per-seed (H, utility) |
|---:|---:|---:|---:|---:|---|
| 10 | 3/3 | 0.7111 | 0.9333 | **0.6180** | (0.9333, 0.3333), (0.3333, 0.6250), (0.8667, 0.8958) |
| 25 | **2/3** | **0.9000** | 0.9333 | **0.6945** | (0.2000, 0.5000), (0.8667, 0.7292), (0.9333, 0.8542) |

**Mean H goes UP from 10 to 25 steps (0.7111 → 0.9000), not down.** Utility also goes slightly up
(0.6180 → 0.6945). So **the non-monotonicity did not reproduce**, and the prediction's first clause
is falsified on its own terms.

**Consequence: `FINDING70` §6b is narrowed.** The non-monotone H is **specific to
`utrap_church_l0.15k8` at this grid**, not a general property of lr 1e-4. Any reading of it as "high
lr always buys the attacker less at longer budgets" is withdrawn.

One thing the replication did add: at 25 steps FRP drops to **2/3 gate-valid**. Heavy damage
interferes with the attacker's ability to fit even its own training prompts, which is a mechanism
worth noting — a damaged model fails the validity gate, so damage shows up as *inconclusive* runs
as well as low H.

## 3. The damage itself **does** generalise — and far beyond two arms

Zero-GPU analysis over **every gate-valid full-UNet attack record in the corpus that has a measured
post-attack utility** (n = 229):

| lr | n | mean H | **mean post-attack utility** | **corr(H, utility)** |
|---:|---:|---:|---:|---:|
| 1e-5 | 169 | 0.9065 | **0.9284** | **+0.4241** |
| 5e-5 | 34 | 0.9137 | **0.7518** | +0.3379 |
| **1e-4** | **26** | 0.8256 | **0.4952** | **−0.0455** |
| all | 229 | 0.8984 | 0.8530 | +0.2952 |

**Mean post-attack utility falls monotonically with learning rate: 0.9284 → 0.7518 → 0.4952.** The
damage claim is therefore confirmed on a far stronger basis than the two-arm evidence it was built
on — **at lr 1e-4 the attacked model's non-target panel is roughly halved, across 26 records.**

**And the correlation structure changes qualitatively.** At lr 1e-5 and 5e-5, H and utility are
**positively** correlated (+0.42, +0.34): attacks that recover the concept also tend to leave the
model working, which is the "recovery without collateral damage" pattern reported throughout this
project. At **lr 1e-4 the correlation vanishes (−0.05)** — H stops carrying any information about
the model's state.

So `FINDING70`'s broader point survives in sharpened form: **H alone cannot order attacks at lr
1e-4**, because there a given H is compatible with an intact model or a halved one. At lr 1e-5 it
can, which is the regime almost all of this project's results were measured in.

## 4. What this means for the existing record

- **Every lr 1e-4 attack in this corpus was measured on a model whose non-target panel averaged
  0.4952.** Those H values are not wrong, but they describe recovery *on a substantially damaged
  model*, and should not be pooled with lr 1e-5 results as if the two were the same measurement.
  The frozen grid's six-point design includes an lr 1e-4 column, so this touches
  `PROTOCOL_FROZEN_v1` itself.
- **The headline results are unaffected**, because they are max-over-valid and the maxima are
  reached at lr 1e-5 and 5e-5, in the high-utility regime. `FINDING70`'s cheapest reproducible
  recovery (lr 5e-5 / 25 steps, H = 1.0000 on all seeds, utility 0.9236) sits squarely in the
  undamaged regime, so it is unaffected by any of this.

## 5. Limits

- **Observational, across heterogeneous arms.** The §3 correlations mix between-arm and within-arm
  variation: different defences, concepts and budgets are pooled within each lr. They establish that
  the lr → utility relationship is strong and consistent, **not** a within-arm causal slope.
- **n = 26 at lr 1e-4**, the cell carrying the most weight in §3's argument.
- The **136 VOID-utility records** (`ERRATUM_text_encoder_eval_bug`) are excluded, as are all records
  without a measured panel — so §3 is over the 229 with valid utility, not all 696 attack records.
- §2 is **one arm, two budgets, three seeds**, at one lr. It falsifies generality; it does not
  explain why UTRAP is non-monotone and FRP is not. That remains open.
- **DEV prompts** throughout; utility is a 48-image closed-set CLIP panel, not an image-quality
  measure (FID still absent).

## 6. Process note

This is the first time in this project that a mechanism claim was **replicated before being relied
on** rather than after being published as a headline — the rule `FINDING69` §7 wrote down. Half of
it failed, which is the point: had `FINDING70` §6b gone unreplicated, "high lr is non-monotone"
would have entered the record as general when it is one arm's behaviour.
