# Findings audit, 2026-10-28 — what survives, what is retracted, what blocks a submission

Triage of all 78 numbered findings plus the pilot and measurement documents. Method: every
`FINDING*.md` read for its title, retraction banner, correction markers and reported `n`. **No
finding's text was altered.** This document classifies; it does not rewrite history.

Project scale: **164 arms · 1,215 attack cells · 374 generation panels · ~30 pre-registrations.**

---

## A. Stands clean — no retraction, no narrowing

| finding | claim | evidence |
|---|---|---|
| **F36** | the loss-based trainability probe measures nothing: the **base model** removes 6.5–6.9% of available loss vs the erased arm's 6.5–7.1% | direct |
| **F45** | it is the allocation's **shape**, not magnitude; F39's attribution is clean | pre-declared falsifier |
| **F51** | entrenchment confirmed on bisected utility-matched points; the gradient **saturates** | bisection |
| **F74** | **six defence families, n=3, none buys the attacker more than 100 gradient steps** | 24 cells, n=3 |
| **U29** | the proposed teacher does not replicate at 0.0000 (0.22/0.11/0.11/0.33) and erases by emitting degenerate texture | 4 eval seeds + images |
| **U30** | the damage-then-distil robustness result is a measurement artifact; the arms still draw churches | 4 seeds + images, 5 arms |

**F74 is the strongest single result in the project.** Six families, n=3, pre-registered prediction,
found by registry query rather than new builds.

## B. Measurements stand, interpretation narrowed

These are usable **if quoted at the narrowed scope**, never at the original headline.

| finding | what stands | what was withdrawn |
|---|---|---|
| **F34** | robust SOTA fully reversed, 6/6 | headline is a **confirmation, not a discovery** — Meta-Unlearning (2410.12777) anticipated it |
| **F35** | 11/11 usable arms reversed; the exception was an n=1 artifact | max-over-attacks phrasing |
| **F43** | FRP, built for relearning robustness, is fully reversed | the retain-only characterisation and the mechanism explanation |
| **F44** | the pruned weights were never relearned — the concept returned through weights FRP left alone | — |
| **F53/F54** | a LoRA adversary at **0.046%** of the UNet, and **768 scalars**, reverse the robust SOTA | the apparent separation between defences |
| **F55** | DreamBooth "resistance" was the **attacker's own regulariser**; at prior-weight 0 it reaches 1.000 | — |
| **F57** | FRP's mask is decisive before the attack, irrelevant after | localisation claims; one seed, count- but not magnitude-matched |
| **F58** | GIFT fully reversed at every usable β | 6 vacuous attacks on an arm that erased nothing, now excluded |
| **F62** | arm still fully reversed; retention 0.9639 ± 0.0127 | "beats FRP on both axes" — target is 0.1778 ± 0.1018 at n=3, worse on erasure |
| **F63** | UTRAP (ours) reversed, 18/18 valid attacks | that it trained against the actual attack — inner process differed on **seven axes** |
| **F64** | SD-1.4, SD-1.5 and BK-SDM-small all reverse to 1.000 | "cross-model generalises"; church only, one build per setting |
| **F70** | learning rate substitutes for budget; **25 steps, zero-variance recovery** | — |
| **F72** | at matched lr, UTRAP recovers as fast as FRP | F68 §6's coarse-grid impression |
| **F73** | all three concepts saturate far below 300 steps | "generalises and gets worse" — tench is slowest, not fastest |
| **F78** | restoring Eq. 8's two missing terms buys **+0.20 to +0.27 retention at identical suppression** | "first arm to meet both mandate halves" — **retracted** |
| **MEASUREMENT** | the 86-arm accounting: 56 reversed, 21 void, 6 destroyed, 3 never erased | — |

## C. Retracted or not usable — do not cite

| finding | status |
|---|---|
| **F37** | "1 image / 50 steps" was a measurement artifact; rewritten, then superseded again |
| **F59** | **retracted headline.** "IMMA partially resists" refuted within the hour — tripling LoRA budget gives 1.000 |
| **F60** | conclusion withdrawn; the cited paper's prediction direction had been **reversed** in our reading |
| **F61** | did not test the published method; it anchors **toward** base weights where the paper maximises distance. Renamed WSR-anchor (ours) |
| **F66** | **withdrawn, not re-verifiable** — inconsistent label text on checkpoints that no longer exist |
| **F67** | derivation stands; the headline **mechanism is wrong** — α 1→4000 buys ≈9.6×, not 4,000× |
| **F68** | superseded by F69/F70 |
| **F69** | **headline retracted by its own §7.** sd = 0.2776 at 10 steps. The curve, the gradedness and the gate result survive |
| **F71** | the non-monotone H did not reproduce; the damage finding (n=229) survives |
| **F75** | **retracted** — its central measurement does not measure what it claimed |
| **F76** | "unroll depth buys nothing" and the mechanism check both withdrawn |
| **F77** | **retracted/superseded** |
| **F39/F40/F41/F42** | various withdrawals; F40 records evidence **destroyed by checkpoint deletion** |
| **F46→F51** | the entrenchment thread reversed itself three times; **only F51 should be cited** |

---

## D. Four defects that block a submission until resolved

These are not presentational. Each one, if a reviewer finds it first, ends the paper.

**D1 — the evaluator used inconsistent label text (F65, F66).**
`gen_eval.py` defines `LABELS` (the CLIP text) and `LAB_KEY` (record keys). **Nine of twelve strings
differ.** `gen_eval_final.py` used the key strings as CLIP text, so FINAL and DEV numbers were
scored against different label sets. F65's own banner: *"The effect on scores and conclusions is
**unresolved** pending corrected evaluation."* **Still unresolved.**

**D2 — the validity gate is biased (F72, n = 520).**
The gate's exclusion rate is **non-monotone in `T_before`**, so `max-over-valid H` — our primary
robustness quantity throughout — is computed over a sample whose filtering **correlates with how
well-erased the arm is**. A reviewer will ask whether the headline survives an unfiltered estimator.
We do not currently know.

**D3 — the UCE baseline is in question (F77).**
Our UCE port and pinned upstream compute edits 10.8–38.4% apart in edited-weight norm, and
**upstream at these settings does not erase church at all.** UCE appears in our tables as the
strongest eraser. That needs settling before it is used as a comparator.

**D4 — n=1 is pervasive.**
Five claims in this project have died on seed replication (F35, F59, F62, F69, F71). Only F74, F70,
F72 and F73 carry n=3 at the headline. Everything else is a candidate for the same fate.

---

## E. What the audit changes about tonight's plan

**U35 (the cost curve) is substantially redundant.** F69 curved 5→300 steps; F74 did six families at
n=3. I proposed it as the untouched axis; it was not. What U35 adds is genuinely narrow: budgets
**below 5** (1, 2, 4), and coverage of **UCE and ESD**, which F74 did not include. It should be
re-scoped to that, not presented as a new axis.

**The representation probe remains the only genuinely new experiment proposed.** Nothing in the
register attempts it.
