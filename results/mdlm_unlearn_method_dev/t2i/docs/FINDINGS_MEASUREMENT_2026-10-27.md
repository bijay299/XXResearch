# Measurement-validity findings for relearning-robustness claims

**2026-10-27.** Consolidates findings established today from re-analysis of this project's existing
attack records — **no new GPU time was spent on any result in this document.** Each is stated with
the cells that support it and the limits that bound it. Companion to
`MANDATE_v2_2026-10-27.md` clauses 1b, 1c and 7, which are the binding protocol versions.

These findings are about **how robustness-to-relearning is measured**, not about any defence. They
arose because the campaign's defence results were negative and the negative result turned out to
depend on measurement choices that are easy to get wrong in either direction.

## 0. The result these findings support

Across **86 concept-matched arms** — the attacked concept is the one the arm erased, taken from each
arm's own `gen_eval` record where it has one (55 arms) and otherwise from its build record (31) — on
5 concepts (church 70, parachute 7, van_gogh 4, garbage_truck 3, tench 2), spanning ESD, UCE, GIFT,
FRAG/FRP, IMMA (adapted and official-code), AdvUnlearn, our Meta-Unlearning port, WSR, the internal
h-series and `undo` variants, and UTRAP at K = 0/1/8/16/32:

> **No arm in this project has been *shown* to both suppress its target and resist relearning.**
> **56** are driven to H = 1.000 by a relearning attack of at most 25 steps; **21** cannot be read at
> all because their defence lives in the text encoder and the post-attack evaluation loaded the base
> one (a documented harness defect); **6** were destroyed or heavily degraded (un-attacked retain8
> 0.0000 to 0.3417, against the teacher's 0.9417); and **3** never erased the target at all
> (un-attacked target accuracy 1.0000). **Every arm is accounted for; no category is left open.**

Every one of the 86 falls into exactly one category; there is no residual group.

**Two stronger forms of this claim were stated earlier today and are both withdrawn.** The first,
"49 of 55", matched concept names against checkpoint paths, which excluded arms whose directory name
omits their concept (`u4_k8_s400_b0`, `u16_k0_lam0.6_b0`, …) and included others on name alone. The
second, "not one of 55 arms both suppressed its target and resisted", was scoped to the arms with a
`gen_eval` target — and that scoping was doing real work. Recovering the remaining targets from
build records widens the tally to 86 and surfaces five arms that cannot be dismissed.

### 0.1 The five unresolved arms

`undo_a0.5_big` (max-over-valid H **0.4667**), `undo_a0.5_retain_fisher` (**0.4000**),
`undo_a0.7_retain_fisher`, `undo_a0.5_pool` and `adv_church` (last three: no gate-valid cell) carry
**no `retain8`, no target-suppression figure and no post-attack utility at all** — they are legacy
cells predating the utility panel.

`undo_a0.5_big` is the one that matters: it suppresses church to `H_before` ~0.0000 and the attacker
reaches only 0.4667 on gate-valid cells across lr 1e-5, with 5e-5, 1e-4 and 3e-4 all failing the
gate. On its face that is the closest thing to resistance in the project. Its collapse at the higher
rates (H 0.0000 -> 0.0000 with `T_rise` 0.083 at 1e-4 and 0.000 at 3e-4) *resembles* the
destroyed-model signature of `h64v2_frag` — **but a resemblance is not a measurement, and the
measurement does not exist.**

Only `undo_a0.5_big` and `adv_church` still have checkpoints on disk; the other three were not
retained and are permanently unresolved. One `gen_eval` panel (~1.4 min) settles `undo_a0.5_big`,
the only unresolved arm that is both sub-1.000 and measurable.

### 0.2 The resolved exceptions

`h64v2_frag_a0.3_parachute` holds max H to 0.7333 over 9 cells at 300 steps and the attacker fails
to relearn even its own training prompts (`T_after` 0.25-0.625) — because its un-attacked retain8 is
**0.0833** against the teacher's 0.9417, `utility_all_nontarget` 0.1455, with post-attack retained
capability 0.000-0.125. A dead model, not a defence. `h64v2_frag_a0.3_church` is the same variant,
destroyed more completely still at retain8 **0.0000**.

`gift_church_b0.1` and `uce_sd15_church_lam50000` are the opposite failure: un-attacked
`target_suppression_acc` **1.0000** with retain8 1.0000. They never erased the concept, so the
24-image T panel has no headroom for a 0.20 rise and the gate can never pass. Uninformative about
the attacker, not evidence of resistance — the saturation mode of finding 1.

The three `u4_k8_s400_b{0,1,2}` builds were sub-1.000 earlier today (0.9333 / 0.9333 / 0.8000, lr
1e-5 only). **U20 completed their sweep and all three reach 1.000** once 5e-5 is measured — `b2`
went from `H_before` 0.0000 to `H_after` 1.0000 in 25 steps. Under-measured, exactly as finding 2
predicts, not resistant.

Scope: SD-1.4 only. Arm names include `sd15` and `bksdm` variants but **no record for those arms
names a base model**, so no base-model generalisation is claimed.

## 1. A negative `T_rise` has two distinct causes and they must not be merged

`T_rise` = rise in target-concept accuracy on the attacker's **own training prompts** (24-image
panel). The validity gate requires `T_rise >= 0.20`: the attacker must demonstrably relearn what it
was given. Of 123 evidence-complete 25-step cells, **11 have `T_rise < 0`**, in two categories:

| mode | `T_before` | signature | n | correct reading |
|---|---|---|---:|---|
| **saturation** | >= 0.90 | small negative rise, `T_after` still high | 1 | the target was never suppressed on this arm, so there is no headroom for a rise; the value is noise at ceiling and says **nothing about the attacker** |
| **destroyed** | < 0.90 | `T_after` -> ~0 **and** post-attack retained capability -> ~0 | 10 | the attacker is outside its useful range and has wrecked the model |

The saturation cell is `u4_k0_s400_b1` attacked on **tench**, a trainability control that was never
erased: `T_before` 1.0000, `T_rise` -0.0417.

The destroyed cells, every one:

| concept | lr | `T_before` | `T_after` | `T_rise` | `rise/T_before` | retain before -> after |
|---|---:|---:|---:|---:|---:|---|
| church | 1e-4 | 0.2083 | 0.1250 | -0.0833 | -0.400 | 0.9792 -> 0.5208 |
| church | 2e-4 | 0.2083 | 0.0833 | -0.1250 | -0.600 | 1.0000 -> 0.0417 |
| church | 2e-4 | 0.2500 | 0.0833 | -0.1667 | -0.667 | 0.9583 -> 0.0000 |
| church | 2e-4 | 0.2083 | 0.0000 | -0.2083 | -1.000 | 0.9792 -> 0.0000 |
| parachute | 5e-5 | 0.7083 | 0.2500 | -0.4583 | -0.647 | 0.7500 -> 0.0625 |
| parachute | 1e-4 | 0.6250 | 0.5000 | -0.1250 | -0.200 | 0.9167 -> 0.2917 |
| parachute | 1e-4 | 0.6250 | 0.2083 | -0.4167 | -0.667 | 0.9583 -> 0.0833 |
| parachute | 2e-4 | 0.6250 | 0.0417 | -0.5833 | -0.933 | 0.9167 -> 0.0000 |
| parachute | 2e-4 | 0.6250 | 0.0000 | -0.6250 | -1.000 | 0.9583 -> 0.0000 |
| parachute | 2e-4 | 0.5417 | 0.2917 | -0.2500 | -0.462 | 0.9792 -> 0.0000 |

**Why this matters.** Reported in isolation a destroyed cell reads as a triumph: on the parachute
`p_r3` arm at 2e-4 the three seeds give H = 0.0000, 0.0000 and 0.2000 with the target panel at
`T_after` ~= 0, which looks like near-perfect erasure that an attack could not undo. It is a model
with **zero** retained capability.

**Binding rule.** A cell is out-of-range only if `T_rise < 0` **and** `T_before < 0.90`, confirmed
by collapsed post-attack retained capability. Such cells are excluded from robustness estimates and
reported as out-of-range attackers. Saturation cells are excluded as uninformative about the
attacker, which is a different exclusion with a different reason.

### 1.1 Withdrawn: "the failure mode is worse on the second concept"

A negative rise is bounded in magnitude by `T_before`, so raw magnitudes are **not comparable
across arms**. Parachute's worst raw value (-0.6250) looked far worse than church's (-0.2083), but
both arms hit `rise/T_before` = -1.000, i.e. both were driven to `T_after` = 0. Normalised, the
concepts are indistinguishable: mean `rise/T_before` **-0.667** on church (n=4) versus **-0.651**
on parachute (n=6). Church simply had less room to fall. **Report `T_after`, or the normalised
ratio — never the raw rise across arms with different `T_before`.**

## 2. The attacker's useful window is arm-dependent, so no single learning rate is valid

Parachute `p_r3`, 25 steps, 3 seeds:

| attacker lr | gate-valid | max-over-valid H | destroyed cells |
|---|---:|---:|---:|
| **1e-5** | 3/3 | **1.000** | 0 |
| 5e-5 | 2/3 | 1.000 | 0 |
| 1e-4 | 1/3 | 1.000 | 2 |
| 2e-4 | 0/3 | — | 3 |

Two independent quantities vary by arm:

- **Where the attacker saturates.** The church teacher needs 5e-5 to reach H = 1.000 and sits at
  0.867 at 1e-5. `p_r3` is already at 1.000 at 1e-5.
- **Where it breaks down.** First destroyed cell at 1e-4 for both church `frp_r3` and parachute
  `p_r3`, but at **5e-5** for parachute `p_r5` — a full rung earlier, *within the same concept*.

So the window must be re-measured per arm, and the sweep must extend **below** a previously-found
optimum as well as above it. A fixed 1e-4 carried over from church would report H = 0.3333 and
0.0000 on parachute from models it had destroyed, as though they were defence. Conversely a fixed
1e-5 understates recovery most on exactly the arms that look best: on church, UTRAP K=0 reads 0.267
at 1e-5 and 0.933 at 5e-5 on the same checkpoint and budget.

**Earlier framing withdrawn:** this was first written as *concept*-dependence. The only arm that
shifts breakdown onset (`p_r5`) differs from `p_r3` within one concept, so *arm*-dependence is the
supported claim.

### 2.1 A cheap predictor was hypothesised and refuted

Hypothesis: the usable learning-rate ceiling tracks the defended model's pre-attack retained
capability (its fragility), which would let one predict the window instead of measuring it.
**Refuted by the existing cells.** `u16_k0_lam0.6_b0`, at retained capability **0.5000** the most
fragile arm measured, has no destroyed cell up to 1e-4, while arms at 0.92-0.96 do. There is no
cheap predictor; the window has to be measured.

## 3. `delta_H` is noise-dominated; report `H_after`

The pre-attack H panel is regenerated per attack seed. `scripts/attack_relearn_v2.py`'s
`scored_panel` seeds each image's generator as
`attack_seed * 1000 + 101 * concept_index + prompt_index`, so changing the attack seed redraws the
whole evaluation panel. On a **bit-identical** checkpoint (`frp_parachute_r5_l05`, verified: one
distinct `provenance.input_unet_sha256` across every cell):

| attack seed | `H_before` | `H_after` | `delta_H` |
|---|---:|---:|---:|
| 0 | **0.2000** | 0.9333 | **+0.7333** |
| 1 | **0.8000** | 0.9333 | **+0.1333** |

Same checkpoint, same attacker learning rate, same budget: `H_before` moves by 0.6000 (9 of 15
images) and `delta_H` by a factor of **5.5**, while `H_after` is **identical**. Within a cell the
before and after panels are *paired* on identical generator seeds, so this is not unpaired-noise
sloppiness — it is that **`H_after` saturates near ceiling on every arm tested, leaving `delta_H`
to inherit all of `H_before`'s variance.**

**Binding consequences.** (a) `delta_H` at n=1 is noise and may not be reported as an effect.
(b) The primary robustness quantity is **max-over-valid `H_after`**. (c) A checkpoint's erasure
depth may not be characterised from one cell — `p_r5` was described in this project's own notes as
"genuinely erased at 0.2000" on the strength of a single seed; **withdrawn**. (d) This is a larger
and different quantity from the previously documented ~2/15 H noise, which was measured on
`H_after` with the attack trajectory held identical. Both stand.

## 4. An absolute post-attack utility drop is confounded with attacker learning rate

Over **95** gate-valid, non-destroyed cells, the change in retained capability from before to after
attack has mean **-0.0268**, median **0.0000**, but **29 of 95 (30.5%)** lose more than 0.05 and the
worst loses **0.7708** (1.0000 -> 0.2292, `U9EV_k0b1_sd6_lr1e-4_s25`). The large losses concentrate
at the top of the useful window (7e-5, 1e-4) and occur on arms containing **no trap mechanism at
all**, including the FRAG/FRP teacher (-0.3958) and plain K=0 baselines.

**Consequence.** A criterion of the form "post-attack utility drops by >= 0.25, therefore the
defence booby-trapped the model" is satisfied by an ordinary attack at a high learning rate against
an ordinary baseline. Such a claim requires a **paired difference against a control arm at matched
attacker learning rate and seed**, measured where the control's own cell is gate-valid and
non-destroyed, at more than one learning rate in the window. This correction was applied to this
project's own pending experiment before its attack phase ran
(`PREREG_S1_AMENDMENT_1_2026-10-27.md` §5).

**Supersedes** the figure "+0.0231 mean across 45 cells, only 3 of 45 lost more than 0.05", which
appears in `MANDATE_v2` §3.3 and §4 and must not be requoted.

## 5. Closed-set accuracy is not an image-quality measure

Previously established and unchanged, restated because findings 1-4 are read alongside it:
`retain8` is 12-way closed-set CLIP accuracy over 8 never-targeted concepts with **no reject
option**. The PI deck's slide 4 shows three methods visibly destroying an unrelated concept while
scoring 0.87-0.97 on it, because a textured blob is still classified as a golf ball. A high H means
"the classifier picked the target", not "the image is a good picture of the target". FID is not
computed; `docs/FID_PROTOCOL.md` records why.

## 6. Reproducibility limit found in an upstream artifact

The pinned Meta-Unlearning repository (`5a2a7600895a2885323cbbf599acaf3f74a96f64`), whose paper is
titled *Preventing Relearning Unlearned Concepts*, **ships no relearning-attack or
attack-evaluation script**: `scripts/` contains `meta.sh`, `esd_meta.sh`, `sdd_meta.sh`,
`esd_baseline.sh`, `sdd_baseline.sh` and `gen_image.sh` — defence training, baselines and
generation only. The three datasets its code loads (`dataset/hrm`, `dataset/irt`, `dataset/tgt`,
`src/meta.py` lines 671-673) are also not shipped.

So the attacker configuration behind the paper's headline claim — learning rate, step count,
corpus — is **not recoverable from the artifact.** This is an observation about the released code,
**not** a claim that the paper's results are wrong. Its practical consequence for us: where an
upstream ships no attacker, "matched attacker effort" cannot be read off the artifact and must be
declared by us and labelled as our reconstruction.

## 7. What these findings are not

- **Not evidence that published results are wrong.** Every finding here is demonstrated on *our*
  pipeline and *our* arms. Findings 1-4 show that a robustness claim measured without these
  guardrails is unsupported, not that any specific paper's claim is false. Establishing the latter
  would require re-running each method's own evaluation, which we have not done.
- **Not a defence.** Nothing here suppresses a concept or resists relearning.
- **Not confirmatory.** Development data throughout; no confirmatory protocol was frozen.
- **One attack family.** Direct fine-tuning on target images with a diffusion loss. No LoRA-based,
  textual-inversion or black-box attacker was tested.
- **Bounded budgets.** 10-300 gradient steps, attacker lr 1e-5 to 2e-4. A stronger or
  better-tuned attacker is not excluded by anything measured here; "worst observed recovery" means
  the maximum over the grid actually run.
