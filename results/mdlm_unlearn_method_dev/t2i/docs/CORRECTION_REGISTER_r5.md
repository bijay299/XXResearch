# Correction register — remediation of the external audit of snapshot `0b435427`

> **Status pointer:** for the **one current verdict per issue**, read `STATUS_CURRENT_v2.md`. `AUDIT_REPAIR_REGISTER_v1.md` is chronological and records hypotheses that were later refuted — two of them ("IMMA composition unproven", "upstream UCE may have written an unedited model") are **refuted** and must not be quoted as live.

**Audited snapshot:** `0b435427557e4fb984cc9acff26e6dacc0613cd2`
**Branch worked against:** `canonwork` → `research/mdlm-unlearn-method-dev`
**Parent commit at start of remediation:** `66ff9d3afec54f446d0750b53f1576a1f10e7ab7`, working tree clean
**Work preserved since the audit:** 49 files changed, +4,643 lines — the cross-model UTRAP /
Meta / UCE runs (`FINDING64`, `ERRATUM_uce_lambda_scale.md`) and the UCE λ-scale correction. **None
of it was reverted.** Every item below says explicitly whether it affects those newer results.

---

## How to read this register

Each row is `issue → evidence → affected → repair → validation → status`. **Status words are used
strictly:**

| status | meaning |
|---|---|
| **CODE FIXED, UNVALIDATED** | the implementation is corrected and unit-tested; **no model has been built or attacked with it.** No scientific claim rests on it. |
| **CLAIM CORRECTED** | a document assertion was wrong and is now corrected, with the original preserved. |
| **VERIFIED CORRECT** | the audit's criticism did not hold up; evidence given. |
| **PENDING EXPERIMENT** | the repair is specified and priced; the run has not been done. |
| **DONE** | fully discharged, with the evidence named. |

**Nothing in this register is marked resolved because a unit test passes.** §C states separately
whether implementation, evaluation, empirical comparison and interpretation are each resolved, and
three of the four are not.

---

## A. Verification of the audit itself

I checked every central finding against our code, the raw output records, the primary papers and
the released reference implementations, rather than accepting them. Results:

| # | audit claim | verdict | evidence |
|---|---|---|---|
| 1 | WSR regulariser has the **opposite direction** | **CONFIRMED** | `erase_wsr.py:161-163` minimises `+λ‖θ−θ_base‖²/‖θ_base‖²`. Reference `vision_unlearner.py:369-373` computes `weight_dist_loss = -sqrt(mean(square(θ_pre−θ))+1e-8)`; paper §6.1: *"We directly aim to maximize the distance"*; §6: *"methods with higher L2 distance exhibit higher robustness"* |
| 2 | Meta objective **omits two defining terms** | **CONFIRMED** | `meta_unlearn.py:122-143` computes `distil_F(θ)+distil_R(θ)−λL_R(θ′)`. Reference `src/meta.py:920-1030` accumulates `+γ11∇L(irt@θ) +γ12∇L(tgt@θ) … −γ21∇L(hrm@θ′) −γ22∇L(tgt@θ′) −γ23∇L(irt@θ′)`. Both `−L_F(θ′)` and `+ζL_R(θ)` are absent from ours |
| 3 | UTRAP's inner ≠ the evaluation attacker | **CONFIRMED on all 7 axes** | inner: `erase_utrap.py:140` SGD rebuilt per step, `:80` one repeated eval prompt, `:55` batch 2, no clip, K=4–16, images from the teacher-loaded pipe (`:75,:88`). Eval: `attack_relearn.py:333` Adam, `:263` base-model generations over 8 `T` prompts, `:335` batch 1, `:360` clip 1.0, 300 steps |
| 4 | New defences reuse an evaluation prompt the checker never sees | **CONFIRMED** | `"a photo of a church"` = `attack_relearn.SPLIT["church"]["H"][0]` = `gen_eval.CONCEPTS["church"][0]`, and appears at `erase_gift.py:119`, `erase_imma.py:86`, `erase_wsr.py:77`, `erase_utrap.py:80`. `verify_splits.py:46-56` imports only `gen_eval`, `attack_relearn`, `erase_uce`, `erase_esd`, `undo_t2i` |
| 5 | "No released code" is false | **CONFIRMED for all three** | IMMA `@2f55041`: `environment.yaml`, `train/defend_text_to_image_lora.py`, `eval/`, README commands **targeting `CompVis/stable-diffusion-v1-4` — our exact model**. Meta-Unlearning `@5a2a760`: `src/meta.py`, 1,087 lines, downloaded and read. From Dormant to Deleted `@7b9a1cc`: `vision_unlearner.py`, 1,935 lines, downloaded and read |
| 6 | FRP described as retain-only | **CONFIRMED** | paper Appendix B.2 "Three relearning attacks": retain-only, forget-only, forget+retain; Appendix C.3 varies optimiser (AdamW/SGD/SGD-momentum/Adafactor/Adagrad), lr 5e-6–1e-4, 1–10 epochs |
| 7 | Stale counts / pooled-arm statistics in r4 | **CONFIRMED, with one exception below** | recomputed from records — see §B.7 |

### A.1 One audit finding that does **not** hold, with evidence

**The audit says `FINDING59`'s "22 valid attacks" on `imma_church_frp_a1e-6_s1500` is really 21**
("six full-UNet, seven LoRA, seven TI, one DreamBooth"). Recomputed from the raw records:

| family | budget | n | gate-valid |
|---|---:|---:|---:|
| full-UNet | 300 | 6 | 6 |
| LoRA | 300 | 5 | 5 |
| LoRA | 900 | 2 | 2 |
| TI | 2000 | 2 | 2 |
| TI | 4000 | 3 | 3 |
| TI | 6000 | 3 | 3 |
| DreamBooth | 300 | 1 | 1 |
| **total** | | **22** | **22** |

TI is **8**, not 7. **`FINDING59`'s count of 22 is correct and is not changed.** The audit
undercounted one TI run. Everything else it says about `FINDING59` — the build pooling and the
0.8667 attribution — **is** correct and is fixed in §B.7.

### A.2 Two leaks the audit did **not** find, which my rebuilt checker did

`scripts/split_registry.py` attributes every overlapping prompt to its source script. Besides the
four `"a photo of a X"` leaks the audit identified, it reports:

1. **`erase_adv` trains on six exact UTILITY-PANEL prompts** — `"a photo of a cassette player"`,
   `"a photo of a chain saw"`, `"a photo of a french horn"`, `"a photo of a gas pump"`,
   `"a photo of a golf ball"`, `"a painting in the style of monet"`. So **AdvUnlearn arms'
   *utility* numbers are partly in-sample**, a leak on the retain side that nothing had flagged.
2. **`undo_t2i` shares six prompts with the ATTACKER's `T` pool** — `"a church"`,
   `"an old stone church"`, `"a parachute"`, `"a garbage truck"`, `"a tench"`,
   `"a van gogh painting"`. For undo/damage composite arms the defence trained on strings the
   attacker is given, which weakens the T-rise gate's interpretation for those arms specifically.

Both are now in `outputs/split_registry_v2.json` and in §B.5's status.

### A.3 A third problem found while rebuilding the aggregates: **vacuous attack records**

The registry flags attacks on defence builds that **erased nothing**, where `H_after` measures the
base model's ability to draw the concept rather than a defence failing:

| build | as-is target | as-is retain8 | attacks run | all reaching |
|---|---:|---:|---:|---:|
| `gift_church_b0.1` | **1.0000** | 1.0000 | **6** | H = 1.000 |
| `uce_sd15_church_lam10000` | 1.0000 | 1.0000 | 1 (killed) | H = 1.000 |
| `uce_sd15_church_lam50000` | 1.0000 | 1.0000 | 2 | gate-INVALID (T rise 0.000) |

The six `gift_church_b0.1` attacks were counted inside the "GIFT fully reversed" aggregate. They
should not be: there was no erasure to reverse. They are now flagged `vacuous_no_erasure` and
excluded from scientific aggregates, which **reduces** our headline attack count. The λ=50000 pair
shows the validity gate catching the same pathology on its own — T could not rise because it was
already 1.000.

---

## B. The register

### B.1 Implementation error — weight-space regulariser direction (audit §2)

| field | content |
|---|---|
| **issue** | `erase_wsr.py` minimises a **positive** normalised squared distance to `θ_base`, pulling the model **toward** the pretrained weights, while describing this as the cited paper's mechanism. The paper and its released code **maximise** that distance. |
| **evidence** | `erase_wsr.py:6` (stated mechanism), `:17` (objective), `:161-163` (the implemented step). Reference: `vision_unlearner.py:366-373`. Paper §6, §6.1. Both fetched and read at the pinned commit. |
| **affected** | `FINDING60` §3–4 (the predictor's *direction* of prediction), `FINDING61`, `FINDING62`, and `erase_wsr.py`'s docstring. **Not affected:** the measured numbers themselves, which are valid observations about anchoring-to-base checkpoints. **Newer results:** none — no cross-model WSR arm exists. |
| **repair** | (a) `ERRATUM_wsr_direction.md` written. (b) The existing arm is **renamed** "WSR-anchor (ours)" throughout the generated tables and is **kept**, not deleted — `erase_wsr.py` is untouched so `FINDING61/62` reproduce bit-for-bit. (c) `scripts/objectives.py` holds **both** terms as separate named functions: `anchor_penalty` (ours) and `wdr_forget_loss` (faithful). (d) `scripts/erase_wdr_v2.py` implements the faithful adaptation, reconciling **eight** axes, not just the sign: direction, RMS-not-sum normalisation, no `‖θ_base‖` division, the distance term **replacing** the forget objective rather than being added to one, unscaled retain coefficient, alternating forget/retain iterations, all-parameter scope, Adam with no clipping. |
| **validation** | 6 deterministic tests, all passing: `t_wdr_increases_distance`, `t_anchor_decreases_distance`, `t_wdr_gradient_is_antiparallel_to_anchor` (cos < −0.99), `t_wdr_matches_reference_formula` (independent transcription, atol 1e-12), `t_wdr_is_not_sum_of_squares`, `t_wdr_eps_guard_at_zero_distance`. |
| **status** | **CODE FIXED, UNVALIDATED** + **CLAIM CORRECTED**. The claim that our experiments discharged this published method is **withdrawn**. No arm has been built with `erase_wdr_v2.py`. |

### B.2 Implementation error — Meta-Unlearning objective (audit §3)

| field | content |
|---|---|
| **issue** | Our port's meta term is `−λL_R(θ′)` only. Eq. 8 is `−L_F(θ′) − ζ[L_R(θ′) − L_R(θ)]`. Two of three components are absent, and the at-θ distillation terms are not algebraically equivalent to either. |
| **evidence** | `meta_unlearn.py:122-143`; reference `src/meta.py:920-1030`, whose `sum_grads` carries the full signed structure with the paper's `R` split into the reference's `irt`+`tgt` pools and `F` = `hrm`. |
| **affected** | `FINDING56`'s **causal** reading ("the trap fires on normal users"); `PREREG_H80`'s objective statement; `FINDING58`'s use of GIFT as an ablation. **Newer results affected:** `meta_sd15_church_lam0.5` and `meta_bksdm_church_lam*` in `FINDING64` were built with the **same** variant-A objective, so they are variant-A measurements too and are relabelled accordingly — they are **not** invalidated, and they are **not** evidence about the published method. |
| **repair** | (a) `ERRATUM_meta_objective.md`. (b) `meta_unlearn.py` preserved untouched as **variant A**. (c) `scripts/meta_unlearn_v2.py` implements full Eq. 8 and additionally repairs: independent `--meta-scope`/`--attack-scope` (the reference uses a selectable scope for the meta gradient and full-UNet for the simulated attack; variant A used full for both and recorded neither); `--target-source base` with a **CLIP content check and `--require-target-acc` refusal** (variant A generated target images through the already-erased teacher and never verified them); benign pool drawn through the split registry with its role declared; `--order` separating first-order from differentiated unroll, with `differentiated` raising `SystemExit` rather than being silently claimed. |
| **validation** | 4 tests, all passing: `t_meta_has_all_three_terms` (∂/∂L_F(θ′) = −1, ∂/∂L_R(θ′) = −ζ, ∂/∂L_R(θ) = **+ζ**), `t_meta_value_matches_equation`, `t_variantA_is_missing_two_terms` (asserts variant A has **zero** gradient w.r.t. both missing terms), `t_meta_reduces_to_variantA_only_if_terms_dropped`. |
| **status** | **CODE FIXED, UNVALIDATED** + **CLAIM CORRECTED**. `FINDING56`'s observation (variant A's benign damage at λ ≥ 0.5) stands; its attribution to the published method's self-destruct mechanism is **withdrawn** — premature benign damage is consistent with variant A's own altered objective. |

### B.3 Incorrect literature description — GIFT is not an ablation of Meta-Unlearning (audit §3)

| field | content |
|---|---|
| **issue** | `FINDING58` used GIFT's failure to rule out the dropped meta-gradient term as the explanation of `FINDING56`. Two different algorithms failing does not isolate one missing term. Separately, `FINDING56` §5a calls GIFT a matched comparator "also editing cross-attention" — but our Meta port optimises the **full UNet** (`meta_unlearn.py:107,109`), not cross-attention. |
| **evidence** | `FINDING58` (inference), `FINDING56:203` (the cross-attention claim), `meta_unlearn.py:107` (`Adam(unet.parameters())`). |
| **affected** | `FINDING56` §5a, `FINDING58`, `PREREG_H81`. |
| **repair** | `ERRATUM_meta_objective.md` §4 withdraws the ablation inference and corrects the scope description. The matched-retain GIFT/TI comparison is **retained as limited evidence** — equal scalar retain accuracy does not match target geometry, conditioning quality or per-concept damage — and the surviving factual observation is stated narrowly: five TI runs on the λ=0.5 variant-A arm, all gate-failing, max ungated H = 0.0667. |
| **validation** | documentary; no code. |
| **status** | **CLAIM CORRECTED.** A controlled within-method ablation (variant A vs v2 at matched retain) is listed in §B.10 as **PENDING EXPERIMENT**. |

### B.4 Implementation error — UTRAP's attacker correspondence (audit §4)

| field | content |
|---|---|
| **issue** | The inner process was called "the ACTUAL full-UNet relearning attack (the one attack_relearn.py runs)" in both code and preregistration while differing on seven axes at once. |
| **evidence** | the seven-axis table in §A, row 3. |
| **affected** | `FINDING63` §2 ("both structural causes were removed"), §3a(a) ("unroll depth was not the missing ingredient"), `PREREG_H84`'s description of its own inner attacker. **Newer results affected:** `utrap_sd15_church_l0.15k8` and `utrap_bksdm_church_l0.15k8` (`FINDING64`) were built with the **same** `erase_utrap.py`, so the identical mismatch applies to them. Their reversal measurements stand; the unroll-depth inference does not. |
| **repair** | (a) `scripts/attack_core.py` — **one** implementation of the relearning update, with `AttackSpec` (optimizer, lr, steps, batch, clipping, scope, precision, persistent optimiser state), `AttackData` (latents, embeddings, prompts, **mandatory `provenance`**, **mandatory `split_role`**), and `divergence_from_evaluation_attacker` computed automatically against the frozen `EVAL_FULL_UNET` reference. `horizon_role` has **no default** and `EVALUATION_REFERENCE` is **refused** for any horizon ≠ 300. (b) `scripts/erase_utrap_v2.py` consumes it, with defaults that now **match** the evaluator (Adam, batch 1, clip 1.0, persistent state, base-model target images) and a `DEF_TRAIN` target pool disjoint from attacker and evaluation prompts. (c) It logs a **fixed-noise target probe immediately before and after each unroll**, so "the inner attacker actually adapts" is measured rather than assumed, plus per-step update norms, grad norms and inner loss trace. (d) `PREREG_H90_utrap_v2.md` preregisters the K=0/K=1/K=8/K=32 comparison with matched compute. |
| **validation** | 10 tests, all passing, including `t_attack_core_runs_and_reduces_loss` (the attacker must lower a fixed-noise probe on a tractable 6-dim model), `t_attack_core_k0_is_a_noop` (K=0 moves nothing, so the control is well defined), `t_adam_and_sgd_are_not_interchangeable` (endpoints separated by ≫1e-3 at the same lr — the audit's point, made measurable), `t_snapshot_utrap_inner_is_flagged_as_divergent` (the snapshot's spec must report **every** divergent axis), `t_utrap_outer_gradient_sign_on_tractable_example` (applied gradient equals `∇L_retain − λ∇L_target(θ_K)` and θ is restored after the unroll), `t_horizon_role_is_mandatory`, `t_short_horizon_cannot_claim_eval_budget`, `t_data_provenance_is_mandatory`, `t_split_role_must_be_declared`, `t_eval_spec_is_the_reference`. |
| **status** | **CODE FIXED, UNVALIDATED** + **CLAIM CORRECTED**. The 18 SD-1.4 and 6 cross-model UTRAP reversal measurements are **preserved unchanged**. Withdrawn: that they ruled out unroll depth. The surrogate whose horizon was varied was not the evaluation attacker, so the sweep did not test the hypothesis it was designed to test. |

### B.5 Evaluation/provenance problem — split leakage and RNG streams (audit §5)

| field | content |
|---|---|
| **issue** | One of three evaluation target prompts is also a defence data-generation prompt for four defences, and `verify_splits.py`'s CLEAN verdict could not cover them. Plus the two further leaks in §A.2. Separately: no DEV/FINAL division existed as separate evaluation paths, and attack-training variation was not crossed independently of evaluation variation. |
| **evidence** | `outputs/split_registry_v2.json`, reproduced below. |
| **repair** | (a) `scripts/split_registry.py` (**v2**) reads prompt **templates out of script source text** — not just importable module-level pools, which is exactly how v1 missed f-string pools built inside `main()`. It **hard-fails** if any `erase_*.py` on disk is unregistered, so a new defence cannot escape the check. Normalisation now strips punctuation as well as case/whitespace. (b) `FINAL` set frozen at `2026-10-25T23:30-04:00`, 3 prompts × 4 concepts, **verified disjoint** from DEF_TRAIN, ATK_TRAIN and DEV by computation, and not yet used for anything. (c) The historical H prompts are recorded as **`DEV` / `USED_FOR_SELECTION`**, not relabelled untouched, with the reason stated: λ, β and K were repeatedly chosen while watching H, which makes H a development set even where no string overlaps. (d) `RNG_STREAMS` + `seed_for()` give `defence_build`, `attack_train`, `data_generation`, `dev_eval`, `final_eval` disjoint offsets, so evaluation seeds are **common across comparator arms** while training variation stays independent. v2 scripts use them. |
| **validation** | 4 tests, all passing — including `t_split_registry_detects_the_known_leak`, which **asserts the leak IS found and attributed to all four scripts** (a registry reporting CLEAN here would be the bug), and `t_final_split_is_disjoint_from_everything`. |
| **status** | **DONE** for detection, registry, FINAL freeze and RNG separation. **PENDING EXPERIMENT** for re-evaluating any arm on FINAL. **CLAIM CORRECTED:** every pre-attack suppression, model-selection and "excellent operating point" figure for GIFT/IMMA/WSR/UTRAP is a **development** measurement with one of three evaluation prompts in-sample, and is relabelled as such. `erase_adv` **utility** figures are partly in-sample. |

### B.6 Incorrect literature description — baseline availability and threat models (audit §6)

| field | content |
|---|---|
| **issue** | "Zero of the five competing defences have usable released code" is false. FRP's threat model is misdescribed as retain-only. |
| **evidence** | §A rows 5–6; all three repositories downloaded at their pinned commits; FRP paper Appendices B.2 and C.3. |
| **affected** | `ERRATUM_r2` C19, `AUDIT_CLOSURE_r2` (lines 37, 51, 199, 224), `FINDING43:106`, `FINDING59:70`, `PREREG_H81:6`, `T2I_REPORT_r4`. |
| **repair** | `T2I_BASELINES_v2.md` — a **per-method table separating four distinct questions**: released code exists / environment compatibility / missing assets / domain adaptation needed. `ERRATUM_baselines_and_threat_models.md` carries the FRP threat-model correction and keeps the AdvUnlearn prompt-space-vs-weight-access distinction intact. |
| **status** | **CLAIM CORRECTED.** Actually *running* the three released implementations is **PENDING EXPERIMENT**; "unusable" is not claimed again without a recorded execution failure or dependency error. |

### B.7 Numerical/reporting mistakes (audit §7)

All recomputed from raw records via `scripts/registry_v1.py`. **Generated tables:**
`docs/TABLES_v1_generated.md`. **Registry:** `outputs/registry_v1.json`.

| claim | verdict | corrected value |
|---|---|---|
| `FINDING59`: 0.8667 LoRA max on `a1e-6_s1500` | **wrong** | that build's 300-step LoRA max is **0.8000** (values 0.2667, 0.5333, 0.6000, 0.6667, 0.8000). **0.8667 belongs to `a1e-6_s300`**, whose retain8 is 0.9417, not 0.8917 |
| `FINDING59`: eight full-UNet attacks on one arm | **wrong** | **6** on `s1500`, **2** on `s300` |
| `FINDING59`: 22 valid attacks on `s1500` | **CORRECT** | 22 confirmed; the audit's 21 undercounts TI by one (see §A.1) |
| `FINDING60`: GIFT β=1.0 "reversed on 22 valid attacks" | **wrong** | that build has **12** records across 4 families, all 12 gate-valid. The 22 belongs to `imma…s1500` |
| r4 GIFT row: β∈{1.0,1.2,1.3,1.35} target range starts 0.0667 | **wrong** | actual **0.2000–0.4667** (0.4667, 0.3333, 0.2667, 0.2000) |
| r4 Meta row: target 0 throughout | **wrong** | λ=0.5 is **0.0667** |
| r4 Meta row: retain range omits λ=0.6 | **wrong** | λ=0.6 retain8 = **0.6250** |
| r4 counts: 609 attacks / 1,088 JSONs | **stale** | snapshot had 618/1,122; **current branch has 637 attack records / 1,191 JSON files**, of which **631 admissible for recovery** (6 vacuous excluded) and only **268 admissible for paired utility** (136 VOID + unmeasured) |
| r4: only FRP rerun cross-model | **stale at the snapshot** | `FINDING64` already had GIFT; the current branch also has **UTRAP, Meta variant A and UCE** |
| `FINDING63`: "first arm ever to meet retain ≥ 0.85" | **wrong as worded** | earlier FRP (0.9417) and IMMA (0.8917, 0.9417) met retain ≥ 0.85. The **conjunctive** claim — target 0.0000 **and** retain8 ≥ 0.85 — is **correct**: `utrap_church_l0.15k8` is the only SD-1.4 arm satisfying both, and the two later ones (`utrap_sd15…` 0.9583, `uce_bksdm_church` 0.9333) came after it |
| `FINDING63`: larger λ "can only be worse" | **wrong** | non-monotone optimisation does not license extrapolation from one setting. Withdrawn |
| `FINDING61/62`: every MSE term "bounded" | **wrong** | MSE is bounded **below**, not above; lower-boundedness does not imply optimiser convergence |
| "13 defences" | **unmapped** | `TABLES_v1_generated.md` §1–2 now enumerates every **build** with its family, base model and operating point. The prose count is replaced by that table |
| 618 records = 618 admissible experiments | **wrong** | the registry separates total / quarantined / VOID-utility / vacuous / admissible, and splits by budget so fixed-grid confirmatory runs are never pooled with adaptive escalation |

**Status: DONE** for the registry and generated tables; **CLAIM CORRECTED** for each row. Report r5
carries the corrected figures; r4 keeps its text under a SUPERSEDED banner.

### B.8 Unsupported interpretations (audit §8)

Each of these is rewritten in `ERRATUM_interpretations_r5.md` and in report r5:

| overreach | replacement |
|---|---|
| "erasure depth / weight-space geometry / unroll depth do not predict resistance" | **"the measured differences did not prevent reaching the ceiling under the attacks run."** A saturated H = 1 endpoint cannot distinguish zero predictive value from predictive value invisible at that budget. Equal endpoints coexist with different recovery rates and minimum sufficient budgets — neither of which we measured |
| re-zeroing shows the pruned coordinates are dispensable | **final-coordinate** dispensability for the **final evaluated model on this panel** only. Not dispensability during the attack trajectory, not localisation to surviving feed-forward weights (the rest of the network was present and trainable throughout) |
| one random matched-count mask shows any such mask is harmless | one seed, one concept, count-matched per tensor but not matched on magnitude or other correlated properties. **No claim about "every 3% mask."** Repeated count- **and** magnitude-matched controls are pending |
| BK-SDM recovery rules out capacity | both models may be above a capacity threshold; pruning also changes architecture, weights **and** starting erasure depth. A two-point observational contrast cannot isolate parameter count |
| H = 1 means the model was restored / H = 0 means the capability is absent | H is **15 images, 3 prompts, 5 each**, closed-set 12-label CLIP argmax, **no reject option**. Quantised to 1/15. An exact binomial interval for 15/15 has lower endpoint ≈ 0.782 — and even that model is inadequate here given prompt clustering, reused seeds and adaptive selection |
| "the arm **destroys the model**" / "broad model destruction" / the erasure is **clean** | **"Retain accuracy collapsed to N/120 under the closed-set scorer."** A closed-set 12-way argmax with no reject option scores a model whose outputs become generically painterly as *perfectly* generating `van_gogh` and *zero* on everything else — indistinguishable from a model that generates nothing (`D-030`). Symmetrically, 0/15 target hits cannot distinguish erasure from degradation, and R4c/R4d measured **no retain axis at all** (`D-034`). **Any claim of broad destruction or of clean erasure requires saved images and an independent quality/semantic assessment.** Neither exists for any arm in this corpus |
| target-image-assisted recovery shows suppression not removal | **does not distinguish** residual-memory reactivation from learning again from the supplied images. This is `ERRATUM_r2` C17, which r4 §7 had re-breached; r5 does not |
| WSR2's utility gain is a frontier improvement | **descriptive, selected-configuration** result: three builds, retain8 0.9500/0.9750/0.9667 (mean 0.9639, sample SD 0.0127); an illustrative one-sample 95% t interval [0.9323, 0.9955] **contains** the FRP teacher's 0.9417. "Statistically indistinguishable erasure" is withdrawn — that needs a prespecified equivalence margin, and failure to show a difference is not equivalence |
| "no valid attack" = the defence fails | an ineffective attack is **inconclusive**. Preventing the attacker from fitting T can itself be defensive. Gated **and** ungated outcomes are now both reported in the generated tables |
| CLIP panel + class accuracy = image quality preserved | FID remains **pending**; prompt adherence and closed-set accuracy do not establish general quality or diversity |

**Status: CLAIM CORRECTED.** The persistent-mask intervention is specified in §B.10 as
**PENDING EXPERIMENT**.

#### B.8a A reject option is a POTENTIAL ADDITIONAL DIAGNOSTIC, not a repair

`AUDIT_REPAIR_REGISTER_v1.md` §5's G13 names "a reject option or open-set margin" among the evidence
it wants, and `D-030` calls a reject-option re-score a "decisive test". **Neither makes a reject
option a fix for anything.** Recording the correct status of the idea, because treating it as a repair
would be a worse error than the one it is meant to address:

- It is a **potential additional diagnostic**. It would tell us something the closed-set argmax
  cannot — whether a collapsed arm's images are off-distribution rather than confidently
  mis-labelled. It does not correct, replace or rehabilitate any existing score.
- If introduced, it must be **calibrated independently of the arms being judged.** Calibrating a
  threshold on the same collapsed arms whose interpretation is in question is circular.
- It must be **versioned** and **documented as to its assumptions** — what distribution the
  rejection threshold is fitted on, what "unusual" means operationally, and what it would do to a
  known-good base model.
- It must **preserve the existing closed-set scores.** Every K1/K2 gating decision in this project is
  defined on `retain8` under the closed-set scorer and was applied consistently; those decisions
  stand and are not reopened by adding a second metric alongside them.
- **No threshold may be tuned to favour a result.** A reject option whose cut-point is selected after
  seeing which arms it rescues is not a measurement.

Until such a diagnostic exists and is calibrated that way, the prose constraint in the table above is
the whole of the remedy: say **"retain accuracy collapsed to N/120 under the closed-set scorer"**, and
make no claim of broad destruction or of clean erasure without saved images and an independent
quality/semantic assessment.

### B.9 Preregistration integrity (audit §9)

Independent timing evidence was sought for `PREREG_H80` and `PREREG_H85`, which first appear in
commits that also contain results (`6caec48e`, `ed1299f0`). **I found none** — no timestamped plan
hash, no job log predating the commit, no separate artifact. Both are therefore labelled
**"precedence unverified"** in `PREREG_PRECEDENCE.md`, with the four preregistrations whose plan
commits **do** precede their checked result commits (H66, H81, H82, H84) labelled verified.

**No preregistration has been edited.** H80's unmet "both concepts before any claim" requirement and
H85's three-point deviation from the six-point grid are recorded as **deviations in a separate
file**, not by rewriting the plans. New protocols (`PREREG_H90`) record stopping rules and primary
settings before outcomes.

**Status: DONE** (labelled, not rewritten).

### B.10 Experiments that remain necessary — configuration, cost, blocker

> **⚠ Corrected 2026-10-26.** The line that stood here — *"None of these has been run"* — was true
> when written and is now **false for five of the ten**. Verified from artifacts:
>
> | id | state | evidence |
> |---|---|---|
> | **E1** | ✅ **RAN**, but at α ∈ {1.0, 4000}, **not** the preregistered {0.3, 1.0, 3.0} | `logs/e1_wdr.log`, `logs/e1b_wdr.log`, `outputs/gen_wdr2_church_a{1.0,4000}.json`; α=4000 fails K1 at retain8 **0.5917** (`FINDING67`, as corrected by `D-017`) |
> | **E2** | ✅ **RAN** as specified, λ ∈ {0.5, 1.0}, plus an 18-cell frozen grid (E2g) and an attacker-lr sweep (E2h) | `logs/e2_meta_v2.log`, `e2g_grid.log`, `e2h_stronger.log`; `FINDING78` |
> | **E4** | ✅ **RAN** as specified, K ∈ {0,1,8,32} at matched compute | `logs/e4_h90.log`, `e4m_matched.log`, `e4mg_grid.log`, 8 `utrap2*` builds; `FINDING76` |
> | **E6** | ⚠ **RAN, CONSUMED THE SPLIT, AND MUST BE RE-SCORED** | `outputs/final_*.json` (18 records); `FINDING65`, `FINDING66`. All 18 were scored with `gen_eval.LAB_KEY` instead of `LABELS` (**`D-022`** below); nine of the twelve label strings differ. **The evaluator used inconsistent label text; the effect on scores and conclusions is unresolved pending corrected evaluation** — for both findings. A **fresh** final set must be frozen before any new configuration is selected, **and the existing records re-scored** — CPU-only, on retained artifacts |
> | **E7** | ⏸ **PARTIALLY RAN** — upstream IMMA (R1, 5000/5000 + eval) and upstream UCE (R4b–d) executed; `vision_relearning` (R3) **not started** | `logs/r1b_imma_train.log`, `logs/r1c_eval.log`, `outputs/r4{b,c,d}_uce_*.json` |
> | E3, E5, E8, E9, E10 | ❌ **still unrun**, blockers unchanged | — |
>
> **None of E7's completed halves is a reproduction.** R1 and R4 ran upstream code but used **our**
> training images, **our** `gen_eval`, **our** 12-label closed-set CLIP scorer and **our** prompt
> pools. So this register's repeated framing — that E7 would "replace *we ported it* with *we
> reproduced it*" — is **not** discharged by them; see `UPSTREAM_FIDELITY_MANIFEST.md` §6, which now
> defines three fidelity tiers instead of two.

Each is specified precisely enough to execute.

| # | experiment | why it is needed | configuration | est. cost | blocker |
|---|---|---|---|---|---|
| E1 | Build + frozen-grid the faithful WDR arm | `erase_wdr_v2.py` is unvalidated; without it the published WSR method remains **untested**, not discharged | `--alpha` ∈ {0.3, 1.0, 3.0}, `--max-rms` 0.02, `--teacher frp_church_r3_l05`, 1500 alternating steps, then 3 attacks @lr1e-5 per arm clearing K1 | ~3 × 1.1 h build + 3 × 0.9 h attack ≈ **6 GPU-h** | none — within authorisation |
| E2 | Build + frozen-grid Meta v2 (full Eq. 8) | decides whether `FINDING56`'s benign damage is variant A's missing terms or the method | `--zeta 1.0 --lam-target 1.0 --meta-scope xattn --attack-scope full --target-source base`, λ sweep {0.5, 1.0}, 1500 steps | ~2 × 2.4 h + attacks ≈ **8 GPU-h** | none |
| E3 | Variant A vs v2 at **matched retain8** | the controlled within-method ablation `FINDING58` lacked | pick operating points with retain8 within ±0.02, run the same TI configuration on both | **~4 GPU-h** | needs E2 first |
| E4 | UTRAP v2: K ∈ {0, 1, 8, 32} at **matched total compute** | the only test of the unroll-depth hypothesis using the actual evaluation attacker | `erase_utrap_v2.py`, λ=0.15, outer steps scaled so Σ(outer × (1+K)) is constant, 4 arms + 3 attacks each | ~**14 GPU-h** | none; preregistered in `PREREG_H90` |
| E5 | Persistent-mask attack | distinguishes transient use of pruned coordinates from final dispensability | clamp the original FRP mask to zero **after every attack step** for all 300 steps; compare with final-only re-zeroing; **repeated** count-matched and magnitude-matched random controls, ≥3 seeds each | ~**9 GPU-h** | none |
| E6 | Re-evaluate the leading arms on the frozen **FINAL** prompts | every current target number is a development measurement | `gen_eval` on `split_registry.FINAL_PROMPTS`, the 4–6 arms that clear K1 | ~**2 GPU-h** | needs a `gen_eval` flag to accept the FINAL pool — small code change, not yet made |
| E7 | Run the three **released** implementations | the only way to replace "we ported it" with "we reproduced it", and to substantiate or retract "unusable" | IMMA `@2f55041` on SD-1.4 (its own target model); Meta-Unlearning `@5a2a760`; `vision_relearning @7b9a1cc` for a classifier-domain sanity check | **unknown** until dependency resolution attempted; IMMA's `environment.yaml` is the cheapest first step | separate conda envs; must not disturb `venv_t2i` |
| E8 | FID | r4's "general quality preserved" is unsupported | 5,000 images, fixed captions/seeds, COCO val2014 reference; **one** shared pre-attack state reused across six attacks ⇒ 35,000 images (40,000 with a fresh base anchor), not 7 × 5,000 | `pip install torchvision scipy` + **~6 GB** COCO download + ~**10 GPU-h** | needs the download and the installs; neither is authorised yet |
| E9 | SD-2.x / SDXL cross-model | the cross-model claim covers SD-1.x and one pruned derivative only | same frozen grid | ~**8 GPU-h** | **BLOCKED: `stabilityai/*` returns HTTP 401. Needs a Hugging Face token.** |
| E10 | A never-trained oracle | `ERRATUM_r2` C17's identification limit; no attack strength can supply it | a tractable model trained from scratch with the concept absent, or narrow the paper to the operational relearning threat | **unscoped** | design decision, not compute |

---

## C. What is resolved, stated per axis

The audit's closing instruction was not to call everything fixed because tests pass. So:

**1. Implementation — SUBSTANTIALLY RESOLVED, AND NOW VALIDATED ON MODELS.**
The three wrong objectives are now implemented correctly and unit-tested (24/24 passing), the
attacker's definition is shared through one module that refuses to let a surrogate claim to be the
evaluation attack, and the split checker covers every defence and fails closed on new ones.

**Corrected 2026-10-26:** the line that stood here — *"Not one corrected implementation has produced
a model"* — is **false**. All three have: WDR v2 (E1, α ∈ {1.0, 4000}), Meta v2 (E2, λ ∈ {0.5, 1.0})
and UTRAP v2 (E4, K ∈ {0,1,8,32}). The tests still show only that the code computes what the
references specify; what the *runs* added is in §B.10 above.

**Running them found two further implementation defects**, which is the argument for validating on
models rather than on tests alone:

- **`D-016`** — `erase_wdr_v2.py` **alternates** the forget and retain steps while its docstring
  claims it does so "as the reference does". The reference **sums** both gradients into one
  `opt.step()` (`vision_unlearner.py:411,637-682`). The docstring is inverted.
- **`D-017`** — because both steps share **one Adam instance**, the second-moment buffer is
  retain-dominated and α is inert at **both** ends of its range. `FINDING67`'s "~4,000×" is an
  artifact (measured ≈**9.6×**), its 1/√N *mechanism* is withdrawn as the cause, its "third instance
  of one scale pathology" framing no longer holds for the WDR member, and `wdr2_church_a4000`'s
  retain8 0.5917 measures **our two-step shared-Adam schedule at saturation**, not "WDR at gradient
  parity". The derivation ‖∇L‖ = α/√N itself stands.

**2. Evaluation and provenance — PARTIALLY RESOLVED.**
Resolved: the leak is detected and attributed; two further leaks found; DEV/FINAL separated with
FINAL frozen before inspection; RNG streams separated; the registry reports gated **and** ungated
outcomes, all three utility denominators, and admissibility flags; checkpoint identity is a sha256
rather than a path. Unresolved: **no arm has been evaluated on FINAL**, so every target number in
the corpus remains a development measurement; the CLIP classifier revision is still unpinned; H
remains a 15-image closed-set measure; per-image predictions are retained but not published for
every historical run.

**3. Empirical comparison — PARTIALLY RESOLVED; still no comparative claim about the published
methods.**

**Corrected 2026-10-26.** All three sentences that stood here were false: corrected implementations
**have** been run (E1, E2, E4); two of the three released references **have** been executed
(upstream IMMA via R1, upstream UCE via R4b–d; `vision_relearning`/R3 has not); and the unroll-depth
question **has** been tested at matched compute with the evaluation attacker (`FINDING76`: flat —
though its §3 probe-mechanism check is overstated, see its §7).

The **verdict is unchanged**, but now rests on a stronger reason rather than on absence of
measurement: **no comparative claim about the published baselines is supported**, because R1 and R4
ran upstream code against **our** training data, **our** scorer and **our** prompt pools. That is
`UPSTREAM_FIDELITY_MANIFEST.md` §6's middle tier — *upstream code, our benchmark* — and it is not a
reproduction of any paper's reported experiment. The claim that all five baselines were discharged
stays withdrawn.

What survives is narrower and still substantial: on the builds actually measured, with the attacks
actually run, every arm that cleared the utility floor was reversed to H = 1.000 — excluding the six
vacuous `gift_church_b0.1` records. `FINDING74` extended this across **six** defence families.

**4. Scientific interpretation — CORRECTED IN TEXT, STRUCTURALLY LIMITED.**
Every saturated-endpoint, mechanism and capacity claim is rewritten to the weaker statement the
data supports. The structural limits remain and are **not** repaired by wording: a ceiling cannot
rank predictors; target-assisted recovery cannot separate reactivation from relearning; a two-point
model contrast cannot isolate capacity. E5, E6 and E10 are the experiments that would move these,
and they have not been run.

**The mandate's bar is unmet, and neither this remediation nor the campaigns since change that.**
No arm reaches retain8 ≥ 0.85 with max-over-valid H ≤ 0.40, **at any budget**.

**This now rests on measurement, and it was a near miss.** `meta2_church_lt0.5` (E2, full Eq. 8)
reaches retain8 **0.9000** at as-is target **0.0000**. At a 25-step budget it clears the robustness
half at attacker lr 1e-5 (max-over-valid **0.0667**) *and* lr 5e-5 (**0.2000**), failing only at
lr 1e-4 (**0.5333**, `logs/e2h_stronger.log`); since max-over-valid is a ratchet (`FINDING35` S4),
the budget fails the bar. `FINDING78`'s companion claim to have produced "the first arm to meet both
mandate halves at any budget" is **retracted** — registered as **`D-008`**, and the same failure
mode as `FINDING63`'s "first arm ever" wording in §B.7 above: a conjunctive superlative published
before its sweep was complete. Its retention result survives and is real: **+0.20 to +0.27** retain8
at identical suppression against variant A's 0.6250 (λ=0.6) and 0.7000 (λ=0.7), attributable to
Eq. 8's **pairing** rather than to the weight of ζ, which is near-inert and has never been swept.

What the register changes is the *standing* of the negative result: it is now a statement about
**our implementations plus two upstream-code runs scored on our own benchmark**, not about the
published methods. **R3 is the one declared cell that would move this**, and it has not run.

---

## D. Change log — 2026-10-26 (Agent 6, provenance audit)

This register was written at the end of the audit remediation, before E1/E2/E4/E6/R1/R4 ran. Its
**claim-level corrections (§A, §B.1–B.9) were re-verified and stand**; its **status statements
(§B.10, §C) had gone stale** and are corrected. Full traceability in
`AGENT6_PROVENANCE_FINDINGS.md`.

| § | what changed | why |
|---|---|---|
| §B.10 | "**None of these has been run**" → per-experiment state table; **five of ten have run** | false. E1 (α ∈ {1.0, 4000}, not the preregistered {0.3,1.0,3.0}), E2, E4, E6 (partial — and it **consumed** the FINAL split), E7 (partial — upstream IMMA and UCE only) |
| §B.10 | added that E7's completed halves **do not** discharge "we reproduced it" | both ran upstream code on **our** benchmark, which is not a reproduction. The register's framing assumed executing upstream code was sufficient |
| §B.10 | E6 relabelled "**RAN, CONSUMED THE SPLIT, AND MUST BE RE-SCORED**" | all 18 FINAL records were scored from `LAB_KEY` rather than `LABELS`, nine of twelve strings differing (`D-022`); the split is spent *and* the numbers are not currently usable, which is the worst of both and needs saying plainly. *(This row originally read "degraded distractor labels"; that wording presupposed a direction and is withdrawn — see §D.0.)* |
| §C.1 | "**Not one corrected implementation has produced a model**" → all three have; the two new defects `D-016` and `D-017` recorded | false, and the defects are the substantive outcome of validating on models rather than on tests |
| §C.2 | FINAL-set consumption, `FINDING72`'s gate bias, the 234/335 `ABSENT` checkpoints, and the 7 `unet: null` records added | "No arm evaluated on FINAL" was false, and three provenance limits were undocumented |
| §C.3 | three false sub-claims corrected; the **verdict kept** and re-grounded | corrected implementations ran, two references executed, unroll depth tested. "No comparative claim is supported" survives for the better reason that upstream-code-on-our-benchmark is not a reproduction |
| §C closing | mandate paragraph rewritten with the `meta2_church_lt0.5` near-miss and `D-008` | the conclusion was right but had been asserted ahead of the evidence that now supports it |

### D.0 Documentation reconciliation, 2026-10-26 — second pass

| what changed | why |
|---|---|
| `D-022` and `D-022a` rows **de-directionalised** | the label-text mismatch is **established** — nine of the twelve strings differ, `church`/`parachute`/`tench` identical — but its **direction and size are not**, pending paired re-scoring. Withdrawn from both rows: "all nine competing distractors were degraded", "`target_hits` is biased upward", "the artifact points the same way as `FINDING65`'s conclusion", "only the FINAL side is inflated", "`FINDING66` survives the defect by construction" and "15/15 can only be lowered by strengthening distractors". Replacement wording for both: *the evaluator used inconsistent label text; the effect on scores and conclusions is unresolved pending corrected evaluation* |
| `D-022` row extended to name **`r4c`/`r4d`** as a second instance | they scored with `list(ge.CONCEPTS)` (= `LAB_KEY`) under `D-006`, so their target accuracies are not comparable to any other H number here and inherit the same unresolved status. Two independent scripts made the same substitution, which makes it systemic |
| §B.8 gained a row on **"destroys the model" / "clean erasure"** | the table corrected the H-interpretation overreach but left the symmetric one: a closed-set argmax with no reject option cannot distinguish erasure from degradation (`D-030`), and R4c/R4d measured no retain axis at all (`D-034`) |
| **§B.8a added — a reject option is a potential additional diagnostic, not a repair** | `D-030` calls it a "decisive test" and G13 lists it as gate evidence; both could be read as making it a fix. §B.8a states the conditions it would have to meet (independent calibration, versioning, documented assumptions, closed-set scores preserved, no result-favouring thresholds) and that it rehabilitates no existing score |
| §B.10's E6 row **de-directionalised** to match | it still read `FINDING66`'s absolute figures as the only casualty |

### D.1 New defect rows

| id | defect | status |
|---|---|---|
| **`D-008`** | `FINDING78` §3's "**first arm in this project to satisfy both halves of the mandate at any attack budget**" was measured at a **single attacker lr** (1e-5). Sweeping lr at the same 25-step budget gives 0.0667 (1e-5), 0.2000 (5e-5) and **0.5333 (1e-4)** — so the arm fails the robustness half, and **no arm has met both halves at any budget**. Same failure mode as `FINDING63`'s "first arm ever" (§B.7): a **conjunctive superlative published before its sweep closed**. | **CLAIM RETRACTED.** `FINDING78`'s title claim withdrawn; its §1 retention result (**+0.20 to +0.27** retain8 at identical 0.0000 suppression, vs variant A's 0.6250/0.7000) **re-verified from raw records and KEPT**. Propagated to `T2I_REPORT_r5.md` §6/§8, `UPSTREAM_STATUS_r1.md` §6, `UPSTREAM_FIDELITY_MANIFEST.md` §2.2, `HANDOFF_r5.md` §6. Independently confirmed that the retracted wording appears in **no** other document. |
| **`D-016`** | `erase_wdr_v2.py` **alternates** the forget and retain steps while its docstring claims it does so "as the reference does". The reference **sums** both gradients into one `opt.step()` (`vision_unlearner.py:411,637-682`). | **CODE/DOC DEFECT, UNFIXED.** Recorded in `UPSTREAM_FIDELITY_MANIFEST.md` §3. Fixing it requires a re-run, which the freeze bars. |
| **`D-022`** | **`gen_eval_final.py` scored every FINAL number with `gen_eval.LAB_KEY` (`:50`) instead of `LABELS` (`:48`)**, prompting CLIP with internal identity keys (*"a photo of a van_gogh"*, *"a photo of a english_springer"*) rather than its text labels. **Nine of the twelve label strings differ. The three that are identical are `church`, `parachute` and `tench`.** That much is **established** by parsing both arrays. What is **not** established is the direction or the size of the resulting change to any score: a different token sequence is a different text embedding, but whether it makes a label a weaker or a stronger competitor for any given image is not determined by the substitution and has not been measured. Recorded per-image facts, without a directional reading: **37 of 180** FINAL images landed on one of the nine labels whose text differs, at per-arm median margins **0.0027–0.0063** (frp 7/15 with 7 such wins; uce_sd15_lam1000 9/15 with 5; utrap 1/15 with 10; utrap_sd15 2/15 with 12; BASE_sd14_anchor 13/15 with 2 at median margin 0.0110). A second axis: `FINDING65`'s ratios put a DEV number scored from `LABELS` against a FINAL number scored from `LAB_KEY`, so the two sides of each ratio did not use the same label text. | **THE EVALUATOR USED INCONSISTENT LABEL TEXT. THE EFFECT ON SCORES AND CONCLUSIONS IS UNRESOLVED PENDING CORRECTED EVALUATION.** `FINDING65`'s "~3×", "~3.5×", "54% undone" and "60% undone" figures are **unmeasured**, and are withdrawn from `T2I_REPORT_r5.md` §8 row 65 and §8a; the finding is neither confirmed nor refuted by this defect. The split-leak and prompt-difficulty reasoning is a separate argument and is **untouched**. DEV numbers were scored through `gen_eval.py`'s `LABELS` path, as were R1–R4 — *except* `r4c`/`r4d`, which scored with `list(ge.CONCEPTS)` (= `LAB_KEY`) under `D-006`, so the same unresolved-effect statement applies to them and their target accuracies are not comparable to any other H number here. The scorer is now repaired; the 18 records predate the repair. **Re-scoring is CPU-only on retained artifacts** and is the cheapest outstanding repair in §B.10. Tracked as resume-gate condition **G12** in `AUDIT_REPAIR_REGISTER_v1.md` §5, **promoted from hygiene to finding-critical**. *(Withdrawn from this row: "all nine competing distractors were degraded", "`target_hits` is biased upward", "the artifact points the same way as `FINDING65`'s conclusion", and "only the FINAL side is inflated". Each infers a sign for the embedding effect that paired re-scoring has not yet supplied.)* |
| **`D-022a`** | The same defect in `FINDING66`. Its central comparison (attacked arms vs the base anchor) runs both sides through the **same** scorer, so the two arms are mutually consistent — but consistency between two arms scored the same inconsistent way establishes nothing about what either score would be under the correct label text. | **THE EVALUATOR USED INCONSISTENT LABEL TEXT. THE EFFECT ON SCORES AND CONCLUSIONS IS UNRESOLVED PENDING CORRECTED EVALUATION.** That covers the absolute FINAL percentages, the "ceiling is not a DEV artifact" conclusion, and the subsidiary claim that the attacked SD-1.4 model (15/15) **beats the unmodified base** (13/15) — a **2-image gap** on n=15. *(Withdrawn: "survives the defect by construction", "remains the more robust of the two findings", and the argument that 15/15 can only fall if distractors are strengthened — that assumes a direction for the embedding effect which is not established.)* `FINDING66`'s standing is **unresolved**, not upheld; §3's robustness results rest on the DEV evidence and are unaffected either way. |
| **`D-017`** | `FINDING67`'s mechanism attribution. The derivation ‖∇L‖ = α/√N **stands**, but it is not why α is inert: both steps share **one Adam instance**, so `v` is retain-dominated and the distance update is ≈ `lr·|g_dist|/√v_retain`. Measured step size as a fraction of `lr` is α=1 → 0.104, α=10 → 0.537, α=4000 → 0.998, i.e. **≈9.6×, not ~4,000×**. α is inert at **both** ends, and the window's upper bound is a **saturation point, not a frontier**. | **CLAIM CORRECTED.** Three withdrawals: the ~4,000× factor, the "**third instance** of one scale pathology" framing (IMMA α and UCE λ are coefficient-scale findings about those *methods*; this one is about our schedule), and "WDR at gradient parity" as a reading of retain8 **0.5917** — which measures our two-step shared-Adam schedule at saturation. **Retained:** 0.5917, the K1 FAIL, and α ∈ (1, 4000) being unmeasured. Propagated to `T2I_REPORT_r5.md` §8 row 67 and §8b, `UPSTREAM_FIDELITY_MANIFEST.md` §3. |

### D.2 Re-verified and found correct (unchanged)

- **§A's four audit verdicts**, including §A.1's "one audit finding that does not hold" and the
  line-cited evidence for the WSR direction, the two missing Meta terms and UTRAP's
  seven-axis attacker mismatch.
- **§A.3's vacuous-record finding** — exactly **9** records, six `gift_church_b0.1` and three
  `uce_sd15_church_lam50000`, confirmed against `outputs/registry_v1.json` and
  `TABLES_v1_generated.md` §3.
- **§B.1–B.9's claim-level corrections**, including §B.7's `FINDING63` superlative row, which
  `D-008` above now cites as precedent.
- **"How to read this register"** and its five status words, which remain the right vocabulary —
  `D-008` and `D-017` are both **CLAIM CORRECTED/RETRACTED**, not experiment failures.
- **§C.4's structural limits** (a ceiling cannot rank predictors; target-assisted recovery cannot
  separate reactivation from relearning; a two-point contrast cannot isolate capacity). E5, E6 and
  E10 remain what would move them, and **E6 has only partially run**.

**Known-stale and not repaired here:** `outputs/registry_v1.json` and
`outputs/correction_register_r5.json`. The former predates 150 JSON files; the latter does not carry
`D-008`, `D-016` or `D-017`. Regenerating the registry needs `scripts/registry_v1.py`, which sha256s
many GB under an active freeze, so both are flagged for the lead rather than run.
