# Correction register — remediation of the external audit of snapshot `0b435427`

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
| H = 1 means the model was restored / H = 0 means the capability is absent | H is **15 images, 3 prompts, 5 each**, closed-set 12-label CLIP argmax, no reject option. Quantised to 1/15. An exact binomial interval for 15/15 has lower endpoint ≈ 0.782 — and even that model is inadequate here given prompt clustering, reused seeds and adaptive selection |
| target-image-assisted recovery shows suppression not removal | **does not distinguish** residual-memory reactivation from learning again from the supplied images. This is `ERRATUM_r2` C17, which r4 §7 had re-breached; r5 does not |
| WSR2's utility gain is a frontier improvement | **descriptive, selected-configuration** result: three builds, retain8 0.9500/0.9750/0.9667 (mean 0.9639, sample SD 0.0127); an illustrative one-sample 95% t interval [0.9323, 0.9955] **contains** the FRP teacher's 0.9417. "Statistically indistinguishable erasure" is withdrawn — that needs a prespecified equivalence margin, and failure to show a difference is not equivalence |
| "no valid attack" = the defence fails | an ineffective attack is **inconclusive**. Preventing the attacker from fitting T can itself be defensive. Gated **and** ungated outcomes are now both reported in the generated tables |
| CLIP panel + class accuracy = image quality preserved | FID remains **pending**; prompt adherence and closed-set accuracy do not establish general quality or diversity |

**Status: CLAIM CORRECTED.** The persistent-mask intervention is specified in §B.10 as
**PENDING EXPERIMENT**.

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

None of these has been run. Each is specified precisely enough to execute.

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

**1. Implementation — SUBSTANTIALLY RESOLVED, UNVALIDATED ON MODELS.**
The three wrong objectives are now implemented correctly and unit-tested (24/24 passing), the
attacker's definition is shared through one module that refuses to let a surrogate claim to be the
evaluation attack, and the split checker covers every defence and fails closed on new ones.
**Not one corrected implementation has produced a model.** The tests show the code computes what the
references specify; they show nothing about whether the resulting defences work.

**2. Evaluation and provenance — PARTIALLY RESOLVED.**
Resolved: the leak is detected and attributed; two further leaks found; DEV/FINAL separated with
FINAL frozen before inspection; RNG streams separated; the registry reports gated **and** ungated
outcomes, all three utility denominators, and admissibility flags; checkpoint identity is a sha256
rather than a path. Unresolved: **no arm has been evaluated on FINAL**, so every target number in
the corpus remains a development measurement; the CLIP classifier revision is still unpinned; H
remains a 15-image closed-set measure; per-image predictions are retained but not published for
every historical run.

**3. Empirical comparison — NOT RESOLVED.**
No corrected implementation has been run. The three published methods with released code have not
been executed. The unroll-depth question is untested with a faithful attacker. Consequently
**no comparative claim about the published baselines is currently supported**, and the claim that
all five were discharged is withdrawn. What survives is narrower and still substantial: on the
builds actually measured, with the attacks actually run, every arm that cleared the utility floor
was reversed to H = 1.000 — now excluding the six vacuous `gift_church_b0.1` records.

**4. Scientific interpretation — CORRECTED IN TEXT, STRUCTURALLY LIMITED.**
Every saturated-endpoint, mechanism and capacity claim is rewritten to the weaker statement the
data supports. The structural limits remain and are **not** repaired by wording: a ceiling cannot
rank predictors; target-assisted recovery cannot separate reactivation from relearning; a two-point
model contrast cannot isolate capacity. E5, E6 and E10 are the experiments that would move these,
and they have not been run.

**The mandate's bar is unmet, and this remediation does not change that.** No arm reaches
retain8 ≥ 0.85 with max-over-valid H ≤ 0.40. What the register changes is the *standing* of the
negative result: it is now a statement about **our implementations** of three of the baselines, not
about the published methods, until E1, E2 and E7 run.
