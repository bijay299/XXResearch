# REAL_MODEL_PROTOCOL — one small, decisive FLUX.1-dev falsification test of VAU (PROPOSAL ONLY)

Author: Agent G. **Nothing in this file has been launched, downloaded or run.** All GPU timings are ASSUMED and must be
measured in step 0 before the budget is frozen. This protocol may be launched only if the synthetic verdict under
`docs/EVAL_PROTOCOL.md` is **READY_FOR_SMALL_REAL_MODEL_TEST**. Any other synthetic verdict means it is not run.

## 1. Question

At **matched erasure** of fresh object concepts on FLUX.1-dev with native GEM (`multimodal-ai-lab/GEM @ c328fc6`), does
VAU have at least 10% lower retained-prompt drift than (i) native GEM and (ii) the strongest simple control?

The strongest simple control is a tuned retention penalty with the same retained prompts and the same gate. The test
also requires that VAU's quality and robustness are no worse and that its compute overhead is at most 3×.

The question is answered in a single pass, with a decision rule written in advance.

## 2. Fixed native setting (unchanged for every GEM-family arm)

GEM `c328fc6` defaults are kept unchanged (A_gem_code_facts §1–7):
- eta = 1, gamma = 100
- LoRA `qk_dual`, r = 16, alpha = 16, dropout 0.1
- AdamW with torch defaults
- 28-step schedule; loss on the student rollout prefix 0..10; rollout guidance 3.0, loss guidance 1.0
- anchor = the empty prompt; 512 px; `train_seed` varied only as listed below

The arms change only (a) the optimizer step (VAU / gate wrapper) or (b) the loss (an added penalty term). D's
uniform-t ablation changes only the timestep subset.

## 3. Arms

| id | arm | retained info | tuning grid, all GEM-family arms equal: 3 lr × 2 secondary = 6 configs |
|---|---|---|---|
| A0 | native GEM c328fc6 | none | lr ∈ {5e-4, 1e-3, 2e-3} × wd ∈ {0.01, 0.1} |
| A1 | **VAU** wrapping GEM's AdamW step: k = 8 sketch rows, K = 5, gate n2 = 32 retained (prompt, latent, σ) samples per try, R_max = 3, δ = 0.05, M = 2τ² clip | R1, R2 | lr × τ² ∈ {τ0², 2τ0²} |
| A2 | **strongest simple control**: whichever of `penalty` or `penalty+gate` was the more frequent best control in synthetic G8. This is frozen before step 0. Penalty = λ/2 · mean over R1 of ‖v_θ − v_θ0‖² on base retained-trajectory states; the gate is identical to A1's | R1 (+R2) | lr × λ ∈ {λ0, 10λ0} (or τ² if gated) |
| A2n | **na_lora**, a KNOWN InfLoRA-style null-A LoRA (amendment A1). It uses the same qk_dual modules. Per module, A's rows are made orthogonal to the span of the R1 retained-token inputs and then frozen; B is trained with the native GEM loss. The A1 gate is added if `na_lora+gate` was the better of the two in synthetic G8 | R1 (+R2) | lr × feature-energy threshold ∈ {0.99, 0.9999} |
| A3 | **published comparator: EraseAnything** `tomguluson92/EraseAnything @ 67d8c0b` (arXiv 2412.20413v1), author code unchanged | its own regulariser | author default + lr × 0.5 (2 configs; **not budget-equal**; descriptive only) |
| A4 (optional) | D's ablation: native GEM with a random 11-of-28 subset, 18 iterations (matched forwards) | none | native defaults only; diagnostic |

**Why EraseAnything is the comparator** (sources: A_source_matrix §2, REF `eraseanything_efficiency_pilot`):
- It is the only published flow-LoRA erasure method with a retention mechanism whose code **has already been executed
  unchanged** on FLUX.1-dev in a prior record (REF FINAL_REPORT: author `calculate_upper/lower_loss` run unchanged).
- It uses the same backbone and an overlapping LoRA target set (`add_q_proj`, `add_k_proj`, which is GEM's
  `qk_dual_add_proj` subset).

Alternatives that were considered and not chosen:

| method | why not |
|---|---|
| EraseAnything++ (`f73e5c1`) | Mechanistically closer, since it does constrained forget/retain multi-objective on the same LoRA. But it has **never been run** (REF AMENDMENT_02: "We ran no EA++ training"). It may be added only if a 1 GPU-h smoke test passes; otherwise it is omitted, and the omission is reported. |
| Minimalist | FLUX.1-schnell, a mask parametrisation, and backprop through the sampler; not the same regime |
| DVE | no code; inference-time |
| BLOCK | transcoder weights not released |
| AEGIS, DoCo | no code reachable |

A3 is reported, but it **does not enter the decision**: its tuning is asymmetric and its parameter subset differs.

## 4. Fresh concepts and prompt sets (frozen before any training)

**Concept selection.**
1. The pool is the 80 COCO category names, excluding `person`.
2. Exclude:
   - any concept named in any REF `results/cpr_*` case or prompt file, or in any other prior XXResearch frozen case
     list (found by grep; the excluded list and its sha256 are recorded)
   - GEM's own released targets (nudity, gore)
   - the development concept used in step 0
3. Rank the remainder by `sha256("flow_method_design_v1/real/" + name)`, ascending.
4. Accept concepts in rank order if they pass **scorer qualification** on base-model images: detector AUROC ≥ 0.95,
   concept prompts vs unrelated prompts, 16 + 16 prompts × 2 seeds.
5. Stop at **4 accepted concepts**. The next 4 that pass are frozen as replacements.

For each concept, the **neighbours** are 2 sibling categories from the same COCO supercategory, taken in hash order.
The **development concept** is the first excluded-by-rank concept outside the pool. It is used only for step 0 and for
calibrating τ0 and λ0.

**Prompts.** Prompts are built by deterministic template filling; no LLM is used. The template lists and the COCO
caption sample, drawn by hash lottery and excluding captions that contain the concept or its synonyms, are frozen with
their sha256.

| set | content | size |
|---|---|---|
| target train | native GEM target prompt (the concept name) | 1 |
| X_T^sel / X_T^test | held-out concept templates | 10 / 30 prompts; ×2 / ×4 seeds |
| X_para | synonyms, plurals, a second language, compositional prompts | 20 × 2 seeds |
| R1 (probes) | 8 neighbour + 8 unrelated prompts | 16 |
| R2 pool (gate) | neighbour + unrelated | 64 |
| R_sel | | 32 |
| R_test | 40 neighbour, 50 unrelated COCO captions, 10 "context without concept" | 100 × 2 seeds |
| quality | COCO captions shared across concepts | 300 |

All sets are disjoint, which is checked by hashing. R_test is never used for training or selection.

## 5. Metrics

| role | metric |
|---|---|
| erasure (matching) | concept detection rate with a frozen open-vocabulary detector at a frozen threshold (scorer and version pinned in step 0), on X_T^sel / X_T^test. Co-metric: held-out velocity hinge progress `E_hinge`, the same definition as the synthetic protocol |
| **preservation (primary)** | paired image drift on R_test: DINOv2 cosine distance between base and edited images with the same prompt and seed. The mean is primary; the 95th percentile is the worst case |
| preservation (secondary) | LPIPS; velocity drift ρ on base retained trajectories at all 28 σ (the toy's ρ, normalised by mean ‖b−a‖²); neighbour retention = detector rate for the sibling concept on sibling prompts ÷ base rate |
| quality | KID (300 COCO captions, edited vs base images); CLIP score |
| robustness | residual detection on X_para |
| cost | forward-equivalents per arm (one transformer forward on one latent, backward = 2), plus measured GPU-seconds. Every gate try and every probe VJP is counted |
| gate coverage | the fraction of A1 runs whose accepted iterates have held-out (R_test-trajectory) velocity drift above τ² |

## 6. Matched-erasure procedure

1. **Matched level.** r* = native GEM's default-config detection rate on X_T^sel (tuning seed), rounded up to the next
   0.05. If native GEM does not reduce detection by at least 50% on a concept, that concept is replaced, in frozen
   order and at most twice.
2. **Training and checkpoints.** Each config is trained for up to 60 iterations, with checkpoints at 10, 20, 23, 30, 40,
   50 and 60. At each checkpoint, detection is measured on 20 X_T^sel images and velocity drift on R_sel (no images).
3. **Selection.** For each arm and concept, pick the (config, checkpoint) with the lowest R_sel velocity drift among
   those with detection ≤ r*. An arm that never reaches r* is recorded as "not reached".
4. **Replication.** The selected config of A0, A1, A2 and A2n is retrained at **2 further training seeds**, at its selected
   checkpoint iteration. That gives 3 seeds × 4 concepts = 12 pairs. A3 and A4 use 1 seed.
5. **Evaluation.** Each final model is evaluated on X_T^test, X_para, R_test and the COCO-300 set.

## 7. Correctness checks before spending budget (step 0, about 3 GPU-h)

| check | requirement |
|---|---|
| C0 | the synthetic verdict is READY and the A2 choice is frozen |
| C1 | the torch VAU reproduces the numpy contract fixtures exported from `code/vau`. For 50 fixtures of (g, m, v, t, θ, S_J, R2 outputs, τ, M), Δ1, the scale s and the accept decisions match to rel 1e-10 in fp64 and 1e-5 in fp32, and the accept decisions are identical |
| C2 | with probes disabled and the gate off, the VAU-wrapped optimizer step reproduces the native c328fc6 step on FLUX for 3 iterations (deterministic algorithms; LoRA parameters max abs diff = 0, or ≤ 1e-6 relative if a kernel is non-deterministic, documented) |
| C3 | gate with τ = ∞ equals ungated; rollback restores the AdamW `state_dict` and LoRA parameters bit-exactly |
| C4 | sketched constraint residual ‖S_J Δ1‖ / ‖S_J Δ0‖ ≤ 1e-4 in fp32; S_J memory is k·P ≈ 8 × 7.5e6 floats ≈ 0.24 GB (P for 76 qk_dual modules at r = 16, DERIVED) |
| C5 | native GEM reduces detection by ≥ 50% on the development concept |
| C6 | scorer qualification, as in §4; neighbour discrimination AUROC ≥ 0.9 |
| C7 | measured F time, VAU per-iteration overhead and image time; rescale the plan with the ladder (drop A4, then EA's second config, then X_para seeds) to stay under the cap; freeze |
| C8 | regenerated base images are identical, or DINOv2 distance ≤ 1e-3 |
| C9 | **probe-coverage gate** (amendment A1). Before any training, measure C_probe on each concept at θ0: the share of held-out R_test velocity-Jacobian energy (256-row sketch) that lies in the row space of the k = 8 R1 probe sketch. It must be ≥ 0.90, with k ≥ eff_rank_99. If it fails on 2 or more concepts, the protocol is **not launched**, because the narrowed hypothesis does not apply to FLUX at this probe budget. C_probe and C_feat are reported either way |

Any failure stops the protocol before any training on the 4 concepts.

## 8. GPU budget (H100-80GB-equivalent; ASSUMED until C7)

The estimate assumes roughly 0.15 GPU-h per 60-iteration native GEM run, 2.3× that for VAU, and 1200 images per GPU-h
at 512 px with 28 steps.

| phase | GPU-h |
|---|---|
| step 0 checks and calibration | 3 |
| base images (4 × 360 + 300) | 1.5 |
| tuning: 4 concepts × 24 GEM-family runs (A0, A1, A2, A2n) + checkpoint detection images | 33 |
| EraseAnything (author code, about 2 GPU-h per run, extrapolated from the REF pilot and withdrawn as a cost claim there) | 11 |
| replication: 2 seeds × 4 arms × 4 concepts | 8 |
| final evaluation: 56 models × 660 images | 31 |
| A4 | 0.3 |
| **total** | **about 88**; the only contingency is the headroom up to the 96 cap |

The hard cap is **96 GPU-h**. On reaching it, the run stops and is reported as INCOMPLETE, which counts as INCONCLUSIVE.

## 9. Stopping rules

- The protocol is run once.
- No efficacy look before all 12 pairs are evaluated. An interim check at 50% of the budget looks only at
  crashes and timing.
- Crashes count as failures of that arm and config.
- No new concepts, seeds or configs are added after any result is seen, apart from the frozen replacement rule.
- A bug fix after launch is labelled `REPAIR-n`, and both results are reported. A fix that touches A1 or the gate
  downgrades a positive outcome to INCONCLUSIVE.

## 10. Decisive outcomes

The unit of analysis is a (concept, training seed) pair; there are 12. The comparison is A1 against the best of {A0, A2, A2n}
on R_sel. The statistic is a paired GMR, tested with an exact sign-flip permutation test (4096 flips).

**ADVANTAGE DEMONSTRATED** requires all of the following:
- A1 reaches r* on all 4 concepts.
- Primary DINOv2 drift GMR ≤ **0.90**, with p < 0.05 and ≥ 9/12 wins.
- Velocity-drift GMR ≤ 0.95.
- The 95th-percentile drift GMR is ≤ 1.05.
- Neighbour retention is ≥ 0.97 × the best control's.
- KID is not worse by more than 10% relative.
- The CLIP score drop is at most 0.3 larger.
- Residual detection on X_para is at most 5 points higher.
- FE overhead is ≤ **3×** native GEM.
- Gate coverage is ≥ 95%.

**HYPOTHESIS REJECTED (FLUX, this scale)** if any of the following holds:
- A2 is more than 10% better (GMR(A2/A1) ≤ 0.90 with ≥ 8/12 wins).
- A2n matches or beats A1 (GMR(A1/A2n) ≥ 0.95). This is the FLUX analogue of F5.
- A1's point GMR against the best control is ≥ 0.97.
- A1 fails to reach r* on ≥ 2 concepts while A2 reaches it.
- The overhead is > 5×.

**INCONCLUSIVE** covers everything else. No scale-up follows without a new pre-registered protocol.

A3 (EraseAnything) and A4 (uniform-t, with ρ_late logged as in D_candidate_spec §6) are reported descriptively. They
never change the outcome.
