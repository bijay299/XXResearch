# A_source_matrix: competitive evidence for concept erasure in flow/DiT text-to-image models

Date 2026-10-19. Agent A. Machine-readable pins: `A_source_cache.json`. GEM code line facts: `A_gem_code_facts.md`.

## 0. Access status and rules (read first)
- arXiv, ar5iv, alphaxiv, HF papers, Semantic Scholar, iclr.cc: unreachable from Bash (403) and WebFetch (DNS error) in this session. Only WebSearch snippets and GitHub (`git ls-remote`, clone) worked. So NO paper equation was re-read by me this session.
- Labels: VERIFIED = read by me at file:line (code) or recorded as read from the primary source in a named REF record (that is secondary verification, not re-read today); CLAIM = authors' stated result (via REF record or abstract-level snippet); DERIVED = my inference; UNVERIFIED = not reached.
- Protocols differ (backbone dev vs schnell vs SD, nudity vs style vs object, metric, detector, step budget). Numbers below are quoted per source with protocol. They are NOT a leaderboard and must not be ranked or compared.
- Reused (not re-searched): REF `modern_erasure_gap_map`, `minimalist_erasure_design_sprint`, `eraseanything_efficiency_pilot/AMENDMENT_01.md`, `block_object_sprint`. Newly searched: GEM paper identity, AEGIS, DoCo, TRACE, EraseFlow, FlowErase-OPD, DVE/AEGIS/DoCo code availability.
- Scope tags: OBJ = object erasure, STY = style, SAF = safety/NSFW, VID = video, ARCH = non-flow architecture.

## 1. Summary matrix (compact)

| id | version / pin | intervention | backbone | scope evaluated | trained? | retention data / teacher | code |
|---|---|---|---|---|---|---|---|
| GEM | arXiv 2606.00140 (v1 05-29, v2 06-10, latest v4 10-04 per snippet); code `c328fc6` (2026-08-26) | LoRA, hinge attract/repel on velocity (EG:242) | Flux.1-dev default; also SD, SD3, Flux2, Qwen-Image in code | SAF (nudity, gore), rights-protected concepts (abstract, UNVERIFIED detail) | yes, ~23 AdamW steps default | frozen-self teacher (adapter off); anchor=empty prompt; NO retention data | yes + 2 HF ckpts |
| EA++ | 2603.00978v1 (2026-03-01); `f73e5c1` | LoRA `add_q/add_k`; bi-level reformulated as constrained multi-objective, implicit gradient surgery | Flux.1-dev (image); video models too (abstract) | SAF, STY (image); VID | yes | retain loss every step (`eupmu.get_weighted_loss`) | yes |
| DVE | 2602.01089v1 (2026-02-01); no code found | inference-time projection of velocity onto `Δv` | Flux.1-dev | SAF, STY, OBJ | NO | needs erase/anchor prompts at inference | UNVERIFIED |
| BLOCK | 2606.25548v1 (ICML 26); `db4ab11` | transcoder swap of a bottleneck layer; edit decoder columns | SD3.5, Flux, Infinity (AR) | SAF, STY, OBJ | transcoder once per model (5428 s, CLAIM Tbl 11) | no per-target data; transcoder weights NOT released | yes (no weights) |
| MCE | 2507.13386v1 (2025-07-16); `d178324` | binary parameter mask on weights; terminal-output distribution objective | Flux.1-schnell primary (VERIFIED REF) | OBJ/SAF/STY mix (see REF) | yes (backprop through K=5 steps) | reference model, prompt filtering | yes |
| DoCo | 2405.15304 v1 ACM MM24, v2 AAAI25; code UNVERIFIED | adversarial domain alignment + gradient surgery ("concept-preserving gradient") | UNVERIFIED (abstract speaks of T2I diffusion; U-Net SD assumed) | instances, styles, offensive concepts (CLAIM) | yes | retention gradient from preserve prompts | UNVERIFIED |
| AEGIS | 2602.06771 (2026-02-06, ICLR 26); no code found | adversarial erasure target + Gradient Regularization Projection | UNVERIFIED (diffusion; flow coverage not seen) | robustness vs adversarial prompts | yes | retention-DATA-FREE (CLAIM) | none found |
| EraseAnything | 2412.20413v1 (ICML 25); `67d8c0b` | LoRA `add_q/add_k` dual blocks; bi-level: forget + InfoNCE attention regulariser | Flux.1-dev | SAF, STY, OBJ-ish | yes | self-contrastive; alternating upper/lower steps | yes |
| X1 TRACE | 2505.23312v1 (2025-05-29) | closed-form cross-attn edit + trajectory-aware finetune acting in late steps | SD and Flux (CLAIM) | OBJ, celeb, STY, SAF (CLAIM) | partly | UNVERIFIED | UNVERIFIED |
| X2 EraseFlow | 2511.00804 (NeurIPS 25); `2fe3496` | GFlowNet trajectory-balance over denoising paths | mainly SD-class (UNVERIFIED), code not read | SAF/STY/OBJ (UNVERIFIED) | yes | policy vs base | yes |
| X3 FlowErase-OPD | 2608.07620v1; ID conflict, see cache | on-policy distillation, anchored, multi-concept | flow matching (title) | multi-concept | yes | anchors | UNVERIFIED |

## 2. Per-method detail

### GEM (native; the mechanism baseline)
- Objective/intervention: VERIFIED `EG:234-249`: `gamma * sum_t relu(||v-a|| - eta*||v-b||) / n_latents`, v = student on target prompt, a = anchor teacher (empty prompt default `EG:23`), b = target teacher, teachers = same net with LoRA disabled (`EG:219-231`). Paper equation numbers: UNVERIFIED. Closed algebra (CPR, matched first-order progress) in REF `cpr_gem_cpu_prototype`.
- Training/inference: VERIFIED LoRA r16/alpha16/dropout .1 on `to_q,to_k,add_q_proj,add_k_proj` of dual-stream blocks (A_gem_code_facts section 4); AdamW lr 1e-3, other args default; no clipping; no scheduler; no retention term; trajectory = 11 latents from the CURRENT student rollout at guidance 3.0, loss evaluated at guidance 1.0; sigma in about [0.77,1.0] only (COMPUTED). Budget `max_step_budget=250` timesteps, about 23 optimizer steps (DERIVED).
- Retention data / teacher / probe: none beyond frozen-self teacher and empty-prompt anchor. Validation concepts only logged.
- Evaluation definitions: UNVERIFIED (paper not read). Code includes evaluators (nudity, FID, CLIP, VLM, Gemini, Q16) in `gem/evaluation/`.
- CLAIM (abstract via search snippet): about 5x faster than prior state of the art on Flux, safer generations; ICML 2026 Spotlight. No numbers verified.
- Limitations (VERIFIED from code): no retention term; deterministic early prefix only (no late-timestep supervision); `negative_guidance_scale` unused; guidance mismatch; LoRA dropout in student only; possible index pairing offset (DERIVED, see facts file section 2).
- Availability: code yes; HF `MAI-Lab/GEM-Flux-Nudity`, `-Bloody-Gore` (README text). Compute: 3 transformer forwards x 11 timesteps + 11-step rollout per iteration, about 23 iterations; wall time UNVERIFIED.
- Scope: SAF-centred (nudity/gore); object erasure not shown in README.

### EraseAnything++ (EA++)
- Version: arXiv 2603.00978v1, 2026-03-01; title "EraseAnything++: Enabling Concept Erasure in Rectified Flow Transformers Leveraging Multi-Object Optimization"; code `nxjiang-jnx/EraseAnything-PlusPlus@f73e5c1`.
- Objective: VERIFIED (REF AMENDMENT_01 section 3, paper III-C): constrained multi-objective; one combined backward; dual variable `lambda_{t+1} = lambda_t - beta_t * delta_t`, `delta_t = (1/alpha_t)(L_p(theta_t) - L_p(theta_{t+1})) + eps_t` (finite difference of forward losses). VERIFIED (REF, diff = 0 lines): `image/utils/calc_loss.py` is byte-identical to original EraseAnything's; the change is the loop (both forget and retain loss every step, `eupmu.get_weighted_loss`, extra evaluations every `eupmu_update_freq`).
- Backbone/targets: Flux.1-dev, LoRA on `add_q_proj`/`add_k_proj` (same as original). Video extension per abstract (anchor-and-propagate, UNVERIFIED detail).
- CLAIM (REF record of paper): nudity 182 parts; FID 21.67 vs 21.32 base; CLIP 30.35 vs 30.87; attack success 68.80% vs EraseAnything 71.55%. Protocol: author's, not matched to GEM.
- Limitations (VERIFIED REF): no wall-clock/GPU cost reported; per-step cost higher by static count (lower loss 19.91 s vs 5.96 s measured by REF for original code); rank/resolution/GPU unreported.
- Scope: SAF/STY images and VID; not an object-erasure paper.
- Search snippets describe a "gradient projection mechanism" in the paper's contribution list as well as implicit gradient surgery; the exact algorithm should be re-read before any claim of distinction.

### DVE (Differential Vector Erasure)
- Version 2602.01089v1 (2026-02-01). No official code found (UNVERIFIED).
- Objective: VERIFIED (REF gap map S2): `Δv = v_anc - v_era` (Eq. 6); `s = <v_user, Δv/||Δv||>` (Eq. 8); correction only when `s < tau` (Eq. 9); Alg. 1; no weights, LoRA or classifier; needs erase, anchor and user prompts at inference.
- Backbone Flux.1-dev; tasks NSFW, style, object (CLAIM via abstract). Baselines per REF: ESD, UCE, CA, MACE, EAP, Meta-Unlearning, EraseAnything.
- Limitations: inference-time cost (extra forwards per step, DERIVED), requires knowing the erase prompt at inference, no weight-level guarantee.
- Scope: OBJ/STY/SAF, inference-time; not comparable to weight-space training budgets.

### BLOCK
- Version 2606.25548v1 (ICML 2026), `sprintml/block-concept-removal@db4ab11`.
- Intervention (VERIFIED REF): sparse transcoder replaces a bottleneck layer; removal edits decoder columns so the target's active latents reproduce the empty-token output. No LoRA/ESD.
- CLAIM (Tbl 11 via REF): upfront transcoder 5428 s (A100 40 GB, App E.1), per-target 1.020 s, +0.43 s/image inference, 1.9 GB. Tbl 7 FLUX nudity detections: EraseAnything 199, BLOCK 153; FID 21.75 vs 22.78. Style IRA 36.1%.
- VERIFIED by REF `block_object_sprint`: a native FLUX.1-dev object reproduction was attempted at small scale; transcoders had to be trained natively (no weights released); results are descriptive and not a reproduction of the paper's table (REF REPORT.md lines 19-26, 120-121).
- Scope: SAF, STY, OBJ across diffusion and AR models.

### Minimalist Concept Erasure (MCE)
- Version 2507.13386v1 (2025-07-16, ICML 2025), `d178324`.
- Objective: VERIFIED (REF ledger P1): Eq. 16 distributional distance on FINAL generated outputs only, tractable loss backpropagated through generation steps (checkpointed, constant memory, section 3.7); weight-space binary mask `theta = M * theta'` (section 3.6). Paper/code divergences (batch 4 vs 1, beta .01 vs .1, 400 vs about 80 steps) recorded in REF COST_MODEL lines 70-71.
- Backbone Flux.1-schnell primary (section 4.1). K=5 sampler steps in training (code `:24`). Single GPU only (section 3.7).
- Scope: mixed; see REF.

### DoCo / DoCoPreG
- Version: arXiv 2405.15304 (v1 ACM MM 2024; v2 AAAI 2025 "Unlearning Concepts in Diffusion Model via Concept Domain Correction and Concept Preserving Gradient"). Content UNVERIFIED beyond the abstract (search snippet).
- CLAIM (abstract): adversarial training aligns output domains of sensitive and anchor concepts for generalisation beyond the unlearned set; a gradient surgery step "mitigates conflicting gradient components" to preserve utility; works across instances, styles, offensive concepts.
- Backbone UNVERIFIED (assumed U-Net SD; ARCH scope). Code claimed at `github.com/yongliang-wu/DoCo`; `git ls-remote` asked for credentials, so repo unreachable. Equations, optimizer, evaluation definitions: UNVERIFIED.

### AEGIS
- Version 2602.06771 (submitted 2026-02-06; ICLR 2026; Li, Li, Wang, Han, Zhou). No code found. Content UNVERIFIED beyond abstract-level snippets.
- CLAIM (abstract snippets): Adversarial Erasure Target (AET) approximating the semantic centre of the erased class; Gradient Regularization Projection (GRP) projects the retention gradient only when it conflicts with the erasure direction (sign of inner product, Alg. 2 / App. C.2 per search tool, NOT read); retention-data-free.
- Backbone UNVERIFIED (diffusion; flow/DiT coverage not seen). Scope: robustness vs adversarial prompts.

### EraseAnything (original)
- Version 2412.20413v1 (ICML 2025), `tomguluson92/EraseAnything@67d8c0b`.
- Objective/intervention: VERIFIED (REF gap map S2c): bi-level, LoRA on `add_q_proj`/`add_k_proj` (App. A), attention-map regulariser and self-contrastive InfoNCE; multi-concept by normalised LoRA blending (App. F.2 Eq. 12). VERIFIED code facts (REF pilot): alternating upper/lower on `global_step % 2`; lower step runs K+1 full 28-step no-grad trajectories and reads one attention map per trajectory (`calc_loss.py:572,595,598`); lower step 19.91 s vs upper 5.96 s on the REF 6-step profile (measured by REF, GPU not comparable to others).
- Backbone Flux.1-dev (section 5.1); NudeNet on I2P 4,703; FID/CLIP MS-COCO 10k; scope SAF + STY + concept erasure.
- Limitation (REF): the pilot could not obtain a qualified style effectiveness measurement (scorer adequacy), so no independent effectiveness verification exists.

### Additional directly threatening references (3; justified)
- X1 TRACE (2505.23312v1): CLAIM trajectory-aware fine-tuning loss acting only in LATE sampling steps, reported for SD and Flux. Threatens any claim that timestep/trajectory-restricted supervision is new. Abstract-level only.
- X2 EraseFlow (2511.00804, NeurIPS 2025; code `2fe3496`): GFlowNet trajectory-balance over denoising paths; GEM states it reduces this to teacher-guided velocity matching (search snippet, UNVERIFIED). Threatens novelty of trajectory-level credit assignment and of the "deterministic sampler, transition prob = 1" derivation used by GEM.
- X3 FlowErase-OPD (2608.07620v1): on-policy (student rollout) anchored distillation for multi-concept flow models. Threatens "student-rollout trajectory plus anchor" and multi-concept shared-parameter claims. Identity conflict with REF's "ARC" label for the same arXiv ID; content UNVERIFIED.
- Searched and NOT found: Adam/optimizer-state-aware erasure for flow models; direction-aware repulsion losses beyond the GEM family (DVE's direction `Δv` is inference-time). Absence is not proof: search was snippet-level only.

## 3. Threat assessment against the three prospective mechanism tracks
Tracks: L = loss geometry; O = optimizer / shared-parameter; T = timestep / trajectory.

**GEM.** It is the baseline for all three tracks. L: the hinge `relu(D+ - eta D-)` with L2 distances on flattened velocities (EG:242) is the object any L-claim must beat; REF already closed that CPR is control-equivalent to a centred smooth radial loss at v=b. O: plain AdamW lr 1e-3 with default weight decay, no clipping, no retention term, so a claim that the shared LoRA parameters or Adam normalisation distort retention is unaddressed by GEM (an opening, not a prior-art threat). T: supervision is only the first 11 points of the student's own rollout (sigma 0.77-1.0), unweighted, with evaluation at guidance 1.0 while the rollout uses 3.0; any T-claim is directly measurable against these facts. Paper-level claims remain UNVERIFIED.

**EraseAnything++.** O threat HIGH: it already treats forget vs retain as constrained multi-objective with gradient surgery and a dual variable on the same LoRA targets, so "reweight/project forget and retain gradients on shared LoRA parameters" is prior art for flow models (CLAIM level; its loss code is unchanged from EraseAnything). L threat low (loss file identical). T threat low. Weakness to exploit: no cost data, no direct comparison with GEM, and gradient surgery uses finite-difference loss, not Adam-aware preconditioning.

**DVE.** L threat MEDIUM: its direction `Δv/||Δv||` and a signed projection `s` with a conditional trigger (Eq. 6, 8, 9) is the inference-time sibling of GEM's axial coordinate used by CPR; any claim that an axial/directional coordinate in velocity space is a new geometry must cite DVE. O none (training-free). T threat low-medium (applied at every step of the ODE, unconditional on timestep). Not a weight-space competitor, so not a matched baseline.

**BLOCK.** Low threat on all three tracks: different paradigm (transcoder, no loss geometry, no optimiser coupling, no trajectory loss). Relevant only as a competing cost regime and as a possible object-erasure comparator; its transcoder weights are not released, so reproducing it needs per-model training.

**Minimalist Concept Erasure.** T threat MEDIUM: terminal-output objective with backprop through K=5 steps versus GEM's per-step velocity supervision; it already argues that intermediate steps need not be constrained. L low (distribution distance, not a velocity hinge). O threat low-medium: weight masking is a different parametrisation that sidesteps LoRA optimiser interactions; the sprint's Adam issues (loss replacement, two grad sweeps) are recorded in REF DERIVATION. Flux-schnell only.

**DoCo.** O threat HIGH for the idea "gradient surgery with a preserve-concept gradient" (2024, before any flow-erasure work); L threat low-medium (adversarial domain alignment between sensitive and anchor outputs is a distributional, not hinge, geometry). Abstract-level only; backbone not flow (ARCH), so transfer to flow is not shown.

**AEGIS.** O threat HIGH: conflict-triggered projection of the retention gradient (sign of inner product) with no retention data is exactly the standard PCGrad-style rule; any "project only when conflicting" retention mechanism is not new in the erasure literature (ICLR 2026). L threat medium: the adversarial erasure target replaces the hand-picked anchor; relevant for target/anchor choice. T none seen. Backbone and equations UNVERIFIED.

**EraseAnything (original).** T threat low-medium: its lower level reads attention maps from full 28-step trajectories at a random index (REF `calc_loss.py:572-598`), so trajectory-sampled regularisation exists; O threat low-medium: alternating upper/lower steps on shared LoRA is a cheap form of forget/retain coupling; L low. Serves as the main existing flow baseline on matching LoRA targets (`qk_dual_add_proj` subset in GEM is its target set, GEM `pipeline_wrapper_flux.py:169-172`).

**X1 TRACE / X2 EraseFlow / X3 FlowErase-OPD.** TRACE: T threat MEDIUM (late-step restriction claimed for Flux). EraseFlow: T threat MEDIUM-HIGH conceptually (trajectory-level objective is the ancestor of GEM's trajectory evaluation; GEM itself cites it). FlowErase-OPD: T and O threat unknown (UNVERIFIED), flagged because on-policy rollouts plus anchoring plus multi-concept overlap with GEM's student-mode trajectory and with any shared-parameter claim.

## 4. Three most threatening findings for novelty (details above)
1. Conflict-aware gradient projection for retention is already published (AEGIS GRP, DoCo gradient surgery, EA++ implicit gradient surgery on flow LoRA). Any O-track claim must be about something else (Adam-preconditioned geometry, shared-parameter interference measured on GEM's own loss), not "project when conflicting".
2. Directional/axial velocity-space geometry exists in DVE (`Δv`, signed projection, thresholded correction) and GEM's own CPR record is closed; no L-track claim can rest on "a signed coordinate along (b-a)".
3. Trajectory/timestep restriction is covered by EraseFlow (trajectory objective, ancestor of GEM), TRACE (late-step restriction, claims Flux), and GEM's own student-rollout prefix; a T-track claim must be about the specific choice (e.g. sigma coverage 0.77-1.0 only, guidance mismatch 3.0 vs 1.0, index pairing) measured against GEM's code facts, not about trajectory-awareness itself.

## 5. What remains UNVERIFIED
GEM paper equations/numbers/eval protocol and which paper version matches `c328fc6`; DoCo and AEGIS full text, code and backbone; DVE code existence; TRACE, EraseFlow, FlowErase-OPD beyond abstract level; EA++ "gradient projection" wording; diffusers callback semantics and flux time-shift formula (used in DERIVED/COMPUTED items). No real-model result was run or reproduced.
