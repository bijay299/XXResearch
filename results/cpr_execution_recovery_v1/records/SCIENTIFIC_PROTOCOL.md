# cpr_execution_recovery_v1 — scientific protocol and numerical criteria (frozen before final validation)

Frozen at the commit that contains this file (see FREEZE_01 record). The values below are mirrored in `code/cprx/protocol.py`;
this document is authoritative if they ever disagree. Nothing is tuned against final-validation failures or efficacy. A failed freeze
or validation is preserved and repaired only by an additive patch followed by a new, separately-labelled freeze.

## Pins
GEM `multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd` (+ attributed patches in `patches/`); FLUX.1-dev `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`;
RT-DETR r50vd `df939e661d8c52e80608d1ec566561aabd25a4e7` (primary, threshold 0.30); DETR resnet50 no_timm `70120ba84d68ca1211e007c4fb61d0cd5424be54` (secondary).
Env: torch 2.4.0+cu121, diffusers 0.30.2, peft 0.12.0, transformers 4.44.2.

## Native correctness criteria (final validation, actual GPU trajectories)
- Native/source/hooked FP32 loss agreement on identical cloned operands: |Δ| ≤ 1e-6 + 1e-5·|ref|.
- FULL student-output gradient w.r.t. `pred_flat` (`autograd.grad(retain_graph=True)`): RMS error ≤ 1e-7 + 1e-4·RMS(ref); max-abs error ≤ 1e-7 + 1e-4·maxabs(ref); shapes equal; all finite.
- Independent FP64 references; synthetic full-dimension cases d=65,536; no minimum-gradient-fraction conditioning; exactly zero gradient for clearly inactive ReLU cases (branch-ambiguity band |margin| ≤ 1e-5·(|D+|+|term|+L) is disclosed, not hidden).
- Asserted unchanged: parameter-gradient buffers, operands, RNG state (probe is read-only). Recorded: ‖v−b‖, bitwise student-vs-target equality (at pre-update indices B=0 so equality is expected; post-update indices are the discriminating ones).
- Also checked per trajectory: teacher detachment, FP32 preparation, reduction `(gamma*per_sample.sum())/n_latents` (same-device bitwise), PEFT inventory (76 modules, 152 tensors, 7,471,104 trainable), frozen-base unchanged digest and optimizer exclusion, dropout/mode (train mode, no-grad student sampling), finite updates, saved-and-reloaded adapter restoration (tensors, keys, shapes), preview-decode skip counter.
- Fresh validation seeds (not used in engineering, engineering seed 6901): native_gem 6301, cpr 6302, centered_radial 6303, truncated_anchor 6304, anchor_only 6305, axial_margin 6306; lr 1e-3, dev target refrigerator.
- Fresh capture indices: native (132 states = 12 updates) [0, 12, 121]; all other arms (44 states = 4 updates) [0, 12, 33]. An 11-state window gives one update per 11 states, so 12/33/121 are post-update.
- The CPR 44-state validation trajectory is the short actual-CPR integration trajectory; production proceeds automatically if every check passes.

## Detectors
Primary RT-DETR threshold 0.30 frozen; 0.20 and 0.50 are sensitivity only. Complete postprocessed predictions are saved down to 0.0. Old ≥16/20 qualification result (failed) is preserved and is not an execution veto; the detectors are NOT called accuracy-qualified.

## Design
Dev target refrigerator (allowed oven, sink, bus, train); test targets toilet and bed (allowed couch, potted plant, bus, train). New case IDs `x.<stage>.<kind>.<cls>.<jj>` and noise seeds in `configs/cases.json` (SHA of noise recorded). Shared initial noise across matched conditions; poor M0 cases are not replaced.
Training seeds: dev 6101; test 6201 and 6202. Settings: qk_dual r/α 16, dropout 0.1, η 1, γ 100, AdamW, lr 1e-3 and 5e-4, max 250 processed states, `max_parallel_timesteps=1`, BF16 backbone, FP32 loss. Evaluation 512×512, 28 steps, guidance 3.5.
Objectives (detached teachers a, b; FP32 flattened student v; ε=1e-6; L=‖b−a‖; e=(b−a)/L if L>1e-6 else 0; α=(b−v)·e; D+=‖v−a+ε·1‖, D−=‖v−b+ε·1‖):
native relu(D+ − ηD−); cpr relu(D+ − ηα); centered_radial relu(D+ − η·sqrt(‖v−b‖²+ε²d)); truncated_anchor relu(D+ − detach(ηL/(1+η))); anchor_only D+; axial relu(L − (1+η)α) (L ≤ 1e-6 is an explicit domain failure, kept as a failed row). Native scaling `(gamma*per_sample.sum())/n_latents` is preserved.
Production order at lr 1e-3: native, cpr, centered_radial, truncated_anchor, anchor_only, axial; then 5e-4.
Dev: 12 trajectories, checkpoints 66/132/250 states, 12 target + 12 allowed cases, 888 images (864 edited + 24 M0). Test: 24 endpoints, 20 target + 20 allowed cases each, 1,040 images (960 + 80 M0). Primary 1,928 images; the 2,200-image ceiling leaves 272 for technical/verification images.

## Selection (development only; locked in Git before any test-model output)
Per arm: among development candidates with S ≥ 0.80 choose max R; ties → fewer states, lower lr, stable ID. If none reaches 0.80, choose highest S then R and disclose `efficacy_unmatched`. If undefined, use the 5e-4/250 fallback and mark `measurement_limited`. The primary non-CPR comparator is chosen by the same rule on the arm-selected controls (fallback centered_radial). Failed rows stay failed.

## Outcomes and analysis
S = target nondetection among M0-positive valid cases; R = allowed-detection retention among M0-positive valid cases (threshold 0.30 strict `score > thr`). All-case and per-class counts, secondary-detector eligibility and common-intersection results are reported. Unreadable, nonfinite, wrong-size or near-constant outputs (max channel std ≤ 1/255) never count as S or R and stay in denominators.
Bootstrap: 10,000 paired case-bootstrap resamples, seed `BOOTSTRAP_SEED` (from the project id), 95% percentile intervals; target and allowed bootstrapped separately, allowed stratified by class; the two training seeds are averaged within each case; individual seeds also reported.
`PROVISIONAL_DETECTOR_DEFINED_ADVANTAGE`: complete measurements; each object primary and common target eligibility ≥16/20 and each allowed class ≥4/5; CPR S ≥ 0.80 on both objects; against all five controls on both objects point ΔS ≥ −0.05, ΔR ≥ 0.10, lower-95% ΔS ≥ −0.05 and lower-95% ΔR > 0; per seed ΔS ≥ −0.05 and ΔR > 0; secondary/common corroboration.
`NO_ADVANTAGE_IN_THIS_SCREEN`: same development-selected comparator on both objects; comparator S ≥ 0.80; point and upper-95% ΔS ≤ 0.05; point and upper-95% ΔR < 0.10; per-seed inequalities; secondary corroboration with adequate denominators. Otherwise `INCONCLUSIVE` or `MEASUREMENT_LIMITED`. Threshold sensitivity (0.20, 0.50) and direction reversals are reported.

## Throughput
Measured on 4 frozen DEVELOPMENT technical images (see projection record) with a 30% planning margin; actual costs come from the ledger.

## Not claimed
No claim of detector accuracy qualification, semantic erasure, or general efficacy. The decision is detector-defined and screen-scoped. The blinded visual audit (VISUAL_AUDIT_ADDENDUM_01) is supplementary and cannot change the primary decision.
