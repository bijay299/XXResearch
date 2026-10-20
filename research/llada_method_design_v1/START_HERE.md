# START HERE — llada_method_design_v1

**Decision: `NO_DISTINCT_METHOD_AFTER_CONSTRUCTION`** (engineering/research decision; not a claim of proven novelty or benchmark superiority).

We constructed, specified, proved what could be proved about, implemented and independently validated one method,
**FX-Mix**. Under the locked CPU validation (3 fresh worlds × 2 seeds, 8 arms, run by the independent verifier G) its only
non-known ingredient — the training-state design — gave **no held-out-leakage gain** over the strongest straightforward
combination **SC-TF** (mean Δ +0.007, 4/6 units; pre-registered margin 0.03 and 5/6) and none over compute-matched SC-TF
(0.027 vs 0.027), at lower retention. The implemented best candidate therefore **reduces to SC-TF = "TNFF-in-MDU"**: the
TNT/TNFF exclusion target (UNDIAL with γ→∞) at the same state, on LLaDA/MDU-style random teacher-forced states, with a
retain SFT loss — all known ingredients. Details: [`RESULTS.md`](RESULTS.md) §3, [`results/final/G_REPORT.md`](results/final/G_REPORT.md).

## The method (one paragraph)
**FX-Mix** — fact-free eXclusion distillation on a native-sampler state Mixture — is a LoRA update for LLaDA. On a
forget prompt, at every masked position of every *fact-free* state (no fact token visible), it distils the frozen
reference's own prediction at the same state with the fact tokens excluded (the closed-form I-projection
r_ε = argmin_{q(F)≤ε} KL(q‖p0); the TNT/TNFF target, UNDIAL with γ→∞). The states are teacher-forced answers with the
fact positions always masked (25 %) plus states recorded from the *current student's own native LLaDA sampler* under a
pool of decoding configurations (greedy low-confidence; random-order T=1 one and several tokens per step), each rollout
stopped at its first fact commit (75 %). A standard LLaDA SFT retain loss is added. Deployment = merged LoRA, plain
sampler, no prompt, no filter. Definitions: [`METHOD.md`](METHOD.md); authoritative spec: [`SPEC.md`](SPEC.md) v1.3.1.

## Hypothesis tested
Primary advantage = **config-robust forgetting**: at matched retention, lower generated-answer fact leakage on
*held-out* native decoding configurations than the strongest straightforward combination of the same ingredients
(SC-TF: same target on LLaDA/MDU-style random teacher-forced states; SC: MDU's loop with its rollout flag). Retention,
forget-answer fluency and compute are constraints. Forgetting = reduced recall (type i) only.

## Implementation status
- CPU reference implementation, controls (SC-TF, SC, SC-TF-cm, FX-Mix-greedy, EMDU, MDU τ=1, NPO), tiny bidirectional
  mask predictor with a LLaDA-faithful SFT loss and a port of the pinned `generate`: `src/mdmu/`; the 38 locked tests pass (G and coordinator runs).
- LLaDA-8B integration (`integration/llada_fxmix.py`, `configs/hyperplane_fxmix.yaml`) with FX-Mix and the comparator
  arms SC-TF, EMDU, MDU τ=1, NPO (comparators added post-validation; 10 further dry-run tests; 48/48 tests pass in total):
  dry-run against a tiny random model only; **real-model execution UNTESTED**; LoRA module names UNVERIFIED (HF blocked here).
- Locked CPU validation (`VALIDATION_SPEC.md`, `configs/LOCK.json`, lock commit `13ec821`) run by the independent
  verifier G: `results/final/` (96/96 runs; one procedural post-lock edit with no numeric effect, DECISION_LOG #20;
  post-lock corrections to locked text in [`SPEC_ERRATA.md`](SPEC_ERRATA.md)).

## Next action
Do not launch a GPU study of FX-Mix as a new method. If Hyperplane time is available after the separate feasibility
report confirms its dependencies, the only justified step is **S0 + S1 of [`HYPERPLANE_PLAN.md`](HYPERPLANE_PLAN.md)**
(≈ 3–3.5 GPU-h): train SC-TF once on injected TOFU facts and test whether it leaves residual held-out-config leakage with
off-manifold decision states at 8B scale; stop if either pre-registered stop rule fires (the CPU analogue fires).

## Closing summary
- **What the method actually does.** It edits LoRA weights so that, on forget prompts, the model's per-position
  predictions match the original model's predictions with the fact tokens removed and renormalised, at every fact-free
  denoising state that its own native sampler visits under several decoding configurations (plus teacher-forced states
  with the fact masked); a retain SFT loss is added. It targets reduced recall in generated answers only.
- **What is potentially new.** Only the state-distribution design (multi-config native-sampler pool + first-commit
  truncation + fact-free teacher-forced states) and its sampler-level justification. The target (TNT/TNFF, UNDIAL γ→∞),
  the target-first KL and the MDU-anchor diagnosis (EMDU), and the on-policy MDM loop (MDU code) are known. The design was
  not shown to matter, so no novelty is claimed.
- **What the mathematics establishes.** Lemma 1: the closed-form I-projection target, its KL value and its θ0 gradient
  (vanishes where the reference puts ≤ ε on the fact) — standard. Prop. 1: an exact first-commit decomposition of
  leakage for the pinned LLaDA sampler and a hazard bound valid for all T ∈ [0,1], both remasking rules and any
  schedule (cfg 0), usable as a post-hoc certificate — standard tools. Prop. 2: in the zero-parameter-sharing class,
  exact training on teacher-forced states leaves worst-case leakage equal to the target sampler's escape probability
  (≥ 1−1/L, or 1 for multi-token facts, under random-order decoding), while training on the sampler's own visitation
  gives zero — an identifiability statement, not a guarantee for networks. No result links the training loss to leakage
  at scale.
- **What the CPU evidence establishes.** In a 2-layer bidirectional mask predictor with fresh synthetic facts: the
  state design is redundant (SC-TF ≈ FX-Mix; compute-matched SC-TF = FX-Mix at higher retention) even though 86 % of
  SC-TF's residual decision states are off-manifold; MDU's single-greedy rollout flag is worse than plain TF states;
  exclusion-target arms keep partly well-formed forget answers (0.35–0.39) where NPO, MDU and EMDU degenerate (0.00),
  while EMDU has the lowest leakage and best retention. None of this is evidence about LLaDA-8B.
- **What remains unknown.** Whether an 8B LLaDA generalises teacher-forced suppression to its off-manifold decision
  states as the toy does; whether a pretrained model's renormalised tail gives fluent alternatives (the toy's tail is
  only ~58 % well-formed under greedy decoding); LLaDA's LoRA module names (HF blocked here); whether the adversarial-RL
  dLLM paper already randomises decoding configs (not accessible).
- **Is one Hyperplane test justified?** Not as a method test. At most the S0+S1 gate (≈ 3–3.5 GPU-h) as a bounded
  falsification of whether the CPU negative transfers to 8B scale; it is optional and nothing else depends on it.

## Map
| need | file |
|---|---|
| decision record, disagreements | [`DECISION_LOG.md`](DECISION_LOG.md), [`agents/AGENT_LEDGER.md`](agents/AGENT_LEDGER.md) |
| mathematics | [`METHOD.md`](METHOD.md), [`THEORY.tex`](THEORY.tex) |
| prior art and novelty | [`PRIOR_ART.md`](PRIOR_ART.md), [`agents/D_prior_art.md`](agents/D_prior_art.md) |
| results (dev vs final) | [`RESULTS.md`](RESULTS.md), [`results/final/G_REPORT.md`](results/final/G_REPORT.md), [`SPEC_ERRATA.md`](SPEC_ERRATA.md) |
| real-model plan | [`HYPERPLANE_PLAN.md`](HYPERPLANE_PLAN.md) |
| resources, evidence | [`ledger/SUMMARY.md`](ledger/SUMMARY.md), [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) |
| agent work products | `agents/` (proposals A/B/C, objections `OBJ_*`, responses `RESP_*`, D, E) |
