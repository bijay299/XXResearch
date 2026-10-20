# Novelty position — what we may and may not claim

Decision-critical. Updated 2026-10-20 after locating a work our first prior-art pass could
not find. Every item is tied to a primary source.

## The work that was missing, and what it occupies

**Zhang, Lin, Wu, Wang, Xin, Li, Lin, Tang, He, Wang. "Adversarial Reinforcement Learning
for Robust Diffusion Large Language Model Unlearning." ICML 2026.**
OpenReview `91bMgklyeQ` (ICML.cc/2026/Conference/Submission1466, camera-ready, public
2026-06-24) · `icml.cc/virtual/2026/poster/65904`. **No arXiv version exists**; it is cited
by EMDU with no id or URL, which is why it was initially unlocatable. Abstract verified by
the coordinator directly from the ICML site.

What the abstract establishes, in their words:

- *"we present the first comprehensive study of unlearning for DLMs"*
- *"unlearning performance in DLMs is highly sensitive to generation hyperparameters,
  highlighting the need for evaluation across diverse generation settings"*
- *"DLMs tend to reproduce unlearned information when target inputs are embedded within
  informative contexts, due to their ability to incorporate both prefix and suffix
  conditioning, which increases vulnerability to elicitation attacks"*
- Method: *"an adversarial reinforcement learning framework, where a context generator
  adversarially produces informative contexts to elicit unlearned knowledge, while the DLM
  is optimized to suppress undesired recall"*, plus *"novel components to address credit
  assignment and stability issues"*

**Their adversary is the input context. Ours is the weights.** They defend against
*elicitation* — an attacker who writes a prompt. We measure *relearning* — an attacker who
fine-tunes on leaked examples (Deeb & Roger's RTT). Different threat model, different
defense, different measurement. That distinction is what remains available to us, and it
must be stated explicitly rather than relied on implicitly.

## UPDATE — the complete algorithm is now VERIFIED

The user supplied the camera-ready PDF (16 pages), extracted to
`outputs/zhang_icml2026.txt`. Everything below is quoted or paraphrased from the paper
itself, not from a summary.

**Their setup.** Warm-up fine-tune `GSAI-ML/LLaDA-8B-Instruct` on the **TOFU forget split**
first, because — in their words — *"this warm-up fine-tuning step ensures that the diffusion
language model captures task-specific knowledge that may not be fully covered during the
pretraining of LLaDA."* They hit the same insufficient-base-recall problem we did and solved
it the same way. Datasets **TOFU and MUSE**; models LLaDA-8B-Instruct and
Dream-v0-Instruct-7B.

**Their objective.** A min-max adversarial RL framework. A trainable context generator
`f_c` emits `K` prefix-suffix pairs; the adversarial prompt is
`p̃ = α ⊕ p ⊕ [MASK] ⊕ β`; the DLM samples `G` rollouts; a **judge model** assigns a binary
reward `R = +1` if the output does *not* match the reference, `−1` if it does (Eq. 4). Both
the DLM and the generator are optimised with **GRPO**. Output probabilities are estimated
with diffu-GRPO's mean-field decomposition, not an exact likelihood.

**Their two named components, and what they actually do:**
- *"Credit assignment"* is **between the prefix and the suffix** — Eq. (5):
  `I_prefix(α) = (1/G) Σ_j [ log π(y|p̃) − log π(y|p̃\α) ]`, and analogously for the suffix.
  It is a causal-influence score over the two halves of the adversarial **context**.
- *"Stability"* is **difficulty-calibrated context generation** via a regret signal against a
  `π_strong` reference, keeping contexts at intermediate difficulty so reward variance does
  not collapse.

**Their baseline is gradient ascent only** (Eq. 3). They explicitly omit the un-unlearned
base model, and do not compare against MDU, EMDU, NPO, or RMU.

**Their infrastructure, which we cannot replicate:** Qwen2.5-14B-Instruct as judge,
Qwen3-4B-Instruct-2507 as the trainable context generator, and Qwen3-30B-A3B-Instruct-2507
as a stronger generator at inference. Three auxiliary models, one of them 30B. We are barred
from additional model-weight downloads and could not hold a 30B generator alongside an 8B
policy on 2x40 GB. **Any comparison to their method is therefore structurally unavailable to
us and must be declared as such, not quietly omitted.**

## What this verification settles

**The live risk on mask-rate / position targeting is RESOLVED IN OUR FAVOUR.** Their "credit
assignment" is prefix-versus-suffix over the adversarial context, not per-position or
per-mask-rate attribution. And both their Eq. (1) and Eq. (3) draw
`t ~ Uniform(0,1)` — the mask rate is untouched by their method. So no published MDLM
unlearning method targets mask rates or positions in the training objective, and that
region remains open.

**Their threat model is confirmed to be input-context elicitation, not weight-space
relearning.** The adversary writes prefixes and suffixes at inference. Nothing in the paper
fine-tunes the unlearned model on leaked examples. Relearning (RTT) robustness for MDLM
remains, to our knowledge, unmeasured outside our own closed study.

**An opportunity their setup reveals.** We had ruled TOFU out because the standard protocol
needs ~250,000 full-parameter SFT steps with no released checkpoint. But Zhang et al. only
warm-up fine-tune on the **forget split** (TOFU `forget01` is 40 QA pairs, `forget05` 200,
`forget10` 400) — which is exactly the inject-then-unlearn structure we already use with
synthetic facts, at a scale we can afford under LoRA. Using TOFU's published forget split as
a Tier-A fact set alongside our synthetic facts would give a **standard, published** fact
set and align our setup with theirs. Referred to the evaluation lead.

**NOT VERIFIED (nothing remaining of consequence):** their numeric results tables, which are
rendered as figures the text extractor could not read. Their relative ordering versus GA is
described in prose; we have not extracted the values and do not rely on them.

## Superseded note

**NOT VERIFIED:** the complete algorithm. The camera-ready PDF sits behind OpenReview's
Cloudflare challenge (403 on seven routes) and PMLR has not published the ICML 2026 volume.
EMDU describes it as GRPO-based. Until the PDF is read, we do **not** know whether it trains
on model-sampled trajectories or teacher-forced masked states, whether "credit assignment"
means per-step/per-position attribution, or what benchmark and metric it uses. **Obtaining
that PDF is the highest-value outstanding action and must happen before any algorithm is
frozen.** A third-party page (`lacuna.tiptreesystems.com`) makes specific method and results
claims about this paper that cannot be verified against the paper; it is marked do-not-cite,
the same failure mode as the "TNT" page that invented a method name.

## Claims now forbidden

- ❌ **"First trajectory-aware MDLM unlearning."** Their framework samples `G` rollouts
  from the DLM per adversarial context and optimises with GRPO, so on-policy DLM output
  sampling for unlearning is occupied. (Note their *probability estimation* is a mean-field
  surrogate, not an exact trajectory likelihood — but the sampling is on-policy.)
- ❌ **"First study of robustness for MDLM unlearning."** They claim the first comprehensive
  DLM unlearning study and explicitly address robustness.
- ❌ **"MDLM unlearning robustness is unexamined."** It is examined, for elicitation.
- ❌ Any claim resting on the premise that no MDLM unlearning method trains on anything but
  uniformly-sampled teacher-forced masks — unsafe until their PDF is read.

## Claims that remain available, each with its qualifier

1. **Relearning (weight-space) robustness for MDLM.** Zhang et al. target elicitation via
   context; MDU and EMDU do not evaluate robustness at all. Our measurement — an unlearned
   LLaDA at the never-memorized floor returning to 0.605 on held-out facts after 10
   fine-tuning steps on 8 leaked facts — is, to our knowledge, not reported anywhere.
   *Qualifier: "to our knowledge", and contingent on their PDF.*
2. **The answer-token coupling mechanism.** That unlearning amplifies same-answer per-fact
   gradient alignment (0.388 → 0.916) while leaving every other structural quantity
   unchanged, and that this predicts a per-fact relearning split (1.000 vs 0.514). Nothing
   in any primary source we have read reports this. *This is our strongest independent
   contribution and it is a finding about mechanism, not a method.*
3. **The ignorance-reference target via verified-unknown-subject substitution** —
   category (c) in the earlier assessment, and the cleanest of the three. MDU and EMDU use
   hand-built masked anchors; Zhang et al. suppress via RL reward rather than target
   matching; nothing resembling substitution-estimated ignorance appears in any primary
   source. *Qualifier: appears open; recheck against their PDF.*
4. **Mask-rate / position targeting.** **OPEN, risk now resolved.** Zhang et al.'s "credit
   assignment" is prefix-versus-suffix over the adversarial context (their Eq. 5), not
   per-position attribution, and both their Eq. (1) and Eq. (3) draw `t ~ Uniform(0,1)`.
   MDU published a stored-knowledge token-role diagnostic it does not use in its objective.
   No published MDLM unlearning method targets mask rates or positions.

## Fan et al. ICML 2025: their GP row and our negative result are not comparable

Verified in the released code (`OPTML-Group/Unlearn-Smooth`, `GA.py`, single commit
`a4ec8e7300`):

```python
251   forget_grad_norm = 0.0
258               forget_grad_norm += f_grad_f.pow(2).sum().item()
260   forget_grad_norm = forget_grad_norm ** 0.5
262   loss = forget_loss + self.gamma * retain_loss + 0.01 * forget_grad_norm
```

`.item()` detaches, so by line 262 `forget_grad_norm` is a Python float and the penalty
contributes a constant to the loss and **nothing to the gradient**. No rescuing path exists:
`autograd.grad` does not accumulate into `.grad`, `base.py` holds no penalty logic, and
`NPO_FT_GNR` inherits HuggingFace's `training_step` (only `SAMTrainer`/`RSTrainer` override
it). The coefficient `0.01` is also outside the paper's own stated grid of [1, 10], and the
launch script passes `gnr_rho=10.0`, which is ignored. `NPO+FT+CR` has no dispatch branch
and crashes as released.

**Consequence, stated carefully.** Their GP row appears to measure NPO+FT with no penalty
applied. So our closed study's `NOT SUPPORTED` verdict on a gradient-norm defense **does not
contradict their GP result — the two are not comparable**, and that non-comparability is the
finding. We do **not** assert the paper errs: their Eq. (10) is standard and correctly
derived. This is a code-versus-method discrepancy, and we cannot determine what code
produced the published numbers. SAM, not GP, is the variant worth testing as a comparator.

## Consequence for our own hypothesis register

- **H2 (decode-trajectory / mask-rate targeting)**: novelty materially damaged on the
  trajectory half; the mask-rate/position half survives with a live risk. H2 may still be
  the right *science*, but it can no longer be framed as opening an unoccupied region.
- **H3 (ignorance-matching rather than suppression)**: now the strongest novelty position,
  and independently the target our own formalization derived from the deconvolution. It also
  makes the falsifiable prediction that separates it from the exclusion target — it should
  **not** amplify same-answer coupling.

The method therefore proceeds from H3 plus Finding 1, with H2's position/mask-rate element
held as a secondary, separately-ablated component rather than a headline.
