# Novelty review — ESD is anticipated by MDU, which is already one of our baselines

**Date:** 2026-10-21
**Verdict: ANTICIPATED. The candidate is closed as a method.**
**Anticipating work:** MDU, *Machine Unlearning for Masked Diffusion Language Models*,
[arXiv 2605.18253](https://arxiv.org/abs/2605.18253), code `github.com/leegeoru/MDU`

---

## 1. What I proposed, and what MDU already published

I proposed "ESD — Entity-Masked Self-Distillation": train the model's answer distribution
toward its **own** frozen-θ₀ distribution under a prompt whose entity-mention tokens are
replaced by the mask token, stop-gradient on the target, cross-entropy on retain; and I claimed
the enabling property was that a masked diffusion LM natively conditions on mask tokens, so no
second model and no retrained reference is needed.

MDU's abstract, verbatim:

> "During fine-tuning, MDLMs learn to recover responses from masked response states conditioned
> on a prompt, thereby shifting their predictions **from a prompt-masked unconditional
> distribution toward a prompt-conditional distribution**. … MDU **minimizes a forward KL
> divergence from the prompt-conditional prediction to a prompt-masked unconditional anchor** at
> every masked response position, with a temperature scaling parameter to control the
> privacy-utility trade-off."

Its anchor is `p_{θ₀}(· | m, y_t)` with `m` a null/masked prompt and θ₀ **frozen** at the start
of unlearning; retain uses ordinary SFT cross-entropy; the released implementation computes the
anchor under `torch.no_grad()` against a frozen reference with KL direction
`"forward : KL(p_c ‖ sg(p_target))"`.

| my claimed property | status |
|---|---|
| target = same model's own distribution with prompt tokens → mask token | **MDU** |
| stop-gradient on that target, frozen θ₀ | **MDU** |
| no second model, no retain-only retrained reference, no extra fine-tuning stage | **MDU** |
| retain preserved with ordinary cross-entropy | **MDU** |
| "MDLMs natively condition on mask tokens, so the prior is available from one model" | **MDU's own stated rationale** |

This is not a partial overlap. **ESD as specified is an ablation of MDU's anchor
construction**, and MDU is a baseline we have already run and already hold matched results
for. I am recording this as a process failure on my part: the candidate was constructed from
first principles without first re-reading the mechanism of a baseline already in our own
comparator set.

The same self-ignorance-teacher pattern also appears independently in multimodal unlearning —
FASD ([2610.14605](https://arxiv.org/abs/2610.14605)), which "distills appropriate refusals from
an ignorance-prompted copy of the model itself" and masks target tokens from the teacher
distribution; and CUSE ([2610.14763](https://arxiv.org/abs/2610.14763)), which builds
counterfactual inputs by removing target-related evidence and self-distils from a frozen copy.
So the idea is not novel as a concept nor as an unlearning mechanism.

## 2. What the review confirmed is NOT prior art

Checked against primary sources, these are clean:

- **ROME** ([2202.05262](https://arxiv.org/abs/2202.05262)) corrupts subject token embeddings
  with Gaussian noise *purely to create a damaged run for causal tracing*; its objective is
  anchored on the **clean** model. The corrupted-subject output distribution is never an
  optimisation target anywhere in ROME. (Verified from the ar5iv rendering of the body; the
  canonical PDF returned HTTP 429. **MEMIT ([2210.07229](https://arxiv.org/abs/2210.07229)) was
  not verified at all** — the claim is confirmed for ROME only.)
- No "null prompt / counterfactual prompt as *editing target*" work was found.
- No work framing arbitrary-position mask conditioning as a general elicitation capability
  beyond generation. That is an available framing, but a framing is not a method.

## 3. What survives — and it is sharper than what I started with

MDU masks the **entire** prompt. Its anchor is therefore the model's **unconditional** prior,
carrying neither the relation nor the syntax of the question. Our own matched baseline run
measured exactly what matching that anchor does:

```
MDU τ=1.0   DISCLOSE 0.000   DEGENERATE 1.000   degeneracy_breakdown {"empty": 64}
```

It emits the **empty string on all 64 forget facts**. Read together with MDU's construction,
that is not a forgetting–utility trade-off being paid. **It is the arm reaching its target,
because the target is degenerate**: on a masked answer canvas the unconditional prior puts its
mass on EOS. And τ cannot fix it, because τ only interpolates toward a degenerate target.

This reframes the project's central measured gap — the strict bimodality of all 15 baseline
grid points — from "forgetting is hard" to a specific, falsifiable claim about anchor
construction:

> **ACG.** MDU's collapse into DEGENERATE is an artifact of **anchor granularity**. An anchor
> that masks only the entity span retains the relation and the syntax, is non-degenerate and
> on-topic, and therefore has its minimum in REPLACE/IGNORANT.

Neither MDU nor EMDU ([2610.25447](https://arxiv.org/abs/2610.25447)) implements, ablates or
discusses entity-selective prompt masking, and **neither evaluates shared-attribute
retention** — forgetting "Alice lives in Paris" while preserving "Bob lives in Paris". That is
the user's own first-principles framing example and it is an evaluation gap in the literature.
Our six-cell partition and ASCR-12s are built to read exactly the quantity the claim is about —
*which cell the mass lands in* — which a scalar forget-quality metric hides.

This is settled at the **anchor level with no training**: `scripts/entity_mask_target.py` now
decodes four arms — `natural`, `all_masked` (MDU's published construction), `entity_masked`,
`control_masked` (length-matched non-entity) — with kill rule **K4** firing if the two anchors'
DEGENERATE rates differ by ≤0.15, in which case entity-selective masking is a hyperparameter of
MDU rather than a mechanism and the line closes.

## 4. Hazards this review raised that must be answered, not sidestepped

| hazard | source | consequence for us |
|---|---|---|
| **EMDU's gradient-vanishing theorem.** Anchor-last KL has `(∇_z L)_j → 0` at memorised near-one-hot predictions; anchor-first gives `→ 1−p_j > 0`. | [2610.25447](https://arxiv.org/abs/2610.25447) Thm 2 | KL direction is not free. `train_esd.py` uses `F.kl_div(log_student, log_target, log_target=True)` = Σ p_t(log p_t − log p_s), i.e. **anchor-first** — the recommended direction. **To be verified against the PDF.** |
| **EMDU already works the masking-granularity axis** — "random answer masking can leave key forget tokens visible to the anchor even when the question is fully masked", fixed by full answer-span masking. | same | EMDU occupies *answer-side* granularity. Our claim is *prompt-side* entity granularity. The distinction must be stated precisely or the contribution collapses. |
| **Even a perfect ignorance oracle relearns.** Oracle matching to a retain-only retrained model still recovers under finetuning attack. | [2506.06278](https://arxiv.org/abs/2506.06278) | Undercuts the whole anchor-matching family on robustness. Our own relearning study (`a67d610044d0`) independently found MDLM unlearning reversible, 0.049 → 0.605 in 10 steps. Any claim must be made on the trade-off we define, not on robustness. |
| **Counterfactual-target hazards.** Training toward fabricated alternatives causes knowledge conflict and hallucination spillover. | [2605.27083](https://arxiv.org/abs/2605.27083) | A predictable objection; the retain/neighbourhood measurement must be able to detect it. |
| **ATWU is orthogonal, not a rival anchor method.** It learns token forget-specificity with a linear scorer over hidden states, claiming SOTA on TOFU and RWKU and explicitly beating auxiliary-model approaches. | [2606.06320](https://arxiv.org/abs/2606.06320) | Still mandatory as a baseline, and **composable** with an anchor change rather than exclusive to it. That matters for positioning. |
| **Added requirement.** Entity-selective masking needs an entity span in the prompt; MDU needs nothing. | review | A strictly added requirement that must be paid for in results. Our own measurement: 365 of 400 forget facts mention their entity, 20/20 authors resolved at ≥0.90 question coverage. |

## 5. PDFs needed (cannot be accessed from here)

The review could not reach these, and two of them could **collapse the remaining difference**,
so they are requested before anything is committed:

1. **arXiv 2605.18253 (MDU)** — full PDF incl. appendix. *If there is an appendix ablation on
   prompt-masking granularity, ACG is closed too.* Highest priority.
2. **arXiv 2610.25447 (EMDU)** — full PDF incl. appendix, to confirm Theorem 2's statement and
   the exact KL direction, and whether prompt-side granularity is touched.
3. **arXiv 2210.07229 (MEMIT)** — to extend the §2 ROME finding to MEMIT, which was not verified.
4. **arXiv 2202.05262 (ROME)** — canonical PDF, to confirm quotes taken from the ar5iv rendering.

Also noted: `WebSearch` was unavailable (session budget exhausted), so discovery ran through the
arXiv API, which searches **abstracts only**. A method described in a paper's body without
abstract-level keywords could have been missed, and OpenReview and ACL Anthology were not
swept. The novelty position in this document is therefore provisional on those four PDFs.
