# Novelty position, final — three candidates closed by primary sources, and what survives

**Date:** 2026-10-21 · all claims below verified against primary sources (PDFs supplied by the
user, plus the ESD training script from the official repo)

---

## 1. Three candidate mechanisms, all anticipated

| candidate | anticipated by | evidence |
|---|---|---|
| **ESD** — entity-masked self-distillation | **MDU**, arXiv 2605.18253 | prompt-masked frozen-θ₀ anchor, stop-gradient, forward KL, CE on retain, and MDU's own stated rationale that MDLMs natively condition on mask tokens. Already one of our baselines. |
| **Attack-aware / meta-unlearning** | **Meta-Unlearning on Diffusion Models** (Gao, Pang, Du, Hu, Deng, Lin) | "if the meta-unlearned DM undergoes malicious finetuning… related benign concepts retained within it will be triggered to self-destruct"; "compatible with most existing unlearning methods, requiring only the addition of an easy-to-implement meta objective"; first-order approximation. |
| **SSA** — schedule-split anchoring | **SPARE** arXiv 2602.07058 and **KSCU** arXiv 2507.06526 (IEEE T-IFS) | SPARE: "restricts the unlearning optimization strictly to these crucial timesteps", with explicit per-concept bands. KSCU: contiguous concept-specific interval, motivated by a coarse-vs-fine split across the schedule. |

**And the robustness problem already has a published solution we had only half-read.** UNDO
(arXiv 2506.06278): `θ_perturbed = (1−α)θ_suppressed + αβN`, then distill the suppressed
teacher's outputs into the damaged student — "the key idea is to globally damage the network to
varying degrees". At its strongest setting it "matches the robustness of a model retrained from
scratch with perfect data filtering". We had been citing this paper **only for its negative
result** (oracle matching relearns). So AFC's 2.7× reversibility is fixable by their method, not
ours. It needs an epoch of unlabeled pretraining data, which we do not have for LLaDA.

**ESD is confirmed uniform-in-t from the official repo** (`utils/esd_trainer.py` line 416:
`run_till_timestep = random.randint(0, config.num_inference_steps - 1)`). So KSCU/SPARE's
banding is a real improvement over ESD — and it is theirs.

**TimeROME-DLM (arXiv 2606.12841) already published the MDLM measurement** that factual content
localises to part of the denoising schedule, including the claim that "this temporal
(denoising-step) localisation has no analogue in the AR causal trace". Our "no AR analogue"
framing was also anticipated.

**Threat to the framing:** arXiv 2409.02908, "Masked Diffusion Models are Secretly Time-Agnostic
Masked Models". A reviewer will cite it to argue t is not a real axis for MDLMs. The defensible
reframing is **mask ratio / context visibility**, not diffusion time — which makes it closer to
BERT mask-ratio curricula than to anything diffusion-specific. Drop "diffusion-specific".

## 2. The evaluation claim is RETRACTED as stated

Claimed: *standard forget-quality metrics cannot distinguish a fluent-wrong answer from a
degenerate one, so a broken model scores as perfectly unlearned.*

**TOFU states this verbatim in its own §5.2 Limitations** (arXiv 2401.06121):

> "models that output gibberish or random words (or even random models) may assign similar
> (very low/random) probabilities to both the correct and the incorrect answers… Hence, **they
> have strong forget quality**… However, for practical purposes, models that output gibberish
> content fall very low on the model quality scale… **its pairing with model utility helps
> identify models that are no longer usable.**"

So the phenomenon is acknowledged by the benchmark authors, who also publish the mitigation.
**AltPO (arXiv 2409.13474)** is closer still: it states TOFU's forget quality "fails to penalize
these errors, as it only measures the probability of predefined sentences rather than analyzing
the generated responses", shows NPO/IdkDPO score well while incoherent, classifies forget-set
outputs by a judge rubric into incoherent / inconsistent / plausible, and KS-tests a degeneracy
statistic against the retrain reference. That covers four of our five categories.

Also relevant: **RWKU has an explicit Fluency axis** (bi/tri-gram entropy); **ROME-lineage
editing has shipped Generation Entropy since 2022**; **SemEval-2025 Task 4** installed an MMLU
floor specifically to exclude broken models; **ELM** elevates non-gibberish to a primary
criterion. **MUSE has no fluency axis** (verified by full-text grep) — that one worry was
unfounded.

## 3. EMDU's behaviour is a design choice, and our framing of it was uncharitable

We measured EMDU at DEGENERATE 1.000 and called it a failure. EMDU's own text argues the
opposite:

> "rather than fluent paraphrases of the sensitive fact. **An empty completion is a safer output
> mode**"; "a fluent paraphrase risks either regurgitating the original fact or confidently
> assert[ing]"; "Relative to MDU, **abstention avoids the gibberish-vs-fluency tradeoff**".

For a privacy threat model that is a defensible position, and it explains why EMDU reports no
fluency metric: it does not regard empty output as a fluency failure. Our partition encodes the
opposite value judgement (DEGENERATE bad, REPLACE good) and we did not flag it as a judgement.

## 4. What actually survives

| surviving claim | why it holds |
|---|---|
| **Profile matching to the retrain reference** | The pristine model does **not** abstain — ABSTAIN 0.113, REPLACE 0.470, IGNORANT 0.388. Empty output is wrong because it is maximally far from what a model that never saw the data does (EMDU TV→pristine = **1.000**), not because gibberish is unattractive. **This argument is immune to the abstention-vs-degeneracy dispute.** |
| The partition's **abstain / replace / ignorant** cuts | AltPO's instrument is one-dimensional (scalar gibberish probability, binary plausibility). Under it, an abstention, a plausible-wrong replacement and genuine ignorance are indistinguishable. Those cuts are what license the rank-instability result. |
| **Rank instability** — three single axes, three winners | Nearest published cousin is OpenUnlearning's unelaborated note of "limitations with the ranking methodology". |
| First degeneracy-aware evaluation applied to **masked diffusion LMs** | MDU and EMDU report no fluency or output-quality axis. |
| **AFC's ablation result** | separating EMDU's two simultaneous changes; the KL direction helps, FAM hurts. |
| **Eight measured negatives** | each closed against a pre-registered rule. |

## 5. Bottom line

**We do not have a novel method that beats existing methods.** Eleven candidates are closed:
eight by measurement, three by primary-source novelty checks. The remaining mechanism space
reachable from our measurements is occupied by published work.

What we have is an **ablation-and-evaluation study**: a configuration (AFC) that beats both
parent methods on mean forgetting at equal utility, a narrower-but-defensible evaluation
argument grounded in retrain-reference profile matching, a valid relearning protocol with a
negative result about our own method, and eight documented dead ends. That is the deliverable,
and it should be written as such.
