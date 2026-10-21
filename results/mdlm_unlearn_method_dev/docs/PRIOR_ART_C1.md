# PRIOR_ART_C1 — Prior art for the C1 mechanism (retain-reference distillation)

**Date:** 2026-10-20
**Scope:** Targeted prior-art adjudication for the C1 base mechanism only.
**Primary sources:** read in full (PDF text extracted and read directly, not via summarizer) unless marked otherwise.

## C1 as assessed here

> Train the student to match, at forget-set states, the output distribution of a reference model
> trained **only on the retain data** (a model that never saw the forget facts). Masked-diffusion
> denoising objective, KL / soft-cross-entropy to that reference's distribution at masked answer
> positions, alongside a retain loss. LoRA, not full-parameter. Base model LLaDA-8B.

The literature already has a name for this training signal: **oracle matching**. Two papers name it
and study it directly. Neither is for a diffusion LM.

---

## 1. Complete algorithm of arXiv 2506.06278

**Title:** Distillation Robustifies Unlearning
**First author:** Bruce W. Lee (ML Alignment & Theory Scholars / UPenn)
**Full author list:** Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath,
Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner
**Year / venue:** NeurIPS 2025. v1 2025-06-06, v2 2025-06-09, **v3 2025-10-24** (version read here: v3)
**URL:** https://arxiv.org/abs/2506.06278
**Code:** https://github.com/AddieFoote/distillation-robustify-unlearning

### 1.0 The critical structural fact

This paper contains **two different algorithms**, and only one of them is the paper's proposal:

| | Section 3 "Oracle Matching" | Section 5 "UNDO" (the proposal) |
|---|---|---|
| Role in paper | **negative-result baseline** | proposed method |
| Teacher | model pretrained on **retain data only** | a model unlearned by GradDiff/MaxEnt/RMU |
| Student init | the reference model (and, separately, random init) | noised copy of the unlearned model |
| Verdict in paper | "does not guarantee robust unlearning" | establishes a new Pareto frontier |

**C1's mechanism is the Section 3 baseline, not UNDO.** This is the single most important finding
in this document and it is addressed in §2.

### 1.1 Section 3 — Oracle Matching (the C1-equivalent procedure)

| Field | Content |
|---|---|
| **Training target / loss** | Forward KL divergence from student to oracle, on output token distributions. The paper writes: *"We match the reference model's outputs with those of the oracle model using the Kullback-Leibler Divergence (KL Divergence) as the loss function."* Appendix A: *"We implemented oracle matching using knowledge distillation with forward KL divergence loss [30] on the pretraining dataset, as also explored in the Gemma 2 technical report."* No equation is numbered for this loss; the paper states it in prose only. |
| **What the teacher is** | The **oracle model**, defined as *"one trained on the reference model's pretraining dataset with all content directly related to the undesirable capability perfectly filtered out… simulating perfect capability removal by never learning undesired capabilities in the first place."* |
| **How teacher is constructed** | **Pretrained from scratch on retain data only.** Language: Gemma-2-0.1B on ~1B tokens of FineWeb-Edu English (base model: same arch on 2B tokens, 1B English + 1B Korean). Arithmetic: Gemma-2-0.3B, 1000 steps, forget set (mul/div) replaced with additional English FineWeb data. This is a true retrained-from-scratch retain-only reference, not a perturbed or unlearned model. |
| **States / inputs distillation applied at** | **The full pretraining dataset — retain AND forget data mixed** (1:1 for language, 1:3 for arithmetic), all token positions, standard next-token distributions. Section 3: *"We finetune a reference model to match an oracle model's outputs on both the retain and forget sets."* Not restricted to forget-set inputs, and not restricted to answer positions. |
| **Gradient path / stop-gradients** | Plain loss descent, AdamW, cosine LR schedule. Teacher is frozen (standard forward-KL distillation; the paper does not discuss a stop-gradient explicitly — **not stated** as such, but implied by the teacher being a separately pretrained fixed model). No meta-learning, no bilevel optimization, no adversarial inner loop anywhere in the paper. |
| **Preservation mechanism** | **No separate retain-loss term.** Preservation is implicit: the KL is computed over a data mixture that includes retain data, so matching the oracle on retain inputs *is* the retain objective. There is no weighted sum of forget and retain losses in Section 3. |
| **Parameterization** | **Full-parameter.** LoRA/PEFT is never mentioned anywhere in the paper (verified by grep over the full 33-page text: zero occurrences of "LoRA"/"PEFT"/"adapter"). |
| **Model sizes** | Custom scaled-down Gemma-2: **Gemma-2-0.1B** (~100M, 14 layers, hidden 320) for language; **Gemma-2-0.3B** (~300M, 14 layers, hidden 768) for arithmetic. |
| **Hyperparameters (Table 3)** | LR 4e-4 (lang) / 7e-4 (arith), min LR 4e-5/7e-5, batch 4/15, grad accum 160/24, weight decay 0.1/0, 1 epoch (arith capped 1000 steps), cosine, 50 warmup, seq len 2048/256. |
| **Benchmarks / metrics** | Synthetic only in Section 3: CE loss on held-out English (retain) vs Korean (forget), 500K tokens each; accuracy on held-out add/sub vs mul/div. |
| **Result — what it buys** | **A negative result.** Distillation loss *"tends to zero"* in both settings — i.e. effectively perfect behavioral agreement with the oracle is achieved. Yet under relearning attack *"Student (Reference) quickly learns the forget distribution compared to the Student (Random) and Oracle Teacher models, which are more robust. This pattern holds both for the worst-case adversary and uniformly across all learning rates."* Conclusion heading: **"Oracle Matching Does Not Guarantee Robust Unlearning."** |
| **Key caveat on that negative result** | The negative result is **specifically about robustness to relearning**, not about behavioral forget quality. The paper is explicit that behavioral agreement *was* achieved ("the distillation loss tends to zero"). Section 3 is silent on, and does not contradict, the proposition that oracle matching gives strong *behavioral* forget metrics. See §2.3. |

Separately, Section 3 reports a **second** oracle-matching arm: distilling the same oracle into a
**randomly initialized** student. That arm **is** robust — *"a model whose relearning speed on the
forget distribution closely matches that of the oracle itself."* So the paper's finding is not
"oracle matching fails" but "oracle matching **into the reference model's own parameters** fails;
oracle matching into fresh parameters succeeds." The failure is attributed to parameter-space
"latent traces," not to the training signal.

### 1.2 Section 5 — UNDO (the paper's actual proposal)

| Field | Content |
|---|---|
| **Three stages** | (i) **Unlearn**: produce a behaviorally suppressed teacher with a standard method (GradDiff, MaxEnt, or RMU). (ii) **Noise**: perturb the suppressed weights to damage the network. (iii) **Distill**: repair the damaged student by distilling the teacher's outputs. |
| **Perturbation equation (as written, unnumbered)** | `θ_perturbed = (1 − α) θ_suppressed + α β N`, with mixing coefficient α ∈ [0,1], noise scale β ∈ ℝ₊, and N sampled via **Xavier initialization**. *"when α = 1 and β = 1, the perturbation reduces to random initialization."* Setting β = 0 (pure shrinkage) is noted to also be effective. |
| **Distillation loss** | **Forward KL divergence on the outputs**, same protocol as Section 3. |
| **Teacher in UNDO** | **The unlearned/suppressed model — NOT the oracle.** This is the whole point: UNDO never requires an oracle, which is what makes it cheaper than data filtering. |
| **Loss terms of the unlearning stage (quoted as the paper writes them)** | GradDiff: `L(θ) := −L_forget(θ) + L_retain(θ)`, both cross-entropy. MaxEnt: `L(θ) := L_uniform(θ) + L_retain(θ)`, where L_uniform is a KL pushing the forget-input output distribution toward uniform. RMU: `L(θ) := L_misdirect(θ) + α · L_preserve(θ)`, MSE on activations toward a random direction (forget) and toward the original model's activations (retain). These are cited baselines, not UNDO's contribution. |
| **Gradient path** | Plain loss descent throughout. No stop-gradient discussion. |
| **Stopping rule** | Distill until the student reaches ≥95% of the teacher's retain performance, or one full epoch of pretraining data, whichever first. |
| **Parameterization** | Full-parameter. No LoRA anywhere. |
| **Hyperparameters used** | Synthetic: α swept over {0.1 … 0.8}, β = 0.1 (β swept {0.05, 0.1, 0.5, 1.0} in Fig. 6). WMDP: **α = 0.25, β = 0**. Headline comparison (Fig. 7): α = 0.6, β = 0.1, teacher = MaxEnt. |
| **Models** | Gemma-2-0.1B, Gemma-2-0.3B (synthetic); **Gemma-2-2B** (WMDP and TOFU); **Phi-1.5** (TOFU). |
| **Benchmarks** | Synthetic language (English retain / Korean forget), synthetic arithmetic (add-sub retain / mul-div forget), **WMDP-Bio and WMDP-Cyber** (retain measured by MMLU), and **TOFU** (Appendix F, added during review). |
| **Baselines compared** | GradDiff, MaxEnt, RMU, RepNoise, SAM, **UNDIAL** (a distillation-based unlearning method), and Data Filtering (gold standard). |
| **Headline claims** | Matches retrained-from-scratch robustness at **60–80% of the compute** with only **0.01% of pretraining data labeled**; new Pareto frontier on synthetic tasks; improves relearning resilience on WMDP. |
| **Stated limitation (their own words)** | On WMDP, UNDO is **4.88% lower on MMLU** on average, and *"other methods achieve a similar trade-off between robust forget and retain performance in this setting."* They attribute this to undertraining (distilling on only 0.015% of the pretraining corpus size) — i.e. **the clean Pareto win is demonstrated on synthetic tasks, and is weaker/not clearly established on the realistic benchmark.** |

### 1.3 Robustness: to *what* adversary

The user's question — relearning/finetuning or prompt-based elicitation — has a clean answer:
**relearning/finetuning, essentially exclusively.**

- Stated elicitation mechanism: *"We use finetuning on the forget set, commonly referred to as a
  relearning attack, as our elicitation method."*
- Strength: 500 relearning steps, with an **adversarial learning-rate sweep over 10 values**
  {1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6}, reporting the **max elicited forget
  performance** (worst-case adversary) at each step. ~8M tokens (language), ~2M (arithmetic).
- WMDP: **seven** relearning configurations varying data mixture (forget/retain corpora,
  forget/retain-QA, wiki-QA), sequence length (256 vs 1024), learning rate, and optionally
  shrink-perturb (α = 0.05) applied *before* relearning. All still finetuning-based.
- Appendix F.2 adds **one non-finetuning attack: INT4 quantization** (GradDiff forget truth ratio
  0.000 → 0.185 under INT4; GradDiff+Distilled 0.001 → 0.017).
- **Prompt-based elicitation / jailbreaks are mentioned only in the introduction as motivation and
  are never evaluated.** No adversarial-prompt, in-context, or many-shot elicitation experiment
  appears in the paper.

### 1.4 TOFU results (Appendix F) — relevant because our benchmark is TOFU

Added during the review period. Protocol differs from the main paper in an important way:

1. Finetune Gemma-2-2B / Phi-1.5 on the **full** TOFU dataset.
2. Apply **GradDiff** to suppress the forget authors.
3. **Distill the unlearned model back onto a fresh Gemma-2-2B / Phi-1.5 base model**, using the full
   TOFU dataset (forget and retain domains).

Here the *student init* is the original base model, used as a stand-in for random init on the grounds
that the base model never saw TOFU's fictional authors — the paper says this *"serves as an effective
substitute for a randomly initialized network with respect to this specific knowledge."*
Note: the teacher is still the **GradDiff-unlearned model, not a retain-only-finetuned reference.**
Run with the `open-unlearning` framework; 7 metrics tracked; relearning over 100 gradient updates.
Reported qualitatively (figures + one quantization table); **no TOFU forget-quality / model-utility
table of the usual kind is reported**, so these TOFU numbers are not directly comparable to
published TOFU leaderboards.

### 1.5 Availability and licence

| Item | Status |
|---|---|
| Code | https://github.com/AddieFoote/distillation-robustify-unlearning — present; `src/`, `run_pretrain_*`, `run_unlearn_*`, `run_distill_*`, `run_partial_distill_*`, `run_relearn_*`, `wmdp_question_extraction.py`, `pyproject.toml`, `uv.lock`. |
| Licence | **MIT** (LICENSE file present). *Repo contents read via summarizer, not line-by-line — treat file list as NOT VERIFIED at the individual-filename level; the MIT licence and the existence of distillation scripts are stated in the repo listing.* |
| Checkpoints | **None released.** No HuggingFace links; models must be reproduced by running the training scripts. (Consistent with the custom pretrained 100M/300M models being integral to the setup.) |
| Paper licence | **Not stated** in the text read. arXiv listing page did not surface a licence string in the fetch. |

---

## 2. Overlap verdict against C1

### 2.1 Field-by-field

| Field | C1 | 2506.06278 §3 Oracle Matching |
|---|---|---|
| Teacher | model trained on retain only | **same** — model pretrained on retain only |
| Training signal | KL / soft-CE to teacher distribution | **same** — forward KL to teacher distribution |
| Applied at | forget-set states, masked answer positions | retain+forget mixture, all token positions |
| Retain preservation | **explicit separate retain loss term** | implicit, via retain data inside the same KL |
| Student init | the model that has the forget knowledge (LLaDA-8B tuned on TOFU) | **same** for the failing arm ("Student (Reference)"); random init for the robust arm |
| Objective family | masked-diffusion denoising | autoregressive next-token |
| Parameterization | **LoRA** | **full-parameter** |
| Scale | 8B | 0.1B / 0.3B (§3); 2B elsewhere |
| Optimization | plain loss descent | **same** — plain loss descent |

### 2.2 Verdict

**Category (a) — exact equivalence — for the training signal and the teacher construction, in the
autoregressive setting.** Not "close to"; the same idea, already named in the literature as *oracle
matching*, with the teacher constructed the same way (retrain on retain only) and the same loss
(forward KL to that teacher's output distribution). 2506.06278 §3 implements it, reports that
behavioral agreement goes to near-zero KL, and names the procedure in a section heading.

The deltas against C1 are: (i) masked-diffusion vs autoregressive objective; (ii) LoRA vs
full-parameter; (iii) restriction of the KL to masked answer positions on forget states rather than
all positions of a mixed corpus; (iv) an explicit separate retain term. Taken on their own, each of
(ii)–(iv) is **category (c) — a familiar combination**: LoRA-plus-KL-distillation, position-masked
distillation losses, and weighted forget+retain sums are all standard and are combined routinely in
the LLM unlearning literature. They do not individually carry a novelty claim.

Delta (i), the port to masked diffusion, is the only one with a route to a real claim, and as of this
survey it is **category (e) — an algorithmic difference whose benefit is unestablished** — or
(d) if, and only if, a substantive interaction with the masked-diffusion structure is demonstrated
rather than asserted. There is a plausible mechanism for such an interaction (an MDLM's teacher
distribution is defined at every mask ratio, so the retain reference supplies a target over a
continuum of corruption states rather than a single factorized next-token target, which is a
genuinely larger supervision signal than the AR case), **but we have not demonstrated it and no
paper found here demonstrates it.** Asserting (d) without that experiment would not be defensible.

### 2.3 The sharpest risk, stated plainly

The risk to C1 is **not** primarily novelty. It is that **the closest prior work ran our mechanism
and published it as a negative result.** A reviewer who knows 2506.06278 will ask: *you are proposing,
as a method, the baseline that Lee et al. titled "Oracle Matching Does Not Guarantee Robust
Unlearning."*

Being scrupulously fair, that objection has a real limit, and the limit is in our favour:

1. Their negative result is **only** about robustness to **relearning/finetuning attack**. It is not
   a negative result about behavioral forget quality. They explicitly obtained near-perfect
   behavioral agreement with the oracle — the distillation loss *"tends to zero."*
2. If C1's claimed benefit is **forget quality under TOFU's own definition** (behavioral
   indistinguishability from a retain-trained model), then 2506.06278 §3 is **evidence for C1
   working on that axis**, not against it. The two findings are consistent, not contradictory.
3. Their §3 is at 0.1B–0.3B on synthetic Korean/arithmetic forget sets, full-parameter. It is not a
   test of C1 at 8B on TOFU in a masked-diffusion model.

What this forecloses: C1 **cannot** be claimed as robust unlearning, or as resistant to relearning,
without an experiment that beats their result. The prior is explicitly against it on that axis.
What it leaves open: C1 as a strong *behavioral* forget-quality method for MDLMs, where the metric
is distance to the retain-trained reference.

---

## 3. Has anyone done retain-reference distillation for a diffusion LM?

**No. Confirmed, with one named verification hole.** Our survey's enumeration is also confirmed —
and extended by one paper we had missed.

### 3.1 The complete MDLM unlearning/editing set

| # | Paper | First author / year | URL | Mechanism | Retain-trained teacher? |
|---|---|---|---|---|---|
| 1 | **Machine Unlearning for Masked Diffusion Language Models** (MDU) | Georu Lee, 2026 | https://arxiv.org/abs/2605.18253 | Forward KL from the prompt-conditional prediction to a **prompt-masked unconditional anchor** at masked response positions, temperature τ∈[0,1]; plus weighted retain SFT reconstruction `L_sft = −E[log p_θ(y_i|x,y_t)]`. | **No.** Anchor is `p_θ₀(·|m, y_t)` where θ₀ is *"the model parameters at the start of unlearning"* — the **pre-unlearning checkpoint** with the prompt nulled, not a retain-trained model. Retain term is plain SFT, not distillation. Baselines GA/GD/NPO/SimNPO/WGA/DPO; **no Retrain/gold baseline.** Code: github.com/leegeoru/MDU |
| 2 | **Enhanced Masked Diffusion Unlearning** (EMDU) | Tomer Garber, 2026 | https://arxiv.org/abs/2610.25447 | Diagnoses two MDU failure modes (random answer masking leaks forget tokens into the anchor; anchor-last KL gradient vanishes at near-one-hot memorized predictions). Fix: **anchor-first KL(p_anchor ‖ p_θ)** with the **entire answer span** masked in the anchor. LLaDA-8B + Dream-7B, TOFU + RWKU. | **No.** Anchor is *"the pre-unlearning model p_θ₀ on unprompted context."* Cites only MDU. **No mention anywhere of a retrained/oracle/gold model** — not as teacher, not as evaluation upper bound, not as future work. |
| 3 | **TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for MDLMs** | Zhengtao Yao, 2026 | https://arxiv.org/abs/2606.12841 | **Training-free, gradient-free, inference-time.** "Temporal Indirect Effect" tracing locates (layer, denoising step, module); installs a closed-form ridge-regularized low-rank residual edit reapplied each denoising step. Backbone untouched. | **No reference/teacher model of any kind.** LLaDA-8B-Base/Instruct, Dream-7B, MMaDA-8B, DiffuLLaMA-7B, LLaDA-MoE-1.4B; TOFU forget01/05/10, RWKU, MUSE, WHP, WMDP-Bio. |
| 4 | **Adversarial Reinforcement Learning for Robust Diffusion LLM Unlearning** | Zhiwei Zhang, ICML 2026 | https://icml.cc/virtual/2026/poster/65904 (OpenReview `91bMgklyeQ`) | DLM unlearning is highly sensitive to generation hyperparameters, and DLMs **re-surface forgotten content when the target appears inside an informative context** (bidirectional conditioning). Proposes **adversarial RL: a context generator produces hard contexts while the DLM suppresses recall**, plus credit-assignment and stability components. | **No retain-trained teacher mentioned.** ⚠️ **Abstract only — full text NOT VERIFIED.** |
| 5 | **Knowledge Editing for Masked Diffusion Language Models** *(we had missed this)* | Haewon Park (with Yohan Jo), 2026, EMNLP 2026 | https://arxiv.org/abs/2606.03924 | Transfers locate-then-edit (ROME-style) to MDMs. Finds edit *location* transfers from ARMs, but multi-token targets degrade far worse in MDMs **because generation passes through partially-unmasked intermediate states the edit was never optimized for.** | **No.** Not training-free, not unlearning, no retain teacher. **Distinct from #3** — there are two MDLM editing papers; ours was the training-free one. |

Our survey's list was therefore correct on all four items and incomplete by one (#5). Our
characterisation of Zhang et al. (adversarial RL, context generator, prompt-based elicitation threat
model) is confirmed from ICML's own abstract.

### 3.2 The gap, stated precisely

Every existing MDLM unlearning method derives its target from the **already-contaminated
pre-unlearning model** (MDU, EMDU: an unconditional/prompt-masked anchor of θ₀) or edits weights
directly (TimeROME-DLM, Park & Jo) or uses adversarial RL (Zhang et al.). **None instantiates a
teacher that never saw the forget data.**

Stronger, and useful to us: **neither MDU nor EMDU reports a Retrain/gold baseline at all** — the
retain-trained reference is absent from the MDLM literature even as an *evaluation* upper bound,
despite TOFU shipping retain-finetuned checkpoints. In the image-diffusion concept-erasure field the
retain-only model is by contrast *ubiquitous as an evaluation gold standard* (SalUn, Fan 2023,
https://arxiv.org/abs/2310.12508: *"we regard Retrain as the MU's gold standard"*; SISS, Alberti 2025,
https://arxiv.org/abs/2503.01034: *"the data unlearning setting has a concrete gold standard:
retraining without the data"*) and **nowhere as a training teacher** — the stated reason being cost
(SD 2.0 filtered retraining ≈150,000 GPU-hours).

So the gap holds a fortiori for diffusion LMs: the retain reference is missing there both as a
teacher *and* as an evaluation baseline.

### 3.3 Image-diffusion concept erasure — kept separate, same answer

Not diffusion *language* modeling, reported only to close the question. No verified
distillation-based eraser uses a retain-only-retrained model as teacher; all take the teacher from
the contaminated pretrained model, an EMA/self copy, the previous unlearning step, or the original
model under an anchor/null prompt: **ESD** (Gandikota 2023, https://arxiv.org/abs/2303.07345),
**Score Forgetting Distillation** (Tianqi Chen 2024, https://arxiv.org/abs/2409.11219 — "Retrain" is
an evaluation row only), **Inverse Distillation** (Leonov 2026, https://arxiv.org/abs/2609.36099 —
teacher explicitly *"trained on the original data"*), **LACU** (George 2025,
https://arxiv.org/abs/2512.02657 — teacher = step *i−1* model), plus closed-form editors
UCE/MACE/SPM/Forget-Me-Not. **Selective Amnesia** (§4.E) targets a user-specified surrogate, not a
retain reference.

### 3.4 Verification holes — named, not glossed

- **Zhang et al. ICML 2026 full text: NOT VERIFIED.** openreview.net returned
  `ChallengeRequiredError` 403 on every route (forum, `/pdf`, `api.openreview.net`,
  `api2.openreview.net`, curl with browser UA); no arXiv preprint exists; absent from Semantic
  Scholar; ICML 2026 is not yet on PMLR. **A retain-reference component buried in its method section
  cannot currently be ruled out.** If we make a novelty claim, this is the one hole to close — via a
  browser session, an author request, or a colleague's OpenReview login.
- **Non-indexed venues**: ICLR/NeurIPS 2026 under-review pools and non-public workshop tracks are not
  fully searchable.
- **Absence of search results is not proof of non-existence.** The arXiv API, arXiv full-text search,
  ICML 2026 virtual search, Semantic Scholar and ~15 web-search variants all saturated on the same
  small set, which is good evidence of completeness but not proof.
- Incidentally, the phrase **"retain-reference distillation" returns zero hits** — our term is not in
  the literature. The literature's term is **oracle matching**. We should adopt it when positioning,
  or we will look like we are renaming a known mechanism.

---

## 4. Other retain-reference / gold-standard-distillation unlearning work

Organised by **what the teacher actually is**, because that is the only field that decides overlap.
A paper only threatens C1 if its teacher is a model *actually trained on retain data only*.

### 4.A Teacher IS a model trained on retain data only — the real precedents

**Attribute-to-Delete: Machine Unlearning via Datamodel Matching** — Kristian Georgiev, 2024/2025 —
https://arxiv.org/abs/2410.23232 (ICLR 2025 camera-ready retitled *Machine Unlearning via Simulated
Oracle Matching*; proceedings text read directly). **Image classification only.**
- Teacher: `f_oracle(x) := f_x(A(S_R))` — a model **retrained from scratch on the retain set only**.
- Loss: **MSE between output logits** and oracle logits — *not* KL, *not* soft cross-entropy.
- Inputs: *"samples x from the forget and retain sets"* (forget set plus a retain subsample).
- Models: ResNet-9 / CIFAR-10, ResNet-18 / ImageNet Living-17. Full-parameter.
- **Demonstrated:** with true oracle access, OM reaches near-oracle unlearning (KLoM metric) at
  **<5% of full-retraining compute**, beating GA, GD and SCRUB. So in classification, the C1
  mechanism is a *published success*, not a published failure.
- They then replace the oracle with datamodel-predicted outputs (DMM), since the oracle is
  unavailable in practice.
- **Explicitly lists language modeling as future work:** *"extending them to work in settings with
  more classes (e.g., full ImageNet or language-modeling) would make them more practical."*
  This is the clearest statement in the literature that the LM case was left open.

**Distillation Robustifies Unlearning** — Bruce W. Lee, 2025 — https://arxiv.org/abs/2506.06278.
Covered in full in §1. Teacher = retain-only pretrained oracle; forward KL; **autoregressive LM**;
used as a **diagnostic baseline that is reported to fail on the relearning-robustness axis**.

**Behavioral Guarantees for Proxy-Based Unlearning** — Virgile Dine (with Teddy Furon), 2026 —
https://arxiv.org/abs/2605.10680. **Classification.** The *ideal* target is defined as the
retain-trained model's posterior and the paper proves an **upper bound on KL to that ideal
posterior**, validated as *"reaching the closest classifier to the model retrained from scratch."*
But the teacher used in practice is a **closed-form statistical proxy** (LDA/QDA/Dirac mixtures),
not an actual retrained network — so this is (a)-adjacent, not (a). CIFAR-10/100, DINOv2-S,
ResNet-18. *A search engine surfaced this under a different title ("Exact Unlearning from Proxies
Induces Closeness Guarantees on Approximate Unlearning"); the fetched abs page gives the title
above — likely a retitle between versions. Treat the alternate title as **NOT VERIFIED**.*

**Forgetting Has Neighbors: Localized Collateral Forgetting in Machine Unlearning** —
Polina Dolgova, 2026 — https://arxiv.org/abs/2605.31317. **Vision classification.**
"Local Teacher Distillation": teacher is a small model trained only on retained *neighbors* of the
forget set; its soft labels replace random labels on the forget set. CIFAR-100 partial-class
deletion. **Loss and architecture details NOT VERIFIED** (abstract level only).

### 4.B The nearest neighbour in the LM setting — and it is close

**Divergence Decoding: Inference-Time Unlearning via Auxiliary Models** — Humzah Merchant
(with Bradford Levy), 2026 — https://arxiv.org/abs/2605.31293. **Verified against the HTML full
text.** Autoregressive.
- Auxiliary models, 1B–1.3B: on TOFU, **q is "trained only on the retain set"** and p on forget+retain
  (on MUSE, q on retain, p on forget only).
- Target model P: **LLaMA-3.1-8B-IT** on TOFU; Sheared-LLaMA-1.3B on MUSE.
- Steered teacher: `ℓ̂_LC(x_<t) = ℓ_P(x_<t) + α·[ℓ_q(x_<t) − ℓ_p(x_<t)]`, plus a rank-based variant.
- **Distillation back into weights:** `L_unlearn = T² · KL( σ(ℓ_P/T) ‖ σ(ℓ̂_LC/T) )`, applied
  **on the forget set**.
- Benchmarks: **TOFU and MUSE.** Full-parameter (LoRA not specified — **NOT VERIFIED**).
- **Why this matters for C1:** this is retain-trained-reference + KL distillation + forget-set inputs
  + TOFU + 8B — four of C1's five defining choices. The one difference is decisive: the teacher is a
  **contrastive combination** of three models' logits, which uses the retain model as a *steering
  direction*, not as the distribution to be matched. C1 matches the retain reference directly.

### 4.C Teacher is the ORIGINAL model (retain preservation only) — not a threat

- **SCRUB / Towards Unbounded Machine Unlearning** — Meghdad Kurmanji, 2023 —
  https://arxiv.org/abs/2302.09880 (NeurIPS 2023). Teacher is explicitly the **original model
  trained on all data**, `f(·,w^o)`. Objective minimizes `D_KL(p(f(x;w^o)) ‖ p(f(x;w^u)))` on retain
  and **maximizes it on forget**. Image classification only (CIFAR-10/5, Lacuna-10/5; ResNet-18,
  All-CNN). **SCRUB is commonly mis-cited as retain-reference distillation; it is not** — the
  retrained model never appears as a teacher. Its forget term is *divergence maximization*, which is
  the opposite sign of C1.
- The ubiquitous "KL retain regularizer" (GradDiff+KL, NPO+KL) — teacher = original model, retain
  inputs only.
- Federated variants (FedQUIT, FedBT) use the pre-unlearning global model. **NOT VERIFIED.**

### 4.D Teacher is random / incompetent — not a threat

- **Can Bad Teaching Induce Forgetting?** — Vikram S. Chundawat, 2022 (AAAI-23) —
  https://arxiv.org/abs/2205.08096. Competent teacher = model trained on retained data; **incompetent
  teacher = randomly initialized model**; `L = (1−l_u)·KL(T_s‖S) + l_u·KL(T_d‖S)` with `l_u=1` on
  forget. The retrained model appears **only as an evaluation reference (ZRF)**, never as a teacher.
  Classification.
- **Zero-Shot Machine Unlearning** — Vikram S. Chundawat, 2022 — https://arxiv.org/abs/2201.05629.
  Teacher = original model, student = random init, generator synthesizes inputs. No retain-trained
  teacher. Loss details **NOT VERIFIED**.

### 4.E Teacher is a synthetic / self-distilled target — the key LM cluster, and the key quote

**Unilogit: Uniform-Target Self-Distillation for LLM Unlearning** — Stefan Vasilev, 2025
(Findings of ACL 2025) — https://arxiv.org/abs/2505.06027. **This paper states the field's reason
for not doing C1:**

> *"the outputs of the retrained model θ_r would serve as the gold standard for distillation.
> However, since θ_r is unavailable in reality, we must approximate these targets."*

- Their target is instead the **model's own logits** with the target token's logit set so its softmax
  probability is `1/|V|`, remaining mass redistributed:
  `h̃ = (1−t)h + t·log( Σ_{i≠k} exp(h_i) / (|V|−1) )`. **Reverse KL** on forget, KL retain regularizer.
- Llama-2-7B (MUSE-News), Llama-3.1-8B-Instruct (RWKU). Full-parameter.
- Their Fig. 5 measures KL between their soft labels and **the actual retrain model's outputs**,
  reporting lower divergence than UnDIAL and baseline — i.e. closeness to the retain reference is
  treated as the thing to approximate. **No mention of TOFU or its forget-quality definition.**

- **UnDIAL: Self-Distillation with Adjusted Logits** — Yijiang River Dong, 2024 (NAACL 2025) —
  https://arxiv.org/abs/2402.10052. Teacher = the model's **own** logits with targeted-token logits
  reduced by a margin. Abstract verified; model sizes/benchmarks **NOT VERIFIED**. Notably this is
  the one distillation-unlearning method 2506.06278 benchmarks against.
- **SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion** — Zizhao Hu, 2026 —
  https://arxiv.org/abs/2605.07482. Teacher = frozen copy of the original with memorized-token logits
  demoted to −∞ and renormalized at selected positions; top-K KL at all positions. Llama-3.2-1B,
  Llama-2-7B, Llama-3-8B; TOFU, MUSE, RWKU, Hubble. Full-parameter (**reports LoRA degrades
  utility** — relevant to C1's LoRA choice). Retain model used only for evaluation.
- **Selective Amnesia** — 2023 — https://arxiv.org/abs/2305.10120. Target is a **user-specified
  surrogate distribution** `q(x|c_f)` (e.g. uniform) + EWC + generative replay. VAEs and Stable
  Diffusion (image generative). **Not retain-reference.** Details **NOT VERIFIED**.

### 4.F Teacher is a model unlearned/steered by another mechanism — not a threat

- **RKLD** — 2024 — https://arxiv.org/abs/2406.01983. Teacher built by continuing training on the
  forget set then subtracting the increase: `l_tea = l_ori − α·ReLU(l_str − l_ori)`; reverse KL
  `RKL(π_tea‖π_θ)` on forget prefixes. Llama2-7B-chat, TOFU, full-parameter.
- **DUET** — 2026 (ICLR 2026) — https://arxiv.org/abs/2601.21283. Teacher = **prompt-steered**
  original model; top-K (K=1000) raw-logit matching with Huber-L1, **at the first decoding position
  only**. Llama-3.2-3B, Zephyr-7B, Mistral-7B, Qwen2.5-3B; MUSE-Books, WMDP. No retain-retrained
  oracle baseline.
- **KGA: Knowledge Gap Alignment** — Lingzhi Wang, 2023 (ACL 2023) —
  https://arxiv.org/abs/2305.06535. Deliberately *avoids* matching one distribution:
  *"Different from previous work that … forces models to perform close to one specific distribution,
  KGA maintains distribution differences."* Worth citing as the explicit counter-philosophy to C1.
- Also: **Visual-Noise Guided In-Context Distillation for MLLM Unlearning** (Junkai Chen, 2026,
  https://arxiv.org/abs/2606.00105); **Bidirectional Logit Distillation** (Yisheng Zhong, 2026,
  https://arxiv.org/abs/2604.15482, details **NOT VERIFIED**); **Robust MLLM Unlearning via Visual
  Knowledge Distillation** (2026, https://arxiv.org/abs/2512.11325, teacher = vanilla MLLM).

### 4.G The TOFU circularity — nobody has said it

**TOFU** — Pratyush Maini, 2024 — https://arxiv.org/abs/2401.06121. The retain model is *"a model
finetuned with retain data only"*, the gold standard. **Forget Quality = the p-value of a
Kolmogorov–Smirnov test comparing the unlearned model's Truth-Ratio distribution to the retain
model's on the forget set.** Llama-2-7B and Phi-1.5, full-parameter.

The user's intuition is correct and sharper than the literature: **TOFU's success criterion is
statistical indistinguishability from the retain model, and C1 trains directly on the retain model's
distribution.** C1 therefore optimizes a close surrogate of the metric it will be scored on.

**We found no paper that states this.** The nearest critiques do not name it:
- **Reference-Specific Unlearning Metrics Can Hide the Truth: A Reality Check** — 2025 —
  https://arxiv.org/abs/2510.12981. Argues reference-dependent metrics are gameable with
  "systematic blind spots"; proposes FADE. Critiques the metric, not the
  distillation-as-metric-optimization equivalence.
- **Position: LLM Unlearning Benchmarks are Weak Measures of Progress** — 2024 —
  https://arxiv.org/abs/2410.02879. Generic benchmark critique; does not name the circularity.

> ⚠️ **Suspected-hallucination warning, recorded deliberately.** A PDF extraction of the survey
> *Unlearning in LLMs: Methods, Evaluation, and Open Challenges* (Tyler Lizzo, 2026,
> https://arxiv.org/abs/2601.13264) returned confident quotes asserting that a retain-only-teacher
> distillation category *and* a TOFU gaming/circularity remark were both present. **Neither appears
> in the HTML full text.** Treat those quotes as **NOT VERIFIED — do not cite.** This is the same
> failure mode that produced a fabricated hyperparameter table for 2506.06278 earlier in this survey;
> all §1 numbers here were therefore re-derived from locally extracted PDF text.

---

## 5. Bottom line

### 5.1 Which of (i)–(iv) applies

**All of (ii) and (iii) apply; (iv) applies only after narrowing to masked-diffusion LMs; (i) is
false.** Precisely:

| Setting | Status of the C1 mechanism | Evidence |
|---|---|---|
| Image/vision classification | **Published, and published as working.** Oracle Matching with a true retain-only retrained teacher reaches near-oracle unlearning at <5% of retraining compute, beating GA/GD/SCRUB. | Georgiev 2024/2025, https://arxiv.org/abs/2410.23232 |
| Autoregressive LMs | **Published, as a named diagnostic baseline, with a negative result on the relearning-robustness axis** (behavioral agreement achieved; robustness not). Not published as a proposed method, and never run at realistic scale — the oracle-matching arm is confined to 0.1B/0.3B synthetic Korean/arithmetic. | Lee 2025 §3, https://arxiv.org/abs/2506.06278 |
| AR LMs on TOFU at 8B | **Nearest neighbour exists but is not C1**: retain-trained auxiliary used as a *steering direction* inside a contrastive teacher, not as the distribution to be matched. | Merchant 2026, https://arxiv.org/abs/2605.31293 |
| **Masked diffusion LMs** | **Apparently unoccupied** — as a teacher and even as an evaluation baseline. One verification hole (Zhang et al. full text). | §3 |

So: **(ii) published for autoregressive models but not diffusion** is the single most accurate
answer, with the important amendments that (a) in AR it was published as a *failure* on one axis, not
a success, and (b) in classification (iii) it was published as a *success*. Claiming (iv) requires
the words "for masked diffusion language models" and the Zhang et al. hole closed.

### 5.2 The framing problem we must solve first

This is the most consequential finding in the document, and it is not about novelty.

**C1 as specified is not a deployable unlearning method. It is an oracle.** It requires a reference
model trained only on retain data, which on TOFU exists only because the benchmark ships it. Both
prior papers that used this mechanism understood it that way and said so:

- Georgiev et al. — *"assuming access to oracle outputs is unreasonable—since our goal is to produce
  an oracle model in the first place"* — and therefore built DMM to **approximate** the oracle via
  data attribution.
- Unilogit (Vasilev 2025) — *"the outputs of the retrained model θ_r would serve as the gold standard
  for distillation. However, since θ_r is unavailable in reality, we must approximate these
  targets."*

Both routes out of this (approximate the reference by attribution; approximate it by a synthetic
target) are **already occupied**. So we cannot present C1 as a practical method without answering
"where does the retain model come from?", and the two standard answers are taken.

Compounding this, §4.G: **TOFU's Forget Quality is the KS-test p-value against the retain model's
Truth-Ratio distribution.** C1 trains on the retain model's distribution. C1 therefore optimizes a
close surrogate of the metric it is scored on. A strong TOFU forget-quality number from C1 is
**close to uninformative** as evidence that C1 unlearns — and a reviewer will see this. Nobody in the
literature has stated this circularity, which makes it both our biggest validity threat and,
inverted, our most defensible contribution.

### 5.3 What a defensible contribution on top of C1 would have to look like

Ordered by how much they survive the above. Items 1–2 are the safe core; 3 is the upside; 4 is what
to stop doing.

1. **Supply the missing baseline, and frame C1 explicitly as the oracle/skyline it is.**
   Neither MDU nor EMDU reports a Retrain/gold baseline; the MDLM field has no retain-reference
   number of any kind. Publishing C1 as *"the oracle-matching skyline for MDLM unlearning"* — the
   best achievable behavioral forget quality when the retain model is handed to you — is honest,
   unoccupied, directly useful to MDU/EMDU, and immune to the deployability objection because it
   never claims to be deployable. This is the same role §3 plays in Lee et al. and §4.2 plays in
   Georgiev et al., which is precedent *for* the framing, not against it.

2. **State the TOFU circularity and quantify it.** Show that C1 nearly saturates TOFU Forget Quality,
   then show *why* that is expected given the KS-test definition, and show what a
   circularity-resistant metric says instead. This converts our biggest weakness into the paper's
   point, it is unoccupied (§4.G), and it composes with our separate finding that the MDLM metric
   landscape is unreproducible (nobody has reproduced MDU's table, including EMDU's authors using
   MDU's own code). A position/measurement paper here is more defensible than a method paper.

3. **If we want a method claim, it must rest on a demonstrated MDLM-specific interaction, not on the
   port itself.** The port alone is category (c): KL distillation into a masked-diffusion LM at
   masked positions is already routine machinery (CD4LM, Yihao Liang 2026,
   https://arxiv.org/abs/2601.02236 — *"align the student with the teacher's softer predictions under
   the same corruption pattern"*, LLaDA, for sampling speed; SelFusion,
   https://arxiv.org/html/2608.22898, self-distillation across masking ratios with KL). Combining two
   pieces of standard machinery is not a claim.
   The one real candidate interaction: **an MDLM reference defines a target at every mask ratio and
   every unmasking order, so a retain reference supplies supervision over a continuum of corruption
   states rather than one factorization.** That is a genuine structural difference from the AR case
   and a plausible reason Lee et al.'s negative result might not transfer. But it is currently a
   hypothesis. To claim (d) rather than (e) we would need the **ablation that isolates it**: restrict
   the reference to a single mask ratio, and/or to AR-order-only factorizations, and show the benefit
   degrades. Without that ablation this stays category (e) — an algorithmic difference whose benefit
   is unestablished. Park & Jo's finding (#5 in §3.1) that MDM edits fail *specifically because of
   partially-unmasked intermediate states* is independent support that these intermediate states
   matter, and is the right citation to motivate the ablation.

4. **Two claims we must not make.**
   - **Not "robust unlearning" / "resists relearning."** Lee et al. §3 is a published negative result
     on exactly this mechanism on exactly this axis, with a worst-case-over-10-learning-rates
     adversary. We would need to beat it, not assert around it. Note their failure mode was
     attributed to *parameter-space latent traces in the student's initialization*, and C1 initializes
     from the contaminated model and trains **LoRA**, i.e. an even smaller perturbation of those same
     parameters than their full-parameter failure case. The prior is strongly against C1 being robust.
   - **Not "first oracle matching" or any framing that implies the mechanism is ours.** It is named in
     the literature, in a section heading, by both precedents. Adopt the term *oracle matching*, cite
     Lee §3 and Georgiev, and claim only the MDLM instantiation.

5. **Scope the threat model explicitly.** The two adversary families in this literature are disjoint:
   Lee et al. test **relearning/finetuning** (plus INT4 quantization) and never test prompt-based
   elicitation; Zhang et al. ICML 2026 target **prompt-based elicitation via informative context**.
   C1 must say which it claims and test it. Given 5.3.4, claiming the relearning axis is the hard road.

### 5.4 One-sentence answer

C1's mechanism is published — as a working method in vision classification and as a named,
explicitly-failing baseline for autoregressive LMs — and is unoccupied for masked diffusion language
models; so the defensible move is to publish it as the MDLM field's missing oracle/skyline plus the
TOFU-circularity measurement result, and to make a method claim only if we can demonstrate that
cross-mask-ratio supervision is what makes it work.

---

## 6. Status and ground rules

- §1, §2: complete, from locally extracted full v3 PDF text of 2506.06278, read directly.
- §3: two independent sweeps (arXiv API, arXiv full-text, ICML 2026 virtual, Semantic Scholar,
  ~15 web-search variants), saturated. One named hole: Zhang et al. full text.
- §4: organised by teacher construction; items marked NOT VERIFIED were read at abstract level only.
- §5: analysis.

**Method note on reliability.** Two summarizer-mediated fetches in this survey returned confident,
fabricated content: a complete hyperparameter table for 2506.06278 (α ∈ [0.01,0.1], λ_distill ∈
[0.5,2.0], TinyLlama, "regain >80%") that contradicted the paper's own abstract, and two quotes
attributed to arXiv 2601.13264 that do not appear in its HTML. Every quantitative claim in §1 was
consequently re-derived from locally extracted PDF text. **Treat any figure in this area that did not
come from directly-read primary text as suspect.**

---

# COORDINATOR'S OWN READ OF 2506.06278 (primary source, user-supplied PDF)

Extracted locally to `outputs/distill_2506.06278.txt`; quotations verbatim. Done because two
summariser-mediated fetches today returned fabricated content.

## Confirmed: oracle matching is C1

> *"We match the reference model's outputs with those of the oracle model using the
> Kullback-Leibler Divergence (KL Divergence) as the loss function."*

Oracle = *"the same 100M model architecture pretrained on 1B tokens of the retain set"*, i.e.
a model that never saw the forget data. Exact equivalence with C1's mechanism, confirmed.

## Three details that change SIO's novelty position in our favour

**1. Their oracle matching is UNIFORM over everything.**

> *"We implemented oracle matching using knowledge distillation with forward KL divergence
> loss on the pretraining dataset"*

All token positions, the whole pretraining mixture. **No token selection, no position
masking, no subset, no weighting.** A targeted grep for selective/subset/saliency/per-token
treatment in their own method returns only *citations to other papers*. So SIO's three
components — selective interdiction over a measured-minimal salient subset, orthogonalised
suppression, and a margin hinge on the kept tokens — **appear nowhere in the paper that
contains C1.**

**2. Scale and granularity are far from ours.** Gemma-2-**0.1B** and **0.3B**; forget sets are
*domain-level* (Korean documents; multiplication and division). Ours is **8B** and
*fact-level* (400 QA about 20 specific authors). Their negative result is not established at
our scale or granularity, and we should not treat it as if it were.

**3. Their negative is about relearning only, and they locate the cause in the
initialisation.** Three models: Student (Reference), Student (Random), Oracle Teacher.

> *"Student (Reference) quickly learns the forget distribution compared to the
> Student (Random) and Oracle Teacher models, which are more robust."*

So distilling the oracle into a **freshly initialised** model is robust; distilling into the
**contaminated reference** is not. The fragility is attributed to the starting point, not to
oracle matching. **This bears directly on us: LoRA on a contaminated backbone is precisely
their fragile configuration.** Any relearning claim we make would sit in their fragile
regime, and that must be declared rather than discovered later.

## And their result is positive evidence on our axis

> *"In both settings, the distillation loss tends to zero."*

Uniform oracle matching achieves effectively perfect **behavioural** agreement with a model
that never saw the data. Our axis is behavioural composition — matching the ignorance
profile with disclosure down. **So the paper that forecloses a robustness claim simultaneously
supports the thing we would use C1 for.** The two axes are separate and both are stated.

## Net position

| component | standing against this paper |
|---|---|
| C1, uniform oracle matching | **(a) exact equivalence** — not a claim, used as the skyline |
| S, selective interdiction over a minimal salient subset | absent from the paper |
| O, orthogonalised suppression | absent |
| K, margin hinge on kept tokens | absent |
| relearning robustness | **foreclosed** — their negative stands, and our configuration is their fragile one |

