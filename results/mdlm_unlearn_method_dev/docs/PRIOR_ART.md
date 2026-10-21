# PRIOR ART AND BASELINES

**Owner:** Prior art and baselines lead
**Date:** 2026-10-20
**Status:** Tier 2 (active development evidence). Not sealed. May be iterated on evidence.

## How to read this document

Every factual claim carries a citation (title, first author, year, URL). Claims I could not
confirm against a primary source are marked **NOT VERIFIED** inline. I distinguish throughout
between what a paper *claims* and what it *demonstrates*.

Primary sources were read as actual paper text (arXiv HTML and/or PDF text extraction) and,
where available, actual repository contents. Secondary descriptions were not used as evidence.

Field convention: "not stated" means the paper does not say. It does **not** mean I did not look.

---

# Executive summary

**Nine findings that change what we should do.**

1. **The MDLM unlearning field is two training objectives.** MDU (arXiv:2605.18253) and EMDU
   (arXiv:2610.25447), plus one training-free edit method (TimeROME-DLM) and one unverifiable RL
   method. Everything else in this document is an autoregressive method somebody has to *port*.
2. **Nobody has reproduced MDU's published table — including EMDU, who tried with MDU's own code.**
   EMDU gets TOFU LLaDA Forget rL 0.346 where MDU reports 0.034–0.135. MDU released no checkpoints,
   no SFT code, no LLaDA RWKU eval script, and **its paper defines ROUGE-L as F1 while its code
   computes recall** — so it is unknown which quantity MDU's table contains. **MDU's numbers are
   not a target we can claim to beat.**
3. **If our implementation uses soft-CE/KL, it is EMDU's divergence, not MDU's.** `∇_z KL(p ‖
   softmax(z)) = q − p` is exactly soft-label cross-entropy (EMDU Proposition 1). MDU's "forward
   KL" means the *opposite* argument order. But **our exclusion target is neither paper's anchor** —
   both anchors are model-derived forward passes; ours is hand-built and sits near MDU's `τ=0`
   (MaxEnt) limit, while both papers' headline results are at `τ=1`. **We are running a third
   method and must stop calling it MDU or EMDU.**
4. **TOFU is out of reach; RWKU is our only viable benchmark.** TOFU needs ~250,000 full-parameter
   SFT steps (1000 epochs) to instil the knowledge, with no released checkpoint — ~70+ GPU-hours
   per backbone against a 4.1-GPU-hour project history. RWKU needs no SFT, and **our local data
   matches EMDU's protocol only partially** — see the **CORRECTION** note below.
5. **EMDU's RWKU recipe is cheap and fully specified**: LR 1e-5, **40 steps**, batch 4, accum 1,
   `α=0`, `τ=1`, per subject. Training is nearly free; **evaluation (MDLM generation at 128
   steps) is the entire budget.**

   > **CORRECTION (2026-10-20, superseding the original claim here and in §2.0/§2.4/Appendix A).**
   > An earlier version of this document stated that our local RWKU data matches EMDU's protocol
   > for all 10 subjects. It does not. Enumerated file by file: the `llada_rwku` copy has **10
   > entities but no `pair.json` forget-training corpus and no F-L3**; the `llada_emdu` copy has
   > the **complete** file set (`pair.json` = 582, F-L1/L2/L3, N-L1/L2, positive/negative = 582)
   > for **only 3 entities** — Stephen King, Confucius, Bruce Lee. So **we can currently train on
   > 3 subjects, not 10, and F-L3 is available for those 3 only.** A 3-entity result is not
   > comparable to MDU's or EMDU's 10-subject macro-averages. Fetching the remaining 7 subjects'
   > `pair.json` from the public RWKU release is a data (not model-weight) download and is the
   > cheapest single unblock — a coordinator decision. Full inventory in
   > `docs/PRIOR_ART_H11.md` §4.1.
6. **Under LoRA the frozen anchor is free** — `p_θ0` is the adapter-disabled forward through the
   same weights, so no second 8B copy is needed. And because EMDU's FAM anchor input `(∅,[MASK]^n)`
   is **independent of `t`**, that anchor forward is **exactly cacheable per example**. Both are
   genuine advantages of the deviation we are forced into.
7. **Three independent sources now agree that parameter-subset unlearning does not survive
   fine-tuning**: RMU's own paper (*"RMU is unable to prevent finetuning from recovering
   performance"*), Fan et al. (RMU is the relearning-robustness loser, attributed to editing only
   layers 5–7), and our own HISTORICAL reversibility result. **LoRA is also a parameter subset.**
   Any relearning number we produce under LoRA is a lower bound and we cannot attribute the gap to
   the method.
8. **Fan et al.'s GP is, as released, a no-op penalty.** `forget_grad_norm += ....item()` detaches
   from the autograd graph (GA.py:258 for GP, :340 for CR — **I verified these lines myself**), so
   the regulariser adds a constant to the loss and nothing to the gradient; GP's coefficient is
   also hardcoded to 0.01, ignoring the `gnr_rho=10.0` the scripts pass. Our HISTORICAL
   `gradient-norm defense NOT SUPPORTED` verdict is therefore **not in tension** with their GP row.
   **If we test one smoothing defense, test SAM** — first-order, implemented as described, and
   consistently their best row.
9. **TNT and TNFF do not exist**, and we caught the likely mechanism: a web summary was headed
   *"Enhanced Masked Diffusion Unlearning (TNT)"* while that paper contains zero occurrences of
   "TNT". **CLP is real but is a 2019 fairness method with zero unlearning citations** (OpenAlex
   sweep of all 200 citing works: 0 hits for unlearn/forgetting/erasure) — if it appears in a draft
   of ours it is a conflation with ULD or Clean Logit Pairing. **WHP+ is real but is not RWKU's
   baseline** — it is Liu et al.'s (arXiv:2407.16997, EMNLP 2024) own N=1 ablation label.

**The four critical questions, in one line each** (full answers in Part 5, corrected in Part 8):

- **Q1 — non-uniform `t`?** No published MDLM unlearning method deviates from `t ~ U(0,1)` on the
  trained-model side. **EMDU's FAM pins the anchor at `t=1`** — a target-side mask-rate
  intervention, and one of the two factors in its ablation. Non-uniform-`t` precedents exist only
  outside unlearning (A2D's floor `λ=(1−ε)t+ε`; D2F's increasing block ratios). **Mask-rate
  selection is unoccupied in unlearning, everywhere.**
- **Q2 — iterative decoding?** No MDLM unlearning objective uses it; **MDU names this as its
  limitation and its stated next goal.** But trajectory-aware unlearning *does* exist for
  autoregressive models (**CUSE**, on-policy, with a measured 0.129→0.372 off-policy ablation;
  **RULE**, GRPO). TimeROME-DLM already occupies trajectory-aware *localisation*. And the
  cited-but-unlocated **Zhang et al. 2026** may occupy the MDLM region outright. **Locate it before
  any novelty claim.**
- **Q3 — the anchors.** Written out explicitly in Part 5. MDU and EMDU differ in exactly **two**
  substantive ways: KL argument order, and whether the anchor sees the partially-masked answer
  (`(∅,y_t)`) or a fully masked one (`(∅,[MASK]^n)`, FAM). The temperature difference is cosmetic —
  `softmax(τ log p)` and `p^τ/Z` are the same operation.
- **Q4 — what would "beating SOTA" even mean?** Only one claim is constructible: a better
  Forget/Neighbor trade-off than **our own re-runs** of MDU, EMDU and the MDU-defined ports, on
  RWKU / 10 subjects / LLaDA-8B-Instruct / **ROUGE-L recall** / RWKU-official MMLU placement /
  fixed stated decoder / identical LoRA / identical selection rule / ≥3 seeds. **A claim against
  published values is not available to us, or to anyone.**

---

# Part 0 — The published MDLM unlearning landscape, exhaustively

Before the comparison table, the scope of what exists. I ran exhaustive arXiv API queries
(`abs:unlearning AND abs:"masked diffusion"`; `abs:unlearning AND (abs:"diffusion language
model" OR abs:dLLM OR abs:LLaDA)`), sorted by date descending, on 2026-10-20.

**The complete set of arXiv-indexed unlearning work on diffusion language models is four items:**

| # | Work | arXiv | Date | Kind |
|---|---|---|---|---|
| 1 | Machine Unlearning for Masked Diffusion Language Models (MDU) | [2605.18253](https://arxiv.org/abs/2605.18253) | 2026-05-18 | training objective |
| 2 | Enhanced Masked Diffusion Unlearning (EMDU) | [2610.25447](https://arxiv.org/abs/2610.25447) | 2026-10-18 | training objective |
| 3 | TimeROME-DLM | [2606.12841](https://arxiv.org/abs/2606.12841) | 2026-06-11 | training-free weight/activation edit |
| 4 | wd1: Weighted Policy Optimization for Reasoning in Diffusion LMs | [2507.08838](https://arxiv.org/abs/2507.08838) | 2025-07-07 | RL for reasoning; "unlearning" is only negative-sample weighting, **not** machine unlearning |

Plus one work that is **cited but not publicly locatable**:

- Zhang, Zhiwei et al., "Adversarial reinforcement learning for robust diffusion large language
  model unlearning", cited by EMDU as *ICML 2026*. I could not find this paper on arXiv or via
  web search on 2026-10-20. **NOT VERIFIED** — it exists only as a citation inside EMDU
  ([2610.25447](https://arxiv.org/abs/2610.25447), §2.3 "Adversarial robustness"). EMDU reports
  that it "finds that unlearning performance in MDLMs is sensitive to generation length and
  denoising steps" and "proposes adversarial RL, in which a separate LLM generates contexts that
  elicit forgotten knowledge while the MDLM learns, via GRPO, to suppress it." That second-hand
  description is the only evidence I have of its content, and it is the single most
  strategically relevant item in this document (see Critical Question 2). **This needs to be
  chased down before we claim novelty on trajectory-aware unlearning.**

Adjacent and important, though not unlearning:

- **A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models**, Jeung et al.,
  2025, [2509.23286](https://arxiv.org/abs/2509.23286) — safety alignment, not unlearning, but
  it is the one MDLM post-training method I found that **does not** sample `t ~ U(0,1)` (see
  Critical Question 1).
- **DiffuGuard**, Li et al., 2025, [2509.24296](https://arxiv.org/abs/2509.24296) — intrinsic
  safety in dLLMs. Cited by MDU as related safety work. Not read in full for this document;
  **NOT VERIFIED** beyond its existence and MDU's characterisation of it.

**Consequence for us:** the competitive field for "unlearning on a masked diffusion LM" is
*two* training objectives (MDU, EMDU), *one* training-free edit method (TimeROME-DLM), and
*one* unverifiable RL method. Everything else in this document is an autoregressive method that
somebody — MDU, EMDU, or us — has to *port* to the MDLM setting. That porting step is where the
"is this comparable?" question lives, and it is addressed in Part 4.

---

# Part 1 — Complete-algorithm comparison

## 1. MDU — Masked Diffusion Unlearning

**Citation:** Lee, Georu; Jeong, Seungwon; Kim, Hoki; Park, Jinseong; Lee, Woojin. "Machine
Unlearning for Masked Diffusion Language Models." arXiv:2605.18253v1, 18 May 2026.
<https://arxiv.org/abs/2605.18253>. Code: <https://github.com/leegeoru/MDU>.

### Training target / supervision

MDU's background definitions, as the paper writes them. Pretraining loss (their Eq. 1):

```
L_pre(θ; D_pre) = -E_{x0, t, xt} [ (1/t) Σ_{i∈M_t} log p_θ(x_0^i | x_t) ]
                =  E_{x0, t, xt} [ (1/t) Σ_{i∈M_t} KL( q(x_0^i) ‖ p_θ(· | x_t) ) ]
```

SFT / fine-tuning loss (their Eq. 2) — this is also the **retain** loss and the building block
of every ported baseline:

```
L_sft(θ; D) = -E_{(x,y), t, y_t} [ (1/t) Σ_{i∈M_t} log p_θ(y^i | x, y_t) ]
            =  E_{(x,y), t, y_t} [ (1/t) Σ_{i∈M_t} KL( q(y^i) ‖ p_θ(· | x, y_t) ) ]
```

with `t ~ U[0,1]`, `y_t ~ q(·|y,t)`, `M_t = {i : y_t^i = MASK}`, prompt `x` always visible.

**The MDU forget objective (their Eq. 5), verbatim:**

```
L_forget(θ) = E_{(x,y)~D_f, t, y_t} [ (1/|M_t|) Σ_{i∈M_t} KL( p^c_θ(· | x, y_t) ‖ (1/Z_i) p^u_{θ0}(· | m, y_t)^τ ) ]
```

where `θ0` are the parameters at the start of unlearning, `Z_i = Σ_v p^u_{θ0}(v | m, y_t)^τ` is a
per-position normalisation constant, `τ ∈ [0,1]` is the anchor temperature, and `m` is a null
(fully masked) prompt.

**Total objective:** `L = L_forget + λ · L_sft(θ; D_r)` with `λ = 1` in their experiments
(stated in Appendix B.3 under GD, and the text says "matching the retain trade-off coefficient
of the main objective").

Note the **normalisation asymmetry**, which matters if you reimplement: the forget term is
averaged `1/|M_t|` over masked positions; the retain term uses the ELBO weight `1/t`. These are
different estimators and they are not interchangeable.

Two reinterpretations MDU offers. First, a tilted distribution (their Eq. 6):

```
p̃_θ(y^i | x, y_t)  ∝  p_θ(y^i | x, y_t)^{1-τ} · p_θ0(y^i | m, y_t)^{τ}
```

which they relate to unsupervised classifier-free guidance in discrete diffusion,
`p̃_θ ∝ p_θ(y^i|x,y_t)^{1+w} p_θ(y^i|m,y_t)^{-w}`, noting their version *interpolates toward* the
anchor where CFG *extrapolates away* from the unconditional. Second, the τ limits: at `τ=1` the
target is exactly `p_θ0(· | m, y_t)`; at `τ=0` the target is the **uniform distribution over the
vocabulary**, which the paper states "is exactly the same as the Maximum Entropy (ME) loss"
(citing Yuan et al., "A closer look at machine unlearning for large language models",
arXiv:2410.08109). This τ=0 equivalence is an explicit, author-stated special-case reduction,
and under the Part 3 rubric it is category (a)/(b) evidence for free.

### State distribution and sampling

- Mask rate: `t ~ U[0,1]`, **uniform, resampled once per training step** (Algorithm 1 line 1).
  Nothing is targeted at specific mask rates.
- Masked state: each response token independently masked w.p. `t`; prompt stays visible for the
  conditional pass.
- **Teacher-forced.** `y_t` is a corruption of the *ground-truth* response `y`. No model-sampled
  continuations, no rollouts. The denoising state is drawn from the forward corruption process,
  not from the model's generation trajectory.
- The **anchor sees the same `y_t`** as the trained model — same random mask, same surviving
  ground-truth tokens — only with the prompt replaced by `m`. This is precisely what EMDU
  attacks as failure mode 1.
- The retain term draws an **independent** `t' ~ U[0,1]` and its own `y_{r,t'}` (Algorithm 1
  line 7).

### Gradient path

Plain loss descent, single backward, AdamW. From Algorithm 1:

```
1: Sample t ~ U[0,1] and y_t ~ q(·|y,t) with masked positions M_t
2: p^c_θ(·)   ← p_θ(· | x, y_t)                 ▷ conditional, trainable
3: p^u_θ0(·)  ← p_θ0(· | m, y_t)                ▷ unconditional anchor, θ0 frozen
4: p*_i(·)    ← (1/Z_i) p^u_θ0(·)^τ with Z_i = Σ_v p^u_θ0(v)^τ   ▷ τ=0: U(V); τ=1: p^u_θ0
5: L_forget   ← (1/|M_t|) Σ_{i∈M_t} KL( p^c_θ(·) ‖ p*_i(·) )     ▷ Eq. (5)
6: if λ > 0 then
7:   Sample t' ~ U[0,1], y_{r,t'} ~ q(·|y_r,t') with masked positions M^r_{t'}
8:   L_sft ← -(1/t') Σ_{i∈M^r_{t'}} log p_θ( y_r^i | x_r, y_{r,t'} )   ▷ retain, Eq. (2)
9: else
10:  L_sft ← 0
11: end if
12: L ← L_forget + λ L_sft
13: θ ← θ - AdamW( ∇_θ L )                       ▷ θ0 remains frozen
```

The anchor branch is a **stop-gradient by construction**: it is computed under `θ0`, a frozen
parameter set, not under `θ` with a detach. There is no implicit stop-gradient inside the KL:
the differentiated argument is the *first* KL slot, so the gradient flows through `p^c_θ`
including through the entropy term of `p^c_θ`. (EMDU's Proposition 1 gives the exact logit
gradient; see §2.)

### Preservation mechanism

A weighted reconstruction (SFT) loss on a retain set, `λ L_sft(θ; D_r)`, `λ=1`. On RWKU there is
no retain split, so **all** methods including MDU run forget-loss-only. MDU argues (and shows in
its Table 3) that the anchor itself provides implicit preservation: the τ sweep moves Neighbor
scores without any retain term. That is a claim they *demonstrate* on RWKU with 10 entities, one
seed.

No gradient projection. No parameter subset. No KL anchor on retain.

### Reference / teacher model

**Required.** `p_θ0(· | m, y_t)` is the pre-unlearning model, frozen, used with a null prompt.
It is initialised from the unlearning starting point: Base SFT on TOFU, off-the-shelf
`GSAI-ML/LLaDA-8B-Instruct` / `Dream-org/Dream-v0-Instruct-7B` on RWKU. One extra no-grad
forward per step.

Note: the "reference model" is **the same weights as the model at step 0**, not a separately
trained teacher. With LoRA this is free (adapter-disabled forward) — see Part 2.

### Inference requirements

**Nothing beyond the edited weights.** No null-prompt pass at deployment, no guidance, no extra
module. The anchor is training-time only.

### Parameterization

**Full-parameter.** Not stated as such in words, but Appendix B.4 reports `4× NVIDIA H200
(141GB HBM3e per device)` with per-device batch 4 and no mention of PEFT/LoRA anywhere in the
paper or in the repository description. EMDU, which ran their code, describes both Base SFTs as
"full-parameter SFT" (EMDU Appendix D.2). Treat as full-parameter.

### Selection procedure

**This is MDU's weakest point and it is independently documented.**

- The paper gives SFT hyperparameters (AdamW, LR `2e-5`, per-device batch 4, grad accum 4,
  1000 epochs LLaDA / 300 epochs Dream) and shared unlearning setup (AdamW, cosine schedule,
  grad clip 1.0, per-device batch 4, accum 4 on TOFU).
- The paper states **no unlearning learning rate and no unlearning epoch/step count** for MDU or
  for any baseline. Confirmed by my own read of Appendix B.4 and independently by EMDU
  (2610.25447 Appendix E.1): *"gives a concrete learning rate and epoch count for Base SFT
  (2×10⁻⁵, 1000/300 epochs for LLaDA / Dream) but no learning-rate value or epoch/step count for
  the unlearning stage itself, for MDU or for any baseline."*
- No checkpoint-selection criterion is stated. No validation split is named. Figure 4 implies
  TOFU unlearning ran to ~9-10 epochs.
- The repository's `configs/mdu_tofu.yaml` / `configs/mdu_rwku.yaml` do carry concrete values
  (LR `1e-5`, epochs 3–9, `τ ∈ {0, 0.25, 0.5, 0.75, 1}`) — so the *code* supplies what the
  *paper* omits. These are config defaults, not a stated selection rule.
- **Reported numbers are single-seed.** EMDU states MDU followed single-seed practice "due to
  limited compute resources"; MDU's own paper does not report seeds or variance.

**Reading:** MDU reports a τ *sweep* as its result (five rows per table), not a single selected
operating point. There is therefore no checkpoint-selection procedure to reproduce, which also
means **there is no single "MDU number" to beat** — you must say which τ.

### Compute cost

- **2 forwards + 1 backward per step** when λ=0 (conditional with grad, anchor no-grad).
- **3 forwards + 1 backward per step** when λ>0 (conditional, anchor, retain), with the retain
  forward also carrying gradients. Algorithm 1 sums both terms before one update, so one
  backward over a graph containing two grad-carrying forwards.
- Wall-clock: **not stated.** Hardware stated: one node, 4× H200 141GB.

### Code availability

- URL: <https://github.com/leegeoru/MDU>. **MIT License** (LICENSE file present).
- Contents: `README.md`, `LICENSE`, `requirements.txt`, `run_main.sh`, `configs/{mdu_tofu,mdu_rwku}.yaml`,
  `src/{unlearn_mdu_llada,unlearn_mdu_dream}.py`,
  `scripts/{eval_tofu_llada,eval_tofu_dream,eval_rwku_dream,convert_rwku_to_tofu,convert_rwku_dream_to_tofu}.py`.
- **No published weights or checkpoints** — neither Base SFT nor post-unlearning, for either
  backbone. Confirmed by README (no checkpoint section) and by EMDU Appendix E.1.
- **Runs on a diffusion LM: yes.** Natively. LLaDA-8B-Instruct and Dream-7B-Instruct.
- **Known gaps in the release**, all from EMDU Appendix E.1 (a hostile but specific source — I
  have read their text directly, not a summary of it):
  - The SFT code is not in the release; the Base SFT checkpoint must be supplied externally.
  - The training script tokenizes forget and retain splits separately, merges, and shuffles, so
    *a training row is either forget or retain* — both terms are **not** seen every step. This
    contradicts MDU's own Algorithm 1 (which pairs them) and differs from TOFU's reference
    implementation and OpenUnlearning.
  - The released config's retain path defaults to `retain_perturbed.json`, which is an
    **evaluation-only** TOFU split, not a training retain set.
  - **The paper defines Rouge-L as F1; the released code computes `.recall`.** (EMDU: *"their
    released evaluation scripts for both TOFU (LLaDA and Dream) and RWKU (Dream) all compute
    `scorer.score(gt, pred)["rougeL"].recall`"*.) TOFU's own official code also reports recall.
    So it is **unclear which metric MDU's published table actually contains.**
  - RWKU conversion scripts reference per-target-model generated forget-pair files that are not
    shipped. There is an RWKU eval script for Dream but **none for LLaDA**, despite shipping a
    LLaDA-targeted RWKU conversion script.
  - MDU's RWKU MMLU prompt places the few-shot "Answer:" cue before the chat template; RWKU's
    official eval places it after the assistant header. EMDU measures off-the-shelf
    LLaDA-8B-Instruct at **0.400** under MDU's placement and **0.680** under RWKU's. MDU's paper
    reports 0.395 for base LLaDA MMLU — consistent with the non-official placement.

### What MDU demonstrates vs. claims

- *Claims* MDU is "the first unlearning framework for MDLMs." Supported — nothing earlier exists
  in my exhaustive search.
- *Claims* the prompt-masked anchor "isolates fact-specific knowledge while preserving standard
  structures." *Demonstrates*: a token-role taxonomy (in-context / structural /
  stored-knowledge) with aggregate KL change of −3.8% / +1.4% / −20.6% (their Table 5) on the
  TOFU forget set. That is one backbone, one dataset, one seed, and the categories are
  hand-annotated. Suggestive, not established.
- *Claims* convergence to a well-defined endpoint, unlike GA/NPO. *Demonstrates*: Figure 4,
  three KL trajectories over 10 epochs, one setting. The qualitative claim (GA/NPO have no
  endpoint; MDU converges to the anchor at τ=1 and to uniform at τ=0) follows directly from the
  objective's unique minimiser and is more a property of the loss than an empirical finding.
- *Claims* best forget-retain trade-off. *Demonstrates* that on both backbones *some* τ beats
  *all* baselines on both axes on TOFU. Note that no single τ dominates on RWKU/Dream — the
  paper says so.
- **Honest limitation the authors state, and the most important sentence in the paper for us:**
  *"As MDU is based on the training scheme, we randomly sampled masked denoising states, while
  generation proceeds through a full iterative denoising trajectory with specific sampling and
  remasking strategies. Since steering methods are frequent in continuous diffusion modeling,
  our next goal is to bridge this gap between training-time unlearning and inference-time
  generation."* (§7 Limitations.) See Critical Question 2.

---

## 2. EMDU — Enhanced Masked Diffusion Unlearning

**Citation:** Garber, Tomer; Tirer, Tom. "Enhanced Masked Diffusion Unlearning."
arXiv:2610.25447v1, 18 Oct 2026. <https://arxiv.org/abs/2610.25447>. Bar-Ilan University / Red Hat.

### The two MDU failure modes EMDU identifies

1. **Knowledge leakage.** *"Random answer masking can leave key forget tokens visible to the
   anchor even when the question is fully masked."* Under MDU, the anchor is evaluated on
   `(∅, y_t)` where `y_t` is the *same* randomly masked ground-truth answer the trained model
   sees. The KL is applied only at masked positions, but the surviving unmasked answer tokens
   are **bidirectional context for the anchor's forward pass**, so the anchor's "unconditional"
   prediction at the masked slots can still concentrate on the forget answer. Their Figure 3
   illustrates this on a TOFU pair: at `t=0.25`, `"The author's [M] name [M] Hsiao Yun-Hwa."`
   leaves the target name fully visible to the anchor.
   They additionally argue anchor-last KL *"matches the model's aggregate output distribution
   rather than directly suppressing the probability of 'forget tokens'."*
2. **Vanishing forget gradient.** *"At a memorized (near-one-hot) prediction, the anchor-last
   KL's gradient vanishes while the anchor-first KL's gradient does not."* Hence *"MDU training
   stalls exactly on the tokens we most need to forget."*

### KL orientation — terminology, and a correction of MDU

EMDU introduces unambiguous names and explicitly flags that MDU's own naming is
non-standard:

- **anchor-last** = `KL(p_θ ‖ p_anchor)` — what MDU minimises.
- **anchor-first** = `KL(p_anchor ‖ p_θ)` — what EMDU minimises.

EMDU: *"MDU calls the anchor-last direction 'forward KL' — the opposite of the widespread
convention in variational inference and knowledge distillation, where 'forward' usually denotes
KL(target‖model)."* **If our reimplementation reads MDU's abstract ("forward KL") and implements
`KL(target‖model)`, we have implemented EMDU, not MDU.** This is the single highest-risk
faithfulness trap in this whole area.

EMDU also pre-empts the obvious hand-wave: both orientations are strictly convex in `q` with the
same unique minimiser `q=p`, so mode-seeking / mass-covering *cannot* explain the difference.
Their argument is explicitly **dynamical**, not about the optimum.

### Training target / supervision

Retain / SFT cross-entropy (their Eq. 1), identical in form to MDU Eq. 2:

```
L_CE(x, y, t; θ) = -(1/t) Σ_{i∈M_t} log p_θ( y^{(i)} | x, y_t )
```

Anchor-last (MDU's forget loss, their Eq. 2, per masked token):

```
L_AL(x, y, t; θ) = KL(p_θ ‖ p_anchor) = Σ_{k=1}^{V} p_θ,k log( p_θ,k / p_anchor,k )
```

**Anchor-first (EMDU's forget loss, their Eq. 3), verbatim:**

```
L_AF(x, y, t; θ) = KL(p_anchor ‖ p_θ) = Σ_{k=1}^{V} p_anchor,k log( p_anchor,k / p_θ,k )
```

applied "only at i ∈ M_t, averaged uniformly over |M_t|" — same masking and same uniform
averaging as MDU's forget loss.

**Full objective (their Eq. 4), verbatim:**

```
L_EMDU(θ) = E_{(x,y)~D_f, t} [ L_AF(x, y, t; θ) ] + α E_{(x,y)~D_r, t} [ L_CE(x, y, t; θ) ]
```

Anchor construction with temperature: `p_anchor = softmax(τ · log p_θ0(· | ∅, y_t^FAM))`,
softmax per token, `τ ∈ [0,1]`. In Algorithm 1 this is written directly on logits:
`p_anchor ← softmax(τ z_u)`.

**Full Answer-span Masking (FAM), §3.2:** *"every answer position is replaced by [MASK] when
forming p_anchor, while the optimized model continues to see randomly masked y_t. FAM changes
the anchor input, and does not affect the computation of p_θ."* Effectively the anchor is
evaluated at `t=1` on the answer span with the prompt also fully masked, i.e. at `(∅, [MASK]^n)`.

### State distribution and sampling

- `t ~ U(0,1]` — **uniform**, note the half-open interval (excludes t=0; MDU writes `U[0,1]`).
- Trained model: each response token independently masked w.p. `t`. Prompt visible.
- Anchor: prompt fully masked **and** the entire answer span masked (FAM). The anchor input is
  therefore **constant across `t`** for a given example — a notable consequence: the anchor
  distribution no longer depends on `t` at all, only the *set of positions where the KL is
  applied* does.
- **Teacher-forced.** Same as MDU: `y_t` corrupts ground truth. No rollouts, no generation
  trajectory.
- Retain minibatch is drawn **independently, same size**, every step (Algorithm 1 line 4) —
  explicitly fixing the forget/retain pairing bug they found in MDU's released code.

### Gradient path

Plain loss descent, one backward per step. Algorithm 1 (verbatim structure):

```
Require: SFT initialization θ; frozen copy for the prompt-masked anchor; D_f, D_r; τ, α; stepsize η; batch size b
 1: for each epoch do
 2:   shuffle D_f into minibatches of size b
 3:   for each forget minibatch B_f ⊂ D_f do
 4:     draw an independently sampled retain minibatch B_r ⊂ D_r of size b
 5:     L ← 0
 6:     for each forget pair (x,y) ∈ B_f do
 7:       sample t ~ U(0,1]; mask answer tokens of y independently w.p. t to get y_t with masked set M_t
 8:       y_t^FAM ← copy of y_t with every answer token set to [MASK]
 9:       z ← model logits on (x, y_t);  z_u ← frozen-anchor logits on (∅, y_t^FAM)
10:       p_θ ← softmax(z);  p_anchor ← softmax(τ z_u)
11:       L ← L + (1/|M_t|) Σ_{i∈M_t} KL(p_anchor ‖ p_θ)
12:     end for
13:     for each retain pair (x,y) ∈ B_r do
14:       sample t ~ U(0,1]; mask answer tokens of y independently w.p. t to get y_t with masked set M_t
15:       L ← L + (α/t) Σ_{i∈M_t} ( -log p_θ(y^i | x, y_t) )
16:     end for
17:     θ ← θ - η ∇_θ L
18:   end for
19: end for
```

Stop-gradient: the anchor is a frozen copy — `z_u` carries no gradient. No detach inside the KL.

**Their Proposition 1 (logit gradients), verbatim** — this is the mathematical core and is worth
quoting because it tells you exactly what each objective does:

```
∇_z L_AL = q ⊙ ( log(q/p) - KL(q‖p) 1 ) ,     ∇_z L_AF = q - p
```

for `q = softmax(z)`, `p` the (frozen) anchor. Both lie in `{v : Σ_k v_k = 0}`. EMDU's reading:
*"The anchor-first gradient ∇_z L_AF equals the standard softmax cross-entropy gradient toward
the frozen target p."* And: *"The anchor-last gradient ∇_z L_AL is weighted by q itself.
Coordinates where the model is very confident contribute a vanishing update regardless of p."*

**This is a result we should internalise:** `∇_z KL(p ‖ softmax(z)) = q − p` is *exactly* soft-label
cross-entropy against target `p`. So **anchor-first KL and soft-CE-toward-the-anchor are the same
update** (they differ only by an additive constant `−H(p)` in the loss value, which has zero
gradient). Under the Part 3 rubric that is category (a), exact equivalence. Directly relevant to
our own reimplementation, which uses soft-CE/KL — see Critical Question 3.

Their Theorem 2 (near-one-hot regime) is stated for any `q` with small `ε = 1 − q_j`, "not only
in the limit, so that it applies throughout training as the model sits near (and is meant to
escape from) the memorized vertex, not only at convergence." Proof in their Appendix A.2; I read
the Proposition 1 proof in full (softmax Jacobian `∂q_k/∂z_j = q_k(δ_kj − q_j)`, the `+1` terms
cancelling because `q` is a probability vector) and it is correct. I did not re-derive Theorem 2
line by line — **NOT VERIFIED** at that level, though its conclusion follows naturally from the
`q ⊙ (·)` prefactor in Proposition 1.

### Preservation mechanism

`α E_{D_r}[L_CE]`, the same retain cross-entropy as MDU, with `α` selected to maximise the
Forget/Retain gap. On RWKU, `α = 0` (no retain set exists), so forgetting relies entirely on the
anchor — same as MDU.

EMDU's additional, non-loss preservation argument: the unlearned model **abstains** by emitting
EOS-only completions. They measure **71.8% (LLaDA) / 99.0% (Dream)** empty completions on TOFU
forget queries, with **≤0.5%** empty on Retain / RA / WF. On RWKU, 35–60% empty. They checked the
full generation canvas past the first EOS and report EOS at every remaining position — i.e. no
leakage hidden behind truncation. This is a genuinely careful check that most unlearning papers
do not do.

### Reference / teacher model

**Required, frozen, initialised from the unlearning starting point** — identical to MDU
("frozen copy for the prompt-masked anchor", Algorithm 1 Require line). Difference from MDU: its
*input* is `(∅, y_t^FAM)` rather than `(∅, y_t)`.

### Inference requirements

**Nothing beyond the edited weights.** EMDU adds an explicit deployment note: *"an empty
completion can be replaced by a generic 'I don't know' response without changing the unlearned
weights"*, optionally templated via a named-entity recogniser. That is a **post-processing
convenience, not a requirement** — do not count it as an inference-time dependency, but do note
that EMDU's qualitative "safe abstention" story partly depends on a wrapper the paper does not
evaluate.

### Parameterization

**Full-parameter.** Appendix D.2: *"Both Bases are full-parameter SFT on TOFU full (prompt loss
masked)"* from `GSAI-ML/LLaDA-8B-Instruct` and `Dream-org/Dream-v0-Instruct-7B`. No LoRA
anywhere. Hardware: **NVIDIA A100 80GB** (§5) — notably smaller than MDU's H200s, and still
2× our per-device memory.

### Selection procedure — the best-documented in this area

- **TOFU (Appendix D.3):** all methods initialised from Base SFT, trained on forget10 (400
  examples) with retain90; batch 4, accum 4 (effective 16, 25 steps/epoch). **Sweep:** LR
  `1e-5 … 4e-5`, training length 3–12 epochs. GA, being forget-only with a step budget, swept
  from 50 steps upward. **Selection rule, verbatim:** *"For each method, we chose the
  configuration from its sweep achieving the largest Forget/Retain gap, among configurations
  where RA and WF remained balanced (neither collapsing relative to the other)."*
  EMDU's chosen config: LR `1e-5`, **10 epochs** (LLaDA) / **9 epochs** (Dream).
- **RWKU (Appendix D.4):** forget-only (`α=0`), per-subject from the off-the-shelf Instruct
  checkpoint, **582 RWKU forget pairs per subject**; batch 4, accum 1 (effective 4, 146
  steps/epoch). Sweep: LLaDA LR `1e-5 … 3e-5`, length 20 steps – 3 epochs; Dream LR
  `{1e-5, 5e-6, 6e-6, 2e-5}`, length 20 steps – 1 epoch. **Selection rule:** largest
  Forget/Neighbor gap among configs where MMLU/TruQA/TriQA stayed balanced.
  EMDU's chosen config: LLaDA LR `1e-5`, **40 steps**; Dream LR `6e-6`, **30 steps**.
- Shared: AdamW, cosine schedule with **10% warmup**, sequence length **1024**, effective batch 16
  on TOFU.
- Baseline loss hyperparameters taken from MDU: NPO `β=0.2`; SimNPO `β=0.2, δ=0`; WGA `γ=1`;
  DPO `β=0.1` with a perturbed answer as preferred response.
- `τ = 1` for both MDU and EMDU in the main tables; Appendix F sweeps `τ ∈ {0,0.25,0.5,0.75,1}`
  without retuning LR/epochs per τ.

**Important honesty note on their selection rule:** "largest Forget/Retain gap" is selected
**on the test metrics themselves.** There is no held-out validation split. This is the standard
practice in this literature (TOFU/RWKU provide no unlearning validation split) but it means
every number in every table here — MDU's and EMDU's — is a *best-of-sweep on the reported
metric*. Any comparison we make must either adopt the same protocol or declare the difference.
This is the most consequential comparability issue in Part 4.

### Compute cost

- **3 forwards + 1 backward per step** (conditional with grad; frozen anchor on `(∅, y^FAM)`
  no-grad; retain with grad), collapsing to 2+1 when `α=0` (RWKU).
- Identical per-step cost to MDU. FAM changes the anchor's *input*, not the number of passes.
- Wall-clock: **not stated.** Hardware: A100 80GB.
- Small efficiency note we could exploit: because the anchor input `(∅, [MASK]^n)` does not
  depend on `t`, **the anchor forward is cacheable per forget example** across epochs and across
  `t` draws (it depends only on the example's answer length and the frozen weights). MDU's
  anchor is not cacheable this way because it depends on `y_t`. EMDU does not mention this.
  This connects directly to our own prior ARC (amortized reference caching) study.

### Code availability

- **No code or checkpoints released at the time of reading.** Contributions list: *"We will
  open-source our project and release (upon acceptance): the models after supervised fine-tuning
  (SFT) on TOFU (no such models are currently available), EMDU checkpoints for LLaDA-8B and
  Dream-7B on TOFU and RWKU, and full evaluation scripts."* Promised, not delivered.
- Their stack: built on the official MDU repository (<https://github.com/leegeoru/MDU>, MIT)
  extended with anchor-first KL + FAM + complete TOFU/RWKU evaluation; Base SFT via the **dLLM**
  library (<https://github.com/ZHZisZZ/dllm>, Zhou et al. 2026).
- **Runs on a diffusion LM: yes.** Natively, LLaDA-8B-Instruct and Dream-7B-Instruct.

### What EMDU demonstrates vs. claims

- *Claims* two MDU failure modes. **Failure mode 2 (gradient saturation) is demonstrated
  analytically** (Proposition 1 is a clean, correct derivation; Theorem 2 extends it off the
  vertex) **and empirically** (their §4 experiments on real data). This is the strongest piece of
  theory in the MDLM unlearning literature so far. **Failure mode 1 (leakage) is demonstrated by
  illustration and by ablation**, not by a direct measurement of anchor mass on forget tokens —
  Figure 3 is a worked example, and Table 2 shows FAM matters. Weaker evidence than mode 2.
- *Claims* AF and FAM are both needed. **Demonstrated** by a 2×2 ablation (Table 2): AL w/o FAM
  (= MDU) vs AF w/o FAM vs AL with FAM vs AF with FAM (= EMDU), on both backbones. Each single
  change gets Forget near zero but pays on Retain or RA/WF; only the combination is good
  everywhere. This is the right experiment and they ran it. Under the Part 3 rubric this is a
  textbook category (d) demonstration — a combination with substantive interaction, evidenced by
  the ablation that both single factors are insufficient.
- *Claims* robustness. **Demonstrated more thoroughly than MDU:**
  - **Generation robustness**: re-evaluated over all six (output length, denoising steps) combos
    `{(32,32),(64,32),(128,32),(64,64),(128,64),(128,128)}`. EMDU Forget rL spans 0.025–0.037
    (LLaDA) and is exactly 0.000 (Dream); MDU spans 0.113–0.171 (LLaDA), 0.210–0.270 (Dream).
  - **Paraphrase robustness**: TOFU's official `paraphrased_question` field. SFT drops
    0.822→0.439 (LLaDA) and 0.931→0.607 (Dream) but stays answerable; EMDU 0.030→0.090 (LLaDA)
    and 0.000→0.008 (Dream).
  - **Seed variance**: 2 additional unlearning seeds (3 total) per backbone per dataset, mean±std
    reported (their Tables 9, 10). TOFU LLaDA Forget rL `0.027±0.005`, Retain rL `0.794±0.008`;
    RWKU LLaDA F-L1 `0.119±0.013`. Main tables remain single-seed.
- **Does not claim or test**: relearning-attack robustness, MIA, any trajectory-aware mechanism,
  larger forget sets, or any backbone beyond LLaDA-8B / Dream-7B.
- *Does not* reproduce MDU's published numbers. Their Table 6: running MDU's official code under
  its suggested config, after fixing the forget/retain pairing, gives TOFU LLaDA Forget rL
  **0.346** — versus **0.034–0.135** across MDU's own τ rows. Even the config EMDU chose to
  report as "MDU" (0.113) differs from MDU's paper. **Nobody has reproduced MDU's table.**

---

## 3. TimeROME-DLM — training-free MDLM edit (important, and almost certainly relevant to us)

**Citation:** Yao, Zhengtao; Song, Liuyang; Zhang, Hongbo; Wei, Chenhao; Xu, Haoyan; Yang, Guang;
Wang, Siheng. "TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge
Editing for Masked Diffusion Language Models." arXiv:2606.12841v1, 11 Jun 2026.
<https://arxiv.org/abs/2606.12841>

Flagged with care: **no code is released** (*"full experimental artifacts to be released upon
acceptance"*, no URL), and I have not seen any independent reproduction. Every number below is
**as reported, NOT independently VERIFIED.**

- **Training target / supervision:** none — **no gradient training at all.** *"training-free,
  gradient-free, inference-time knowledge-editing."* An offline closed-form ridge solve.
- **State distribution and sampling:** this is the one MDLM method that **explicitly probes the
  denoising-step axis.** Its Temporal Indirect Effect (TIE) causal-tracing protocol runs three
  diffusion passes per fact — clean (cache residuals at every layer, step and module), corrupted
  (Gaussian noise into subject-token embeddings), patched (restore clean activations at a single
  coordinate `(ℓ,k,m)` during the corrupted run) — and measures the object's log-probability gain
  **at later denoising steps `k' > k`, aggregated with exponential decay over subsequent steps.**
  So the *selection of where to edit* is trajectory-aware, even though the edit itself is not a
  loss.
- **Gradient path:** no gradients. Closed-form ridge regression, offline.
- **Preservation mechanism:** ridge regularisation `λ` plus a sparsity threshold `q`
  (`TopQ_q` keeps the q largest-magnitude rows), rather than a retain loss.
- **Edit, as reported:** `h ← h + α · TopQ_q( Dᵀ G U h )` applied to the residual stream, with
  `G = (U Uᵀ + λI)^{-1}` precomputed once, `U` stacked subject keys (|F|×H), `D` stacked target
  deltas (|F|×H), `α ∈ [0.25, 2.0]`.
- **Reference/teacher model:** none.
- **Inference requirements:** **yes — this method is not weight-only.** A forward hook at
  coordinate `(ℓ*, m*)` applies the residual update **at every denoising step**, and the
  precomputed `U, D, G` must be shipped. Backbone weights are frozen and never updated. For us
  this is a *different deployment contract* from MDU/EMDU and must be scored as such.
- **Parameterization:** residual stream at a single layer `ℓ*` (reported as layer 19 of 32 on
  LLaDA). No parameters updated.
- **Selection procedure:** three hyperparameters (`α, λ, q`) "tuned on a small validation split."
  More disciplined than MDU in principle; the split is not characterised in what I read.
- **Compute cost:** edit installation 3.87 s; end-to-end construction for |F|=40 ≈ 26 s;
  **0 additional GB VRAM** vs. 35–51 GB for gradient methods; 4–14× wall-clock speedup vs.
  training baselines; 542 runs in ~48 GPU-hours on 4× A100-80GB.
- **Benchmarks/metrics:** TOFU forget01 reported in **log-probability (nats)**, not ROUGE-L:
  ForgetLP −118.94 (vs −35.7 baseline), RetainLP −89.08, i.e. ~83 nats of forgetting with
  retain within ~1 nat. Also reports MUSE-Books/News, WHP, WMDP-Bio, scaling to 400 facts, and
  six MDLM backbones (LLaDA variants, Dream-7B, MMaDA-8B, DiffuLLaMA-7B).
- **Comparability warning:** **ForgetLP in nats is not comparable to anything in MDU or EMDU.**
  TOFU forget01 ≠ forget10. There is no shared metric between this paper and MDU/EMDU on any
  shared split. Do not place it in the same ranking.

**Why it matters to us:** it is prior art for *"use the denoising-step axis to decide what to
change"* — which overlaps the trajectory-aware direction we suspect is open. It is **not** prior
art for a trajectory-aware *training objective*. The distinction is load-bearing for our novelty
claim and is formalised in Part 3.

---

## 4. A2D — the one MDLM post-training method that does not use t ~ U(0,1)

**Citation:** Jeung, Wonje; Yoon, Sangyeon; Cho, Yoonjun; Jeon, Dongjae; Shin, Sangwoo; Hong,
Hyesoo; No, Albert. "A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models."
arXiv:2509.23286, 2025. <https://arxiv.org/abs/2509.23286>

**This is safety alignment, not machine unlearning.** Included because it answers Critical
Question 1 and because its parameterization is directly relevant to our hardware.

- **Training target:** the standard MDLM masked cross-entropy, with harmful examples supervised
  toward `[EOS]` instead of their original tokens:
  `L(θ) = E_{λ, x0, xλ} [ -(1/λ) Σ_{i=1}^{L} 1[x_λ^i = MASK] log q_θ(x_0^i | x_λ) ]`
  over a harmful set `D_harm` (masked tokens → `[EOS]`) and a retain set `D_retain` (masked
  tokens → original targets).
- **Mask-rate sampling — the key detail:** *"Sample timestep t ~ U(0,1)"* then
  ***"Set mask ratio λ ← (1−ε)t + ε"***. So the effective mask rate is `λ ~ U(ε, 1)`, with a
  **floor `ε`** that prevents near-zero masking. This is a deliberate, documented departure from
  plain `U(0,1)`.
- **Parameterization:** **LoRA, rank r=32, α=64**, on projection modules across attention and
  feedforward. This is a public precedent for PEFT on an 8B-class MDLM for a
  forgetting-adjacent objective, which helps our Part 2 deviation argument considerably.
- **Trajectory:** *"No explicit 'any-step' mechanism exists beyond uniform timestep sampling
  during training. The any-step robustness emerges from training across variable corruption
  rates rather than explicit trajectory modeling."* So: **reshaped `t` distribution, yes;
  trajectory, no.**
- **Claims** near-zero DIJA attack success. Not reproduced here; the loss/sampling/LoRA details
  above come from the arXiv HTML. The numeric safety claims are **NOT VERIFIED** by me.

---

## 5. Fan et al. ICML 2025 — SAM / RS / GP / CR / WA for relearning robustness

**Citation:** Fan, Chongyu; Jia, Jinghan; Zhang, Yihua; Ramakrishna, Anil; Hong, Mingyi; Liu,
Sijia. "Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization
Perspective and Beyond." ICML 2025. arXiv:2502.05374 (v1 2025-02-07, v4 2025-05-27).
<https://arxiv.org/abs/2502.05374>. Code: <https://github.com/OPTML-Group/Unlearn-Smooth> (MIT).

### Core thesis

A relearning attack perturbs **weights**, not inputs. So robust unlearning is a min–max problem
over weight perturbations — which is exactly SAM applied to the **forget** loss. With
`ℓ_relearn = −ℓ_f`, their Eq. 3:

```
min_θ  [ max_{‖δ‖_p ≤ ρ} ℓ_f(θ+δ | D_f) ]  +  λ ℓ_r(θ | D_r)        (p = 2 by default)
         └──────── =: ℓ_f^SAM(θ) ─────────┘
```

Analytically (their Eq. 6) `ℓ_f^SAM(θ) ≈ ℓ_f(θ) + ρ‖∇_θ ℓ_f(θ)‖₂`, whose gradient involves `Hv`
with `H = ∇²_θθ ℓ_f`, `v = ∇ℓ_f/‖∇ℓ_f‖₂` (Eq. 7). **Thesis: relearning robustness tracks the
smoothness/curvature of the *forget* loss in weight space.** Supporting observation (their
Appendix B, Fig. A1): the *retain*-loss landscape is nearly unchanged by any of the methods, so
the effect is specific to the forget loss. Secondary claim: smoothing also defends input-level
jailbreaks (enhanced-GCG), explained via "shallow unlearning alignment" — NPO's per-token KL
against the origin model is small on the first few tokens, and smoothing fixes that (Fig. 6b).

### Training target / supervision — the five variants, verbatim

Base: `min_θ ℓ_f(θ|D_f) + λ ℓ_r(θ|D_r)` (Eq. 3).

| Variant | Equation as written | Symbols / values |
|---|---|---|
| **SAM** | Eq. 4: `δ*(θ) = argmax_{‖δ‖₂≤ρ} ℓ_f(θ+δ) ≈ ρ · ∇_θℓ_f(θ)/‖∇_θℓ_f(θ)‖₂`; Eq. 5: `min_θ ℓ_f^SAM(θ) = min_θ ℓ_f( θ + ρ ∇_θℓ_f(θ)/‖∇_θℓ_f(θ)‖₂ )` | `ρ = 0.01` (§5.1); grid `[1e-3, 1e-1]`. Ablation Table A3 (MUSE Books): 0.001 too weak, 0.1 breaks unlearning, 0.01 optimal |
| **RS** | Eq. 9: `ℓ_f^RS(θ) = E_{δ∼N(0,σ²)}[ ℓ_f(θ+δ) ]` | `σ`; 3 perturbation samples (App. C). Repo: `σ = sam_rho = 1e-4` for WMDP |
| **GP** | Eq. 10: `ℓ_f^GP(θ) = ℓ_f(θ) + ρ‖∇_θ ℓ_f(θ)‖₂` ("penalty SAM", after Dauphin et al. 2024) | `ρ` in the equation; App. C instead tunes `γ ∈ [1,10]` for CR/GP — **a symbol inconsistency in the paper** |
| **CR** | Eq. 11: `ℓ_f^CR(θ) = ℓ_f(θ) + γ‖ ∇_θℓ_f(θ+μv) − ∇_θℓ_f(θ) ‖₂`, `v = ∇_θℓ_f(θ)/‖∇_θℓ_f(θ)‖₂`; from the finite-difference `Hv ≈ [∇_θℓ_f(θ+μv) − ∇_θℓ_f(θ)]/μ` (Eq. 8) | `γ > 0` (grid `[1,10]`); `μ > 0` step |
| **WA** (SWA) | Eq. 12: `θ_{WA,t} = (θ_{WA,t}·n + θ_t)/(n+1)`, `θ_t = θ_{t−1} + Δθ_t` | `n` = #checkpoints. App. C: start at step 100, average every 5 steps |

**Answering our standing question about GP directly: GP is a penalty on the ℓ₂ norm of the
gradient of the FORGET loss only, taken with respect to the model WEIGHTS** — not the retain
loss, not the total objective, and not an input gradient. The paper explicitly contrasts it with
adversarial-robustness input-gradient penalties ("in this scenario, the gradient is computed
with respect to the model's input rather than its weights"). The same holds for SAM and CR: the
perturbation and the curvature are on the forget loss in weight space, and the retain term
`λ ℓ_r` is left unsmoothed in all five variants.

Algorithm A1 (SAM-enhanced unlearning) confirms the update:
`δ ← ρ ∇ℓ_f(θ_u)/‖∇ℓ_f(θ_u)‖₂` ; `g_f ← ∇ℓ_f(θ_u+δ)` ; `g_r ← ∇ℓ_r(θ_u)` ;
`θ_u ← θ_u − η (g_f + λ g_r)`.

### State distribution and sampling

Autoregressive teacher-forced cross-entropy on `D_f` and `D_r`. No sampling-of-states question
arises. **No generation trajectory anywhere.**

### Gradient path

- **SAM:** a custom two-step update (ascent step to build `δ`, then gradient at the perturbed
  point). **Remark 1 of the paper explicitly states the scalable implementation bypasses the
  second-order term** — SAM is pure first-order here.
- **RS:** averaged gradients at 3 noise draws. Plain descent on a smoothed loss.
- **GP, CR:** require **double backprop** (`create_graph=True`), i.e. differentiating through a
  gradient.
- **WA:** no change to the gradient at all — a running parameter average over checkpoints,
  applied as a callback.

### Preservation mechanism

The ordinary retain loss `λ ℓ_r(θ | D_r)`, unsmoothed, where `D_r` = Wikitext for WMDP. For the
RMU host, the parameter subset is RMU's own (unlearn layers 5–7, perturbations applied to layers
1–7, App. C). No projection, no KL anchor.

### Reference / teacher model

Inherited from the host objective: NPO needs a frozen reference (`self.infer_model` in the repo);
GradDiff does not; RMU needs a frozen copy for its retain term. The smoothing wrappers themselves
add no reference model.

### Inference requirements

**Nothing beyond the edited weights** for all five. (WA produces a single averaged weight set.)

### Parameterization

**Full-parameter, explicitly not LoRA** — every released config sets `"use_lora": false` and LoRA
appears nowhere in the paper. Optimizer `adamw_torch`, `bf16=True`,
`warmup_steps = max_steps//10`, `full_determinism=True` (`WMDP/src/model/unlearn.py`). The paper
itself never names the optimizer.

### Base objectives the variants sit on

- **All five smoothing variants are applied to NPO.**
- **SAM additionally** to **GradDiff** and **RMU** (Fig. 3, Table A1).
- RMU-LAT and TAR appear as *robust-unlearning baselines*, not as smoothing hosts.
- Noted finding: RMU is the robustness loser despite the best utility (UT 0.57 vs ~0.45); the
  paper attributes this to RMU editing only layers 5–7 while the attacker fine-tunes everything.
  **Directly relevant to any parameter-subset defense we might propose.**

### Selection procedure

App. C grids: LR `[2.5e-6, 1e-5]`; `λ ∈ [1, 2.5]`; NPO `β ∈ [0.01, 0.05]`; RMU `λ ∈ [800, 1600]`;
CR/GP `γ ∈ [1,10]`; SAM `ρ ∈ [1e-3, 1e-1]`. WMDP: 125 unlearning steps for NPO and GradDiff, 150
for RMU. Repo tuned values (`run_wmdp_unlearn.sh`, all `max_steps=125`, `seed 1001`): NPO lr 5e-6
/ β 0.0225 / γ 1.0; NPO+SAM lr 5e-6 / β 0.015 / γ 2.25 / ρ 0.01; NPO+RS lr 1e-5 / β 0.0125 / γ 2.0
/ σ 1e-4; NPO+CR lr 8.5e-6 / β 0.035 / γ 1.0 / gnr_rho 10.0; NPO+GP lr 7.5e-6 / β 0.04 / γ 1.0 /
gnr_rho 10.0; NPO+WA lr 5.5e-6 / β 0.05 / γ 2.0.
**No held-out validation split is described** — selection is on the reported metrics, same
weakness as MDU/EMDU.

### Evaluation protocol — complete

- **Benchmarks:** **WMDP (Bio subset only)** and **MUSE (Books + News)**. **TOFU is only cited in
  Related Work and is not used.** **RWKU is not used.**
- **Models:** Zephyr-7B-beta (WMDP primary), LLaMA-3 8B (WMDP, Table 1 only), LLaMA-2 7B
  fine-tuned on BBC News (MUSE News), ICLM 7B fine-tuned on Harry Potter (MUSE Books). Retain set
  for WMDP = Wikitext.
- **Metrics:** WMDP — `UE = 1 − accuracy on WMDP-Bio` (↑), `UT = zero-shot MMLU accuracy` (↑).
  MUSE — UE via **KnowMem** and **VerbMem** on `D_f` (↓), `UT = KnowMem on D_r` (↑).
  **Note: no ROUGE-L anywhere, and no exact match. This paper shares *no metric* with MDU or
  EMDU.**
- **Relearning attack:**
  - WMDP: fine-tune the unlearned model on data sampled from the forget set. Two axes:
    **N = 20/40/60** forget samples at 1 epoch; **M = 1/2/3** epochs with 20 samples. Also
    forget-*unrelated* relearn sets — **AGNews, GSM8K, SST2** (60 samples, 1 epoch).
    **Averaged over 5 independent random trials.**
  - MUSE: Fig. 5 caption gives 200/300/400 samples (Books) and 400/500/600 (News). The §5.2 body
    text instead says "(75, 100, 125)" — **an internal inconsistency in the paper.** The repo
    resolves it in favour of the caption (`BOOKS_MAX_STEPS=(50 75 100)`,
    `NEWS_MAX_STEPS=(100 125 150)` at `per_device_batch_size 4`); the body numbers appear to be
    step counts mislabelled as samples.
  - Relearn hyperparameters come from the **repo, not the paper**: lr `1e-5`, `adamw_torch`, bf16,
    warmup = max_steps/10. WMDP `Relearn+Forget.json`: batch 1, grad_accum 4, epochs 3,
    max_steps 100, seed 1; sample count `= max_steps × batch × accum × num_devices`
    (`WMDP/src/model/relearn.py`). MUSE: lr 1e-5, per-device batch 4, max_len 2048.
- **Jailbreak attack:** enhanced-GCG / FLRT (after Łucki et al.;
  `ethz-spylab/unlearning-vs-safety`), `--attack_layers 20 --max_iter 1500
  --optimize_prompts 0,2,3,4,5`.
- **How they decide a method is robust: purely comparatively.** Post-attack UE vs. the
  non-smoothed base, with "results closer to 'Origin' indicate weaker unlearning robustness",
  best per column highlighted. **No threshold, no statistical test, no confidence intervals**
  (only "averaged over 5 trials" for the WMDP relearn-set sampling). Worth internalising: the
  field's strongest relearning-robustness paper has **no significance machinery at all.**
- **Headline numbers (Table 2, WMDP, Zephyr-7B-beta):** UE without attack ≈ 0.73–0.75 for all.
  Under `N=20`: NPO 0.57, +SAM **0.70**, +WA 0.69, +RS 0.65, +CR 0.62, +GP 0.61. Under `M=3`:
  NPO 0.37, +SAM **0.59**, +CR 0.52, +GP 0.43, +WA 0.43, +RS 0.42. UT (MMLU) 0.41–0.46.
  Table 1 (LLaMA-3 8B, N=60): NPO 0.41 → NPO+SAM **0.70**, matching TAR (0.70) and beating
  RMU-LAT (0.44).

### Compute cost

- **SAM = 2 forget forward-backwards + 1 retain = 3 passes/step.** Confirmed in Algorithm A1 and
  in `SAMTrainer.training_step` (`WMDP/src/unlearn/base.py:754`): `forget_step` →
  `after_forget_step` (build `ε` from the global ℓ₂ grad norm) → `pre_forget_step_perturbed`
  (`p.data.add_(ε)`) → `forget_step_perturbed` → undo → `retain_step` → `combine_and_update`
  (`final_grad = f_grad + γ·r_grad`). The empirical confirmation is Table 1: 5.8 → 11.5 min.
- **RS = 3 noisy forget passes + 1 retain = 4 passes/step** (`RSTrainer.forget_step_randomized(...,
  n_samples=3)`, noise `torch.randn_like(p) * sam_rho`, gradients averaged).
- **GP, CR: double backprop** (`torch.autograd.grad(..., retain_graph=True, create_graph=True)`),
  and in the released code **inside a Python loop over every named parameter individually**
  (`WMDP/src/unlearn/GA.py:255` and `:337`). The paper never quantifies GP/CR/RS cost.
- **Wall-clock reported only in Table 1** (LLaMA-3 8B, WMDP, minutes): NPO **5.8**, RMU-LAT
  **10.3**, NPO+SAM **11.5**, TAR **7441.9**. No timings for RS/GP/CR/WA.
- **Hardware: the paper states none.** Scripts use `CUDA_VISIBLE_DEVICES=0,1,2,3` for
  unlearn/relearn and 2 GPUs for the jailbreak. **Which GPU model: NOT VERIFIED — not stated in
  the paper or the repo.**

### Code availability

- <https://github.com/OPTML-Group/Unlearn-Smooth>, **MIT License** ("Copyright (c) 2024 OPTML
  Group").
- **Checkpoints published: yes**, 10 models —
  <https://huggingface.co/collections/OPTML-Group/smooth-unlearned-model-67a92bb04d402b6ca3b2fb01>
  (`OPTML-Group/{NPO,NPO-SAM,NPO-RS,NPO-GP,NPO-CR,NPO-WA,GradDiff,GradDiff-SAM}-WMDP`,
  `NPO-SAM-MUSE-NEWS`, `NPO-SAM-MUSE-BOOKS`). Per-model-card licenses not stated on the
  collection page — **NOT VERIFIED**.
- **Runs on a diffusion LM: no.** The string "diffusion" appears **zero** times in the full v4
  text — all sections, appendices, references. Everything is autoregressive.

### ⚠️ Two implementation findings in the released code — I verified these myself

I fetched `WMDP/src/unlearn/GA.py` on 2026-10-20 and read lines 218–346 directly.

1. **As released, the GP and CR penalties contribute no gradient.** The norm is accumulated as
   `forget_grad_norm += f_grad_f.pow(2).sum().item()` (line 258 for GP, line 340 for CR).
   `.item()` returns a **Python float**, which detaches from the autograd graph. The subsequent
   `loss = forget_loss + self.gamma * retain_loss + 0.01 * forget_grad_norm` (GP, line 262) and
   `... + self.gnr_rho * forget_grad_norm` (CR, line 344) therefore add a **constant** to the
   loss value. The regulariser changes the logged loss but **not the update direction.** CR still
   differs from plain NPO, but for a different reason: line 299 does
   `self.accelerator.backward(forget_loss, retain_graph=True)` and the returned loss *also*
   contains `forget_loss`, so the forget gradient is **double-counted**.
   **This is my reading of the public code at the state I fetched it, not a claim the paper
   makes, and not a claim about any other commit.**
2. **GP's coefficient is hardcoded.** `run_wmdp_unlearn.sh` passes
   `NPO+FT+GNR.gnr_rho=10.0`, but line 262 uses a literal `0.01` and ignores `gnr_rho`. Also,
   `NPO+FT+CR` is imported in `__init__.py` but has **no dispatch branch** in
   `get_unlearn_method`, so the documented `--unlearn.unlearn_method NPO+FT+CR` command raises
   `ValueError("No unlearning method")` as the repo stands. (GP is reachable; it is named
   `NPO+FT+GNR`, gradient-norm regularization.)

**Why this matters enormously to us.** Our own HISTORICAL study
(`research/relearning-robust-mdlm` @ `a67d610044d0`) concluded that a gradient-norm defense was
**NOT SUPPORTED** for MDLM unlearning. That conclusion stands and is not revisable here. But it
is now clear that **the published GP numbers in Table 2 were produced by code whose penalty term
has zero gradient contribution** — so GP's reported 0.61 / 0.43 is, on this reading, *NPO with a
differently-tuned LR and β plus a constant in the loss*, not NPO with a weight-space
gradient-norm penalty. Three consequences:

- Our negative GP-style result is **not in tension** with Fan et al.'s GP row, because that row
  may not be measuring what its equation says.
- If we ever want to compare against GP, we must implement Eq. 10 **correctly** (true double
  backprop) and declare that we did, because that is **not** the published baseline.
- The genuinely load-bearing variant in that paper is **SAM** (first-order, implemented as
  described, and the consistently best row), not GP. If we test one smoothing defense on MDLM,
  it should be SAM.

---

## 6. D2F — two unrelated methods share this acronym

**(a) D2F in the diffusion-LM literature has nothing to do with unlearning.**

**Citation:** Wang, Xu; Xu, Chenkai; Jin, Yijie; Jin, Jiachun; Zhang, Hao; Deng, Zhijie.
"Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing."
arXiv:2508.09192, 8 Aug 2025. <https://arxiv.org/abs/2508.09192>.
Code: <https://github.com/zhijie-group/Discrete-Diffusion-Forcing> (MIT), mirrored at
<https://github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing>, which badges ICLR 2026
(**repo-asserted venue, not in the arXiv listing — NOT VERIFIED**).

A pure **inference-acceleration** method: (1) block-wise AR generation so KV cache can be used —
block-wise causal attention, bidirectional within a block, causal across blocks; (2) prediction
of later-block tokens before earlier blocks finish, enabling inter-block parallel decoding.
Trained by **asymmetric distillation** from a pretrained dLLM, with block-level noise ratios
sampled to be **monotonically increasing**; plus a pipelined parallel decoding algorithm. Reports
>2.5× faster than LLaMA3/Qwen2.5 on GSM8K and >50× vs vanilla LLaDA/Dream at comparable quality.
**Checkpoints released as LoRA adapters for LLaDA-Instruct-8B and Dream-Base-7B**, plus an HF
Spaces demo.

**Relation to unlearning: none.** Unlearning, forgetting, and privacy are absent from the
abstract and both repos' documentation. **Any claim coupling this D2F to unlearning would be
fabricated.** It is, however, relevant to us in two other ways: it is a public precedent for
**LoRA adapters on LLaDA-8B**, and its "block-level noise ratios sampled to be monotonically
increasing" is a non-uniform `t` schedule in the dLLM literature (see Critical Question 1).

**(b) "D2F" does exist in unlearning — a different, unrelated paper.**

**Citation:** Cong, Jingyue; Song, Andy; Yu, Yang; Sarwar, Tabinda; Li, Xinshu; Xu, Ziqi; Yi,
Xun; Wang, Ke; Han, Bo; He, Estrid. "Decouple to Forget: Semantic Mediation for Sensitive
Unlearning in Large Language Models." arXiv:2610.17654, 13 Oct 2026.
<https://arxiv.org/abs/2610.17654>. Code: <https://github.com/JingyueCong/D2F>. Preprint, **no
venue**.

Fine-grained LLM unlearning: extract a fact-level "mediator" from each forget answer, flag which
parts are sensitive, delete only those, generate counterfactual answers preserving non-sensitive
content, apply preference-based contrastive optimization. **Not a diffusion model paper.** It
belongs with the span-level / selective family (§ below). Its detailed algorithm was not
extracted for this document — **NOT VERIFIED** beyond title, authors, URL and this summary.

**Keep the two D2Fs strictly separate.** Discrete Diffusion Forcing (inference speed, dLLMs,
2025) vs. Decouple to Forget (LLM unlearning, 2026).

---

## 7–14. Remaining methods — see Parts 6 and 7

The remaining requested methods are covered at full field depth in:

- **Part 6** — NPO (§6.1), SimNPO (§6.2), WHP and WHP+ (§6.3), UNDIAL (§6.4)
- **Part 7** — RMU (§7.1), CLP (§7.2), CUSE (§7.3), TNT/TNFF (§7.4),
  selective/span-level: SPAN, SU, WGA/TNPO/WTNPO, SEUL and others (§7.5)

### The MDLM ports, as MDU defines them (this is the de-facto baseline standard)

Any claim of the form "we beat NPO on a diffusion LM" is a claim about *a particular port*. MDU
Appendix B.3 defines them, and EMDU reuses them, so these two definitions constitute the
field standard. The substitution rule is stated verbatim as:

> "Throughout, 'AR ⇒ MDLM' indicates the substitution `−log p^AR_θ(y|x) → L_sft(θ; (x,y))`."

where `L_sft(θ; (x,y))` is the Eq. (2) integrand on a single pair, and `L^ref_sft(θ; (x,y))` the
same under frozen reference parameters. MDU also states the eval-time estimator `L̂_rec` is used
**only** for the TOFU answer-probability metric and *"is never back-propagated through."*

| Baseline | AR form (MDU Appendix B.3) | MDLM port used by MDU and EMDU | Hyperparam |
|---|---|---|---|
| **GA** | `L = -E_{D_f}[ -log p^AR_θ(y\|x) ]` | `L = -E_{D_f}[ L_sft(θ;(x,y)) ]` | — |
| **GD** | `L_GD = L_GA + λ L_sft(θ; D_r)` | same, with MDLM `L_sft` | `λ=1` |
| **NPO** | `-(2/β) E_{D_f} log σ( -β [ log p^AR_θ(y\|x) - log p^AR_ref(y\|x) ] )` | `-(2/β) E_{D_f} log σ( β [ L_sft(θ;(x,y)) - L^ref_sft(θ;(x,y)) ] )` | `β=0.2` |
| **SimNPO** | `-(2/β) E_{D_f} log σ( β [ (1/\|y\|)(-log p^AR_θ(y\|x)) - δ ] )` | `-(2/β) E_{D_f} log σ( β [ (1/\|y\|) L_sft(θ;(x,y)) - δ ] )` | `β=0.2, δ=0` |
| **WGA** | `-E_{D_f} Σ_i w_i(-log p_θ(y^i\|x,y_{<i}))`, `w_i = p_θ(y^i\|x,y_{<i})^γ` | `-E_{D_f,t,y_t} Σ_{i∈M_t} w_i(-log p_θ(y^i\|x,y_t))`, `w_i = p_θ(y^i\|x,y_t)^γ` | `γ=1` |
| **DPO** | `-E log σ( β [ log r^AR_θ(y^+\|x) - log r^AR_θ(y^-\|x) ] )`, `log r^AR_θ = log p^AR_θ - log p^AR_ref` | `-E log σ( β [ r̂_θ(y^+\|x) - r̂_θ(y^-\|x) ] )`, `r̂_θ(y\|x) = -L_sft(θ;(x,y)) + L^ref_sft(θ;(x,y))` | `β=0.1`, `y^+` = perturbed answer, `y^-` = forget answer |

Observations that matter for our comparisons:

1. **Every port inherits `t ~ U[0,1]` teacher-forced masking.** None of them is trajectory-aware.
   The port is purely a substitution at the level of the per-sample NLL.
2. **NPO and DPO ports need the frozen reference**, so they cost an extra forward per step just
   like MDU/EMDU. GA, GD, SimNPO and WGA do not.
3. **WGA's port changes meaning.** In AR, `w_i` is the confidence at position `i` given a *causal
   prefix*; in the port it is the confidence at a *masked* position given bidirectional context
   at a random `t`. Same symbol, different quantity. Also note the MDLM WGA port drops the `1/t`
   ELBO weight entirely (it is a bare `Σ_{i∈M_t}`), unlike `L_sft`.
4. **SimNPO's `1/|y|` length normalisation is applied to `L_sft`**, which *already* contains a
   `1/t` weight and a sum over `|M_t| ≈ t·n` positions. So the MDLM SimNPO port normalises an
   already-normalised quantity. This is a defensible choice but it is *not* the AR method's
   semantics, and it is the kind of thing that should be checked before claiming a SimNPO
   comparison is fair.
5. **DPO's port is not DPO.** `r̂_θ` substitutes an ELBO surrogate for a log-likelihood ratio
   inside a sigmoid. The bound direction is not preserved. Treat MDLM-DPO as "a DPO-shaped
   objective", not DPO.

</content>

---

# Part 2 — What is actually reproducible on our hardware

## 2.0 Our constraints, restated

- **4× A100-SXM4-40GB; at most 2 usable concurrently** (GPUs 2–3 while another user holds 0–1;
  confirmed live on 2026-10-20 via `nvidia-smi`). Never interfere with other users' jobs.
- `GSAI-ML/LLaDA-8B-Instruct` at revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Local RWKU data, **re-verified file by file on 2026-10-20 and NOT as uniform as first
  reported**: `/data/bijaypandey/llada_rwku/data/rwku/Target/` holds **10 entities**
  (`1_Stephen_King` … `10_Prince_Harry,_Duke_of_Sussex`) but only `forget_level{1,2}.json` and
  `neighbor_level{1,2}.json` — **no `pair.json` and no `forget_level3.json`**.
  `/data/bijaypandey/llada_emdu/data/rwku/Target/` holds the **complete** set (`pair.json` = 582,
  `forget_level{1,2,3}`, `neighbor_level{1,2}`, `positive`/`negative` = 582) for **only 3
  entities**: Stephen King, Confucius, Bruce Lee. **Trainable today: 3 subjects. F-L3 reportable
  today: 3 subjects.** See `docs/PRIOR_ART_H11.md` §4.1.
- PEFT/LoRA available; bf16.
- No paid services, no new model-weight downloads, no infrastructure permission changes.
- Resource ledger: **4.117 GPU device-hours consumed across all six prior studies combined.**
  Any plan costing tens of GPU-hours is out of scale for this project.

## 2.1 The memory wall, independently re-derived

Our HISTORICAL conclusion `PAPER_ALIGNED_RUN_BLOCKED` (`research/llada-emdu-mdu-rwku-v1` @
`e2c87cae87e2`) states full-parameter 8B training does not fit in 2×40GB, citing ~89.6 GiB of
FP32 optimizer state alone. **I re-derived this independently and it reproduces exactly.** For
8.02e9 parameters:

| Item | Size |
|---|---|
| bf16 weights | 16.0 GB / 14.9 GiB |
| bf16 gradients | 16.0 GB / 14.9 GiB |
| fp32 master copy | 32.1 GB / 29.9 GiB |
| AdamW `m` (fp32) | 32.1 GB / 29.9 GiB |
| AdamW `v` (fp32) | 32.1 GB / 29.9 GiB |
| **Optimizer state (master + m + v)** | **96.2 GB / 89.6 GiB** |
| Full-parameter steady state (w + g + opt) | 128.3 GB / 119.5 GiB |
| **Our budget (2×40GB)** | **80.0 GB / 74.5 GiB** |
| ZeRO-3 sharded /2, per GPU | 64.2 GB / 59.8 GiB — **still exceeds 40 GB** |

So full-parameter is blocked **even with ZeRO-3 across both GPUs**, before any activations. The
only full-parameter route is ZeRO-3 with CPU/NVMe optimizer offload, which needs ~96 GB of host
RAM and costs roughly an order of magnitude in step time. Given a 4.1 GPU-hour project history,
that is not a route; it is a different project.

By contrast, **LoRA r=32 on attention + MLP projections** (the A2D configuration, and the
configuration D2F ships adapters in) is ~84 M trainable parameters:

| Item | Size |
|---|---|
| frozen bf16 weights | 16.0 GB / 14.9 GiB |
| adapter + AdamW state | ~1.2 GB / 1.1 GiB |
| **LoRA steady state** | **~17.2 GB / 16.0 GiB** |
| sharded over 2 GPUs | ~8.6 GB / 8.0 GiB |

This fits on **one** 40 GB card with room for activations at seq-len 1024, batch 4, with
gradient checkpointing. **Therefore every full-parameter method in Part 1 must be reproduced
here as LoRA, and that is a deviation, not an exact reproduction.** This phrasing is binding:
we never call a LoRA run a reproduction of a full-parameter result.

## 2.2 A benefit of the LoRA deviation worth naming

MDU and EMDU both need `p_θ0`, the **frozen pre-unlearning model**, evaluated with a null
prompt. Under full-parameter training that is a second set of 8B weights (+16 GB bf16). **Under
LoRA it is free: `p_θ0` is exactly the adapter-disabled forward pass through the same frozen base
weights.** So on our hardware the LoRA variant of MDU/EMDU is:

- one forward with the adapter enabled, under grad;
- one forward with the adapter disabled, under `no_grad`;
- one backward.

with **no second weight copy at all**. This is a real advantage of the deviation and should be
stated as such — it is also what makes MDU/EMDU *more* tractable for us than GA/GradDiff would
naively suggest, since the reference is not a memory cost.

**A memory cost neither paper mentions, and it is not small.** LLaDA-8B-Instruct's config (read
from our local copy: `n_layers 32`, `d_model 4096`, `mlp_hidden_size 12288`, **`vocab_size
126464`**) means the anchor is a full-vocabulary distribution of **126,464 entries per masked
position**. At batch 4 × sequence 1024, one such tensor in bf16 is
`4 × 1024 × 126464 × 2 B ≈ 1.04 GB`. A KL between two of them, with the intermediate `log(p/q)`,
touches **three or four** such tensors — several GB of transient activation on top of everything
else, for *both* MDU and EMDU and for *any* soft-target objective including ours. Mitigations to
plan for: compute the KL in chunks over the sequence, mask to `M_t` **before** materialising the
full-vocab tensors (only ~`t·n` positions actually contribute), and keep the anchor in bf16 rather
than promoting to fp32. **This, not the model weights, is the likely OOM site for a soft-CE/KL
MDLM objective on a 40 GB card** — and it is a reason our exclusion target may be cheaper than a
model-derived anchor, since a uniform-over-complement target need never be materialised densely.

Additional EMDU-specific saving: because FAM makes the anchor input `(∅, [MASK]^n)`
**independent of `t`**, the anchor forward is **cacheable per forget example** — it depends only
on the answer length and the frozen base weights. MDU's anchor is *not* cacheable this way
(its input is `(∅, y_t)`, which changes every step). This connects directly to our HISTORICAL
ARC study (`research/arc-amortized-reference-caching` @ `28154b2f001c`), whose exactness and
1.257× speedup held while per-fact efficacy parity failed at 128 facts. For EMDU the caching is
*exact by construction*, not an approximation — a different situation from what ARC tested.

## 2.3 The deviations that are consequential, with evidence

Not all deviations matter equally. Three do.

**(D1) LoRA restricts the reachable parameter space, and there is direct published evidence this
costs relearning robustness.** Fan et al. 2502.05374 report that **RMU is the worst method for
relearning robustness despite the best utility**, and attribute it to RMU editing only layers
5–7 while the relearning attacker fine-tunes the whole model (their §5.2, Fig. 3). LoRA is also
a parameter-subset restriction. Our own HISTORICAL finding that MDLM unlearning is **reversible**
(0.049 → 0.605 in 10 relearning steps; `research/relearning-robust-mdlm` @ `a67d610044d0`) was
itself obtained under our constraints. **Declaration required:** any relearning-robustness
number we produce under LoRA is a lower bound relative to a full-parameter equivalent, and we
cannot attribute the gap to the method.

**(D2) TOFU is out of reach, so our benchmark is RWKU only.** TOFU requires a Base SFT stage to
instil the fictitious knowledge: MDU and EMDU both use **1000 epochs** on TOFU-full for LLaDA at
LR 2e-5, effective batch 16. TOFU-full is 4000 QA pairs → 250 steps/epoch → **~250,000 SFT
steps**, full-parameter. At even 1 s/step that is ~70 GPU-hours *for the SFT alone*, before any
unlearning or evaluation, and it would be a LoRA SFT, which is **not** the same memorisation
regime — a LoRA Base SFT might not even reach the near-one-hot memorised predictions that EMDU's
whole gradient-saturation argument is about. Neither paper released a Base SFT checkpoint
(EMDU Appendix E.1 confirms for MDU; EMDU's own are promised "upon acceptance"). **Conclusion:
TOFU is not reproducible here, and no TOFU comparison should be attempted.**

**(D3) Our base-recall regime differs from the papers', and the metric is why.** Our HISTORICAL
conclusion `INSUFFICIENT_BASE_RECALL` (`research/llada-rwku-baseline-v1` @ `aecefcd662c5`)
measured LLaDA-8B-Instruct at **22/100 exact match** on RWKU probes over 10 entities. MDU reports
base F-L1 of **0.488** and EMDU **0.484** on the same backbone and the same 10 subjects. These are
**not in conflict**: F-L1 is **ROUGE-L recall**, which gives partial credit for any overlapping
subsequence, whereas exact match does not. **Operational consequence: if we want numbers
comparable to MDU/EMDU we must report ROUGE-L recall, not exact match.** Reporting EM and
comparing to their tables would be an error. Both metrics can be reported; only one is
comparable.

## 2.4 Per-method verdict

Legend — **Y**: runnable essentially as published. **Y\***: runnable with the LoRA deviation
(and nothing else consequential). **P**: partially runnable; a stated component cannot be
reproduced. **N**: not runnable here.

| # | Method | Runnable here? | Consequential deviations |
|---|---|---|---|
| 1 | **MDU** (RWKU track) | **Y\*** | LoRA instead of full-parameter (D1). Anchor comes free (§2.2). **Also unavoidable:** the paper states no unlearning LR or step count, so we must either adopt the repo config (`1e-5`, 3–9 epochs) or EMDU's sweep — *either way it is our selection, not theirs.* MDU is a τ sweep, not a point; we must sweep τ or declare which τ. MDU ships **no LLaDA RWKU eval script** (EMDU Appendix E.1), so the evaluation harness is ours. The ROUGE-L F1-vs-recall ambiguity means we cannot match their reported figure even in principle — report recall and say so. |
| 1b | **MDU** (TOFU track) | **N** | Base SFT unreachable (D2); no released checkpoint. |
| 2 | **EMDU** (RWKU track) | **Y\*** | LoRA (D1). **Best-specified target in this area:** LR `1e-5`, **40 steps**, batch 4, accum 1, `α=0`, `τ=1`, per-subject, 582 pairs/entity — and the `pair.json` files we do have contain **exactly 582 entries**, matching EMDU. **But we only have `pair.json` for 3 of the 10 subjects** (see the CORRECTION in the Executive Summary and `docs/PRIOR_ART_H11.md` §4.1), so a 10-subject reproduction is **data-blocked, not compute-blocked**, until the remaining 7 subjects' forget corpora are obtained. No code released, so we implement from Algorithm 1 — which is complete enough to do faithfully. 40 steps × 3 entities is **trivially affordable**; evaluation dominates cost. |
| 2b | **EMDU** (TOFU track) | **N** | Same as 1b. |
| 3 | **TimeROME-DLM** | **P** | No code, no URL, release "upon acceptance". Gradient-free and 0 extra VRAM, so it would be *the cheapest* thing here if it existed — reimplementation from the paper is a real project (TIE causal tracing needs 3 diffusion passes per fact with full residual caching at every layer and step, which is a memory problem of its own on 40 GB). Also reports in **nats on TOFU forget01**, so even if reimplemented it gives no number comparable to our RWKU track. **Different deployment contract** (inference hook required). Treat as related work, not a baseline. |
| 4 | **A2D** | **Y\*** (but off-task) | Safety alignment, not unlearning. Already LoRA r=32 as published, so *no* parameterization deviation. Worth running **only** as a probe of the mask-rate-floor question (Critical Question 1), not as an unlearning baseline. |
| 5 | **GA / GD / NPO / SimNPO / WGA / DPO** (MDU's MDLM ports) | **Y\*** | LoRA (D1). Loss hyperparameters are fully specified by MDU Appendix B.3 and reused by EMDU (`β=0.2`; `β=0.2, δ=0`; `γ=1`; `β=0.1`), so these are the cheapest and most faithful baselines available to us. LR/step budget must be swept (EMDU's RWKU grid is a defensible template: LR `1e-5…3e-5`, 20 steps–3 epochs). GD is **not applicable on RWKU** (no retain split) — EMDU says so explicitly. DPO needs a perturbed-answer construction that neither paper fully specifies for RWKU — **P** for DPO specifically. |
| 6 | **Fan et al. SAM** | **Y\*** | LoRA (D1), and **in SAM's case the deviation is directly adverse**: SAM perturbs weights, and under LoRA the perturbation lives in an 84 M-dimensional subspace rather than 8 B dimensions. Also a host-objective change: Fan et al. apply SAM to AR NPO/GradDiff/RMU; we would apply it to an **MDLM** NPO port or to EMDU — a *new* combination, not a reproduction. Cost: **2 forget forward-backwards + 1 retain per step**, i.e. ~2× an EMDU step. Affordable at RWKU scale. |
| 6b | **Fan et al. GP** | **Y\*, but do not call it GP** | Eq. 10 needs true double backprop through an 8B graph. Under LoRA the penalty is on `‖∇_LoRA ℓ_f‖₂`, not `‖∇_θ ℓ_f‖₂` — **a different quantity**. And see the §5 code finding: the *published* GP is a no-op penalty, so a correct implementation is **not** the published baseline and must not be presented as reproducing it. Combined with our HISTORICAL `NOT SUPPORTED` verdict, GP is low value for us. |
| 6c | **Fan et al. RS** | **Y\*** | 4 passes/step. Noise `σ=1e-4` applied to weights; under LoRA it is applied to adapter weights, which is again not the same perturbation. Cheap enough to try. |
| 6d | **Fan et al. CR** | **P** | Double backprop; and `NPO+FT+CR` has **no dispatch branch** in the released repo (raises `ValueError`), so there is no runnable reference implementation to check ours against. |
| 6e | **Fan et al. WA** | **Y** | **The one variant with no deviation at all.** A running parameter average over checkpoints is orthogonal to LoRA (average the adapters), adds ~zero compute, needs no reference model, and changes nothing at inference. Start at step 100, average every 5 steps — but note EMDU's RWKU budget is only **40 steps**, so Fan et al.'s schedule must be rescaled, and that rescaling is a deviation we must declare. |
| 7 | **Fan et al. relearning-attack protocol** | **Y, and we should adopt it** | lr `1e-5`, AdamW, bf16, warmup = steps/10; axes `N ∈ {20,40,60}` samples at 1 epoch and `M ∈ {1,2,3}` epochs at 20 samples; **5 independent trials**; plus forget-unrelated relearn sets (AGNews/GSM8K/SST2). All of this is cheap, is the field standard, and gives us a protocol we did not have in the HISTORICAL relearning study. The relearn-set construction must be adapted from WMDP to RWKU. **Note their decision rule has no statistical test** — we can do better cheaply by reporting the 5-trial spread. |
| 8 | **RMU** (§7.1) | **Y, with one engineering change** | RMU is *already* a parameter-subset method — it updates exactly three matrices (`layers.{ℓ-2,ℓ-1,ℓ}.mlp.down_proj.weight`), ≈0.3 GB trainable + ≈1.2 GB AdamW state for LLaDA-8B. **So RMU needs no LoRA at all** and is the cheapest *full-weight* method available to us. The published blocker is its **second full frozen model copy** (2× 16 GB); because only three matrices change, `M_frozen` can be recovered by retaining just the three original matrices (~0.3 GB) — **exactness-preserving, not an approximation.** Deviations to declare: ℓ must be re-searched (LLaDA's depth/MLP shape ≠ Zephyr's); the forget corpus would be RWKU pairs as raw text, not RMU's published data regime; and **the WMDP forget corpora are gated** behind an HF token we should not accept. One genuine advantage: RMU's loss is on **activations**, so it is **mask-rate agnostic** and its MDLM port is less ambiguous than any likelihood-based port. |
| 9 | **NPO / SimNPO** as originally published (AR) | **N** (as published) / **Y\*** (as MDU's MDLM port) | The AR papers run Llama-2-7b-chat on TOFU with a Base SFT we cannot produce (D2). The *ports* are row 5 and are available. **SimNPO is the only method in this document with MIT code *and* MIT published weights** — useful as a reference implementation to check our port against, even though its checkpoints are AR. Note NPO's repo has **no license at all**. |
| 10 | **WHP / WHP+** | **N** | WHP needs a **reinforced model** (the base deliberately fine-tuned *onto* the forget set) plus **GPT-4** for anchor-term extraction — a paid external service, which our constraints forbid. **No official code was ever published.** Its weights are Microsoft-Research-license, non-commercial, **redistribution expressly forbidden**. WHP+ is Liu et al.'s own N=1 ablation and needs their causal-intervention stack. Also: the MUSE reading of WHP is a **test-time** method requiring the reinforced model at every decoding step — a different deployment contract. **Not a candidate.** |
| 11 | **UNDIAL / FUnDIAL** (§6.4) | **Y\*** and worth doing | Structurally the closest AR method to MDU/EMDU — a *target-matching* objective against a frozen, single-logit-perturbed teacher. **Needs no retain set**, which is exactly the RWKU situation. Already published **with LoRA** (r=8 on LLaMA-2-7B for MUSE), so the parameterization deviation is small. γ sweep `{1,3,10,30}` is cheap. Deviations: never run on RWKU or on any diffusion LM, so the port is ours; the repo has **no license**; and the MDLM version must decide whether the teacher sees the prompt (UNDIAL's does; MDU/EMDU's does not) — **that choice is a substantive design decision, not a detail.** |
| 12 | **CLP** | **N/A — not an unlearning method** | A 2019 fairness penalty on CNN text classifiers with zero unlearning citations (§7.2). Nothing to reproduce. If the *idea* was wanted, cite ULD (arXiv:2406.08607) or Offset Unlearning (arXiv:2404.11045) instead. |
| 13 | **Selective / span-level: SU** (§7.5b) | **Y\*** and cheapest of the three | SU's n-gram selector is **~20 MB** and the paper's own conclusion is that it beats a fine-tuned Mistral-7B at deciding what to forget. Trains two tiny assistants on RWKU splits, thresholds the score difference, dilates to a 5-gram. Deviations: **no code released**; the host objective becomes an MDLM port; and "5-gram dilation" has to be redefined for a bidirectional masked model, where there is no left-to-right phrase boundary. |
| 13b | **Selective: SPAN** (§7.5a) | **P** | Needs **EK-FAC** inverse-Hessian approximation over an 8B model — a real memory and engineering cost on 40 GB — plus K=10 sampled generations per anchor token, which for an MDLM at 128 denoising steps is expensive. The repo **withholds the training step** and ships the self-consistency script for TOFU only, with **no license**. Its diagnosis (prior selective methods cause *"severe over-forgetting"*, collapsing retain KnowMem to 0.00) is valuable to us regardless of whether we run it. |
| 13c | **Selective: WTNPO / TNPO** (§7.5c) | **Y\*** | MDU and EMDU already run **WGA at γ=1** — the α-only, reference-free variant. **TNPO and WTNPO have never been ported to MDLM**, and both the source paper and the SPAN paper report WTNPO as the strongest of the family. A cheap, concrete gap — but category **(c)** under Part 3, so a baseline, not a contribution. |
| 14 | **CUSE** (§7.3) | **N as published; mechanism is the template we care about** | Multimodal benchmarks (MLLMU-Bench, CoVUBench) and backbones we do not have; **no code**; **`λ` never given a value**; decoding config never given. But it is the field's existence proof that on-policy trajectory-conditioned target matching beats teacher-forced unlearning (off-policy ablation 0.129 → 0.372), it uses **anchor-first KL** like EMDU, and it is **already LoRA r=8**. Porting it to MDLM is a research proposal, not a baseline. |
| 15 | **TNT / TNFF** | **N — do not exist** | See §7.4. No paper uses either name for an unlearning method. The closest real method behind a possible "TNT" mislabel is **Temper-Then-Tilt / T3-Unlearning** (arXiv:2602.10217, ICML 2026), which keeps the base model frozen and **needs a classifier at generation time** — a different deployment contract from everything else here. |
| 16 | **TimeROME-DLM-style trajectory localisation** | **P** | Already covered in row 3, but worth restating as a *mechanism* we might borrow: it is the only published MDLM method that reads the denoising-step axis. No code, and full residual caching at every layer × step is its own 40 GB problem. |

## 2.5 The affordable plan this implies

- **Benchmark: RWKU, ROUGE-L recall** (F-L1/L2/L3, N-L1/L2) plus MMLU / TruthfulQA / TriviaQA.
  Not TOFU. **Entity count: 3 today, 10 only if the missing forget corpora are obtained** — see
  the CORRECTION in the Executive Summary.
  **Also note (from `docs/PRIOR_ART_H11.md` §4.2): every RWKU forget answer is short** — F-L1 mean
  1.57 words / median 1 / 86.1% ≤2 words; F-L2 mean 2.16; F-L3 mean 2.10 / 89.7% ≤2 words. RWKU's
  ROUGE-L recall is therefore effectively entity-level exact match, which is good for
  interpretability but means **RWKU cannot distinguish surgical forgetting from whole-answer
  destruction.** Any claim of that form needs long-form answers.
- **Training is nearly free.** EMDU's published RWKU budget is 40 steps at batch 4; 10 entities
  is 400 steps per method. Even at 2 s/step that is well under half a GPU-hour per method.
- **Evaluation dominates.** MDLM generation at 128 denoising steps over F-L1/L2/L3 + N-L1/L2 +
  three utility suites, for 10 entities × every method × every config in a sweep, is the actual
  budget line. Plan the sweep around generation cost, not training cost.
- **MMLU prompt placement is a trap.** Use **RWKU's official placement** ("Answer:" after the
  assistant header), not MDU's. EMDU measured off-the-shelf LLaDA-8B-Instruct at **0.400** under
  MDU's placement vs **0.680** under RWKU's — a 0.28 swing on a utility metric, from prompt
  formatting alone. MDU's paper reports 0.395. **If we use the official placement our MMLU
  numbers will not match MDU's table, and that is correct behaviour, not a bug.**
- **Generation hyperparameters must be fixed and declared** before any comparison: EMDU shows
  MDU's forget score moves from 0.113 to 0.171 across (length, steps) combinations. Pick one
  canonical setting (their main protocol is **128 steps, 128 output tokens, answer canvas 128**)
  and additionally report the six-combination grid for anything we claim is robust.

---

# Part 3 — The novelty framework

A rubric for placing a candidate algorithm `A` relative to a published method `B`. It is meant
to be **operational**: each category states the evidence that places `A` there and the checks
you must run to earn that placement. Default assumption is the *most conservative* category the
evidence supports. **The burden of proof is on novelty, not on sameness.**

Apply it per *pair* `(A, B)`, then take the strongest (lowest-letter) category over all `B` in
the prior art as `A`'s standing. A method is only as novel as its closest neighbour.

## (a) Exact equivalence

`A` and `B` induce **the same update** on the same inputs, for all parameter values, up to
reparameterisation and additive constants in the loss.

**Evidence that places it here — any one suffices:**
1. **Algebraic identity of the gradient.** Derive `∇_θ L_A` and `∇_θ L_B` and show they are equal,
   or equal up to a positive scalar that the learning rate absorbs. *Worked example:*
   `∇_z KL(p ‖ softmax(z)) = q − p` (EMDU Proposition 1) is exactly the soft-label cross-entropy
   gradient toward `p`. So **anchor-first KL ≡ soft-CE against the anchor.** They differ only by
   `−H(p)` in the loss value, which is constant in `θ`. Category (a).
2. **Numerical gradient agreement.** On ≥3 random seeds and ≥2 input batches, compare the full
   parameter-gradient vectors: cosine similarity ≥ 1 − 1e-6 and relative ℓ₂ difference ≤ 1e-6 in
   FP64, or at the documented FP32 tolerance from our HISTORICAL gate-tolerance work. **This is
   the check to actually run**, because algebra is where people fool themselves. Match
   element-wise on the real dimension, not on a toy dimension.
3. **Bit-level loss-value agreement after subtracting the constant**, across a `t` sweep.

**Checks you must run before claiming you are NOT in (a):** numerical gradient comparison
against every prior-art method whose loss has the same functional type (both KLs, both CEs,
both sigmoid-of-log-ratio). If you have not run it, you may not claim better than (a).

**Reporting rule:** an (a) finding is not a failure. It is a *result*, and it is the one we are
most likely to produce. Publish it as an equivalence, with the derivation and the numerical
check.

## (b) A special or limiting case

`A` is `B` with a hyperparameter fixed, a term dropped, or a limit taken — or vice versa.

**Evidence that places it here:**
1. **Exhibit the substitution.** Name the parameter and the value. *Worked examples from this
   document:* MDU at `τ=0` **is** the Maximum Entropy loss (MDU states this itself: "exactly the
   same as the ME loss"); MDU at `τ=1` is anchor-last KL to the untempered unconditional model;
   EMDU is MDU with the KL arguments swapped **and** the anchor input changed, so EMDU is *not* a
   special case of MDU — it is (c)/(d), see below.
2. **Show the family contains both.** Write a single parameterised objective and show `A` and `B`
   are two settings of it. If you can, you are in (b), and the burden becomes showing the
   *interior* of the family is better than both endpoints — which is a (d) or (e) claim, not a
   (b) one.
3. **Limiting behaviour.** `A = lim_{λ→∞} B` or similar, with the limit shown to be well defined
   (not just plugged in).

**Checks you must run:** sweep the identified parameter and confirm `A`'s behaviour appears at
the stated value. If `A`'s claimed benefit **does not** appear at the endpoint and **does** appear
in the interior, you are making a (d)/(e) claim about the interior and must say so.

## (c) A familiar combination

`A` combines mechanisms each of which is already used, in a pairing that is standard practice,
and the combination's effect is **additive / separable** — each component does what it does
elsewhere, and no interaction is demonstrated.

**Evidence that places it here:**
1. **Each component has prior art** — cite it, with the paper where that component is used for
   this purpose.
2. **The pairing is conventional** — e.g. "forget loss + retain CE" (MDU, EMDU, GD, NPO+retain,
   and Fan et al. all do this); "objective + LoRA"; "objective + frozen reference".
3. **Ablations show separability:** `A(both) − A(neither) ≈ [A(c1) − A(neither)] + [A(c2) −
   A(neither)]` within noise, on the metric you care about, across ≥3 seeds.

**Checks you must run to stay out of (d):** the full 2×2 (or 2^k) ablation **with seed variance**.
Without the variance you cannot tell additivity from interaction, and the default is (c).

## (d) A potentially novel combination with substantive interaction

`A` combines mechanisms whose **joint** effect is demonstrably not the sum of the parts.

**Evidence required — all three:**
1. **Full factorial ablation.** Every subset of the `k` mechanisms, same budget, same selection
   rule. *Worked example — EMDU meets this:* their Table 2 reports AL-without-FAM (= MDU),
   AF-without-FAM, AL-with-FAM, AF-with-FAM (= EMDU) on **two** backbones. Each single change
   drives Forget to ~0 but pays on Retain (LLaDA) or RA/WF (Dream); only the combination is good
   on every metric.
2. **A superadditivity measurement with uncertainty.** The interaction term must exceed the
   seed-to-seed spread. EMDU reports 3-seed mean±std for the chosen configuration (their Tables
   9, 10) but **not** for the ablation cells — so even EMDU's (d) claim is one experiment short
   of fully earned. *We should not repeat that gap.*
3. **A mechanism story that predicts the interaction, stated before the measurement.** EMDU has
   this: Proposition 1 predicts AF escapes memorised vertices where AL stalls, and FAM removes
   anchor-side leakage; the two address *different* failure modes, which is why neither alone
   suffices. A post-hoc story does not count.

**Checks you must run:** (1) and (2) above, plus a negative control — a *third*, unrelated
mechanism added to the pair, to confirm the interaction is specific and not "any two things help".

**"Potentially" is load-bearing.** (d) means the combination is novel *and* interacting. It does
**not** establish that the benefit generalises, which is (e)'s concern.

## (e) An algorithmic difference whose benefit is unestablished

`A` genuinely differs from all prior art — the update is not equivalent, not a special case, and
not a familiar combination — but the **benefit is not established**.

**This is the default landing place for a new idea, and most of our candidates will start here.**

**Evidence that places it here — any one:**
1. **The difference is real but unmeasured.** No head-to-head against the closest prior art under
   a shared protocol.
2. **Measured, but not separably from confounds.** Different LR, different step budget, different
   selection rule, different generation settings, or different metric from the comparator. *Note:
   every number in MDU vs EMDU vs Fan et al. is confounded this way across papers — see Part 4.*
3. **Measured and separated, but within noise.** The effect does not exceed the seed spread.
   *Worked example from our own history:* TD-SSD vs the SSD control — best point +0.0625, which
   is **1 probe out of 16**, and worse at 150 steps (`research/llada-poc-v1` @ `c3578a908d6c`).
   That is category (e), and we recorded it as such.
4. **Measured, separated, significant — but only in a regime that does not transfer.** One
   backbone, one benchmark, one forget-set size, one generation setting.

**What it takes to leave (e):** a head-to-head against the closest prior art, same backbone, same
benchmark, same selection rule, same generation protocol, ≥3 seeds, with the effect exceeding the
spread — and then a robustness axis (paraphrase, generation-setting grid, relearning) showing it
is not setting-specific. EMDU does roughly this and is the standard to meet.

**(e) is not a verdict of worthlessness.** It is a statement about evidence. Most published
methods in this document sit in (e) on at least one axis, and saying so is the honest reading.

## 3.1 Assignment procedure

1. Write `A`'s loss and update explicitly, with all stop-gradients and all normalisations.
2. List every prior-art `B` whose loss has the same functional type.
3. For each: attempt the algebraic identity. If plausible, **run the numerical gradient check.**
   → (a) if it passes.
4. Else: attempt the parameter substitution / limit. → (b).
5. Else: decompose `A` into mechanisms, cite prior art for each. If all have prior art and the
   pairing is conventional, run the factorial ablation with seed variance.
   → (c) if additive, (d) if interacting with a pre-stated mechanism story.
6. Else: `A` differs. Assess the *evidence* for benefit. → (e) until the leaving-(e) bar is met.
7. **Record the category, the check that produced it, and the check you did not run.**
   An unrun check is reported, not assumed to pass.

## 3.2 Pre-applied to the two facts we already know about our own work

- Our reimplementation uses "an exclusion target plus soft-CE/KL". **Soft-CE against a frozen
  target and `KL(target ‖ model)` are category (a) — exactly equivalent in gradient.** So the
  soft-CE/KL part of our implementation is EMDU's anchor-first loss, by identity, *provided our
  target equals EMDU's anchor*. See Critical Question 3 — it probably does not, and the
  difference is in the target, not the divergence.
- A gradient-norm penalty for relearning robustness is **(a)-or-(b) relative to Fan et al.'s
  Eq. 10** — we established the match previously. Combined with our HISTORICAL `NOT SUPPORTED`
  verdict and the released-code finding in §5, this direction is closed.

---

# Part 4 — Are the reported numbers comparable? (Short answer: mostly no)

This matters because "beating SOTA" has to mean something. Here is what breaks comparability,
ordered by severity.

**1. There is no shared metric across the three paper families.**

| Family | Forget metric | Utility metric |
|---|---|---|
| MDU, EMDU | **ROUGE-L** on TOFU/RWKU + MDLM answer probability `p` | TOFU Retain/RA/WF; RWKU N-L1/L2, MMLU, TruthfulQA, TriviaQA |
| Fan et al. 2502.05374 | **WMDP-Bio accuracy** (`UE = 1 − acc`); MUSE **KnowMem / VerbMem** | zero-shot MMLU; MUSE KnowMem on `D_r` |
| TimeROME-DLM | **log-probability in nats** on TOFU forget01 | RetainLP in nats |

No two of these share a forget metric. A cross-family ranking is not constructible.

**2. ROUGE-L is ambiguous even *within* the MDU/EMDU family.** MDU's paper defines ROUGE-L as
**F1**; MDU's released code computes **`.recall`**; TOFU's official code computes recall; EMDU
reports recall. **It is therefore unknown which quantity MDU's published tables contain** (EMDU
Appendix E.1 makes exactly this point). Any claim of the form "we beat MDU's 0.488" is unsound
until that is resolved. Our position: report **ROUGE-L recall**, state it, and compare only to
EMDU's recall numbers and to our own re-run of MDU.

**3. MDU's and EMDU's own numbers disagree on the same method.** EMDU's Table 6: MDU's official
code under its suggested config gives TOFU LLaDA Forget rL **0.346**; the config EMDU reports as
"MDU" gives **0.113**; MDU's own paper reports **0.034–0.135** across its τ rows. And base-model
rows differ (TOFU LLaDA Forget rL base: MDU **0.884**, EMDU **0.822**) because the Base SFT
checkpoints are different — MDU released none. **Nobody has reproduced MDU's table, including
EMDU, who tried.** Treat MDU's published numbers as not-yet-replicated.

**4. Every number in this area is a best-of-sweep on the reported metric.** EMDU selects "the
configuration achieving the largest Forget/Retain gap" from a sweep over LR and epochs, with no
held-out validation split (Appendix D.3/D.4). Fan et al. tune LR, `β`, `γ`, `λ`, `ρ` with no
described validation split. MDU reports a τ sweep rather than a selected point. **There is no
held-out-selection result anywhere in this literature.** If we introduce one, we will look worse
than everyone, and we must say why.

**5. Decoder settings change the answer by more than methods differ.** EMDU's Table 7: MDU's
Forget rL spans 0.113–0.171 (LLaDA) and 0.210–0.270 (Dream) across six (length, steps)
combinations; Dream RA spans 0.081–0.413. A method comparison at an unstated decoder setting is
uninterpretable for an MDLM. **Nothing in the AR unlearning literature has this problem, which is
why ported baselines cannot simply inherit AR evaluation conventions.**

**6. Split and scale differences.** TOFU forget10 (400 examples) vs forget01; RWKU 10 of 200
subjects, 582 pairs each; WMDP-Bio; MUSE Books/News. MDU and EMDU use the same 10 RWKU subjects
(listed in EMDU Appendix E: Stephen King, Confucius, Bruce Lee, Warren Buffett, Christina
Aguilera, Cindy Crawford, Marie Osmond, Paris Hilton, Justin Bieber, Prince Harry) — **the same
10 we have locally.** That is the one place where a like-for-like comparison is available to us.

**7. Forget-corpus provenance differs on RWKU.** RWKU's protocol generates the forget corpus with
the target model. MDU's scripts reference per-backbone generated forget-pair files that **are not
shipped**. EMDU instead uses RWKU's **publicly released Llama-generated** corpus for both
backbones, precisely so its numbers do not depend on unreleased data. **Our local `pair.json`
(582 entries/entity) is the public Llama-generated corpus, so we are aligned with EMDU and not
with MDU.** Declare this.

**8. Seeds.** MDU: single seed, no variance reported. EMDU: single seed in the main tables, 3
seeds for its own chosen configs only. Fan et al.: 5 trials over relearn-set sampling, but a
single unlearning seed (`seed 1001`). **No paper in this area reports seed variance for its
baselines.** So any gap smaller than an unmeasured baseline spread is not a real gap.

## 4.1 What "beating SOTA" could legitimately mean for us

Given the above, exactly one claim is constructible on our hardware:

> On **RWKU**, **10 subjects**, **LLaDA-8B-Instruct** at a pinned revision, **ROUGE-L recall**,
> **RWKU-official MMLU placement**, a **fixed and stated** generation setting, **LoRA**
> parameterization applied identically to every method, a **single stated selection rule**
> applied identically to every method, and **≥3 seeds** with spreads reported — method `X`
> achieves a better Forget/Neighbor trade-off than our own re-runs of MDU, EMDU, and the
> MDU-defined MDLM ports of GA/NPO/SimNPO/WGA/DPO.

Every qualifier in that sentence is load-bearing. In particular it is a claim against **our
re-runs**, not against published table values, because of points 2, 3, 7 and 8. A claim against
published values is not available to us and would not be available to anyone.

---

# Part 5 — The four critical questions, answered

## Q1. Does ANY published MDLM unlearning method train on anything other than uniformly-sampled `t ~ U(0,1)`?

**No. Not one.** But the precise shape of the "no" is more useful than the "no", so here it is in
full.

### The mask-rate axis, method by method

| Method | Mask rate for the **trained** model | Mask rate for the **target/anchor** | Position weighting inside `M_t` |
|---|---|---|---|
| **MDU** | `t ~ U[0,1]`, uniform, resampled per step | **same `t`, same `y_t`** (only the prompt is replaced by `m`) | uniform `1/\|M_t\|` |
| **MDU retain** | **independent** `t' ~ U[0,1]` | n/a | ELBO weight `1/t'` |
| **EMDU** | `t ~ U(0,1]`, uniform | **pinned at `t = 1` on the answer span (FAM)** — anchor input is `(∅, [MASK]^n)`, constant in `t` | uniform `1/\|M_t\|` |
| **EMDU retain** | `t ~ U(0,1]` | n/a | ELBO weight `1/t` |
| **MDU's MDLM ports** (GA, GD, NPO, SimNPO, WGA, DPO) | `t ~ U[0,1]` inherited from `L_sft` | NPO/DPO reference: same `t`, same `y_t` | `1/t` for most; **WGA drops `1/t`** and reweights by `w_i = p_θ(y^i\|x,y_t)^γ` |
| **TimeROME-DLM** | no training | n/a | n/a — but **its edit *localisation* is step-aware**, aggregating the object's log-prob gain at later denoising steps `k' > k` with exponential decay |
| **A2D** (safety, *not* unlearning) | **`λ ← (1−ε)t + ε` with `t ~ U(0,1)`** — i.e. `λ ~ U(ε,1)`, a **floored** mask rate | n/a | `1/λ` |
| **D2F** (acceleration, *not* unlearning) | **block-level noise ratios sampled to be monotonically increasing** | asymmetric distillation teacher | n/a |

### What this means for us — three findings, in order of value

**(i) EMDU already moved the mask rate — but only on the target side, and that is exactly what
worked.** FAM pins the anchor's answer span at `t = 1` while leaving the trained model's `t`
uniform. This is the single clearest existing evidence that the mask-rate axis is productive for
MDLM unlearning. It is also *half* of EMDU's contribution and one of the two factors in its
ablation. **No published unlearning method has touched the mask-rate distribution on the
trained-model side.** The `t`-distribution of the states we take gradients at remains uniform in
every published unlearning method.

**(ii) The non-uniform-`t` precedents in the dLLM literature are all outside unlearning**, and
both go in directions worth noting: A2D **floors** the mask rate (never trains on nearly-clean
states) and D2F makes the noise ratio **monotonically increasing across blocks**. Neither is an
unlearning method, so neither is prior art against an unlearning claim — but both establish that
reshaping `t` is an accepted, publishable move in this model family, which weakens any "nobody
would think to do that" framing and strengthens any "this is the natural thing to try" framing.

**(iii) Nobody targets specific positions, and MDU published the diagnostic that says they
should.** MDU's own §5/§6.3 analysis divides response tokens into **in-context**, **structural**
and **stored-knowledge** roles and measures the conditional-anchor KL by category. Their Figure 3
shows KL spiking exactly on in-context tokens early and **stored-knowledge tokens late**, and
their Table 5 shows the post-training KL change concentrated on stored-knowledge tokens
(**−20.6%**, vs −3.8% in-context and +1.4% structural). **They then do not use any of this in the
objective** — the forget loss remains a flat `1/|M_t|` average over *all* masked positions. This
is an explicitly identified, explicitly measured, explicitly unexploited signal in the
state-of-the-art method's own paper. The only position-dependent weighting in the whole area is
WGA's confidence weighting `w_i = p_θ^γ`, which is data-dependent rather than positional, and
which was ported from AR where it means something different.

**Bottom line for Q1:** the mask-rate-and-position axis is **open on the trained-model side**,
partially occupied on the target side by EMDU's FAM, and has a published-but-unused diagnostic
(MDU's token roles) pointing straight at it. If our candidate targets specific mask rates or
specific positions on the *gradient* side, the closest prior art is EMDU-FAM (target side) and
WGA (confidence weighting), and under the Part 3 rubric that is most likely category (d) if we
can show interaction, or (e) until we measure it.

## Q2. Does any method exploit the ITERATIVE DECODING process (the actual generation trajectory) rather than only the per-position denoising loss?

**No published unlearning *training objective* does. The gap is real. But it is narrower than it
looks, in two specific ways you need to know about.**

### The clean part of the answer

- **MDU: no**, and the authors say so in print. Their §7 Limitations, verbatim: *"As MDU is based
  on the training scheme, we randomly sampled masked denoising states, while generation proceeds
  through a full iterative denoising trajectory with specific sampling and remasking strategies.
  Since steering methods are frequent in continuous diffusion modeling, our next goal is to
  bridge this gap between training-time unlearning and inference-time generation."*
  **The SOTA method's own stated next step is the thing we suspect is open.** This both confirms
  the gap and tells us a competent group is working on it.
- **EMDU: no.** Teacher-forced throughout. EMDU *evaluates* across decoding settings (six
  (length, steps) combinations, their Table 7) and makes robustness-to-decoder a selling point,
  but nothing in `L_EMDU` touches a trajectory. Its anchor is, if anything, *further* from the
  trajectory than MDU's: FAM pins the anchor at the fully-masked state, which is the trajectory's
  *first* step only.
- **All MDLM ports (GA/GD/NPO/SimNPO/WGA/DPO): no.** The port is a substitution at the level of
  the per-sample NLL (`−log p^AR_θ(y|x) → L_sft(θ;(x,y))`). It cannot be trajectory-aware; there
  is nothing in the substitution that could be.
- **Fan et al. 2502.05374: no, and not applicable.** Autoregressive only — "diffusion" appears
  zero times in the full v4 text.
- **A2D: no.** Explicitly: *"No explicit 'any-step' mechanism exists beyond uniform timestep
  sampling during training. The any-step robustness emerges from training across variable
  corruption rates rather than explicit trajectory modeling."*

### The two caveats that actually matter

**Caveat 1 — TimeROME-DLM occupies trajectory-aware *localisation*, though not a trajectory-aware
*loss*.** Its Temporal Indirect Effect protocol runs clean/corrupted/patched **diffusion passes**
and scores each coordinate `(ℓ,k,m)` by the object's log-probability gain **at later denoising
steps `k' > k`, exponentially decayed over subsequent steps**. That is genuinely reading the
denoising trajectory to decide what to change. It is not a training objective — there are no
gradients anywhere in the method — and its deployment contract differs (a forward hook fires at
every denoising step). **Precise consequence for our novelty framing:** "use the denoising-step
axis to decide *where* to intervene" is **taken**. "Use the denoising trajectory inside a
*training objective*" is **not taken**. We must draw our claim on the right side of that line and
cite arXiv:2606.12841 when we do. Note also that this paper has **no released code and no
independent reproduction**, so it is prior art on the record but not prior art we can verify.

**Caveat 2 — and this is the one to resolve before any novelty claim: Zhang et al. 2026.** EMDU
cites, as ICML 2026, *"Adversarial reinforcement learning for robust diffusion large language
model unlearning"* (Zhiwei Zhang, Yudi Lin, Linlin Wu, Fali Wang, Yi Xin, Xiaomin Li, Minhua Lin,
Xianfeng Tang, Qi He, Suhang Wang). EMDU's description: it *"finds that unlearning performance in
MDLMs is sensitive to generation length and denoising steps"* and *"proposes adversarial RL, in
which a separate LLM generates contexts that elicit forgotten knowledge while the MDLM learns,
via GRPO, to suppress it."*

GRPO is an **on-policy** method: it optimises against the model's own generations. If that
description is accurate, this paper trains an MDLM unlearning objective **on model-sampled output
rather than teacher-forced masked states** — which is the closest thing to trajectory-aware MDLM
unlearning in existence, and it would land squarely on top of part of our suspected gap.

**I could not find this paper.** Not on arXiv (three separate API queries), not via web search,
on 2026-10-20. Status: **NOT VERIFIED — exists only as a citation inside EMDU.**

**Action required, flagged as the single highest-priority open item in this document:** locate
this paper (ICML 2026 proceedings / OpenReview / the authors' pages) and read its method before we
commit to a trajectory-aware novelty claim. Until then, our honest position is: *"no publicly
available MDLM unlearning method uses the generation trajectory in its training objective; one
cited-but-unlocated work may."* Do not shorten that to "nobody does this".

### What is genuinely, defensibly open

Putting the caveats together, the unoccupied region is:

> A **training objective** for MDLM unlearning whose gradient is taken at states drawn from, or
> weighted by, the **actual iterative decoding process** — its step schedule, its remasking rule,
> its confidence-based unmasking order — rather than at i.i.d. teacher-forced corruptions of the
> ground-truth answer at `t ~ U(0,1)`.

Three facts support that this is worth doing rather than merely undone:
1. MDU names it as the gap and as their next goal (§7).
2. EMDU's Table 7 shows forget scores **move substantially with the decoder** (MDU 0.113→0.171 on
   LLaDA; 0.210→0.270 on Dream) — i.e. the train/deploy mismatch has a measurable footprint.
3. The unverified Zhang et al. result, as EMDU relays it, is that unlearning is *"highly
   sensitive to generation hyperparameters, particularly output length and the number of denoising
   steps"* — independent corroboration of the same mismatch, from a second group.

And one fact that should temper us: our own HISTORICAL PoC in this direction, **TD-SSD, did not
separate from the SSD control** (`research/llada-poc-v1` @ `c3578a908d6c`; best point +0.0625 =
1 probe of 16, worse at 150 steps). The gap being open is not evidence that filling it works.
Under Part 3 that result is category (e) and it stays there.

## Q3. For MDU and EMDU: what exactly is the "null anchor" / "anchor-first KL"? And how faithful is our reimplementation?

### Written out explicitly

Let `x` be the prompt, `y` the ground-truth forget answer of length `n`, `θ0` the frozen
pre-unlearning parameters, `m = ∅ = [MASK]^{|x|}` the null prompt, `τ ∈ [0,1]`.

Sample `t`, then `y_t` by masking each answer token independently w.p. `t`; let
`M_t = {i : y_t^i = [MASK]}`.

**MDU's null anchor** — the frozen model's *prompt-masked* prediction on **the same partially
masked answer** the trained model sees:

```
p^MDU_anchor,i(·)  =  (1/Z_i) · [ p_θ0(· | m, y_t) ]^τ  at position i,     Z_i = Σ_v p_θ0(v | m, y_t)^τ
```

**MDU's forget loss (anchor-last):**

```
L^MDU_forget = E_{(x,y)~D_f, t, y_t} [ (1/|M_t|) Σ_{i∈M_t} KL( p_θ(· | x, y_t) ‖ p^MDU_anchor,i(·) ) ]
```

**EMDU's anchor** — the frozen model's prompt-masked prediction on the **fully** masked answer
span (FAM):

```
p^EMDU_anchor,i(·)  =  softmax( τ · log p_θ0(· | ∅, y_t^FAM) )_i  ,   y_t^FAM = [MASK]^n
```

equivalently `softmax(τ · z_u)` on the frozen model's logits `z_u` (their Algorithm 1 line 10).
Note this is **independent of `t`** and of which tokens survived in `y_t`.

**EMDU's forget loss (anchor-first):**

```
L^EMDU_forget = E_{(x,y)~D_f, t} [ (1/|M_t|) Σ_{i∈M_t} KL( p^EMDU_anchor,i(·) ‖ p_θ(· | x, y_t) ) ]
```

**Totals:**
```
L^MDU  = L^MDU_forget  + λ · E_{D_r,t'} [ -(1/t') Σ_{i∈M_{t'}} log p_θ(y_r^i | x_r, y_{r,t'}) ]   (λ=1)
L^EMDU = L^EMDU_forget + α · E_{D_r,t}  [ -(1/t)  Σ_{i∈M_t}    log p_θ(y^i | x, y_t) ]
```

**So there are exactly three differences between MDU and EMDU, and that is all:**
1. **KL argument order.** MDU: `KL(model ‖ anchor)` (anchor-last). EMDU: `KL(anchor ‖ model)`
   (anchor-first).
2. **Anchor input.** MDU: `(∅, y_t)` — surviving ground-truth answer tokens remain visible to the
   anchor. EMDU: `(∅, [MASK]^n)` — nothing of the answer is visible (FAM).
3. **Temperature application site.** MDU tempers the **probabilities** and renormalises
   (`p^τ/Z`); EMDU tempers the **logits** (`softmax(τ z_u)`). At `τ=1` these coincide; at
   `τ∈(0,1)` they are the same family (`softmax(τ log p)` = `p^τ/Z` when `p = softmax(z)`, since
   `τ log p = τ z − τ logΣe^z` and the constant drops in softmax) — **so (3) is actually the same
   operation written two ways, not a difference.** Confirmed by algebra; both papers' `τ=0` limit
   is uniform and `τ=1` limit is the untempered anchor, as both state.

Two more things worth having on record:

- **The "null" is masking, not a learned null embedding.** `m = ∅` is the prompt tokens replaced
  by `[MASK]`. Both papers justify this as the discrete analogue of classifier-free guidance's
  conditional/unconditional split (EMDU §2.2 says exactly that), relying on the fact that an MDLM
  natively supports an unconditional forward pass.
- **Anchor-first KL is, in gradient, soft-label cross-entropy.** EMDU's Proposition 1:
  `∇_z KL(p ‖ softmax(z)) = q − p`. The two objectives differ by `−H(p)`, a constant in `θ`.
  So *"anchor-first KL"* and *"soft-CE toward the anchor"* are **the same algorithm** (Part 3
  category (a)).

### How faithful is our reimplementation?

Our reimplementation uses **an exclusion target plus soft-CE/KL**. Verdict, in two parts:

**The divergence half is faithful — exactly so.** Soft-CE against a frozen target has gradient
`q − p`, identical to `KL(p ‖ q)`. So our divergence *is* EMDU's anchor-first KL, by identity,
not by approximation. **And it is therefore NOT MDU's.** If anyone on the project believes our
implementation follows MDU because MDU's abstract says "forward KL", that belief is wrong:
EMDU §2.3 documents that MDU's "forward KL" means `KL(model ‖ anchor)`, the *opposite* of the
distillation convention. **Soft-CE/KL ⇒ we have EMDU's direction.**

**The target half is not faithful to either paper.** Neither MDU's nor EMDU's target is an
exclusion target. Both targets are **model-derived**: a frozen forward pass of `p_θ0` with a
masked prompt. An exclusion target (mass removed from the ground-truth token, renormalised over
the rest) is **hand-constructed** and never consults `θ0`. Consequences:

- Our target is closer to **MDU at `τ=0`** than to either paper's headline setting. MDU states
  `τ=0` makes the anchor **uniform over `V`**, "exactly the same as the Maximum Entropy (ME)
  loss" (Yuan et al., arXiv:2410.08109). An exclusion target is **uniform-over-the-complement**,
  i.e. MaxEnt with the forget token removed. Under Part 3 that is category **(b)** relative to
  MDU's `τ=0` limit — a near-neighbour of a published special case — **not** a reproduction of
  MDU or EMDU at their reported settings (`τ=1` for both in EMDU's tables).
- **Both papers' headline results use `τ=1`**, i.e. a fully model-derived anchor. Our target is
  structurally at the opposite end of their τ axis. Any statement that we are "running EMDU"
  is incorrect. We are running *EMDU's divergence with a MaxEnt-exclusion target*, which is a
  third method.
- **FAM is inapplicable to us, and so is failure mode 1.** EMDU's FAM changes the *anchor's
  input*; with a hand-constructed target there is no anchor forward pass, so there is nothing to
  mask. The flip side is genuinely favourable: **failure mode 1 (anchor-side leakage) cannot
  occur in our formulation at all** — a hand-built target cannot be cued by surviving
  ground-truth tokens. We get EMDU's leakage fix for free, structurally. This is worth stating as
  a property, and it is testable: measure target-side mass on forget tokens for MDU's anchor,
  EMDU's FAM anchor, and our exclusion target, at matched `t`.
- **Failure mode 2 does not apply to us either**, because it is an anchor-last problem and we are
  anchor-first. Our gradient is `q − p`, which does not vanish at a memorised vertex.
- **We lose MDU/EMDU's claimed preservation mechanism.** Both papers argue the anchor preserves
  fluency and response structure *because* it is the model's own unconditional prediction — MDU's
  Table 1 shows the anchor "preserves the response structure while omitting target-specific
  facts". A uniform-over-complement target carries no such structure, and MDU's own τ sweep is
  the evidence for what that costs: at `τ=0` their TOFU forget pseudo-perplexity goes to
  **25,452** and outputs become corrupted or empty (their Tables 6, 7). **We should expect
  gibberish-type forgetting unless something else supplies the structure** — and we should check
  for it explicitly, because MDU reports it as the characteristic `τ→0` failure.

**Action items this implies for us:** (1) state in all internal documents that our method is
"anchor-first/soft-CE with a MaxEnt-exclusion target", not MDU and not EMDU; (2) if we want a
faithful EMDU baseline we must implement the frozen-anchor FAM forward — which under LoRA is
cheap (adapter-disabled pass on `(∅, [MASK]^n)`, cacheable per example); (3) run the
`τ ∈ {0, 0.25, 0.5, 0.75, 1}` sweep for both anchors, since our target sits at the `τ=0` end and
both papers' results sit at `τ=1`; (4) check output degeneracy, not just ROUGE-L — MDU's §6.3
makes the point that refusal, corruption, and fluent-but-wrong are different behaviours that
aggregate forget metrics score identically.

## Q4. Which methods have published numbers on which benchmarks, and with what metric?

| Method | Benchmark | Backbone | Forget metric | Utility metric | Comparable to MDU/EMDU? |
|---|---|---|---|---|---|
| **MDU** | TOFU forget10; RWKU subjects 1–10 | LLaDA-8B-Instruct, Dream-7B-Instruct | **ROUGE-L** (paper says F1; code computes **recall** — ambiguous) + MDLM answer probability `p` (MC reconstruction, `N=128`) | TOFU Retain/RA/WF; RWKU N-L1/L2, MMLU, TruthfulQA, TriviaQA | — (the reference point, but its own metric is ambiguous and unreplicated) |
| **EMDU** | TOFU forget10 + retain90; RWKU subjects 1–10 | LLaDA-8B-Instruct, Dream-7B-Instruct | **ROUGE-L recall** (explicit) + `p` (Eq. 13, `N=128`) | same as MDU, RWKU-official MMLU placement | **Yes** — same splits, same subjects, explicit metric. **This is our comparator.** |
| **GA/GD/NPO/SimNPO/WGA/DPO** (MDLM ports) | as above, in both papers | as above | as above | as above | Yes, within each paper. Across the two papers, **no** (different Base SFT, different MDU configs) |
| **TimeROME-DLM** | TOFU forget01; MUSE-Books/News; WHP; WMDP-Bio | LLaDA variants, Dream-7B, MMaDA-8B, DiffuLLaMA-7B | **ForgetLP in nats** (−118.94 vs −35.7 base) | **RetainLP in nats** (−89.08) | **No.** Different metric (nats), different split (forget01). Also unreplicated, no code. |
| **Fan et al. 2502.05374** | **WMDP-Bio**; MUSE Books + News. **Not TOFU, not RWKU** | Zephyr-7B-beta, LLaMA-3 8B, LLaMA-2 7B, ICLM 7B — **all autoregressive** | `UE = 1 − WMDP-Bio accuracy`; MUSE **KnowMem**, **VerbMem** | zero-shot **MMLU accuracy**; MUSE KnowMem on `D_r` | **No.** No shared metric, no shared benchmark, no shared model class. Its *protocol* transfers; its *numbers* do not. |
| **A2D** | safety benchmarks / DIJA attack success | LLaDA-class dLLMs | attack success rate | — | **No.** Not an unlearning benchmark. |

### Headline numbers it is legitimate to quote

**RWKU, LLaDA-8B-Instruct, 10 subjects, ROUGE-L recall** — the only like-for-like pairing, and
the only one we can reproduce:

| | F-L1 ↓ | F-L2 ↓ | F-L3 ↓ | N-L1 ↑ | N-L2 ↑ | MMLU ↑ |
|---|---|---|---|---|---|---|
| Base (MDU's table) | 0.488 | 0.433 | 0.469 | 0.551 | 0.372 | 0.395 |
| Base (EMDU's table) | 0.484 | 0.393 | 0.475 | 0.535 | 0.344 | **0.680** |
| MDU `τ=0` (MDU's table) | **0.015** | **0.006** | **0.015** | 0.330 | 0.225 | 0.415 |
| MDU `τ=1` (MDU's table) | 0.108 | 0.087 | 0.072 | 0.431 | 0.334 | 0.416 |
| MDU `τ=1` (EMDU's re-run) | 0.320 | 0.151 | 0.179 | 0.441 | 0.235 | 0.730 |
| **EMDU `τ=1`** (EMDU's table) | 0.120 | **0.046** | **0.141** | **0.486** | 0.242 | 0.719 |
| EMDU, 3 seeds | 0.119±0.013 | 0.047±0.004 | 0.144±0.007 | 0.475±0.016 | 0.239±0.003 | 0.711±0.007 |

**Read this table carefully, because it contains the whole comparability problem in one place.**
The two papers disagree on the *base model's* MMLU by 0.285 (0.395 vs 0.680) — that is the prompt
placement issue, not a model difference. They disagree on MDU `τ=1` F-L1 by a factor of three
(0.108 vs 0.320). MDU's own `τ=0` row is far lower on forget than EMDU's headline — but EMDU's
Figure 1 argues MDU's low-`τ` forgetting is entangled with incoherent generation that **still
surfaces forget tokens**, which a ROUGE-L recall number does not capture. **So "MDU `τ=0` has the
best F-L1" is true as a number and contested as a claim.**

### What a forget-metric target should be for us

- **Primary: ROUGE-L recall** on RWKU F-L1/L2/L3, because it is the only metric EMDU states
  unambiguously and the only one with a published 3-seed spread (`±0.013` on F-L1).
- **Do not use exact match as the comparable metric.** Our HISTORICAL 22/100 EM figure is a valid
  measurement that is simply *not* on the same scale as these tables (see §2.3 D3).
- **Report a degeneracy check alongside it.** Both the MDU `τ=0` case and EMDU's empty-completion
  analysis show that ROUGE-L recall alone cannot distinguish refusal, EOS-abstention, corrupted
  output, and fluent-but-wrong. EMDU's own method — inspect the **full generation canvas past the
  first EOS** — is cheap and is the right check to copy.
- **A number is not a result without the decoder setting.** State length and step count, and for
  any robustness claim report EMDU's six-combination grid.

---

# Part 6 — The autoregressive methods, in full

Fields follow the same schema. Equations are transcribed from the papers' own math, not
paraphrased. "not stated" means the paper does not say.

## 6.1 NPO — Negative Preference Optimization

**Citation:** Zhang, Ruiqi; Lin, Licong; Bai, Yu; Mei, Song. "Negative Preference Optimization:
From Catastrophic Collapse to Effective Unlearning." arXiv:2404.05868 (v1 Apr 2024, v2 10 Oct
2024). <https://arxiv.org/abs/2404.05868>. **Venue: not stated on the abs page — NOT VERIFIED.**
(EMDU's bibliography lists it as CoLM 2024; MDU's lists it as an arXiv preprint. The papers
disagree.)

**Training target.** GA baseline (their Eq. 1): `L_GA(θ) = E_{D_FG}[ log π_θ(y|x) ]`.
NPO loss (their Eq. 3), derived as DPO with the `y_w` term deleted:

```
L_{NPO,β}(θ) = -(2/β) E_{D_FG}[ log σ( -β log( π_θ(y|x) / π_ref(y|x) ) ) ]
             =  (2/β) E_{D_FG}[ log( 1 + ( π_θ(y|x)/π_ref(y|x) )^β ) ]
```

`σ(t)=1/(1+e^{-t})`, `β>0` inverse temperature, `π_ref` a reference model. Deployed as
**NPO+RT**: `L_NPO + L_RT` with `L_RT(θ) = -E_{D_RT}[log π_θ(y|x)]`. NPO+KL variant uses
`K_RT(θ) = E_{D_RT}[ D(π_θ(·|x) ‖ π_ref(·|x)) ]`.

Gradients (their Eqs. 4–5) — the reason NPO works:

```
∇_θ L_GA      = E_{D_FG}[ ∇_θ log π_θ(y|x) ]
∇_θ L_{NPO,β} = E_{D_FG}[ W_θ(x,y) ∇_θ log π_θ(y|x) ],
   W_θ(x,y) = 2 π_θ^β(y|x) / ( π_θ^β(y|x) + π_ref^β(y|x) )
```

Proposition 1: `lim_{β→0} ∇_θ L_{NPO,β} = ∇_θ L_GA` — **NPO is a strict generalisation of GA, and
GA is its `β→0` limit.** Under Part 3 that pair is category (b), and the paper says so itself.
Theorem 2: GA diverges linearly in `t`, NPO logarithmically.

**State distribution / sampling.** Fully teacher-forced on ground-truth `(x,y) ∈ D_FG`. **No
length normalisation** — verified in code (`TOFU/data_module.py:242-250`): `get_batch_loss` uses
`CrossEntropyLoss(ignore_index=-100, reduction='none')` then `.sum(dim=-1)`, so `log π_θ(y|x)` is
a **sum** over answer tokens with the prompt masked. The retain term uses HF's `outputs.loss`, a
token **mean**. **The two terms are on different scales and the paper does not remark on it** —
worth knowing if we port NPO ourselves.

**Gradient path.** Plain descent, AdamW. Only stop-gradient is `π_ref` inside `torch.no_grad()`
(`dataloader.py:279-282`). `W_θ` arises analytically from the sigmoid, not from a hand-written
reweighting — so NPO is plain loss descent, not a custom update.

**Preservation.** Retain CE (NPO+RT) or KL-to-`π_ref` on the retain set (NPO+KL). The paper argues
*against* maximising forget-set KL: *"a larger Forget KL is not necessarily advantageous... Forget
KL may not be a suitable objective function to maximize."* Retain-weight sweep: forget quality
improves 0→2, degrades 2→5.

**Reference model.** Required, frozen, initialised from the TOFU-finetuned model before unlearning
(`ref_policy: fine_tuned`; the code raises `NotImplementedError` otherwise). Re-run every step,
not cached.

**Inference requirements.** Nothing beyond edited weights.

**Parameterization.** Full-parameter — `TOFU/config/forget.yaml` sets `LoRA: {r: 0}` and `forget.py`
gates PEFT on `cfg.LoRA.r != 0`. Llama-2-7b-chat, AdamW wd 0.01, **lr 1e-5**, effective batch 32
(4 × accum 4 × 2 GPUs), **10 epochs**, linear warm-up then decay. **`beta: 0.1` appears only in
the repo config, not in the paper.** Retain coefficient raised to 5.0 (Forget50) / 12.0 (Forget90).

**Selection.** Weak: Figure 5's caption says points are plotted "at the epoch where each method
attains its **peak forget quality**" — selection on the headline metric, no held-out split. β and
LR by grid search. 5 seeds.

**Compute.** 2 forwards on the forget batch (policy + frozen ref) + 1 forward/backward on retain,
1 backward. Hardware "two A100 GPUs." **No wall-clock in the paper.** MUSE measures it (8×A40,
batch 32, 2048 tokens): GA 4.14 s/step, **NPO 5.68**, NPO_GDR 7.59, NPO_KLR 9.11 — the reference
forward costs ≈37% over GA.

**Code.** <https://github.com/licong-lin/negative-preference-optimization>. **No LICENSE file at
all** — the `/license` endpoint 404s, API `license` is `null`, nothing in the tree of `main`.
**This is a real constraint: the code carries no license grant.** No weights, 0 releases.

**Benchmarks / metrics.** **TOFU only** (Forget01/05/10 + author-constructed 20/30/50/90),
Llama-2-7b-chat.
- *Forget quality* = **p-value of a Kolmogorov–Smirnov test** on the Truth Ratio distribution,
  unlearned vs retrained, on the forget set; >0.05 counted as meaningful.
- *Model utility* = **harmonic mean of 9 metrics** — normalised conditional probability
  `P(a|q)^{1/|a|}`, **ROUGE-L recall**, Truth Ratio, each on Retain / Real Authors / Real World.
  Truth Ratio: `R_truth = [ (1/5) Σ_{i=1}^{5} P(â_i|q)^{1/|â_i|} ] / P(ã|q)^{1/|ã|}`.
- **NPO publishes no results table.** Headline TOFU numbers are **figure-only**. Prose claims:
  NPO+RT is the sole method above 0.05 forget quality on Forget05/10/20/30; Forget50 vanilla
  NPO+RT reaches ~1e-3; with tuned retain weights it "easily attains a forget quality exceeding
  0.05 and model utility above 0.55 on Forget50 and Forget90."
- Third-party numbers (**note they disagree, because each paper retuned NPO**): SimNPO Table 1
  (TOFU Forget05) NPO at FQ 0.79 / MU 0.57; SimNPO Table A3 (Forget10) FQ 0.29 / MU 0.55. MUSE
  Table 3: NPO at VerbMem 0.0 / KnowMem_f 0.0 / PrivLeak 24.4 / Utility 0.0 (News) — total
  collapse. RWKU Table 1 (LLaMA3-Instruct 8B, 100 targets, ROUGE-L recall): NPO (Full) Forget
  FB/QA/AA/All = 33.4/21.0/24.8/26.2 from 85.9/76.4/77.7/79.6; Neighbor 76.0/69.9/72.6.
  **Caveat: RWKU writes the NPO loss without the `2/β` prefactor.**

**Diffusion LM: YES** — ported and scored on LLaDA-8B and Dream-7B by MDU, EMDU, and
TimeROME-DLM (see Part 1's port table for the exact MDLM form).

## 6.2 SimNPO

**Citation:** Fan, Chongyu; Liu, Jiancheng; Lin, Licong; Jia, Jinghan; Zhang, Ruiqi; Mei, Song;
Liu, Sijia. "Simplicity Prevails: Rethinking Negative Preference Optimization for LLM
Unlearning." arXiv:2410.07163 (v4, 17 Oct 2025). **NeurIPS 2025** (confirmed from the abs-page
Comments field). <https://arxiv.org/abs/2410.07163>. **It exists.**

**Training target.** Regularized problem
`min_θ E_{D_f}[ ℓ_f(y|x;θ) ] + λ E_{D_r}[ ℓ_r(y|x;θ) ]`, `ℓ_r = -log π_θ(y|x)`. Forget loss
(their Eq. 4):

```
ℓ_SimNPO(θ) = E_{(x,y)∈D_f}[ -(2/β) log σ( -(β/|y|) log π_θ(y|x) - γ ) ]
```

`β>0` temperature; `|y|` response length; **`γ≥0` the reward-margin parameter from SimPO**, set to
`γ=0` "unless otherwise specified... to align with the NPO loss"; `λ≥0` retain weight. The move:
swap DPO's reward `β log(π_θ/π_ref)` for SimPO's **reference-free, length-normalised** reward
`(β/|y|) log π_θ(y|x)`.

Gradient (their Eq. 5, γ=0):

```
∇_θ ℓ_SimNPO = E_{D_f}[ w'_θ(x,y) ∇_θ log π_θ(y|x) ],
  w'_θ(x,y) = [ 2 π_θ(y|x)^{β/|y|} / ( 1 + π_θ(y|x)^{β/|y|} ) ] · (1/|y|)
```

As `β→0`, `∇_θ ℓ_SimNPO → E[ (1/|y|) ∇_θ log π_θ(y|x) ]` — a **length-weighted** GA, not plain GA.
So SimNPO's `β→0` limit differs from NPO's. Appendix G: NPO needs `Õ(log(1/ε)/(|y|η))` steps for
a length-`|y|` sample (unlearns long responses faster); SimNPO needs `Õ(log(1/ε)/η)`,
length-agnostic.

**⚠️ Paper/code discrepancy.** Eq. 4 has the margin as `−γ`, unscaled. The reference
implementation (`TOFU/dataloader.py:283-284`) computes
`forget_loss = get_batch_loss(...)/loss_mask.sum(-1) - self.gamma` then
`-F.logsigmoid(self.beta * forget_loss).mean() * 2/self.beta`, so the margin enters as **`−βγ`**.
Immaterial at the default γ=0; **any γ>0 ablation in the code is not on the paper's scale.**

**State distribution.** Teacher-forced on `D_f`. **Explicit length normalisation by
`|y| = loss_mask.sum(-1)`** (count of answer tokens) — the whole point of the method.

**Gradient path.** Plain descent on `ℓ_SimNPO + λ ℓ_r`. **No stop-gradients anywhere** — there is
no second model to freeze. That is the structural simplification.

**Preservation.** Additive retain CE (`simnpo_grad_diff`), weight `grad_diff_coeff`. No KL anchor
in headline results, no projection, no parameter subset.

**Reference model: not needed. This is the contribution.** Motivating failure modes: (L1) uneven
allocation of unlearning power across forget data of differing memorisation and response length;
(L2) `W_θ ≈ 1` at epoch 1 because `θ` is initialised at `θ_ref`, so early NPO behaves like GA. A
warm-up experiment shows NPO's forget quality dropping **0.79 → 0.27** on TOFU Forget05 when
`θ_ref` is replaced by a Gaussian-perturbed average, while utility barely moves (0.57 → 0.52).

**Inference requirements.** Nothing beyond edited weights.

**Parameterization.** Full-parameter (`LoRA: {r: 0}`; `"use_lora": false` in WMDP configs).
TOFU: Llama-2-7B-chat, lr 1e-5, 10 epochs, effective batch 32 (1 × accum 4 × 8 GPUs). Grids:
Forget05 SimNPO `β∈[1.5,3.5]`, `γ∈[0,2]`, `λ∈[0.05,0.25]`; NPO baseline `β∈[0.05,0.2]`,
`λ∈[0.5,1.5]`, up to 20 epochs. Forget10 SimNPO `β∈[2.5,5.5]`. MUSE: `β∈[0.5,1.0]`,
`λ∈[0.05,0.25]`, `γ∈[0,2]`, 10 epochs, lr 1e-5. WMDP: Zephyr-7B-beta, 125 steps,
lr ∈ [2.5e-6, 5e-6], `β∈[0.05,7.5]`, λ=5.0. Shipped defaults: forget05 `beta 2.5, gamma 0.0,
npo_coeff 0.1375`; forget10 `beta 4.5, npo_coeff 0.125`; WMDP `{gamma 1.0, beta 0.1}`, lr 9.5e-6;
MUSE README `--beta 0.7` (News) / `0.75` (Books).

**Selection.** *"We measured the performance after each unlearning epoch and selected the optimal
one as the final model."* Same weakness as NPO: selection on the evaluation metric, no held-out
split. 5 trials on TOFU.

**Compute.** **1 forward/backward on forget (no reference forward) + 1 on retain** — strictly
cheaper than NPO. Hardware "8 NVIDIA A6000 in a single node." **No wall-clock.**

**Code.** <https://github.com/OPTML-Group/Unlearn-Simple>, **MIT License** (© 2024 JC Liu).
**Weights published:** six unlearned models at
<https://huggingface.co/collections/OPTML-Group/simnpo-unlearned-models-6721751fb02ab0e490ab0017>
(TOFU forget05/10 Llama-2-7b-chat, MUSE-Books iclm-7b, MUSE-News Llama-2-7b, WMDP zephyr-7b-beta,
WMDP llama3-8b), each `license: mit`. **This is the only one of the four AR methods here that is
cleanly reusable.**

**Benchmarks / metrics / numbers.** TOFU Forget05/10 (Llama-2-7B-chat) with FQ, MU, and
1−ROUGE-L / 1−Prob / Truth ratio on forget:

| TOFU Forget05 | 1−RL↑ | 1−Prob↑ | TR↑ | **FQ↑** | **MU↑** |
|---|---|---|---|---|---|
| Original | 0.04 | 0.01 | 0.49 | 0.00 | 0.62 |
| Retrain | 0.61 | 0.85 | 0.66 | 1.00 | 0.62 |
| GA | 1.00 | 1.00 | 0.66 | 1.9e-9 | 0.00 |
| GradDiff | 1.00 | 1.00 | 0.60 | 3.6e-9 | 0.56 |
| IDK | 0.98 | 0.40 | 0.55 | 1.9e-9 | 0.57 |
| RKLD | 0.69 | 0.96 | 0.66 | 0.79 | 0.56 |
| NPO | 0.73 | 0.94 | 0.67 | 0.79 | 0.57 |
| **SimNPO** | 0.74 | 0.97 | 0.69 | **0.99** | **0.58** |

Forget10: NPO FQ 0.29 / MU 0.55; **SimNPO FQ 0.45 / MU 0.62** (Retrain 1.00/0.62).
MUSE News (Llama-2-7B): NPO 2.53 / 56.93 / 108.91 / 37.58; **SimNPO 2.34 / 44.84 / 72.93 /
39.65** (Retrain 20.75/33.32/0.00/53.79). Books (ICLM-7B): NPO 0.00/0.00/−31.17/23.71;
**SimNPO 0.00/0.00/−19.82/48.27** (Retrain 14.30/28.90/0.00/74.50). **Do not mix these NPO rows
with MUSE's own** — SimNPO retuned NPO and gets very different values.
WMDP (Zephyr-7B-beta), 1−Acc_Bio↑ / MMLU↑: Original 0.35/0.59, RMU 0.68/0.57, NPO 0.74/0.44,
**SimNPO 0.74/0.48**. The paper is candid that RMU preserves utility better *"because RMU performs
unlearning only on layers 5, 6, and 7, whereas NPO and SimNPO apply unlearning on the entire
model."* Also: relearning robustness (20% of Forget05, 3 epochs; shortest- vs random-relearn),
where SimNPO degrades more slowly.

**Honesty note:** v4's conclusion claims SimNPO "demonstrates additional advantages in unlearning
robustness and **the application to reasoning model unlearning**", but no section or appendix in
v4 (A–J) presents a reasoning-model experiment. **NOT VERIFIED.**

**Diffusion LM: YES** — ported by MDU, EMDU and TimeROME-DLM.

## 6.3 WHP — "Who's Harry Potter?" — and WHP+

**Citation:** Eldan, Ronen; Russinovich, Mark. "Who's Harry Potter? Approximate Unlearning in
LLMs." arXiv:2310.02238 (v2, 4 Oct 2023). <https://arxiv.org/abs/2310.02238>. Microsoft Research /
Microsoft Azure.

**Training target — and this is the key structural fact: WHP is not a divergence objective at
all.** It is **label synthesis followed by ordinary cross-entropy fine-tuning**. Two ingredients
build "generic predictions", defined as the answer to *"What would a model that has not been
trained on the Harry Potter books have predicted as a next token in this sentence?"*

*(a) Reinforcement bootstrapping.* Fine-tune the baseline further on the unlearn target to get a
**reinforced** model, then combine logits. Naive form `v_generic := v_baseline − α(v_reinforced −
v_baseline)`; the form actually used (their Eq. 1):

```
v_generic := v_baseline - α · ReLU( v_reinforced - v_baseline )
```

*"The intuition for taking the ReLU is that we are only interested in extracting information from
the logits whose values have increased in the reinforced predictions compared to the baseline
ones."*

*(b) Anchored terms.* GPT-4 extracts ~1,500 idiosyncratic anchor terms with generic translations
(`'Hogwarts'→'Mystic Academy'`, `'Quidditch'→'Skyball'`, `'Harry'→'Jon'`). Each block is
translated, run through the baseline, and those next-token predictions become generic predictions.
Their Algorithm 1 composes both on the translated predictions:

```
predictions_on_translated ← baseline_model.forward(translated_block)
predictions_on_translated ← predictions_on_translated[position_mapping]
reinforced_predictions    ← reinforced_model.forward(b)
reinforcement_offset      ← ReLU(reinforced_predictions - predictions_on_translated)
generic_predictions       ← predictions_on_translated - α · reinforcement_offset
```

**There is no loss equation in the paper.** Step 3 of §2.3: "take the token with maximal logit to
produce the generic prediction labels"; step 4: "We fine-tune the baseline model with the original
text as input tokens and the generic labels as target tokens." So the objective is **standard
next-token cross-entropy against hard argmax labels**; the α-combined logits are an offline
label-construction device, discarded before training. No `β`, no margin, no KL.

**State distribution.** Teacher-forced, with a twist: **inputs are the original unlearn-target
text; targets are the generic labels.** "for every target label in this example, the context given
to the model is the entire original text which precedes this token." 512-token blocks. Two
consistency hacks: *"(i) Make sure that any instance of an anchored term that appeared previously
in the same block will not be integrated into the loss function from the second appearance
onward, (ii) We reduce the probabilities of the logits corresponding to the translations of
anchored terms that appeared previously."* So per-token masking, no length normalisation.

**Gradient path.** Plain CE descent on the student. Both label-producing models are used only in
`forward` during offline dataset construction — **stop-gradient by construction**, and the labels
are fixed before training starts.

**Preservation.** **No retain loss, no KL anchor, no parameter subset.** Preservation comes
entirely from the *content* of the labels: the generic prediction is a plausible in-distribution
continuation, so language competence is never pushed against. Residual risk flagged honestly:
*"our technique may end up unlearning a super-set of the unlearn target... Our assumption is that
this can easily be mitigated by fine-tuning the model on any related content in order to
re-learn it."*

**Reference/teacher model.** **Two auxiliary models at dataset-construction time** — the frozen
baseline (`Llama-7b-chat-hf`) and the **reinforced** model (baseline fine-tuned on the unlearn
target for 3 epochs). Plus **GPT-4** for entity extraction. The reinforced model is a genuine
liability: it is a model deliberately trained to know the forget set *better*.

**⚠️ Inference requirements — the literature diverges here and it is easy to get wrong.**
In the **original paper**: nothing beyond edited weights. But the **MUSE benchmark**
(Shi et al., arXiv:2407.06460) implements "WHP" as an entirely different, **test-time** method:
`p_unlearn(·|x) = p_target(·|x) − α( p_reinforce(·|x) − p_target(·|x) )`, and states that *"WHP is
a test-time technique where the unlearning operation involves no optimization by itself."*
**MUSE's WHP therefore needs the reinforced model at every decoding step** and is incompatible
with MUSE's GDR/KLR regularizers. Liu et al. (arXiv:2407.16997) read WHP the opposite way, as the
name-change/teacher-distillation half; UNDIAL describes it as contrastive decoding.
**Any citation of "WHP" must say which reading.**

**Parameterization.** Unlearn dataset = the books (2.1M tokens) + synthetic
discussions/blog-posts/wiki entries (1M tokens). Reinforced model: Llama-7b-chat-hf, 3 epochs,
context 512, **lr 3e-6**, batch 8, 16 accum. Labels built with **α = 5**. Final student:
**lr 1e-6**, batch 8, 16 accum, **two epochs**, "roughly 150 gradient descent steps suffice." HF
card: 120 steps, 4M tokens, fp16, 4×A100, 0.5 h. **Full-FT vs LoRA is never stated — NOT
VERIFIED**, though the card ships a single full `pytorch_model.bin` with no adapter files.

**Selection.** Ad hoc and acknowledged: *"GPT-4's role was further leveraged to analyze the
completions during parameter search, but due to its apparent inaccuracy at the task, for our final
training, **a manual inspection was conducted** on the completions."*

**Compute.** Two offline forwards per block + one forward/backward per training step. Reported
wall-clock is the paper's headline efficiency claim: **"in about 1 GPU hour of finetuning"** vs
"over 184K GPU-hours to pretrain"; elsewhere "roughly 30 minutes on 4 A100-GPUs."

**Code.** **No official repository exists.** `github.com/microsoft/Llama2-7b-WhoIsHarryPotter`
404s; a GitHub search for `WhoIsHarryPotter` returns `total_count: 0`; the arXiv Comments field is
empty. Awkward, because the paper twice defers technical detail to a repo that was never
published (§2.2: "Please refer to the GitHub repository for a more detailed account").
**Weights published:** <https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter>, license
**`other` / microsoft-research-license-agreement** — non-commercial research only, and expressly
*"you may use the model(s), but **you may not distribute the models**"*. **Not open source;
redistribution of weights or derivatives is forbidden.** Non-official reimplementations:
`MUSE/baselines/baselines/whos_harry_potter.py` inside the SimNPO repo (MIT), and
<https://github.com/UCSB-NLP-Chang/causal_unlearn>.

**Benchmarks / metrics.** Idiosyncratic — WHP predates TOFU/MUSE/WMDP and invented its own.
- *Completion familiarity*: 300 GPT-4-generated indirect prompts, completions classified by GPT-4
  into four categories; *"We counted only the first two categories, with a multiplier of 5 for the
  first, and summarized the score."*
- *Probability familiarity*: 30 prompts, next tokens manually split "idiosyncratic" vs "generic";
  score = total probability mass on idiosyncratic tokens.
- *General capability*: ARC-C/E, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande via LM Harness.

| Steps | 0 | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| Familiarity (completion) | 0.290 | 0.040 | 0.020 | 0.017 | 0.007 | 0.007 | 0.007 |
| Familiarity (probability) | 0.244 | 0.062 | 0.022 | 0.012 | 0.011 | 0.008 | 0.006 |
| ARC-C | 0.440 | 0.431 | 0.420 | 0.417 | 0.416 | 0.416 | 0.414 |
| HellaSwag | 0.577 | 0.569 | 0.565 | 0.562 | 0.560 | 0.559 | 0.557 |
| BoolQ | 0.807 | 0.802 | 0.801 | 0.798 | 0.798 | 0.797 | 0.796 |
| WinoGrande | 0.663 | 0.676 | 0.669 | 0.666 | 0.665 | 0.661 | 0.657 |

Completion familiarity **0.290 → 0.007** with ≤2.6 points lost on any benchmark. Ablations:
reinforcement bootstrapping alone "never dropped by more than a factor of 0.3 for any combination
of parameters"; anchoring alone (α=0) matched familiarity but cost benchmarks. Honest residual:
*"if the model is prompted to give a list of fictional schools, 'Hogwarts' will be one of the
answers."* Third-party: MUSE scores WHP at News 19.7/21.2/109.6/28.3 and Books 18.0/55.7/56.5/63.6
(α = 2² News, 2⁸ Books). **RWKU does not use WHP as a baseline** — it appears only in Related Work
and as a benchmark row.

### WHP+ — resolved, and the common hypothesis is wrong

**WHP+ exists as a named method with an explicit definition, but it is not an independently
published method, and it is NOT from RWKU.**

- **Not in RWKU** (Jin et al., arXiv:2406.10890). Full-text check: exactly 4 occurrences of "WHP",
  **zero of "WHP+"**, all four non-method. RWKU's six baselines are ICU, RepE, GA, DPO, NPO, RT.
- **Not in MUSE.** Zero occurrences of "WHP+"; MUSE uses plain WHP as the interpolation method.
- **It IS defined in:** Liu, Yujian; Zhang, Yang; Jaakkola, Tommi; Chang, Shiyu. "Revisiting Who's
  Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective."
  arXiv:2407.16997, **EMNLP 2024 main**. <https://aclanthology.org/2024.emnlp-main.495/>, code
  <https://github.com/UCSB-NLP-Chang/causal_unlearn>.
- **But note how.** It is the authors' own label for an ablation of their own framework, always
  written "WHP + (Ours-1)". Verbatim, their §4.2 baseline ❼: *"**WHP + (Ours-1)**, which is an
  instance of our framework where all improved designs in section 3.6 are included except for
  aggregating multiple distributions."* So **WHP+ ≡ their causal-intervention method with N=1
  counterfactual distribution** (their full method uses N=20), keeping "changing the name back"
  and "counter-factual prompting" and dropping multi-distribution aggregation. Their framework
  targets `p(Y|do(X=x)) = Σ_e p(Y|X=x,E=e) p(E=e)` and distils with
  `L = Σ_k KL( p̂(Y|do(X=x_{1:k})) ‖ p_{θ'}(Y|X=x_{1:k}) )`.
- **Cite it as:** "an improved WHP variant introduced as the N=1 ablation of Liu et al.'s
  causal-intervention framework (arXiv:2407.16997, EMNLP 2024)" — **not** as RWKU's baseline and
  **not** to Eldan & Russinovich. Downstream reuse with that attribution exists, e.g. Sun et al.,
  "Unlearning vs. Obfuscation" (arXiv:2505.02884, EMNLP 2025). Numbers are mostly plots; Table A7
  prints WHP 0.75/0.80/0.67/0.54/0.49/0.54 vs WHP+ 0.93/0.90/0.63/0.67/0.82/0.82. Liu et al.'s own
  WHP baseline is a reimplementation "without the reinforcement bootstrapping, to keep consistency
  with our framework" — **so it is not WHP as published either.**
- Two search results attributing WHP+ to arXiv:2510.25732 and arXiv:2506.05735 were checked and
  are **false** (0 occurrences in both) — search-engine conflation.

**Diffusion LM: NO** as a method. **One exception that is easy to misread:** TimeROME-DLM lists
"WHP (Harry Potter completions)" as a **dataset** and reports "on WHP the Harry Potter completion
log-probability drops by 9 nats." The WHP *method* is not implemented anywhere on a diffusion LM.

## 6.4 UNDIAL / UnDIAL

**Citation:** Dong, Yijiang River; Lin, Hongzhou; Belkin, Mikhail; Huerta, Ramon; Vulić, Ivan.
"UnDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models."
arXiv:2402.10052 (v2, 16 Oct 2024). <https://arxiv.org/abs/2402.10052>. Cambridge / Amazon / UCSD.
**Retitled between versions** — v1 (15 Feb 2024) was "Unmemorization in Large Language Models via
Self-Distillation and Deliberate Imagination", which is why the method is `DI` in the code.
**Venue: not stated — NOT VERIFIED.**

**Training target.** Teacher logits from the frozen original model,
`logit_original ~ M_original(·|x_{<t})`, adjusted by subtracting a scaled one-hot at the target
token:

```
logit_adjusted ~ logit_original - γ · e_{x_t}
```

*"where `e_{x_t}` is a one-hot vector for token `x_t` and `γ` is a hyperparameter controlling the
unlearning strength."* Softmax gives `p_adjusted`. The loss:

```
L = min_θ  E_{x ~ D_unlearn} [ Σ_{t=1}^{T} H( p_adjusted , p_{M_θ} ) ]
```

*"where `H` is the cross-entropy between the adjusted and model-generated distributions. As
`p_adjusted` is fixed, minimizing this loss corresponds to minimizing the KL-divergence between
the two distributions."* `x_t` "is typically the highest logit token among the entire vocabulary."

**FUnDIAL** (Focused UnDIAL) adds an entity indicator `𝟙_e` (spaCy nouns + named entities):
`L_f = min_θ E_x[ Σ_t 𝟙_e(x_t) H( p_adjusted, p_{M_θ} ) ]`.

**Of all methods in this document, UNDIAL is the structurally closest to MDU/EMDU** — it is a
*target-matching* objective against a frozen, deliberately-perturbed teacher distribution, exactly
the shape MDU and EMDU argue is right for masked diffusion. See the note at the end of this part.

**State distribution.** Teacher-forced over **every position** `t = 1..T` of the forget sequences,
against **soft** targets (full-vocabulary distributions). `reduction='none'`, summed over
positions; **no length normalisation.** FUnDIAL is exactly a per-token weighting by the
noun/entity indicator — **the one position-selective mechanism in this whole document that is
semantic rather than confidence-based.**

**Gradient path.** Plain descent; teacher under `torch.no_grad()`, `p_adjusted` fixed. The paper's
core argument is the same one MDU makes: *"In GA and NPO, it is difficult to determine the optimal
stopping point because the model lacks a clear convergence target... In contrast, UnDIAL employs a
well-defined target distribution that guides the model toward a stable outcome."*

**Preservation. None explicit — and this is claimed as a feature.** *"UnDIAL has the unique
ability to achieve state-of-the-art performance **without even relying on the Retain set at
all**."* Preservation comes from the target: only one logit per position is perturbed, so the rest
of the vocabulary mass is preserved by construction. **This is directly relevant to the RWKU
setting, where no retain split exists.**

**Reference/teacher model.** Required and frozen — but it is the *original* model, so this is
self-distillation, not an auxiliary adversary (contrast WHP's reinforced model). Loaded separately
(`self.unlearn_teacher_model = AutoModelForCausalLM.from_pretrained(...)`). Because the teacher is
frozen and inputs fixed, `p_adjusted` **could** be precomputed once; neither the paper's listing
nor the repo does so.

**Inference requirements.** Nothing beyond edited weights.

**Parameterization.** **Both** — `main.py:56`: `--peft ∈ {ft, lora, adapter}`, default `ft`; LoRA
`--rank 8 --lora_alpha 16`. GPT-Neo 125M / 1.3B / 2.7B; Table 2 is explicit that 125M/1.3B use
full FT and 1.3B/2.7B also use LoRA. MUSE: **LLaMA-2 7B with LoRA rank 8** "to fit the fine-tuning
onto one NVIDIA A100". AdamW. Batch 64 (125M); 16 × accum 4 = 64 (1.3B). **γ sweeps: 1, 3, 10, 30**
on extraction data; **γ = 2, 4, 8, 16** on MUSE; LRs **1e-4 and 1e-5** on MUSE. Repo default
`--di_strength 5`, but the README's reproduction commands use `--di_strength 3`, with
`--lr_di 1e-06 --num_epochs_di 10` (125M full FT) and
`--lr_di 5e-06 --num_epochs_di 100 --peft lora --rank 8 --lora_alpha 16` (1.3B LoRA).
**Epoch counts are never stated in the paper** — repo only.

**Selection.** The paper claims none is needed: *"This stability allows for **flexible stopping
points without any risk of degradation**."* γ is swept to trace Pareto curves rather than
selected. **But the repo contradicts the paper's framing**: `main.py:64-65` exposes `--early_stop`
with `--early_stop_criteria` default `1.03`, documented inline as "A number > 1 that indicates the
percentage of ppl change", wired into
`EarlyStoppingCallback(initial_perplexity=..., ppl_change=...)` — i.e. stop when validation
perplexity exceeds 1.03× its initial value, **a utility-based criterion computed on a held-out
validation set**, and the 1.3B README command enables it. The paper does not mention this.
**Noteworthy: this is the only held-out-validation stopping rule found anywhere in this
document's literature.** If we want a defensible selection rule, this is the precedent to cite.

**Compute.** 2 forwards (student with grad, frozen teacher no-grad) + 1 backward — structurally
the same as NPO, with the extra cost that soft targets are full `|V|`-wide distributions.
Hardware "one NVIDIA A100" for MUSE. **No wall-clock reported anywhere.**

**Code.** <https://github.com/dong-river/LLM_unlearning> — owner is the first author; README
self-identifies as "the official code implementation for our paper **UNDIAL**". **Caveat: the repo
is not linked from the paper** (no github link on the v2 abs page, empty Comments); it was located
via the author's GitHub account. Near-duplicate earlier repo:
<https://github.com/dong-river/DI_LLM_unlearning>. **No LICENSE file** — API `license: null`,
`/LICENSE` 404s, nothing in the tree. No weights, 0 releases.

**Benchmarks / metrics / numbers.**
*Case study 1 — Training Data Extraction Challenge* (15,000 × 200-token Pile sequences; GPT-Neo):
`MA(x) = [Σ_{t=1}^{T-1} 𝟙[argmax p_θ(·|x_{<t}) = x_t]]/(T-1)`;
`EL_n(x) = [Σ_{t=1}^{T-n} Overlap_n(M(·|x_{<t}), x_{≥t})]/(T-n)` with
`Overlap_n(a,b) = |n-gram(a) ∩ n-gram(b)|/|n-gram(a)|`. Usefulness via **MAUVE**, perplexity, Rep3
on WikiText-103; NLU on HellaSwag/WinoGrande/COPA/ARC/PIQA/PubMedQA.

GPT-Neo 125M (baseline EL3 0.202, MAUVE 0.718, PPL 17.192, Rep3 0.035, NLU 0.503), rows at
matched EL3 ≈ 0.1:

| Method | Coeff | EL3↓ | MAUVE | PPL↓ | Rep3↓ | NLU |
|---|---|---|---|---|---|---|
| +TA | 0.05 | 0.117 | 0.305 | 24.089 | 0.174 | 0.504 |
| +DP | 0.2 | 0.090 | 0.522 | 76.428 | 0.002 | 0.478 |
| +CD | 0.25 | 0.089 | 0.333 | 52.202 | 0.056 | 0.495 |
| +GA | 3 | 0.092 | 0.119 | 10.478 | 0.163 | 0.489 |
| +NPO | 1 | 0.114 | 0.685 | 26.538 | 0.077 | 0.515 |
| **+UnDIAL** | γ=3 | 0.111 | 0.674 | 32.584 | 0.010 | 0.500 |
| +UnDIAL | γ=10 | 0.019 | 0.450 | 65.591 | 0.005 | 0.499 |
| +UnDIAL | γ=30 | 0.013 | 0.437 | 64.594 | 0.005 | 0.498 |

The real claim is the γ=10/30 rows: UnDIAL holds MAUVE ≈ 0.44–0.45 at EL3 ≈ 0.013–0.019, where
NPO at EL3 0.038 has already collapsed to MAUVE 0.026.
*Case study 2 — MUSE News* (LLaMA-2 7B + LoRA r=8): Knowledge Memorization vs Utility
Preservation, KL-to-reference training dynamics, forget-set scaling, sequential unlearning.
**Honest gap: all MUSE results are figures (5, 6, 7) — no numerical MUSE table.**
**UNDIAL is never run on TOFU, WMDP, RWKU, or Harry Potter completions.**

**Diffusion LM: NO.** Zero occurrences of "UNDIAL" in the full texts of MDU, EMDU, or
TimeROME-DLM; arXiv and web queries for UNDIAL + LLaDA / Dream / masked diffusion return only MDU.
This negative is strong but **not provably exhaustive** — arXiv API search covers
title/abstract/author only.

## 6.5 Two traps in the AR-method literature worth recording

1. **Name collision on "NPO".** "Diffusion-NPO" (arXiv:2505.11245) and "Self-NPO"
   (arXiv:2505.11777) use "NPO" for **preference-aligned image generation**, not unlearning, and
   do not cite Zhang et al. Do not count these as diffusion applications of NPO.
2. **Image-diffusion concept erasure is a separate literature and these methods are largely
   absent from it.** The concept-erasure survey arXiv:2502.14896v2 has zero occurrences of "NPO",
   "SimNPO", or "preference optimization"; SimNPO cites ESD / Forget-Me-Not / AdvUnlearn /
   UnlearnCanvas in its references only and runs no image-diffusion experiments. **Do not import
   image-diffusion erasure results as evidence about text diffusion.** MDU and EMDU both make
   this point themselves (EMDU §2: continuous Gaussian schedules over continuous latents *"do not
   directly transfer to the discrete absorbing-state masking process of text-based MDLMs"*).

## 6.6 The opening this part surfaces

**UNDIAL is the only AR method here that is a *target-matching* objective rather than a
divergence-maximizing one — which is precisely the property MDU and EMDU argue is the right shape
for masked-diffusion unlearning — and it has never been run on a diffusion LM.** Nor has WHP's
label-synthesis approach.

Two further specifics make this concrete for us:

- **UNDIAL's target is a single-logit edit of the frozen teacher's own distribution.** That is
  structurally *between* MDU/EMDU's model-derived anchor (a whole separate forward pass) and our
  exclusion target (hand-built, model-free). An MDLM UNDIAL would be: frozen base model forward on
  `(x, y_t)` **with the prompt visible**, subtract `γ` from the ground-truth logit at each masked
  position, softmax, match. **Note what that is not: it is not prompt-masked.** It preserves the
  prompt-conditional structure while suppressing exactly the forget token — which is a different
  preservation hypothesis from MDU's "collapse to the unconditional trajectory".
- **FUnDIAL's entity indicator is a semantic position-selection rule** (spaCy nouns + named
  entities). MDU's own token-role analysis (in-context / structural / **stored-knowledge**) is a
  near-identical idea that MDU measured and did not use. An MDLM FUnDIAL is therefore a
  *specifically licensed* way to act on MDU's unexploited diagnostic.

Under Part 3, "UNDIAL ported to MDLM" would be category **(c)** at best on its own — a familiar
combination of a published objective with the published AR⇒MDLM substitution rule — and that is
worth knowing before anyone proposes it as novel. Combining it with a mask-rate or position
mechanism could reach (d), but only with the factorial ablation and the pre-stated mechanism
story Part 3 requires.

---

# Part 7 — RMU, CLP, CUSE, TNT/TNFF, and span-level unlearning

**Headline of this part:** of the five requested topics, **two are real methods with full published
detail (RMU, span-level unlearning), one is real but is a fairness method never used for
unlearning (CLP), one is real but is a ten-day-old multimodal preprint (CUSE), and two do not
exist (TNT, TNFF).**

## 7.1 RMU — Representation Misdirection for Unlearning

**Citation:** Li, Nathaniel; Pan, Alexander; Gopal, Anjali; Yue, Summer; … Hendrycks, Dan (55+
authors). "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning."
arXiv:2403.03218 (v1 2024-03-05, v7 2024-05-15), **ICML 2024**.
<https://arxiv.org/abs/2403.03218> · <https://wmdp.ai> · code
<https://github.com/centerforaisafety/wmdp>.

Naming note: RMU was first published as **CUT**. Appendix B.4, verbatim: *"we identified that the
performance of CUT is derived from increasing the norm of the activations, rather than steering
towards a particular direction. Thus, we introduce RMU, a simplification to CUT which steers
towards random vectors (of the same norm that CUT steered towards) and retains the same
performance."* **That is a published (a)/(b)-style finding about their own method, and it is a
good model for how to report one.**

**Training target.** Transcribed from the arXiv LaTeX source (`sections/4-method.tex`):

```
L_forget = E_{x_f ~ D_forget}[ (1/L_f) Σ_{token t ∈ x_f} ‖ M_updated(t) - c · u ‖₂² ]
L_retain = E_{x_r ~ D_retain}[ (1/L_r) Σ_{token t ∈ x_r} ‖ M_updated(t) - M_frozen(t) ‖₂² ]
L        = L_forget + α · L_retain
```

`L_f`/`L_r` = token counts; `c` *"some hyperparameter that controls activation scaling"*; `α`
weights retention. **The loss is on activations, never on logits** — this is the one method in
this document that never touches an output distribution.

**The control vector**, verbatim: *"we compute `u`, a random unit vector with independent entries
sampled uniformly at random from `[0,1)`. Note that `u` is held fixed throughout training."* One
vector per forget topic. Code (`rmu/unlearn.py:36-38`) draws `torch.rand`, L2-normalises, scales
by `steering_coeff`.

**Layer and parameter subset**, verbatim: *"In practice, we find it sufficient to compute the loss
only on layer `ℓ` and update gradients only on layers `ℓ-2`, `ℓ-1`, and `ℓ`."* Appendix B.4:
*"We focus unlearning only on the MLPs, as those encode knowledge in the model."*

**The exact parameter set is only recoverable from code, and was verified two ways.** `rmu/utils.py`
selects `model.model.layers[layer_id].parameters()` by positional index `param_ids=[6]`, which for
a Mistral/Llama decoder layer is `mlp.down_proj.weight`. Independently, `locuslab/open-unlearning`
(MIT) reimplements RMU citing the exact WMDP source line and spells it out literally:
`trainable_params_regex=["model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"]`.
**So RMU updates exactly three weight matrices — the MLP down-projections of layers ℓ−2, ℓ−1, ℓ.**
(Mixtral is the exception: `--param_ids 7`, because the MoE block's parameter ordering differs.)

**Layer selection**, verbatim: *"We perform a hyperparameter search over the layer `ℓ` to perform
the unlearning loss on, starting from the third layer and going to the last layer."*

**Hyperparameters** (Appendix B.4; the repo's reproduction notebooks agree exactly):

| | Zephyr-7B-β | Yi-34B-Chat | Mixtral-8x7B-Instruct |
|---|---|---|---|
| layer ℓ | 7 | 15 | 7 |
| updated layers | 5,6,7 | 13,14,15 | 5,6,7 |
| steering coeff `c` | 6.5 | 300 | 300 |
| `α` | 1200 | 350 | 1600 |
| LR | 5e-5 | 5e-5 | 5e-5 |
| batch size | 4 | 2 | 2 |
| batches (= grad steps) | 150 | 400 | 400 |

Repo **defaults differ from the paper** (`--steering_coeffs 20,20 --alpha 100,100`) — use the
notebook commands. Epochs = 1 (hardcoded `for epoch in range(1)`). Max length 512 bio / 768 cyber
forget, 512 retain. Seed 42. AdamW. One stated inconsistency: the paper says the batch-count grid
was `[150, 300, 500]`, but Yi and Mixtral were run at **400**, not in that grid.

**State distribution / sampling.** Raw corpus text, **no labels and no generation whatsoever**.
Forget = PubMed papers (bio) and a filtered GitHub crawl (cyber); retain = **Wikitext**, chosen
deliberately: *"we find in practice that RMU is more performant when `D_retain` has qualitatively
distinct content from `D_forget`, so as not to relearn the unlearned knowledge."* Topics
interleave: *"update model weights on the biosecurity distribution, then update on the
cybersecurity distribution, then repeat."*

**Gradient path.** Hidden states at layer ℓ captured by a forward hook; plain AdamW descent on
`unlearn_loss + alpha*retain_loss`. The frozen model's forward is under `torch.no_grad()`. No
stop-gradients inside the differentiated path.

**Preservation.** ℓ² activation anchoring against the frozen model, weighted by a very large α
(350–1600), **plus** the three-matrix parameter subset.

**Reference/teacher model.** Required and frozen — a **second full copy of the same base model**
(`load_model` called twice on the same checkpoint), used only for `M_frozen` on retain batches.
**Memory cost is 2× the model** — directly relevant to our 40 GB constraint.

**Inference requirements.** None. Output is an ordinary HF checkpoint.

**Compute.** Per step **3 forwards** (forget w/ grad, retain w/ grad, retain no-grad) + **1
backward**, with backprop restricted to ℓ−2…ℓ. Wall-clock is not in the paper, but the committed
notebook output gives it: **Zephyr-7B, 150 steps in 2:23 (~1.05 it/s)** — RMU is minutes, not
hours. Hardware not stated.

**Selection.** Grid over ℓ (layer 3 → last), batch count, and α, selected on WMDP-Bio/Cyber
accuracy vs MMLU/MT-Bench. **Tuned directly on the evaluation benchmarks; no held-out split** —
same as everything else here.

**Code / license / weights.** <https://github.com/centerforaisafety/wmdp>, **MIT License**
(© 2024 centerforaisafety, LICENSE file read directly). RMU implementation last touched
2024-04-27 (stable). **Published weights, all MIT:** `cais/Zephyr_RMU`, `cais/Yi-34B-Chat_RMU`,
`cais/Mixtral-8x7B-Instruct_RMU`. **Forget corpora are gated** (`cais/wmdp-bio-forget-corpus`,
requires `HF_TOKEN`) — a real reproducibility friction point, and one we cannot clear without
accepting new data terms.

**Benchmarks / metrics / numbers** (Table 1; WMDP/MMLU in %, 25% = random):

| Model | WMDP-Bio ↓ | WMDP-Cyber ↓ | MMLU ↑ | MT-Bench ↑ |
|---|---|---|---|---|
| Zephyr-7B base | 63.7 | 44.0 | 58.1 | 7.33 |
| + LLMU | 59.5 | 39.5 | 44.7 | 1.00 |
| + SCRUB | 43.8 | 39.3 | 51.2 | 1.43 |
| + SSD | 50.2 | 35.0 | 40.7 | 5.48 |
| **+ RMU** | **31.2** | **28.2** | **57.1** | **7.10** |
| Yi-34B-Chat → +RMU | 75.3 → 30.7 | 49.7 → 29.0 | 72.6 → 70.6 | 7.65 → 7.59 |
| Mixtral-8x7B → +RMU | 74.8 → 34.0 | 52.0 → 30.8 | 68.2 → 67.1 | 8.30 → 8.17 |

Collateral damage is concentrated and visible: MMLU **Virology** 52.4 → 25.9 and **Computer
Security** 65.0 → 45.0 on Zephyr. Robustness: resists GCG (2,500 steps yields only degenerate
output) and linear probes.
**Honest negative result the paper reports about itself:** *"we find that RMU is unable to
prevent finetuning from recovering performance"* — fine-tuning on the cyber forget corpus restores
the capability. **This is independently consistent with Fan et al.'s finding that RMU is the
relearning-robustness loser, and with our own HISTORICAL result that MDLM unlearning is
reversible.** Three independent sources now agree that parameter-subset unlearning does not
survive fine-tuning. **That is the strongest cross-validated fact in this document, and it is a
direct warning about our LoRA deviation (Part 2, D1).**

**Diffusion LM: NO** for RMU as published (Zephyr/Yi/Mixtral only), and **RMU is absent from the
baseline lists of both MDU and EMDU**. However the **mechanism has been transplanted to image
diffusion**: Lee, Uichan; Kim, Jeonghyeon; Hwang, Sangheum. "Localized Concept Erasure in
Text-to-Image Diffusion Models via High-Level Representation Misdirection." arXiv:2602.19631,
**ICLR 2026**. It reproduces RMU's layer decoupling exactly — loss on the final CLIP text-encoder
block `h^(L)`, updates only `θ₁` (first block), `L_HiRM-R = (1/T) Σ_t ‖h_t^(L) − c·r̂_t‖²`, **no
retain term**. Code <https://github.com/Coffeeloveman/HiRM>, **no LICENSE file**. **Caution: this
is continuous image diffusion over a CLIP text encoder, not a text MDLM** — do not import it as
evidence about LLaDA (both MDU and EMDU explicitly warn against that transfer).

**Generation trajectory: NO.** RMU never decodes. Also no for its reasoning extension **R²MU**
(Wang, Changsheng et al. "Reasoning Model Unlearning: Forgetting Traces, Not Just Answers…"
arXiv:2506.12963, **EMNLP 2025**, <https://github.com/OPTML-Group/Unlearn-R2MU>, no LICENSE),
whose objective is `ℓ_RMU + α ℓ_unthink + β ℓ_CoT` with
`ℓ_unthink = E[(1/N) Σ_i ‖M_θ(r_i) − c·u‖²]` over CoT traces **generated offline and then
teacher-forced**, not sampled in the loop.

**RMU follow-ups worth a line:** **Adaptive RMU** (arXiv:2408.06223) replaces the fixed `c` with
`β‖h^(l)_{θ_frozen}(x_F)‖`; **HiRM** above; the G-effect paper (§7.5c) finds RMU's G-effect
unstable and reports **RMU losing to NPO on TOFU in 5 of 6 settings**.

**Runnable here?** **Y\*** with an important twist. RMU is *already* a parameter-subset method, so
it needs no LoRA — we could update `model.layers.{ℓ-2,ℓ-1,ℓ}.mlp.down_proj.weight` directly, which
for LLaDA-8B is ~3 × (4096 × 12288) × 2 bytes ≈ 0.3 GB of trainable weights plus ~1.2 GB of
AdamW state. **That fits trivially.** The blocker is different: RMU needs a **second full frozen
copy** (32 GB of bf16 weights across two models) — unless we exploit the same trick as for
MDU/EMDU and note that if we restrict trainable parameters to three matrices, `M_frozen` can be
obtained by swapping those three matrices back, or by keeping only the *original* three matrices
in reserve (~0.3 GB) rather than a whole second model. **That is a legitimate exactness-preserving
engineering change, not an approximation**, and it makes RMU the cheapest full-weight method
available to us. Deviations to declare: LLaDA's layer count and MLP shape differ from Zephyr's, so
ℓ must be re-searched; RMU's loss is on activations and is therefore **mask-rate agnostic**, so the
MDLM port is *less* ambiguous than the likelihood-based ports; and the forget corpus for RWKU would
have to be the RWKU forget pairs as raw text, which is not RMU's published data regime.

## 7.2 CLP — Counterfactual Logit Pairing: real, but **not an unlearning method**

**Citation:** Garg, Sahaj; Perot, Vincent; Limtiaco, Nicole; Taly, Ankur; Chi, Ed H.; Beutel, Alex.
"Counterfactual Fairness in Text Classification through Robustness." arXiv:1809.10610, **AIES
2019** (DOI 10.1145/3306618.3317950). <https://arxiv.org/abs/1809.10610>

It exists under that exact name, **in the fairness literature only.** The abstract names it:
*"we offer three approaches, blindness, counterfactual augmentation, and counterfactual logit
pairing (CLP), for optimizing counterfactual token fairness during training."*

**CTF is the *metric*; CLP is one of three *training methods* for it.** Exact loss, verbatim:

```
Σ_{x∈X} J(f(x), y)  +  λ Σ_{x∈X} E_{x' ~ Unif[Φ(x)]} | g(x) - g(x') |
```

with `f(x) = σ(g(x))`, `J` = cross-entropy on the binary toxicity label, and *"For computational
tractability, during training, we randomly sample a single counterfactual example for each
input."* The metric is the same difference on **probabilities** rather than logits:
`CF GAP_Φ(x) = E_{x'~Unif[Φ(x)]} | f(x) - f(x') |`.

**Provenance and a naming trap.** The paper states *"The robustness term is given by logit pairing
[Kannan, Kurakin, and Goodfellow]"* — CLP is an L1 adaptation of **Adversarial Logit Pairing**
(arXiv:1803.06373), **whose §4.2.2 already uses "CLP" to mean *clean* logit pairing.** A third,
unrelated CLP exists in model repair: **Channel Lipschitzness-based Pruning** (arXiv:2208.03111,
ECCV 2022), data-free backdoor removal — genuinely adjacent to unlearning and the likeliest source
of confusion. (License of that repo: **NOT VERIFIED**.)

**Other fields.** `λ ∈ {0.05, 1, 5}`. CNN classifiers trained from scratch, full-parameter, 5
epochs, best-on-dev, 10 runs averaged; **LR / batch / architecture not stated**. No reference
model. Nothing needed at inference. Compute not stated (~2× baseline, derived). Train data = 160K
Kaggle Jigsaw Wikipedia comments; eval on a **private** forum dataset + synthetic templates; 50
identity terms split 35 train / 12 held-out. Headline: AUC held at 0.962–0.964 across all methods;
CTF gap on training terms 0.140 → **0.002** (CLP λ=5); on **held-out** terms 0.091 → 0.076.
Mechanism diagnostic: identity-token cosine similarity 0.25 → 0.87.
**No diffusion. No generation trajectory** — a single-step, non-autoregressive classifier penalty
with no decoding at all.
**Code:** the paper releases none, but CLP ships in **TensorFlow Model Remediation**,
<https://github.com/tensorflow/model-remediation>, **Apache-2.0** (LICENSE read directly), as
`counterfactual/losses/pairwise_absolute_difference_loss.py`. **Caveat: that repo is archived,
last commit 2023-03-06** — cite it as an archived reference implementation, not live tooling. No
weights.

### Has anyone applied CLP to LLM unlearning? **No.** Five independent negative probes

1. arXiv API `all:"counterfactual logit pairing"` → **exactly 1 result**, Garg et al.
2. **OpenAlex citation sweep.** Garg et al. = `W2893240881`, 200 citing works. Filtering those 200
   for `unlearn` / `unlearning` / `forgetting` / `erasure` / `machine unlearning` → **count 0 for
   every term.**
3. `locuslab/open-unlearning` (the de facto toolkit; NeurIPS D&B 2025; covers TOFU + MUSE + WMDP)
   — methods are exactly GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, AltPO, SatImp, WGA,
   CE-U, PDU. **No CLP.**
4. `chrisliu298/awesome-llm-unlearning` README, 190,917 bytes — **0 hits** for `CLP`,
   `logit pairing`, `Counterfactual Logit`, and even bare `Pairing`.
5. Targeted searches surface genuinely *different* methods that could be mistaken for it:
   **ULD / "Unlearning from Logit Difference"** (arXiv:2406.08607, NeurIPS 2024) and **Offset
   Unlearning** (arXiv:2404.11045).

**Verdict: if any draft of ours lists "CLP" as an unlearning baseline, it is a conflation — most
likely with ULD, with Clean Logit Pairing, or with Channel-Lipschitzness pruning. Remove it.**
If the *idea* is wanted (pair logits under a counterfactual perturbation of the prompt), the real
prior art to cite is ULD and Offset Unlearning, not CLP.

## 7.3 CUSE — real, an unlearning method, and ten days old

**Citation:** Ding, Chenlu; Wu, Jiancan; Luo, Yanchen; Zhang, Tianyu; Yuan, Yancheng; Wang, Xiang;
He, Xiangnan. "CUSE: Fine-Grained Multimodal Unlearning with Counterfactual Guidance and On-Policy
Self-Distillation." **arXiv:2610.14763v1, submitted 2026-10-10.**
<https://arxiv.org/abs/2610.14763>. cs.AI + cs.CL, **no journal_ref — an unrefereed preprint.**
Expansion verbatim: *"a **C**ounterfactual-guided **U**nlearning framework via on-policy
**SE**lf-distillation."*

It is absent from `awesome-llm-unlearning` only because that list was pushed 2026-10-06, **four
days before CUSE appeared** — so that absence is not evidence. An arXiv sweep of `all:"CUSE"`
returns 17 entries of which **16 are copper-selenide condensed-matter physics**; 2610.14763 is the
sole ML hit. There is no CUSE *benchmark*, no "confidence/certainty-based unlearning", nothing
under that name in concept erasure.

**Exact losses, verbatim:**

```
Eq 6:  Δ_t = log p_{θ0}( y_t | x, Y_{<t} ) - log p_{θ0}( y_t | x^cf, Y_{<t} )
Eq 7:  a_t = clip( max(Δ_t, 0) / ρ , 0, 1 )
Eq 8:  w_t = 1 + γ a_t ,     ŵ_t = w_t / ( (1/T) Σ_{j=1}^{T} w_j )
Eq 12: L_CUSE(θ) =
   Σ_{x^f ∈ D_f} E_{Y ~ p_θ(·|x^f)} (1/|Y|) Σ_{t=1}^{|Y|} ŵ_t · D_KL( p_φ(·|x^f, c_f*, Y_{<t}) ‖ p_θ(·|x^f, Y_{<t}) )
 + λ Σ_{x^r ∈ D_r} E_{Y ~ p_θ(·|x^r)} (1/|Y|) Σ_{t=1}^{|Y|}      D_KL( p_φ(·|x^r, c_r*, Y_{<t}) ‖ p_θ(·|x^r, Y_{<t}) )
```

**One-line characterisation: CUSE = forward-KL on-policy self-distillation + a
counterfactual-derived per-token gradient scaler.** There is **no separate counterfactual loss
term** — `x^cf` appears in no loss, only inside Eq 6 under the frozen `θ0`.

**Note the KL orientation: `D_KL(teacher ‖ student)` — anchor-first, the same orientation EMDU
argues for.** Two independent groups, in the same month, on different model families, converging
on anchor-first target matching. Worth knowing.

**Hyperparameters.** `γ = 2`; `ρ` = *"the 75th percentile of positive counterfactual
log-probability differences in each local batch"* (adaptive per batch); **`λ` is never given a
numerical value anywhere in the paper** — a genuine reproducibility hole.

**State distribution / sampling — the interesting part.** **Model-sampled, on-policy, NOT
teacher-forced.** Verbatim: *"We first sample a response `Y=(y_1,…,y_T)` from the current policy
`p_θ(·|x)`"* and *"responses are sampled online from the evolving policy `p_θ` rather than using
fixed offline responses."* Both branches sample. The frozen teacher and `θ0` are then
teacher-forced **onto the student's own sampled tokens**. Rollouts per step: **not stated**
(Algorithm 1's singular draw plus `1/|Y|` implies one). **Decoding config (temperature / top-p /
greedy / max_new_tokens) is never given** — zero hits for any of those strings. That is a real
hole, and it sits awkwardly beside a robustness experiment premised on stochastic decoding.

**Gradient path.** Only `θ`, only through the second KL argument. Appendix G makes `ŵ_t` a pure
gradient scaler: `∇_θ( ŵ_t D_KL(q_t ‖ p_{θ,t}) ) = ŵ_t ∇_θ D_KL(q_t ‖ p_{θ,t})`. "stop-gradient"
and "detach" appear **zero** times; the no-grad property is structural (teacher and `θ0` frozen).
**Flag worth recording:** despite `E_{Y ~ p_θ}` in the objective there is **no score-function /
REINFORCE term** through the sampling distribution, and the paper never acknowledges it — **the
implemented gradient is a biased surrogate.** Algorithm 1 writes plain descent
`θ ← θ − η ∇_θ L`; Appendix B.4 says AdamW.

**Preservation.** The retain KL branch under a retention-oriented privileged context, plus Eq 8's
normalisation forcing mean token weight to 1 (redistributes rather than amplifies forgetting
pressure). Motivating statistic: *"Only a small subset of generated tokens is highly sensitive to
target-related evidence (about 20% in our preliminary analysis)."* **Compare MDU's own
stored-knowledge-token finding — the same observation, from a different model family.**

**Reference/teacher model.** Required, frozen, and it is the frozen original — **definitively not
an EMA.** Algorithm 1: `M_θ ← M_{θ0}`, `M_φ ← M_{θ0}` (frozen); zero hits for `EMA` / `momentum`.
"Self" means same weights, different conditioning: *"instantiated from the same underlying model
with different conditioning contexts."* Privileged contexts are hand-written (Appendix F); the
paper concedes *"the construction of privileged contexts currently requires task-specific
design."*

**Inference requirements.** Nothing beyond the LoRA adapter — **NOT VERIFIED as an explicit claim**;
inferred from the eval protocol.

**Parameterization, verbatim.** Vision encoder and projector frozen; **LoRA on LM attention + MLP
(`q,k,v,o,gate,up,down`), rank 8, α 16, dropout 0.05**; AdamW, 5 epochs, global batch 16, weight
decay 0.01; LR 5e-4 (LLaVA-1.5-7B) / 5e-5 (Qwen2.5-VL-7B) on MLLMU-Bench, 5e-4 / 2e-4 on
CoVUBench.

**Selection: not stated** — no validation split, no early stopping, `while not converged do` with
no test defined. The only tuning is a single-variable `γ` sweep run **on MLLMU-Bench itself**.

**Compute.** *"four NVIDIA A100 GPUs with 80 GB"*; no wall-clock or step count. Limitations admit
*"additional computational costs due to online sampling and privileged teacher inference."*
Derived per-step: **~2 generation rollouts + ~6 teacher-forced forwards + 1 backward**, vs 1+1 for
GA/NPO.

**Code: none.** No URL, no "code will be released" statement. No weights.

**Benchmarks.** MLLMU-Bench and CoVUBench; LLaVA-1.5-7B, Qwen2.5-VL-7B-Instruct,
Qwen3-VL-4B-Instruct; forget ratios 5/10/15%; baselines GA, GA_Diff, KL_Min, NPO, MMUnlearner,
MANU, ASRU. LLaVA 5% (ClsFgt↓ / ClsRet↑ / GenFgt↓ / GenRet↑): Vanilla 59.20/60.70/0.645/0.584 ·
best baseline ASRU 30.40/50.58/0.358/0.513 · **CUSE 11.20/60.51/0.129/0.554**. CoVUBench 5% LLaVA:
CUSE 99.23/99.75/98.68/72.29/96.97/78.28 vs ASRU 91.34/87.92/89.48/65.72/90.86/75.93.

**⚠️ Discrepancy to flag.** The abstract's headline *"reduces forget-set ROUGE-L by 33.6%"* is
never restated or derived in the body, and "33.6" appears nowhere else. Table 1 against the
strongest baseline gives **64%** (LLaVA) and **53%** (Qwen). The retain half (+1.2%) does check
out. **The forget half of the headline is unsubstantiated by the paper's own tables.**

**Diffusion: NO** — "diffus" has zero hits (the `GA_Diff` baseline is gradient-ascent-plus-
*difference*, not diffusion); all backbones are autoregressive and every term is defined over
left-to-right prefixes `Y_{<t}`.

**Generation trajectory in the loss: YES, centrally.** Appendix G: *"Since generation is
autoregressive, the sensitivity at each decoding step is evaluated under the current generation
state conditioned on the preceding trajectory."* Load-bearing, not incidental: the **off-policy
ablation degrades forget ROUGE from 0.129 to 0.372.** Lineage: GKD/OPD (Agarwal et al. 2024) →
OPSD (Zhao et al. 2026) → CUSE.

**Why this matters to us: CUSE is the single most important entry in this part.** It is an
*existence proof* that on-policy, trajectory-conditioned target matching beats teacher-forced
unlearning — with a measured ablation (0.129 vs 0.372) — on autoregressive multimodal models. It
is **not** on a diffusion LM. See the Q2 refinement below.

**Runnable here?** **N as published** (multimodal benchmarks we do not have, no code, `λ`
unspecified). But its *mechanism* is portable and it is the closest published template for a
trajectory-aware unlearning objective. Porting it to MDLM is a research proposal, not a baseline.

**Closest acronym neighbours, for the record:** **SAUCE** (arXiv:2503.14530), **CUFG**
(arXiv:2509.14633), **SAeUron** (arXiv:2501.18052), **CE-U** (in open-unlearning).

## 7.4 TNT and TNFF — **these do not exist**

**Honest answer: "TNFF" does not exist as any ML acronym at all, and while "TNT" exists in ML,
every real TNT paper is in an unrelated field. Neither is an unlearning method.**

Evidence, all primary:

- **arXiv API** `ti:"TNT"` (all time, 60 max) → **17 papers, zero unlearning-related**: TnT
  Attacks (adversarial patches), TNT: Target-driveN Trajectory Prediction, TnT-LLM, TNT-KID, TnT
  POS tagger, Nested-TNT, TNT turbulence ViT, Dense-TNT, TNT ternary quantization, TNT MPLS
  tunnels, plus literal TNT-explosive chemistry.
- `abs:"TNT" AND abs:"unlearning"` → **0**. `abs:"TNT" AND abs:"forget"` → **0**.
  `all:"TNFF"` → **totalResults 0**.
- **GitHub repo search API:** `TNT+unlearning` → **0**; `TNFF+unlearning` → **0**; `TNFF` → 22
  repos, all Toronto Nepali Film Festival / fantasy football.
- `awesome-llm-unlearning` README (190,917 bytes): `TNT` **0**, `TNFF` **0**.
- `open-unlearning` method list: no TNT, no TNFF.
- Full texts of MDU (2605.18253), EMDU (2610.25447) and Forgetting Neural Networks (2410.22374)
  grepped: `TNT` 0, `TNFF` 0.

**Expansions tested and found empty:** "Task-Normalized…", "TNT: Target-aware…",
"Token-level/Token-Normalized…", "TNT: Train aNd Trim", "Targeted Neuron…", "Token-Normalized
Forget-Firstness", "Target Negation + Forget-Free", "Targeted Neural Trojan".

**One finding worth flagging loudly, because it explains where these names probably came from: we
caught the acronym being fabricated live.** A web search returned a summary headed *"Enhanced
Masked Diffusion Unlearning (TNT)"*; downloading that paper's full text (arXiv:2610.25447, 838 KB)
gives **zero occurrences of "TNT"** — it calls itself EMDU throughout. **A search summarizer
attached the query's acronym to an unrelated real paper.** That is very likely the mechanism that
produced "TNT"/"TNFF" in whatever source they reached us from. **Treat any method name that cannot
be traced to a paper's own text as unverified until it can.**

**Closest real work, with the collisions documented:**
- **TnT Attacks!** Doan, Bao Gia et al., arXiv:2111.09999, IEEE TIFS — "universal **Na**Turalistic
  adversarial pa**T**ches". An *attack*, not trojan removal or unlearning.
- **There is no paper titled "TNT: Targeted Neural Trojan."** The real bit-trojan paper is **TBT**
  (Rakin, He, Fan, arXiv:1909.05193, CVPR 2020). That string is a conflation of TBT with TnT
  Attacks.
- **Most plausible real unlearning paper behind a "TNT" mislabel: Temper-Then-Tilt**, Block, Jacob
  L.; Mohri, Mehryar; Mokhtari, Aryan; Shakkottai, Sanjay. arXiv:2602.10217, **ICML 2026**, code
  <https://github.com/jacob-block/T3-Unlearning>. It abbreviates itself **T3-Unlearning, never
  TNT** — but "Temper-aNd-Tilt → TNT" is the obvious route. Base model frozen; inference-time
  tempering + classifier tilting `p̂_r^{(T)} ∝ p^{1/T} · f̂`; **needs the classifier at generation
  time**, so it is not a weights-only edit. *If someone on this project meant a real method by
  "TNT", this is the most likely one, and its deployment contract differs from MDU/EMDU's.*
- Nearest "Task-Normalized" method: **NGDiff** (arXiv:2410.22086). Nearest "Target Negation":
  **ALKN** (OpenReview tcK4PV3VN4). Nearest "Targeted Neuron Forgetting Factor": **Machine
  Unlearning using Forgetting Neural Networks** (arXiv:2410.22374) — grepped, `TNFF` 0 hits; the
  acronym is not theirs. **None is ever called TNT or TNFF by its authors.**

**Scope caveat, stated honestly:** arXiv's API searches title/abstract, not full text, so a method
named TNT only in a paper's body would escape probes 1–3. The two full-text greps, the 190 KB
bibliography, the toolkit method list, and GitHub repo search close that gap from different angles
and all return zero. **Semantic Scholar returned HTTP 429 and is an unchecked index.**

## 7.5 Selective / span-level unlearning

### (a) SPAN — Selective Span-Level Unlearning (the flagship, and the most on-target paper for us)

**Citation:** Yoon, Chaewon; Kim, Dongjun; Song, Hyun-Je (Jeonbuk National University). "Selective
Span-Level Unlearning for Large Language Models." **ACL 2026, Volume 2: Short Papers, pp.
423–431.** <https://aclanthology.org/2026.acl-short.35/> · code
<https://github.com/lluvecwonv/Span-level-Unlearn>

Its pitch is directly relevant: prior selective methods *"depend on signals provided by external
models"* and are therefore misaligned with the target model's internals; SPAN is *"grounded
entirely in model-intrinsic information."*

**Exact equations, from the PDF.** Generic objective (Eq 1):
`min_θ E_{D_f}[ℓ_f(y|x;θ)] + λ E_{D_r}[ℓ_r(y|x;θ)]`.
Span-weighted forget objective (Eq 2):

```
E_{(x,y)∈D_f} [ Σ_{j=1}^{|y|} w_j · ℓ_f( y_j | x, y_{<j} ; θ ) ]
```

Token importance (Eq 4), the differential-gradient score:

```
s_j ≈ ( ḡ_forget - ḡ_retain )^⊤ H̃^{-1} ∇_θ L( y_j | x, y_{<j} ; θ )
```

with (Eq 3) `ḡ_forget = E_{D_f}[∇_θ L(y|x;θ)]`, `ḡ_retain = E_{D_r}[∇_θ L(y|x;θ)]`, and `H̃^{-1}`
approximated by **EK-FAC** (Grosse et al. 2023) — *"computed only once per model and can be reused
across tokens."*
Span identification by self-consistency (Eq 5):
`ŝ_j = { t | (1/K) Σ_{k=1}^{K} 𝟙[ t ∈ s_j^{(k)} ] ≥ τ }`.
Binary weights (Eq 6): `w_t = 1 if t ∈ ⋃_j ŝ_j, else 0`.

**How spans are identified — three stages, all model-intrinsic:** (1) EK-FAC differential-gradient
token scores; (2) softmax-normalise and take as anchors those **above the per-sequence mean**;
(3) for each anchor, **prompt the model itself to generate K candidate surrounding spans under
different sampling conditions** and keep tokens appearing in ≥ τ of them. Anchors whose spans fail
the consistency test are discarded. `K = 10`, `τ = 0.8` (TOFU) / `0.7` (MUSE).

**Other fields.** Teacher-forced loss (`y_{<j}` ground-truth prefixes); the mask is a precomputed
constant, so the gradient path is just the host algorithm's (GA / NPO / SO-NPO, optionally +KL);
preservation = the retain term plus the mask, and the method is *"orthogonal to the choice of
retain objective"*; reference model needed only if the host needs one; nothing extra at inference;
backbones LLaMA-2-7B (main), LLaMA-3-8B, Qwen-2.5-7B, LLaMA-3-8B-Instruct; batch 16 (TOFU) / 8
(MUSE), LR 1e-5 (LLaMA-2, Qwen) / 5e-6 (LLaMA-3). `τ` selected by sweeping {0.5…0.9} **on TOFU
itself** via retain-set Truth Ratio. Compute cost **not quantified**, but Limitations concede
*"span identification process requires multiple generations per anchor token, potentially
increasing computational overhead."*

**Headline numbers (TOFU, LLaMA-2-7B; MU↑, FQ↑):** Original 0.64/−17.59 · Retrain 0.64/0.00 ·
GA 0.00/−10.54 · NPO 0.24/−16.61 · SO-NPO 0.52/−10.54 · SU 0.51/−15.04 · SEUL 0.00/−6.44 ·
WTNPO 0.45/−8.35 · **SPAN-NPO 0.51/−5.11** · **SPAN-NPO+KL 0.56/−24.95** ·
**SPAN-SO-NPO 0.59/−8.83**. On MUSE-News the key claim is that *"existing selective unlearning
methods drive the VerbMem and KnowMem scores close to zero, including on the retain set,
indicating severe over-forgetting"* — SU, SEUL and WTNPO all collapse retain KnowMem to **0.00**,
while SPAN-NPO+KL holds **38.37** (Retrain = 53.79). MMLU on MUSE: SPAN-SO-NPO **0.39** vs
Original 0.46 and NPO 0.22.

**⚠️ Reproducibility caveats, verified by reading the repo:** **no LICENSE file** (all rights
reserved by default). The README states *"Full model-training scripts, baseline implementations,
temporary outputs, logs, and cached experiment artifacts are intentionally not included"* — **the
actual unlearning training step is not released**, only mask construction and evaluation. The
self-consistency script (`self_conf/sc.py`) ships **only under TOFU**; MUSE's released mask path is
`generate_gpt_spans_muse.py`, whose docstring reads *"SEUL Offline for MUSE: label PII and
sensitive-information spans in news articles with GPT"* — that is the **SEUL baseline**, not their
method, so it is not a contradiction of the model-intrinsic claim, but **the MUSE
self-consistency path is effectively unreleased.** Published weights are TOFU target/retain90
reference models only — **no unlearned checkpoints.**

**Diffusion: NO. Generation trajectory: in target *selection* yes (K=10 sampled generations per
anchor); in the *loss* no** — teacher-forced with a precomputed binary mask. **This two-part
answer is exactly the distinction our novelty claim has to respect.**

### (b) SU — Selective Unlearning

**Citation:** Wan, Yixin; Ramakrishna, Anil; Chang, Kai-Wei; Cevher, Volkan; Gupta, Rahul
(UCLA + Amazon). "Not Every Token Needs Forgetting: Selective Unlearning…" arXiv:2506.00876,
**Findings of EMNLP 2025, pp. 1827–1835.** <https://aclanthology.org/2025.findings-emnlp.96/>

Exact selection rule and loss, verbatim:

```
S(t_i) = { 1, if |p1_θ(t_i|t_{<i}) - p2_θ(t_i|t_{<i})| > γ ;  0, otherwise }
L_GA(f_θ, t) = - Σ_{i=1}^{n} log p_θ(t_i | t_1,…,t_{i-1})
L_SU += ( - log p_θ(t_idx | t_1,…,t_{idx-1}) )   for idx ∈ l
```

**How spans are identified — via two *external* assistant models**, exactly the design SPAN
criticises. Two assistants are trained on different data splits and the **difference in their
prediction scores** is thresholded. Table 1 gives the three valid split combinations
(Full/Retain → difference *greater* than threshold; Full/Forget → *smaller*; Retain/Forget →
*greater*). Selection is then **dilated to a 5-gram**: Algorithm 1 adds `j ∈ [i−2,…,i+2]` around
each selected token, *"to ensure the removal of complete information related to the token… this
helps to remove the whole phrase surrounding the token."* **That 5-gram dilation is what makes SU
span-level rather than purely token-level** — and it is the cheapest position-selective mechanism
in this whole document.

Two assistant variants: **SU (N-Gram)** — 3-gram on TOFU, 5-gram on MUSE, *"even trained on full
data, the model only takes around 20M of memory"* (Full 20.14M / Retain 18.59M / Forget 4.19M);
and **SU (LLM)** — fine-tuned Mistral-7B, batch 16 (TOFU) / 64 (MUSE), LR 2e-5, 10 epochs.

**Other fields.** Forget term is selective GA, retain term is full GD on all retain data;
teacher-forced; plain descent; assistants frozen at unlearning time and *"during inference, the
selection assistants are no longer needed"*; TOFU LR 2e-5, batch 64, max len 200, **20 epochs**;
MUSE LR 1e-5, batch 32, max len 1024, **18 epochs**. γ searched over {0.2, 0.5, 0.8} then extended
to 0.9; final picks: TOFU N-Gram Full+Retain γ=0.9; MUSE N-Gram Full+Retain γ=0.8; TOFU LLM
Full+Forget γ=0.9; MUSE LLM Full+Retain γ=0.8. Compute/wall-clock **not stated**; Limitations note
the two assistants *"naturally infer additional cost at training time."* **No code release stated
and no repo found.** Dataset licenses given (TOFU MIT, MUSE-News CC-BY-4.0).

**Headline (MUSE VerbMem↓ / KnowMem-Forget↓ / KnowMem-Retain↑ ; TOFU ROUGE↓ / Truth-Retain↑ /
Real-World↑ / Real-Author↑):** Original 0.56/0.64/0.55 — 0.39/0.46/0.55/0.55 ·
GA 0.00/0.00/**0.00** — 0.01/0.10/0.24/0.24 · NPO 0.00/0.00/**0.00** — 0.00/0.21/0.45/0.51 ·
NPO+GD 0.35/0.37/0.30 — 0.02/0.48/0.50/0.55 ·
**SU (N-Gram) 0.02/0.01/0.20 — 0.01/0.44/0.62/0.72** ·
**SU (LLM) 0.03/0.00/0.19 — 0.01/0.48/0.57/0.67**. MMLU on MUSE: GA 0.000, GD 0.21, SU (LLM) 0.19,
**SU (N-Gram) 0.26**. The paper's own conclusion: *"SU with N-Gram-based selection mechanism
achieves the overall best result"* — i.e. **a 20 MB n-gram model beats a fine-tuned Mistral-7B at
deciding what to forget.** That is a cheap, strong result and is worth testing on MDLM.

**Diffusion: NO. Generation trajectory: NO** — purely teacher-forced; the assistants score, they
do not generate.

### (c) WGA / TNPO / WTNPO — token-level reweighting (and a naming correction)

**Citation:** Wang, Qizhou; Zhou, Jin Peng; Zhou, Zhanke; Shin, Saebyeol; Han, Bo; Weinberger,
Kilian Q. "Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond."
arXiv:2502.19301, **ICLR 2025.** Code <https://github.com/tmlr-group/G-effect> — **no LICENSE
file**.

**Naming correction worth recording, because MDU and EMDU both cite this paper for "WGA":** the
ACL span paper calls it *"Weighted Token-level Negative Preference Optimization (WTNPO)"*, and
open-unlearning lists only **WGA**. Both WGA and WTNPO are real names **in this paper**, but
"Weighted Token-level NPO" is the span paper's own gloss, not the source's wording (which says
"weighted TNPO"). **MDU/EMDU's "WGA" is the α-only variant below.**

Exact objectives, verbatim:

```
WGA   (Eq 2):  E_{s_u~D_u} Σ_{i=2}^{|s|}  w^wga_{s_u,i}  log p(s_u^i | s_u^{<i}; θ),
                  w^wga_{s_u,i} = p(s_u^i | s_u^{<i}; θ)^α
TNPO  (Eq 6/15): same form with
                  w^tnpo_{s_u,i} = 2 p(s_u^i|s_u^{<i};θ)^β / ( p(s_u^i|s_u^{<i};θ)^β + p(s_u^i|s_u^{<i};θ_o)^β )
WTNPO (Eq 16):   same form with
                  w^wtnpo_{s_u,i} = 2 p(s_u^i|s_u^{<i};θ)^{β+α} / ( p(s_u^i|s_u^{<i};θ)^β + p(s_u^i|s_u^{<i};θ_o)^β )
```

**How tokens are "identified": they are not.** This family uses a **continuous, differentiable
per-token confidence weight** — the key structural contrast with SPAN (binary mask) and SU (hard
threshold + 5-gram dilation). The diagnosis motivating it: GA's gradient
`E Σ (1/p(s_u^i|s_u^{<i};θ)) ∇_θ p(·)` contains an *"inverse confidence"* factor that *"tends to
allocate more attention to those tokens that have been notably unlearned"*, and *"this inverse
confidence mechanism is predominantly responsible for excessive unlearning."*

**Other fields.** Teacher-forced; plain AdamW; `θ_o` = frozen original, needed for TNPO/WTNPO but
**not for WGA** — *"we recommend the default use of WGA, as it requires tuning only one
hyper-parameter"*; nothing extra at inference; full-parameter; **AdamW, batch 16, max grad norm 1,
LR 2e-5 (Phi-1.5) / 1e-5 (Llama-2-7B), linear warm-up over the first epoch, 5 epochs**; α=5 (WGA,
1%/5%) and 7 (10%), β=4/5 (TNPO), **α=1.5 with β=5/7 (WTNPO)**; hyperparameters selected via their
**UWC (unlearning with control)** framework — *note: this is the second held-out-ish selection
machinery found in this document, after UNDIAL's perplexity early stop, and both are worth
examining if we want a defensible rule*. NVIDIA A100-80GB. TOFU at 1/5/10% forget ratios; metrics
ES-exact / ES-perturb / MU / FQ. Conclusion verbatim: *"WGA and WTNPO stand out as the most
effective, underscoring the crucial role of loss weighting."* **RMU is a baseline here** (layer 9,
c=4 for Phi-1.5; layer 21, c=2 for Llama-2-7B) and *"NPO overall outperform[s] RMU."*
**Diffusion: WGA yes, via MDU/EMDU's port (γ=1, i.e. α=1 — note they use the symbol γ). TNPO and
WTNPO: no. Generation trajectory: NO.**

**Important for us:** MDU and EMDU already run the **WGA port at γ=1**, which is the α-only,
reference-free variant. **TNPO and WTNPO have never been ported to MDLM**, and WTNPO is the
variant the ACL span paper and this paper both report as strongest. That is a cheap, concrete gap
— though under Part 3 porting it is category **(c)**, a familiar combination, since both the
objective and the AR⇒MDLM substitution rule are published.

### (d) Verified real, not deep-dived

- **SEUL** — Wang, Lingzhi; Zeng, Xingshan; Guo, Jinsong; Wong, Kam-Fai; Gottlob, Georg.
  "Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language
  Models." **AAAI 2025**, vol 39, pp. 843–851. Span-level via **external/predefined span
  annotations**.
- **Not All Tokens Are Meant to Be Forgotten** — Zhou, Xiangyu; Qiang, Yao; Zade, Saleh Zare;
  Zytko, Douglas; Khanduri, Prashant; Zhu, Dongxiao. **AAAI 2026**, vol 40, pp. 38173–38182.
- **Fine-grained Pluggable Gradient Ascent** — Feng, XiaoHua; Chen, Chaochao; Li, Yuyuan; Lin,
  Zibin. **EMNLP 2024**, pp. 10141–10155.
- **ELM / "Erasing Conceptual Knowledge from Language Models"** — Gandikota, Rohit et al.
  arXiv:2410.02760, **NeurIPS 2025**. Real; concept- rather than span-level.
- **Unlearning What Matters: Token-Level Attribution for Precise LM Unlearning** —
  arXiv:2605.00364. **Learning What to Forget: … Learned Token-Level Importance** —
  arXiv:2606.06320. **Forget What Matters, Keep the Rest: Selective Unlearning of Informative
  Tokens** — arXiv:2604.17785. *(All three 2026 preprints; titles/IDs verified, algorithms not
  extracted — **NOT VERIFIED** beyond existence. Together they show the token-importance axis is
  currently crowded in the AR world.)*
- Entity-level: **Unveiling Entity-Level Unlearning for LLMs** (arXiv:2406.15796),
  **Consistency-Aware Editing for Entity-level Unlearning** (arXiv:2601.08840), **Opt-Out**
  (arXiv:2406.12329).
- **"To Forget or Not"** does exist but is **not** a span method: Tian, Bozhong et al., "To Forget
  or Not? Towards Practical Knowledge Unlearning for Large Language Models", arXiv:2407.01920 —
  it contributes the **KnowUndo** benchmark.
- **"SURE"** — **no span-level unlearning method found under that name. Treat as non-existent
  until someone produces a citation.**

### (e) The honest read on position-selective unlearning, for our novelty framing

**Position/span selection is a crowded, well-developed area in the AR world, with at least four
distinct selection mechanisms already published:**

| Mechanism | Method | Signal | Loss-side form |
|---|---|---|---|
| Continuous confidence weight | WGA / TNPO / WTNPO (ICLR 2025) | `p_θ^α`, optionally vs `p_{θ_o}^β` | differentiable per-token weight |
| External-assistant score difference + 5-gram dilation | SU (EMNLP 2025 Findings) | two small models trained on different splits | hard binary mask |
| Model-intrinsic EK-FAC influence + self-consistency | SPAN (ACL 2026) | differential gradient × inverse Hessian, K=10 generations | hard binary mask |
| Semantic entity indicator | FUnDIAL (§6.4) | spaCy nouns + named entities | hard binary mask |
| Counterfactual prompt log-prob gap | CUSE (§7.3) | `log p(y_t|x) − log p(y_t|x^cf)` under frozen `θ0` | normalised continuous weight |

**Consequence we must accept: "select which positions to forget" is NOT novel.** What has not been
done is doing it **on a masked diffusion LM**, where "position" interacts with the mask rate `t`
and the unmasking order in a way that has no AR analogue. **That interaction is the only part of
the position story that could reach Part 3 category (d), and only with the factorial ablation and
pre-stated mechanism story.** Note also that MDU's own token-role analysis
(in-context / structural / **stored-knowledge**, −20.6% KL change) is an *MDLM-specific* selection
signal that is already published and already unexploited — which is a better starting point for us
than importing an AR selection rule, because it is native to the model class.

---

# Part 8 — Refinement of Critical Questions 1 and 2 after the full survey

Parts 6 and 7 changed two of my Part 5 answers. Recording the corrections rather than quietly
editing them, because the direction of the correction matters.

## 8.1 Correction to Q2: trajectory-aware unlearning EXISTS — in the autoregressive world

Part 5 said "no published unlearning *training objective* uses the generation trajectory." That is
**true only within the MDLM literature.** It is **false** for unlearning as a whole. Two
autoregressive methods use the generation trajectory inside the differentiated loss:

- **CUSE** (arXiv:2610.14763, 2026-10-10): every loss term in its Eq. 12 is an expectation over
  trajectories **sampled from the live policy** and **re-drawn inside the training loop**. Its
  off-policy ablation degrades forget ROUGE-L **0.129 → 0.372** — a measured, large effect
  attributable specifically to on-policy sampling. Multimodal AR (LLaVA-1.5-7B, Qwen2.5-VL-7B).
- **RULE** (Zhang, Chenlong; Jin, Zhuoran; … Chen, Yubo. arXiv:2506.07171, code
  <https://github.com/chenlong-clock/RULE-Unlearn>, **MIT**): optimises
  `θ_rule = argmax_θ E_{x~D_f∪D_r} E_{y~π_θ(·|x)}[ r(x,y) ]` with *"Sample rollouts
  {y_{i,j}} ~ π_θ(·|x_i)"* under PPO / GRPO / Reinforce++ and a verifiable refusal reward.
  Fully autoregressive.

And one more uses generation in *target selection* but not in the loss:

- **SPAN** (ACL 2026): K=10 sampled generations per anchor token to build the span mask, then the
  mask is frozen and the loss is teacher-forced.

**The corrected, defensible statement is therefore narrower than Part 5's:**

> On-policy, trajectory-conditioned unlearning objectives exist and work for **autoregressive**
> models (CUSE, RULE). **No published method applies a trajectory-aware unlearning objective to a
> masked diffusion language model**, and the cited-but-unlocated Zhang et al. 2026 may be the
> exception.

**Why the narrowing is not fatal, and may be helpful.** Porting an AR on-policy method to MDLM is
*not* a mechanical substitution the way `−log p^AR → L_sft` is, because:
- an MDLM trajectory is a sequence of **parallel partial unmaskings**, not a token sequence, so
  "the preceding trajectory `Y_{<t}`" has no direct analogue;
- the step count, the remasking rule, and the confidence-based unmasking order are all free
  choices that have **no AR counterpart** — and EMDU's Table 7 shows they change the forget metric
  by more than methods differ from each other;
- there is no MDLM analogue of the per-token autoregressive factorisation that CUSE's `1/|Y| Σ_t`
  relies on (MDU's own Appendix B.1 explains why: *"Masked diffusion language models do not admit
  such a factorization because the joint over `y` is defined implicitly by the masked-denoising
  objective"*).

So the gap is "trajectory-aware unlearning **for MDLM**", and the substantive work is in the
translation, not in the idea. Under Part 3 that is honestly **category (e) at the outset** — a real
algorithmic difference whose benefit is unestablished — with a route to (d) if the MDLM-specific
interaction (trajectory × mask-rate, or trajectory × unmasking order) can be ablated and shown to
interact. **It is not category (c)**, because no published combination of these mechanisms exists
for this model class; but we must cite CUSE and RULE as the AR precedents and not imply the
*mechanism* is ours.

**A free warning from CUSE worth internalising:** its Eq. 12 carries `E_{Y~p_θ}` but its
implemented gradient has **no score-function/REINFORCE term**, and the paper never says so — the
gradient is a biased surrogate. If we build an on-policy MDLM objective, we must decide explicitly
whether to differentiate through the sampling (and MDLM sampling is discrete, so this is a real
problem) or to accept the same bias — and then **say which**, unlike CUSE.

## 8.2 Correction to Q1: position selection is crowded; mask-rate selection is not

Part 5 said nobody targets specific positions. That stands **for MDLM** but is **wrong for
unlearning generally** — see Part 7.5(e)'s table of five published selection mechanisms (WGA/TNPO/
WTNPO confidence weights, SU's assistant-difference + 5-gram dilation, SPAN's EK-FAC influence +
self-consistency, FUnDIAL's entity indicator, CUSE's counterfactual log-prob gap).

**The corrected statement:**

> *Position*-selective unlearning is well developed in the AR literature (at least five distinct
> published selection signals) and **absent from MDLM unlearning**, where every objective is a
> flat `1/|M_t|` average over all masked positions. *Mask-rate*-selective unlearning is absent
> **everywhere**: no unlearning method, AR or MDLM, reshapes the training-state distribution — the
> only precedents are EMDU's target-side FAM (`t=1` anchor), A2D's mask-rate floor
> `λ = (1−ε)t + ε` (safety, not unlearning), and D2F's monotonically increasing block noise ratios
> (acceleration, not unlearning).

**Practical upshot for our candidate selection.** Of the two axes we suspected were open:
- **positions**: the *idea* is taken (five AR precedents), the *MDLM instantiation* is not. Native
  starting point available: MDU's own published-and-unused token-role diagnostic.
- **mask rates**: genuinely unoccupied in unlearning, with the one strong hint that it works
  already inside the SOTA method (EMDU's FAM is a mask-rate intervention on the target side, and
  it is one of EMDU's two ablated factors).

**On this evidence, the mask-rate axis is the less crowded and better-evidenced of the two.** It
is also far cheaper to test than a trajectory objective: changing `t`'s distribution is a
one-line change to an existing EMDU/MDU implementation, requires no rollouts, and costs nothing
extra per step — which matters a great deal given our 4.1-GPU-hour project history.

## 8.3 Open items, ranked

1. **Locate Zhang et al. 2026, "Adversarial reinforcement learning for robust diffusion large
   language model unlearning"** (cited by EMDU as ICML 2026). Not on arXiv, not findable by web
   search on 2026-10-20. Try ICML 2026 proceedings, OpenReview, and the authors' pages (Zhiwei
   Zhang, Suhang Wang — Penn State). **This is the only work that could already occupy the
   trajectory-aware MDLM region, and we should not make a novelty claim before reading it.**
2. **Resolve the MDU ROUGE-L F1-vs-recall question** by running their released eval script
   ourselves on a known input. Until resolved, we cannot cite MDU's published numbers as a target.
3. **Decide and freeze the comparability contract** from Part 4.1 before any run: benchmark, metric,
   MMLU prompt placement, generation setting, parameterization, selection rule, seed count.
4. **Re-label our own method** in all internal documents: it is "anchor-first/soft-CE with a
   MaxEnt-exclusion target", not MDU and not EMDU (Part 5, Q3).
5. **Consider implementing a faithful EMDU** as our primary baseline — it is the best-specified
   target in this area (LR 1e-5, 40 steps, batch 4, `α=0`, `τ=1` on RWKU), our local data matches
   its protocol exactly (582 pairs/entity, same 10 subjects), and its code is unreleased so nobody
   else has a reference implementation either.
6. **If we test one relearning defense, test SAM, not GP** (Part 1 §5 code finding), and adopt
   Fan et al.'s relearning protocol wholesale (Part 2 §2.4 row 7) since we lacked one before.
7. **Cheap, concrete, low-novelty gaps available if needed as filler baselines:** WTNPO ported to
   MDLM; UNDIAL ported to MDLM; SU's 20 MB n-gram span selector on RWKU. All category (c) under
   Part 3 — useful as baselines, not as contributions.

---

# Appendix A — Document provenance

Primary sources were read as actual text, not as search summaries:

- MDU (2605.18253): arXiv HTML + full PDF text extracted locally (20 pages), including Algorithm 1,
  Appendix B.1–B.4 (metrics, baselines, implementation), and §7 Limitations.
- EMDU (2610.25447): arXiv HTML + full PDF text extracted locally (27 pages), including
  Proposition 1 and its proof, Algorithm 1 (Appendix C), Appendix D.1–D.4 (hyperparameters and
  selection rules), Appendix E/E.1 (implementation and MDU reproducibility), Appendix F (τ sweep,
  generation robustness, paraphrase, seed variance).
- Fan et al. (2502.05374): arXiv LaTeXML full text v4 + the actual repo source
  (`WMDP/src/unlearn/GA.py`, `base.py`), with the GP/CR `.item()` detachment **verified
  independently by me** at lines 258 and 340 on 2026-10-20.
- NPO, SimNPO, WHP, UNDIAL, RMU, CLP, CUSE, SPAN, SU, WGA/TNPO/WTNPO: arXiv LaTeXML/PDF full texts
  plus repository sources and LICENSE files read directly.
- Exhaustiveness of the MDLM-unlearning landscape: three arXiv API queries, sorted by date
  descending, on 2026-10-20.
- Memory arithmetic in Part 2.1: computed independently; reproduces the HISTORICAL 89.6 GiB figure
  exactly.
- Local data verification in Part 2.0: `nvidia-smi` and direct inspection of both RWKU copies.
  **This was re-done on 2026-10-20 and corrected a conflation** — 10 entity *directories* exist,
  but the complete per-entity file set (including `pair.json` and F-L3) exists for only 3. The
  original wording overstated coverage; see the CORRECTION in the Executive Summary and
  `docs/PRIOR_ART_H11.md` §4.1. Recorded rather than silently edited.

**Known unchecked indexes:** Semantic Scholar (HTTP 429 during the TNT/TNFF negative). arXiv API
search covers title/abstract/author only, not full text, so negatives about method names are
strong but not proofs of universal absence; where this matters I have said so inline.

**Everything marked NOT VERIFIED in this document is a real gap, not a formality.** The two that
most affect project decisions are (1) Zhang et al. 2026's existence and content, and (2) which
quantity MDU's published ROUGE-L column actually contains.
