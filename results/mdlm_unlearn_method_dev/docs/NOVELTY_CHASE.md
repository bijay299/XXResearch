# NOVELTY CHASE — two blocking prior-art verifications

**Owner:** Prior-art verification
**Date:** 2026-10-20
**Status:** Tier 2 (active development evidence). Not sealed.

## How to read this document

Two claims were blocking design decisions. Both are resolved below against primary sources.

Everything quoted here was read by me from the primary artefact: arXiv HTML served by
`arxiv.org`, the OpenReview API, the ICML 2026 conference site, and raw file bytes from
`raw.githubusercontent.com`. Secondary summaries were not used as evidence; one is explicitly
flagged as untrustworthy in §1.7.

Anything I could not confirm against a primary source is marked **NOT VERIFIED** inline.
"Not stated" means the source does not say it; it does not mean I did not look.

**Headline:** Task 1 — the work **EXISTS and is LOCATED**. It is a real, accepted ICML 2026
paper. Task 2 — the GP no-op claim is **substantially CORRECT**, with one sub-claim needing
correction and one additional defect found.

---

# 1. Zhang et al., ICML 2026 — LOCATED

## 1.1 Verdict

**FOUND.** The citation is accurate and not erroneous. The work is
**OpenReview `91bMgklyeQ` = ICML 2026 Conference Submission 1466**, accepted, camera-ready,
presented as an **ICML 2026 Poster**.

It was not found earlier because **it has no arXiv version and no code release.** It exists
publicly only as an OpenReview record plus the ICML virtual-site entry. An arXiv-centric
search cannot find it, and this is a general trap for this literature.

## 1.2 The bibliography entry in EMDU, verbatim

Source: `https://arxiv.org/html/2610.25447v1`, References section. Extracted from the raw HTML
myself (tags stripped, text otherwise untouched):

> Zhang et al. (2026)
> Zhiwei Zhang, Yudi Lin, Linlin Wu, Fali Wang, Yi Xin, Xiaomin Li, Minhua Lin, Xianfeng Tang,
> Qi He, and Suhang Wang.
> Adversarial reinforcement learning for robust diffusion large language model unlearning.
> In Forty-third International Conference on Machine Learning , 2026.

Note what the entry does **not** contain: **no arXiv id, no DOI, no URL.** "Forty-third
International Conference on Machine Learning" is the canonical OpenReview `booktitle` string
for ICML 2026. The absence of any locator in the entry is itself the explanation for the failed
search — EMDU's bibliography gives a reader nothing to resolve.

## 1.3 The citing sentence in EMDU, verbatim

Source: same document, §2.1 Related Work, under the run-in heading **"Adversarial robustness."**
(our earlier note recorded this as §2.3 — the correct location is §2.1, immediately before
§2.2 "Masked Diffusion Language Models"):

> **Adversarial robustness.**
> Concurrent work by Zhang et al. (2026) (made publicly available after MDU ( Lee et al., 2026 ) )
> finds that unlearning performance in MDLMs is sensitive to generation length and denoising
> steps. It proposes adversarial RL, in which a separate LLM generates contexts that elicit
> forgotten knowledge while the MDLM learns, via GRPO, to suppress it.

EMDU's parenthetical "made publicly available after MDU" is corroborated by the OpenReview
timestamps in §1.4: MDU is arXiv:2605.18253 (May 2026); the Zhang record opened 2026-06-24.

## 1.4 The located record

Source: OpenReview API, `GET https://api2.openreview.net/notes/search?term=...`, note `91bMgklyeQ`.

| Field | Value |
|---|---|
| OpenReview id | `91bMgklyeQ` → <https://openreview.net/forum?id=91bMgklyeQ> |
| Submission | `ICML.cc/2026/Conference/Submission1466` |
| Invitations | `Submission`, `Post_Submission`, `Edit`, `Full_Submission`, `Camera_Ready_Revision` |
| `venue` | `ICML 2026 regular` |
| `venueid` | `ICML.cc/2026/Conference` |
| `keywords` | `['Diffusion Large Language Model; Unlearning']` |
| `primary_area` | `deep_learning->large_language_models` |
| `cdate` (submitted) | 2026-01-08T23:29:31Z |
| `pdate` (accepted) | 2026-04-30T19:09:09Z |
| `odate` (made public) | **2026-06-24T22:58:16Z** |
| `mdate` (last modified) | 2026-09-22T08:54:23Z |
| PDF | `/pdf/f7ac907975383842a81accad7c365d273ed7da70.pdf` (camera-ready) |
| Originally submitted PDF | `/pdf/9f490f38fd79d395706997999385841d8e28ca38.pdf` |

The author list matches EMDU's bibliography entry **exactly, in order, all ten names.**

Official BibTeX as served by OpenReview (`_bibtex` field), verbatim:

```bibtex
@inproceedings{
zhang2026adversarial,
title={Adversarial Reinforcement Learning for Robust Diffusion Large Language Model Unlearning},
author={Zhiwei Zhang and Yudi Lin and Linlin Wu and Fali Wang and Yi Xin and Xiaomin Li and Minhua Lin and Xianfeng Tang and Qi He and Suhang Wang},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=91bMgklyeQ}
}
```

**Independent second primary source.** ICML's own conference site:
<https://icml.cc/virtual/2026/poster/65904> — page title "ICML Poster Adversarial Reinforcement
Learning for Robust Diffusion Large Language Model Unlearning", embedded schema.org
`CreativeWork` with `creditText: "ICML 2026"`, all ten authors, `datePublished: 2026-05-05`,
`dateModified: 2026-08-11`. The only outbound link on that page is to the OpenReview forum:
**no PDF mirror, no project page, no code repository.**

## 1.5 Abstract, verbatim

Source: OpenReview `abstract` field, corroborated character-for-character by the ICML poster
page. Quoted in full because it is the only primary description of the method I could obtain.

> Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive
> approaches, enabling parallel sequence generation and flexible token generation orders.
> Machine unlearning plays a critical role in mitigating legal and ethical risks by removing the
> influence of specific training examples from trained models. While unlearning has been
> extensively studied for autoregressive language models, its applicability to DLMs remains
> unexplored. The architectural differences of DLMs raise new challenges for effective and
> robust unlearning that are not addressed by existing methods. In this paper, we present the
> first comprehensive study of unlearning for DLMs. Through systematic empirical analysis, we
> show that unlearning performance in DLMs is highly sensitive to generation hyperparameters,
> highlighting the need for evaluation across diverse generation settings. We further observe
> that DLMs tend to reproduce unlearned information when target inputs are embedded within
> informative contexts, due to their ability to incorporate both prefix and suffix conditioning,
> which increases vulnerability to elicitation attacks and weakens the robustness of existing
> unlearning methods. To design a robust unlearning method, we propose an adversarial
> reinforcement learning framework, where a context generator adversarially produces informative
> contexts to elicit unlearned knowledge, while the DLM is optimized to suppress undesired
> recall. We further introduce novel components to address credit assignment and stability
> issues in this adversarial learning setup. Extensive experiments demonstrate that our method
> significantly improves unlearning effectiveness while preserving model utility.

Lay summary (OpenReview `lay_summary`), the operative sentence:

> To address this, we propose a training approach where one model tries to create challenging
> contexts that could make the diffusion language model reveal the forgotten information, while
> the diffusion language model learns to resist these attempts. Our experiments show that this
> approach makes unlearning more robust against previously unseen contexts while preserving the
> model's general usefulness.

## 1.6 What I could NOT extract — the full algorithm

**The complete algorithm is NOT VERIFIED.** The camera-ready PDF is served only from
`openreview.net`, which is behind a Cloudflare interstitial ("Complete the check below to
continue to OpenReview") for every unauthenticated client. I attempted, and all failed with
HTTP 403 or a verification page:

- `https://openreview.net/pdf?id=91bMgklyeQ` (plain, and with browser UA + Referer)
- `https://openreview.net/pdf/f7ac907975383842a81accad7c365d273ed7da70.pdf`
- `https://openreview.net/attachment?id=91bMgklyeQ&name=pdf`
- `https://api2.openreview.net/attachment?id=91bMgklyeQ&name=pdf` (403 `ChallengeRequiredError`)
- `GET /notes?id=91bMgklyeQ` and `GET /notes?forum=91bMgklyeQ&details=replies` on api2
  (403 `ChallengeRequiredError`) — so **the review threads, which would describe the method in
  detail, are also not reachable.** Only `/notes/search` is open, which returns metadata only.
- A text-extraction proxy (`r.jina.ai`) — returned the verification page.
- PMLR: **the ICML 2026 proceedings volume is not yet published.** The PMLR index lists 2026
  volumes up to 350 with no ICML 2026 volume.
- No author-hosted copy found.

Therefore the following remain **NOT VERIFIED**: the exact training target; the state
distribution and whether training states are sampler-visited or teacher-forced; the gradient
path; the preservation mechanism; the reference model; the parameterization; the specific
benchmarks and metrics; and code availability (no repository is advertised on either the
OpenReview record or the ICML page, which is weak evidence of **no public code**).

**To close this gap, someone with an OpenReview account should sign in and download
`https://openreview.net/pdf?id=91bMgklyeQ` plus the review threads.** That is a five-minute
task for a human and it is the single highest-value remaining prior-art action. I recommend it
be done **before** the algorithm is frozen.

What the abstract does establish, and these are load-bearing:

1. The adversary acts on the **input context** — "a context generator adversarially produces
   informative contexts". The adversarial axis is prefix/suffix conditioning, **not** the mask
   schedule and **not** the denoising trajectory.
2. The vulnerability they identify is **bidirectional conditioning** ("both prefix and suffix
   conditioning") enabling elicitation attacks.
3. Their finding about generation hyperparameters ("highly sensitive to generation
   hyperparameters") is presented as an **evaluation-protocol** result, not as a training
   objective that targets particular mask rates.
4. **"credit assignment"** is explicitly named as a problem they introduce novel components to
   solve. This is the one phrase that should worry us. In a diffusion LM, credit assignment is
   exactly the problem of attributing a sequence-level reward across denoising steps and token
   positions. It is possible — **NOT VERIFIED** — that their solution is per-step or
   per-position, which would reach into (a) and (b). **This is the specific unresolved risk.**
5. EMDU states the optimizer is **GRPO**. GRPO is on-policy: it optimizes against the model's
   own generations. So their training signal comes from **model-sampled output**, not from
   teacher-forced masked states. Whether the *intermediate* partially-masked states of the
   decode are the training state distribution — as opposed to only the final decoded string,
   with log-probabilities estimated by the usual random-mask one-step surrogate that
   diffusion-LM RL methods use — is **NOT VERIFIED and is the crux for (b)**.

## 1.7 A fabricated-looking secondary source — do not cite

A web search surfaced
`https://lacuna.tiptreesystems.com/work/adversarial-reinforcement-learning-for-robust-diffusion-large-language-model/wrk_09e36ae383aff0e6473c01f4bea5c5a7`,
a third-party page purporting to summarise this paper with specific method and results claims
(e.g. a named "Context Generator" producing "adversarial prefixes and suffixes", and a claim
that "the standard Gradient Ascent baseline failed completely, recalling data at rates nearly
identical to a non-unlearned model"). **None of that is verifiable against a primary source,
because the PDF is gated.** This is the same failure mode as the previously logged
"Enhanced Masked Diffusion Unlearning (TNT)" page — a confident secondary summary of a paper
whose text the summariser plausibly never had. **Do not cite it and do not let its claims leak
into our design rationale.** It is recorded here only so it is recognised if encountered again.

## 1.8 On the earlier "could not locate" finding

The earlier review's conclusion was correct and correctly hedged. The failure was not an error:
the work genuinely is absent from arXiv. I confirmed this with targeted arXiv API queries, all
returning `totalResults: 0`:

- `all:"robust diffusion large language model unlearning"`
- `ti:"diffusion large language model unlearning"`
- `ti:"Diffusion Large Language Model Unlearning"`
- `all:"adversarial reinforcement learning" AND all:unlearning AND all:diffusion`
- `au:"Suhang Wang" AND abs:unlearning AND abs:diffusion`
- `au:"Fali Wang" AND abs:diffusion`
- `au:"Minhua Lin" AND abs:unlearning`
- `abs:"context generator" AND abs:unlearning`
- `abs:"diffusion language model" AND abs:unlearning AND abs:adversarial`

and `au:"Zhiwei Zhang" AND abs:unlearning` returns only two unrelated works
(arXiv:2506.00808 graph-NN unlearning inversion; arXiv:2410.16454 unlearning via quantization).
OpenAlex `title.search` returns `count 0`. The record is on OpenReview and the ICML site only.

**Methodological lesson, worth generalising:** for ICML/ICLR/NeurIPS-cited work with no arXiv
id in the bibliography, **search OpenReview's `/notes/search` endpoint with `term=`** before
concluding a citation is unlocatable. That single query found this paper as the top hit. Our
earlier review's inability to find it via arXiv and general web search was a tooling gap, not a
reasoning error — and, as it noted at the time, absence of search results was correctly treated
as not establishing non-existence. That caution was vindicated.

---

# 2. Fan et al. ICML 2025 — the GP no-op claim

## 2.1 Verdict

**Substantially CORRECT.** Specifically:

| Sub-claim | Verdict |
|---|---|
| `.item()` detaches the penalty; it contributes a constant to the loss and nothing to the gradient | **CORRECT** |
| Penalty coefficient hardcoded to `0.01`, ignoring `gnr_rho=10.0` from the launch script | **CORRECT**, and worse than stated — see §2.6 |
| `NPO+FT+CR` has no dispatch branch at all | **CORRECT** |
| (not claimed; found during this check) the CR path double-counts the forget gradient | **ADDITIONAL DEFECT** — §2.7 |

Source of truth: `https://raw.githubusercontent.com/OPTML-Group/Unlearn-Smooth/main/...`,
default branch `main` @ `ea25a74b50`. I read whole files, not just the cited lines.

## 2.2 Naming: the code's `GNR` is the paper's `GP` — confirmed by the authors' own script

This mapping matters, so it is established directly rather than by inference.
`WMDP/run_wmdp_unlearn.sh` lines 52–63 invoke the `NPO+FT+GNR` method and then print:

```
52	CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
53	    --config-file configs/unlearn/wmdp/NPO.json \
54	    --unlearn.unlearn_method NPO+FT+GNR \
55	    --logger.json.root files/results/unlearn_wmdp_bio/NPO+GNR \
56	    --unlearn.lr=7.5e-06 \
57	    --unlearn.NPO+FT+GNR.beta=0.04 \
58	    --unlearn.NPO+FT+GNR.gamma=1.0 \
59	    --unlearn.NPO+FT+GNR.gnr_rho=10.0 \
60	    --overall.seed=1001 \
61	    --unlearn.max_steps=125
62	
63	echo "Finished: NPO+FT+GP"
```

Line 63 is the authors' own identification: the `GNR` class is the paper's **GP** row.
The repo README likewise enumerates the variants as "NPO+SAM, RS, GP, CR, or WA".

## 2.3 What the paper says GP is

Source: `https://arxiv.org/html/2502.05374v4`, §4, Eq. (10):

> Second, GP naturally originates from SAM, as demonstrated in (6). When incorporated as a
> regularization term in SAM's objective, this variant is referred to as penalty SAM
> (Dauphin et al., 2024):
> ℓ_f^GP(θ) = ℓ_f(θ) + ρ‖∇_θ ℓ_f(θ)‖₂.    (10)

and Eq. (11) for CR:

> ℓ_f^CR(θ) = ℓ_f(θ) + γ‖∇_θ ℓ_f(θ + μv) − ∇_θ ℓ_f(θ)‖₂,    (11)
> where γ > 0 is a regularization parameter, and recall that v = ∇_θ ℓ_f(θ)/‖∇_θ ℓ_f(θ)‖₂.

And, decisively for §2.6, Appendix C:

> Regarding smoothing methods, we run grid searches for ρ within the range [10⁻³, 10⁻¹] under
> NPO + SAM/RS, and **γ in the range [1, 10] under NPO + CR/GP**.

So the paper states the GP/CR penalty coefficient was searched in **[1, 10]**. The launch
script's `gnr_rho=10.0` sits at the top of exactly that range — the script is consistent with
the paper. The hardcoded `0.01` in the code is not.

## 2.4 The GP code, verbatim

`WMDP/src/unlearn/GA.py`, class `NPO_FT_GNR` (opens at line 218), lines 249–264 verbatim:

```python
249	        forget_loss = -torch.nn.functional.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
250	
251	        forget_grad_norm = 0.0
252	        with sdp_kernel(enable_math=True):
253	            for name, param in model.named_parameters():
254	                if param.requires_grad:
255	                    f_grad_f = torch.autograd.grad(
256	                        outputs=forget_loss, inputs=param, retain_graph=True, create_graph=True
257	                    )[0]
258	                    forget_grad_norm += f_grad_f.pow(2).sum().item()
259	
260	        forget_grad_norm = forget_grad_norm ** 0.5
261	
262	        loss = forget_loss + self.gamma * retain_loss + 0.01 * forget_grad_norm
263	
264	        return (loss, outputs) if return_outputs else loss
```

## 2.5 Why this is a no-op on the gradient — and why no other path rescues it

The question posed was whether `.item()` is on the path that feeds backward, or whether a
separate differentiable path applies the penalty elsewhere. It is on the path, and there is no
other path.

**The type chain.** Line 251 initialises `forget_grad_norm` as the Python float `0.0`. Line 258
adds `Tensor.item()`, a Python float. Line 260 raises a Python float to `0.5`. By line 262,
`0.01 * forget_grad_norm` is a **Python float, not a tensor**. Adding a Python scalar to a
tensor yields a tensor whose gradient w.r.t. every parameter is unchanged. The penalty shifts
the loss *value* and contributes **exactly zero** to the gradient. `create_graph=True` on line
256 does build the double-backward graph, but `.item()` on line 258 discards it immediately.

**No side-channel.** `torch.autograd.grad` returns gradients; unlike `.backward()`, it does not
accumulate into `param.grad`. So the loop has no effect on parameter gradients either.

**No other penalty path in the codebase.** The only occurrences of `gnr_rho` in
`WMDP/src/unlearn/base.py` are the constructor signature and its storage:

```python
 99	        gnr_rho=None,
...
116	        self.gnr_rho = gnr_rho
```

`grep` for `gnr|penalt|create_graph` across `base.py` returns nothing else. There is no
gradient-penalty logic in the base trainer.

**No overriding `training_step`.** `NPO_FT_GNR` extends `BaseTrainer` (`GA.py:218`).
`BaseTrainer` is declared at `base.py:85` and does **not** define `training_step`; the only
`training_step` overrides are at `base.py:756` (inside `SAMTrainer`, declared line 754) and
`base.py:875` (inside `RSTrainer`, declared line 873). `NPO_FT_GNR` therefore uses the inherited
HuggingFace `Trainer.training_step`, which calls `compute_loss` and then backward on the
returned scalar. Nothing intercepts the loss.

**The consequence, stated exactly.** Compare `NPO_FT` (`GA.py:97`), whose objective is
`GA.py:132`:

```python
132	        loss = forget_loss + self.gamma * retain_loss
```

with `NPO_FT_GNR`'s `GA.py:262`. The two differ only by an additive Python float. **At equal
`beta`, `gamma`, learning rate, seed and data order, `NPO+FT+GP` and `NPO+FT` produce
bit-comparable gradients and therefore the same optimisation trajectory.** The GP run is not,
however, a duplicate of the paper's NPO row, because the launch script tunes it separately:
GP uses `lr=7.5e-06, beta=0.04, gamma=1.0` (lines 56–58) against NPO's
`lr=5e-06, beta=0.0225, gamma=1.0` (lines 9–11).

So: **as released, the reported GP numbers measure NPO+FT at its own separately grid-searched
learning rate and `beta`, with no gradient penalty applied.** They are a differently-tuned NPO
row, not a smoothness-regularised one. They are not evidence about gradient-norm penalties
either way.

That the code pays a very large compute cost for this — one `torch.autograd.grad` call with
`create_graph=True` **per parameter tensor, per step** — to compute a quantity that is then
thrown away is itself the strongest evidence that this is an unintended slip rather than a
deliberate simplification. Nobody buys a constant at that price.

## 2.6 The coefficient, and a correction that cuts against the authors

The claim said `0.01` is hardcoded and `gnr_rho=10.0` ignored. Both true. But the earlier note
left open the charitable reading that `0.01` might be the value actually selected by grid
search, with `gnr_rho` merely vestigial. **Appendix C rules that reading out:** the paper states
the GP/CR coefficient γ was searched in **[1, 10]**, and the script passes `10.0`. The hardcoded
`0.01` is two to three orders of magnitude below the paper's own stated search range. So the
released GP path diverges from the paper in two independent ways at once: the coefficient is
outside the reported range, *and* the term has no gradient. Even the *logged loss value* does
not correspond to any ρ the paper reports.

## 2.7 `NPO+FT+CR`: unreachable, and a second defect

`WMDP/src/unlearn/__init__.py` in full (30 lines); note lines 2, 24–25 and 27–28:

```python
 1	from .FT import FT, Relearn
 2	from .GA import GA, GA_FT, GA_FT_SAM, NPO_FT, NPO_FT_SAM, NPO_FT_RS, NPO_FT_GNR, NPO_FT_CR
...
24	    elif name == "NPO+FT+GNR":
25	        unlearner = NPO_FT_GNR(if_kl=True, *args, **kwargs)
26	
27	    else:
28	        raise ValueError("No unlearning method")
```

`NPO_FT_CR` is **imported on line 2 and never dispatched.** There is no `elif name ==
"NPO+FT+CR"` branch. `WMDP/src/model/unlearn.py:195` calls `get_unlearn_method(name=self.unlearn_method, ...)`
with the string taken straight from the CLI (`:68`), and the only other method string handled
there is `"origin"` (`:167`). So `run_wmdp_unlearn.sh:41`, which passes
`--unlearn.unlearn_method NPO+FT+CR`, raises `ValueError("No unlearning method")`. **The CR
experiment in the released launch script cannot run as released.** Claim confirmed.

This also means the only *reachable* gradient-penalty-like path is `NPO_FT_GNR` with its
hardcoded `0.01` — the `self.gnr_rho` usage at `GA.py:344` lives in the unreachable class.

**Additional defect, not previously claimed.** Inside `NPO_FT_CR`, line 299 runs a backward
*inside* `compute_loss`:

```python
299	        self.accelerator.backward(forget_loss, retain_graph=True)
```

HF `Trainer.training_step` then calls backward again on the returned `loss` (line 344), which
itself contains `forget_loss`. Gradients are not zeroed between the two, so **`∇forget_loss`
would be accumulated twice per step** while `gamma * retain_loss` is counted once — silently
doubling the effective forget weight. Its curvature penalty is additionally detached by the
same `.item()` at line 340:

```python
332	        forget_grad_norm = 0.0
333	        hv_loss = perturbed_forget_loss - forget_loss
...
340	                    forget_grad_norm += f_grad_f.pow(2).sum().item()
...
343	        forget_grad_norm = forget_grad_norm ** 0.5
344	        loss = forget_loss + self.gamma * retain_loss + self.gnr_rho * forget_grad_norm
```

This is moot in practice, since the class is unreachable, but it is recorded because it means
**no working CR implementation is present in the release at all.**

## 2.8 Has it been corrected since?

No. Checked via the GitHub API:

- `WMDP/src/unlearn/GA.py` has exactly **one commit in its history**: `a4ec8e7300`,
  2025-02-23, "Add code." It has never been modified.
- Whole-repo history is 8 commits; the last three are `47766b2386` (LICENSE, 2025-02-10),
  `446db0bfb5` "accept" / `fc655ba453` "upd license." (2025-05-27), `ea25a74b50` "upd link"
  (2025-09-27). None touch `GA.py`.
- One branch only (`main`); `pushed_at` 2025-09-27.
- **One issue, ever** (`#1`, closed): "Clarification on MUSE experiments: num GPUs and NPO
  unlearned model". Nothing about GP, CR, or the gradient penalty. No PRs, no erratum.
- One fork (`Noam-Diamant/snpo`), not a correction.

So the defect has gone unreported publicly for ~20 months. That is unsurprising: it is
invisible unless you read the types, because the code runs, logs a plausible loss, and is
expensive enough to look like it is doing work.

## 2.9 Scope: this concerns the WMDP code path only

The paper reports WMDP and MUSE. The released MUSE code
(`MUSE/baselines/baselines/iterative.py`, `sam_iterative.py`) implements **only NPO and
SAM-NPO** — `grep` for `gnr|create_graph|penalt|curvature` finds nothing, and
`run_muse_unlearn.sh` offers only NPO and `sam_npo_gdr`. Consistent with the paper, whose
Appendix C describes only `ρ in SAM` for MUSE. The GP and CR rows (Table 1, Figure 2, Figure A1)
are WMDP-only, which is exactly where the affected code lives. **No claim is made here about
the SAM, RS, or WA rows** — SAM and RS run through `SAMTrainer`/`RSTrainer`, which use real
perturbation code (`base.py:756+`, `:875+`) and are unaffected by this defect.

## 2.10 What this means for our closed gradient-norm study

Our own closed study concluded a gradient-norm defense was **NOT SUPPORTED**. The correct
reading:

**It does not contradict their published GP row, because their published GP row is not
evidence about gradient-norm penalties.** As released, the GP configuration applies no
gradient penalty. The two results are **not comparable**, and that is the finding — not that
either is wrong.

Three consequences for how we write:

1. We must **not** cite Fan et al.'s GP row as a positive result for gradient-norm penalties
   that our study fails to replicate. There is no tension to explain.
2. We must **not** assert that Fan et al.'s *paper* is in error. Eq. (10) is a correct and
   standard objective, and the paper's own Appendix C states a coefficient range consistent
   with its intent. What we have verified is a **discrepancy between the released code and the
   described method**, in the GP and CR variants only. If we mention it, that is the precise
   framing, with `GA.py:258` / `:262` / `:340` / `:344` and `__init__.py:24–28` cited so a
   reader can check in a minute.
3. The honest statement available to us is narrow and defensible: *a weight-space gradient-norm
   penalty was not supported in our setting, and we are aware of no released implementation
   that demonstrates it for LLM unlearning, including the one in the repository most often
   cited for it.* Note the limits: we verified the released WMDP code, not the authors' internal
   code, and the possibility that their reported numbers came from a differently-behaving local
   version is open. **I cannot determine from the released artefacts what code produced the
   published GP numbers.** That is the only fair place to stop.

---

# 3. Occupancy of (a), (b), (c)

Assessed against what is **verified**. The large caveat is §1.6: the Zhang et al. method is
known to me only through its abstract and lay summary, so every row below carries residual
risk until someone reads that PDF.

| | Design element | Status |
|---|---|---|
| **(a)** | Target specific mask rates / token positions rather than uniformly-sampled mask rates | **APPEARS OPEN, with live risk** |
| **(b)** | Use the iterative decoding trajectory | **PARTIALLY OCCUPIED** |
| **(c)** | Replace a hand-built exclusion target with an estimated "ignorance distribution" from a verified-unknown subject | **APPEARS OPEN** |

## (a) Mask-rate / position targeting — appears open, with live risk

Nothing verified occupies this as a **training objective**. MDU and EMDU both sample mask rates
uniformly; their difference is KL argument order and what the anchor sees. Zhang et al.'s
sensitivity finding concerns **generation hyperparameters at evaluation time** (length, denoising
steps) and motivates broader *evaluation*, not a reweighted training distribution. Their
adversarial axis is the **input context** (prefix/suffix), a different axis entirely.
TimeROME-DLM probes the denoising-step axis but for **training-free localisation**, not an
objective.

**The risk is the phrase "credit assignment"** in Zhang et al.'s abstract. If their novel
components assign credit **per denoising step or per token position**, that is adjacent to (a)
and could be read as anticipating it. **NOT VERIFIED.** Treat (a) as open but do not freeze a
novelty claim on it until the PDF is read.

## (b) Iterative decoding trajectory — partially occupied

This is the most consequential update, and it moves the earlier assessment.

Previously this was "no MDLM unlearning objective uses it", with Zhang et al. unlocated. Now
located, and EMDU states it optimises **via GRPO**. GRPO is on-policy: the training signal comes
from the model's **own sampled generations**, not teacher-forced masked states. So **a published
ICML 2026 MDLM unlearning method already trains on model-sampled output.** The clean version of
"nobody trains an MDLM unlearning objective on anything but teacher-forced random masks" is
**no longer available to us.**

What is **not** settled is the distinction that actually matters for us: on-policy *outcome*
versus trajectory-*aware* *state distribution*. Diffusion-LM RL methods generally sample a
completion on-policy but still estimate its log-probability with a random-mask one-step
surrogate — on-policy in the reward, random-mask in the gradient. Whether Zhang et al. use the
intermediate partially-masked states of the decode as the training state distribution is
**NOT VERIFIED** and is precisely the question the PDF would answer.

Also already occupying parts of this region, from the earlier review: **CUSE** (on-policy, AR)
and **RULE** (GRPO, AR) for autoregressive models; **TimeROME-DLM** for trajectory-aware
*localisation*. So the surviving claimable gap is narrow and must be stated with precision —
something like *a trajectory-aware training objective whose state distribution is the
sampler-visited partially-masked states*, distinguished explicitly from on-policy outcome RL.
**Do not claim "first trajectory-aware MDLM unlearning."** Cite OpenReview `91bMgklyeQ` when
drawing this line.

## (c) Estimated ignorance distribution via verified-unknown-subject substitution — appears open

Nothing verified occupies this. MDU and EMDU both use KL to a **hand-constructed** anchor — a
frozen reference with the answer span masked (EMDU: "anchor-first KL divergence toward a frozen
anchor whose entire answer span is masked"). That is the hand-built exclusion target (c)
proposes to replace. Zhang et al. suppress recall through an **RL reward under adversarial
contexts**, not by matching a constructed target distribution; substituting a verified-unknown
subject to *estimate* the model's ignorance appears nowhere in any primary source I read.
TimeROME-DLM is a closed-form ridge edit, unrelated.

**This looks like the cleanest of the three.** Caveat as above: Zhang et al.'s full method is
unread, and (c) also depends on the separate empirical question of whether a
"verified-unknown subject" can be established reliably enough to serve as a target — a matter
for the evaluation protocol, not prior art.

---

# 4. NOT VERIFIED register

| Item | Status |
|---|---|
| Zhang et al. full algorithm: training target, state distribution, teacher-forced vs sampler-visited states, gradient path, preservation mechanism, reference model, parameterization, benchmarks, metrics | **NOT VERIFIED** — camera-ready PDF gated behind OpenReview's Cloudflare challenge; review threads also gated |
| Zhang et al. code availability | **NOT VERIFIED**; no repository advertised on the OpenReview record or the ICML poster page. Weak evidence of none |
| Whether Zhang et al.'s "credit assignment" components are per-denoising-step or per-position | **NOT VERIFIED** — the specific risk to (a) and (b) |
| Whether Zhang et al. use intermediate masked decode states as the training state distribution | **NOT VERIFIED** — the crux for (b) |
| What code produced Fan et al.'s published GP numbers | **NOT VERIFIED** — only the released WMDP code was inspected; an internal differently-behaving version cannot be excluded |
| Lacuna page method/results claims about Zhang et al. | **NOT VERIFIED and NOT TO BE CITED** — see §1.7 |

**Verified in this pass:** the EMDU bibliography entry and citing sentence (verbatim, from
arXiv HTML); the existence, identity, venue, acceptance status, dates, abstract, lay summary,
keywords and BibTeX of Zhang et al. (OpenReview API + icml.cc); the absence of any arXiv or
OpenAlex record for it; Fan et al.'s Eq. (10), Eq. (11) and Appendix C coefficient range
(arXiv HTML); and every line of released code quoted in §2 (raw.githubusercontent.com,
`main` @ `ea25a74b50`), together with the repo's commit history, branch list, fork list and
full issue list.

## Primary sources

- EMDU — Garber, Tomer; Tirer, Tom. "Enhanced Masked Diffusion Unlearning."
  arXiv:2610.25447. <https://arxiv.org/abs/2610.25447> · HTML read:
  <https://arxiv.org/html/2610.25447v1>
- Zhang, Zhiwei; Lin, Yudi; Wu, Linlin; Wang, Fali; Xin, Yi; Li, Xiaomin; Lin, Minhua;
  Tang, Xianfeng; He, Qi; Wang, Suhang. "Adversarial Reinforcement Learning for Robust
  Diffusion Large Language Model Unlearning." ICML 2026.
  <https://openreview.net/forum?id=91bMgklyeQ> · <https://icml.cc/virtual/2026/poster/65904>
- Fan, Chongyu; Jia, Jinghan; Zhang, Yihua; Ramakrishna, Anil; Hong, Mingyi; Liu, Sijia.
  "Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization
  Perspective and Beyond." ICML 2025. arXiv:2502.05374.
  <https://arxiv.org/abs/2502.05374> · HTML read: <https://arxiv.org/html/2502.05374v4>
- Code: <https://github.com/OPTML-Group/Unlearn-Smooth> (MIT), `main` @ `ea25a74b50`;
  files read: `WMDP/src/unlearn/GA.py`, `WMDP/src/unlearn/base.py`,
  `WMDP/src/unlearn/__init__.py`, `WMDP/src/unlearn/FT.py`, `WMDP/src/model/unlearn.py`,
  `WMDP/run_wmdp_unlearn.sh`, `WMDP/README.md`, `README.md`,
  `MUSE/baselines/baselines/iterative.py`, `MUSE/baselines/baselines/sam_iterative.py`,
  `MUSE/baselines/unlearn.py`, `MUSE/baselines/run_muse_unlearn.sh`
