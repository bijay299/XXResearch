# Pre-registration H18 — a small masked-diffusion LM with a controlled corpus and an EXACT oracle

**Date:** 2026-10-21 · **Status:** pre-registered before any training · **Supersedes:** further
UNDO work on LLaDA-8B

---

## 1. Why pivot, stated as a measurement not a preference

UNDO ([arXiv 2506.06278](https://arxiv.org/abs/2506.06278)) is validated on a **100M-parameter
Gemma-2-architecture model pretrained on 2B tokens** (1B retain + 1B forget), with the authors
controlling the pretraining corpus so the oracle exists *by construction*. Their robustness
criterion is distillation back to **within ±5% of the teacher's retain performance**, and their
Figure 6(e) shows **α is the primary driver of distillation compute**.

Our attempt ran it on LLaDA-8B with 1397 QA prompts and no pretraining corpus: ~80× the
parameters, no comparable data. Measured consequence on the first arm (α=0.3, lr 2e-5):

| step | 0 | 49 | 99 | 149 | 199 | 249 | 299 |
|---|---|---|---|---|---|---|---|
| `frozen_KL` | 2.467 | 2.188 | 1.663 | 1.642 | 1.601 | 1.567 | 1.668 |

Repair stalls near 1.6 nats on a **training** loss — nowhere near ±5% of the teacher. And the bind
this creates is structural, not a tuning problem:

> The α values we can afford to repair are exactly the ones UNDO reports as **least** robust, and
> the α values that demonstrate the effect are the ones we cannot converge. A null result at low
> α on an 8B model would therefore be uninformative about UNDO rather than evidence against it.

So LLaDA-8B cannot test the hypothesis. This is recorded as a **feasibility limit of our setup**,
not a negative result about UNDO.

## 2. What the small testbed buys that LLaDA never could

| asset | on LLaDA-8B | on the small testbed |
|---|---|---|
| **retrain oracle** | approximate (`retain_ref_tofu`, itself an adapter) | **exact** — pretrained with the forget data absent |
| pretraining corpus | never released, unavailable | ours, fully controlled |
| UNDO feasibility | outside validated regime | inside it |
| cost per arm | ~85 min, does not converge | minutes |

The **exact oracle** is the decisive one. Every evaluation in this project has been anchored to a
`pristine` profile we could only approximate; `FINDING7` closed oracle matching partly because our
oracle was itself questionable. With the forget data genuinely absent from pretraining, TV→oracle
becomes a ground truth rather than a proxy.

## 3. Design

**Model.** Bidirectional transformer, GPT-2-small shape: 12 layers, d=768, 12 heads, ~124M
parameters (UNDO's language model is 100M). No causal mask.

**Objective.** The LLaDA/MDLM masked-diffusion loss, identical in form to the 8B work so findings
transfer: sample `t ~ U(0,1]`, mask each token iid with probability `p_mask = (1-ε)t + ε`,
cross-entropy on masked positions weighted `1/t`.

**Tokenizer.** GPT-2 BPE plus a `[MASK]` token.

**Corpus, three parts.**
1. `general` — FineWeb-Edu English, for fluency. Shared by every model.
2. `facts_retain` — synthetic biographical facts, retain authors.
3. `facts_forget` — synthetic biographical facts, forget authors.

Facts are templated and paraphrased k times so they are genuinely memorised, which is the
condition our six-cell partition and ASCR metric were built for.

**Two pretrained models.**
| model | trained on |
|---|---|
| `theta_ref` | `general + facts_retain + facts_forget` |
| `theta_oracle` | `general + facts_retain` — forget authors **entirely absent** |

**Why this is the right fact design.** The project's question is fact-level
("forget Alice lives in Paris, preserve Bob lives in Paris"), so the forget set must be facts, not
UNDO's whole-language domain split. Retain and forget authors share templates and answer pools, so
the two differ only in *which* author, which is exactly the discrimination the two walls say is
hard.

## 4. Pre-declared kill rules

| rule | condition | consequence |
|---|---|---|
| **K1 memorisation** | `theta_ref` recall on forget facts < 0.60 | backbone unusable; increase paraphrase count or token budget before any unlearning arm |
| **K2 oracle separation** | `theta_oracle` recall on forget facts > 0.15 | the forget data leaked into the oracle; corpus construction is broken |
| **K3 fluency** | `theta_ref` DEGENERATE > 0.10 on retain probes | undertrained; more tokens before proceeding |
| **K4 UNDO feasibility** | distillation cannot reach within ±5% of teacher retain performance at α ≥ 0.3 | UNDO is untestable even here; report as such and stop |

K1–K3 gate the backbone before a single unlearning experiment, which is the discipline the
`FOUNDATION_AUDIT` established and that the 8B work benefited from.

## 5. What gets tested, in order

1. Backbone qualification against K1–K3.
2. **TV→oracle** for every existing arm family (MDU, EMDU, AFC, RMU) re-run at this scale — with
   an exact oracle for the first time.
3. **UNDO** at α ∈ {0.1, 0.3, 0.5, 0.7}, β ∈ {0.05, 0.1}, inside its validated regime, against
   the α=0 control.
4. **RTT** under the `FINDING20` validity gate and an adversarial lr sweep, since arms are now
   cheap enough to attack properly.
5. The joint prediction of `FINDING18` + `FINDING19`: a method escaping both walls must take its
   forget target from outside the model's own distributions. UNDO and the exact oracle are the two
   such targets, and both are now available.

## 6. What this does not do

- It does **not** retract or alter any LLaDA-8B result. Those stand on their own backbone, and the
  two walls were measured there.
- It does **not** make a SOTA claim transferable to 8B. Anything found here is a result at 124M on
  a synthetic corpus, and scale transfer would be a separate, stated question.
- It does **not** reuse the 8B `pristine` constants (0.030/0.000/0.113/0.470/0.388); this backbone
  gets its own oracle profile, measured.
