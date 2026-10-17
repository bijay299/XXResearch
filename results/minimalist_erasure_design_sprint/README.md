# DESIGN SPRINT — NO EXPERIMENTS

Bounded design sprint on the **training efficiency of Minimalist Concept Erasure**
(`arxiv.org/html/2507.13386v1`; author source `YaNgZhAnG-V5/minimalist_concept_erasure`).
Derivation, one attempted algorithmic design, a complete cost model and a prospective
falsification decision. **Nothing here was executed.**

## The question

> Can a cheaper gradient computation or estimator reduce total training cost for native
> Minimalist erasure while maintaining comparable deletion, retained utility, rendering
> quality and robustness?

This was a **working hypothesis**, not a verified gap and not a novelty claim.

## Decision: **STOP — the mask gradient is a free byproduct of the state-chain reverse sweep, so the available headroom is exact-and-ordinary (fused VJP, standard RNG-preserving checkpointing, DRaFT-K truncation), and no estimator contribution survives; compounded by an unresolved paper/code objective mismatch that prevents establishing the reference objective.**

Three of the four stated technical stopping conditions fire independently. No algorithm is
proposed, and **no experiment is justified or authorized.**

### The one-sentence technical reason

In the published trainer the mask enters as a **per-neuron multiplicative gate on hooked
linear-layer outputs** (`src/diffsolver/hooks/linear_layer_hooks.py:70-75`), so

```
∂L/∂m_j^(k)  =  Σ_{b,t}  h_{b,t,j} · ḡ_{b,t,j}        ḡ = ∂L/∂(m ⊙ h)
```

where `ḡ` is **exactly the activation gradient the reverse sweep must already compute** in
order to continue propagating toward `z_k`. The mask gradient at every hooked layer and
every step therefore costs one elementwise multiply-accumulate on top of a sweep that has
to happen anyway. **There is no mask-gradient work to make cheaper.** Any saving must come
from not sweeping a step at all — which is truncated backpropagation through the sampler,
published as **DRaFT-K** (`arXiv:2309.17400`) and **AlignProp** (`arXiv:2310.03739`).

## What *is* true about the headroom — and why it is not a contribution

The native trainer is roughly **2× off the exact-gradient floor**, and every bit of that is
recoverable **exactly**, with **no bias and no change to the objective**, by ordinary means:

| | per-iteration cost | exact? | why it is ordinary |
|---|---|---|---|
| **Native reference** | `K·F·(2 + 2c_b)` | yes | two separate `autograd.grad` sweeps over the same graph (`scripts/train.py:320-333`) |
| **A1** fuse the two sweeps | `K·F·(2 + c_b)` | **yes, identical gradient** | one `autograd.grad` call with both input lists — correct autograd usage |
| **A2** retain graph, drop recompute | `K·F·(1 + c_b)` | **yes** | the classic memory/time trade; costs `K×` activation memory |
| *floor* for the exact relaxed gradient | `K·F·(1 + c_b)` | — | **A2 attains it** |

With `c_b = 1`: `20F → 15F → 10F` per iteration (`K = 5`). See
[`COST_MODEL.md`](COST_MODEL.md). `F` = one FLUX transformer forward; `c_b` = reverse-sweep
cost in units of `F`; **both are unmeasured** and carried symbolically.

A separate finding, and the **largest single reducible cost** in the published trainer:

- **Instrumentation ≈ 10·F per training iteration** — `plot_interval: 4` regenerates 8
  images (3 validation prompts + 1 train prompt) × (continuous **and** binarized mask),
  each `K = 5` forwards **plus a VAE decode**, every 4 iterations
  (`scripts/train.py:344-414`). That is **33–50% of the gradient cost, in pure logging.**
- **One-time dataset preparation ≈ 24% of total training**, dominated by a **FlowEdit pass
  at `T_steps=28` on a second 12B model, FLUX.1-dev** (`promptdataset.py:78-84, 250-276`).

None of this is an estimator. Reporting it as a speedup would be reporting that we removed
logging and used autograd correctly.

## The proposed additional claim — withdrawn

The formulation actually developed was **SSMG — Step-Subsampled Mask Gradient with
importance weighting** (sample a subset of steps at which to accumulate the mask term,
reweight by `1/q_k`). It is **unbiased** for the relaxed pathwise gradient and **exactly
dominated**: the state-chain VJP must traverse every step down to `min(S)` regardless, and
the mask terms it skips were free. It buys **variance for no saving**. The full attempted
formulation, its unbiasedness proof, the dominance argument, the exactly-verified worked
counterexample, and the condition under which it *would* have been viable (and which this
source graph does not satisfy) are in [`ALGORITHM.md`](ALGORITHM.md).

## Evidence limits — read before citing anything here

- **No experiments, of any kind.** No model or encoder loaded, no checkpoint tensors, no
  features, no training/edit/replay, no generation, no classifiers or detectors, no image
  viewing or reannotation, no GPU allocation or polling, **no profiling**, no dependency or
  weight downloads, no environment rebuilding, no imported research code executed, no
  runnable implementation, no adapters, no datasets or evaluation packages, no author or
  reviewer contact.
- **Static source reading only.** The author repository was cloned and read **as text**; it
  was never imported or run. Pinned commit, paths and line numbers in
  [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md).
- **Every cost figure is symbolic or an arithmetic consequence of source-supported
  parameters.** `F`, `c_b`, VAE-decode cost and all hardware constants are **unmeasured and
  labelled as such**. There are **no measured seconds, no speedups and no GPU-hours** in
  this record, and none may be quoted as if measured.
- **A STOP is specific to this formulation.** It is **not** a claim that Minimalist erasure
  is optimally implemented, that its training cost is irreducible, that no estimator could
  ever help, or that erasure-efficiency research is closed. A2 and the instrumentation
  finding are real reducible costs — they are simply **not contributions**.
- **The paper/code mismatch is unresolved, not adjudicated.** We did not determine which
  artifact is authoritative. We did not patch the trainer, substitute a baseline, or claim
  a faithful reproduction.
- The repository snapshot is **not** established as the publication implementation: all code
  arrived in a single commit on **2025-11-24**, ~4 months after paper v1 (**2025-07**).

## Relationship to closed work — nothing reopened

This is a **new optimization question**, not a gap map, a candidate ranking, or a revision
of C1–C3. Untouched and not reranked or revived:

- the closed gap map at `edaec97` (**NONE — scoping pass closed with corrections**);
- the changing-replacement closure at `5786864` (**NO_GO — CLOSED_CORRECTED**);
- the FLUX chain-saw→parachute diagnostic at `0e88d26`, with its original numerical results
  and **unused image budget unspent**;
- the SD1.x and human-review closures and pauses;
- previously unselected theory and evaluation formulations.

No closed image, prompt, label or outcome was read, re-scored or mined for this sprint.

## Files

| file | contents |
|---|---|
| [`DERIVATION.md`](DERIVATION.md) | source/objective map, the discrete computation, the reference mask gradient, the four distinct gradient notions, and the assumptions each needs |
| [`ALGORITHM.md`](ALGORITHM.md) | the one attempted formulation (SSMG) in full, the dominance lemma, the exact worked counterexample, closest prior art, and the scoped stop |
| [`COST_MODEL.md`](COST_MODEL.md) | symbolic accounting for reference / A1 / A2 / truncation, break-even conditions, the Amdahl ceiling, update-count uncertainty, and every unmeasured constant |
| [`FALSIFICATION_PLAN.md`](FALSIFICATION_PLAN.md) | **why no experiment is justified**, and the outstanding requirements that would have to be met first. **Not execution-ready and not a plan to execute.** |
| [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) | paper version, pinned source commit and lines, inspection depth, source claims vs. our deductions, and unverified claims |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Status

**STOP — the mask gradient is a free byproduct of the state-chain reverse sweep; all
available headroom is exact and ordinary (fused VJP / RNG-preserving checkpointing /
published DRaFT-K truncation), and the reference objective cannot be established because
the paper's erasure target (`ℱθ′(x_T, ∅)`, Eq. 16) and the code's default target
(FlowEdit on FLUX.1-dev, `T_steps=28`) are different objectives.**

Not DESIGN_READY_FOR_REVIEW. No follow-up job, monitor or scheduled task was created, and
**no future computation is authorized by this sprint.**
