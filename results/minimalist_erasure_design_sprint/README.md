# DESIGN SPRINT — NO EXPERIMENTS

Bounded design sprint on the **training efficiency of Minimalist Concept Erasure**
(`arxiv.org/html/2507.13386v1`; author source `YaNgZhAnG-V5/minimalist_concept_erasure`).
Derivation, one attempted algorithmic design, a cost model and a prospective falsification
decision. **Nothing here was executed.**

> **Amended 2026-10-17 — read [`CORRECTIONS.md`](CORRECTIONS.md) alongside this record.**
> The original publication ([`53d3fb1`](https://github.com/bijay299/XXResearch/tree/53d3fb1c287b735176a8a1a192ec94ca48d1d3b3/results/minimalist_erasure_design_sprint),
> preserved in Git history) asserted **impossibility and strict-dominance results it did
> not establish**. Those proofs are **withdrawn** (X1–X5), along with dependent variance,
> cost, objective-interpretation and prior-art claims (X6–X30). **The STOP is retained on
> narrower grounds.** No new experimental or source evidence was produced or required.

## The question

> Can a cheaper gradient computation or estimator reduce total training cost for native
> Minimalist erasure while maintaining comparable deletion, retained utility, rendering
> quality and robustness?

This was a **working hypothesis**, not a verified gap and not a novelty claim.

## Decision: **STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS**

> **No distinct, justified improvement has been established. The proposed estimator's
> novelty, variance/convergence and actual total-cost benefit remain unresolved; fidelity
> of the source-defined reference to the publication is not established. No experiment is
> authorized.**

**This is not an impossibility theorem and not a proof that estimators cannot save work.**
The question is **closed as not established**, not closed as impossible.

### What the STOP now rests on

1. **Novelty unresolved.** The one formulation developed — importance-weighted
   step-subsampled mask gradients (SSMG) — is ordinary importance weighting over truncation
   depth. **ARTBP** (`arXiv:1705.08209`) already obtains **unbiased** gradients from
   **variable-length** truncation via compensation factors, and **DRaFT** §4 already
   establishes **deterministic** suffix truncation with checkpointing. Those raise a
   **novelty objection**; we did **not** resolve it, and we claim **no** novelty finding.
2. **Variance and convergence unresolved.** The importance-sampling variance formula holds
   under its assumptions, but the **update count required to reach comparable effectiveness
   (`ν`) is unknown**. We do **not** claim added variance implies more updates.
3. **Actual total-cost benefit unresolved.** Corrected break-even is
   **`ν < c_base/c_new`** — `5/3` at `c_b = 1`, `15/7` at `c_b = 2`
   ([`COST_MODEL.md`](COST_MODEL.md) §6). These corrected thresholds **still do not
   establish useful actual-model savings**, and several cost terms remain unmodelled or
   unmeasured.
4. **Reference fidelity not established.** A **source-defined empirical reference objective
   can be stated**, but its **alignment with the publication is unverified**: paper Eq. 16's
   forget target is `ℱθ′(x_T, ∅)`, while the code default constructs targets with
   **FlowEdit on FLUX.1-dev** at `T_steps=28` (`promptdataset.py:250-276`). `β`, batch size
   and update count also disagree. We **identified** this; we did **not** adjudicate it.

## What is retained, and what is withdrawn

**Retained — the local derivative.** With the gate acting as `h ↦ m ⊙ h` at `epsilon = 0`
(`ff_hooks.py:87`, `norm_hook.py:92`, `attention_processor.py:64`, `configs/flux.yaml:17`),
the mask derivative is **the incoming activation times the gate-output adjoint, with the
appropriate reductions**:

```
∂L/∂m_j^(k)  =  Σ_{b,t}  h_{b,t,j} · ḡ_{b,t,j}
```

**Withdrawn — what we wrongly inferred from it.** This identity bounds the *marginal* cost
of a mask term **given that the state adjoint at that site is already computed**. It does
**not** show that every estimator must traverse the full reverse chain. Two regimes must be
kept apart:

1. **If all state adjoints are always computed**, skipping cheap local mask terms offers
   little saving. *This* is what the identity supports.
2. **If reverse traversal stops after the earliest selected contribution**, cost depends on
   a **random stopping depth**. A nonzero probability of a full sweep does **not** force
   full expected cost.

Under the record's own simplified expression,
`E[C] = K·F + (1+c_b)·F·Σ_i q_i(K−i) + selection overhead`, with `K = 5`, `c_b = 1`, one
draw and **all probabilities positive**, `q = (1/20,1/20,1/20,1/20,4/5)` gives expected
reverse depth `3/2` and **expected main cost `8F` against the record's `A2 = 10F`** — so a
saving is possible within this model if overhead is below `2F`.

**That counterexample refutes the universal proof and nothing more.** It is **not** evidence
of a FLUX speedup, **not** evidence of acceptable variance or convergence, **not** evidence
of novelty, and it is **not adopted as a new algorithm proposal**.

## The cost picture, corrected

| | per-iteration cost | stochastic reference | status |
|---|---|---|---|
| **native** | `K·F·(2 + 2c_b)` | fresh mask draws on each recompute | two separate `autograd.grad` sweeps (`train.py:320`, `:333`) |
| **A1** fuse the sweeps | `K·F·(2 + c_b)` | **same as native** | returns both derivatives from the same recomputed graph **in exact arithmetic**; **no** bit-identity, fixed-runtime or identical-trajectory claim |
| **A2** retain graph | `K·F·(1 + c_b)` | **coherent graph — a *different* reference** | **not** an exact substitute for the native estimator; ordinary reverse-mode cost, **not a lower-bound theorem** |

Corrected conditional ceilings for a hypothetical **free** gradient, each **with its actual
included costs** (`c_b = 1`): vs native gradient-only **`4×`**; vs `A2` gradient-only
**`2×`** (*not* the `1.33×` previously published, which was the including-instrumentation
figure); vs native including instrumentation **`2×`**; vs `A2` including instrumentation
**`4/3×`**.

**Withdrawn:** the `54%/27%/19%` composition, the whole-run predicted ratios, and
"preparation is the pipeline's global peak memory". Two resident models establish a
**preparation memory burden**, not proof it exceeds the unknown training activation peak.
Newly accounted or explicitly excluded: **per-iteration prompt encoding**
(`inference_preparation_phase` calls `encode_prompt`, `models/flux.py:250`), decodes,
optimizer work, I/O, batching, initialization, **separate `c_state`/`c_mask`/`c_joint`
costs**, and graph-retention memory feasibility.

Real, source-supported, and still **not** contributions: fusing the two sweeps; raising
`plot_interval`; RNG-preserving replay (which **changes the reference contract** — it is
**not** proof of an unchanged native gradient).

## Evidence limits — read before citing anything here

- **No experiments, of any kind.** No model or encoder loaded, no checkpoint tensors, no
  features, no training/edit/replay, no generation, no classifiers or detectors, no image
  viewing or reannotation, no GPU allocation or polling, **no profiling**, no dependency or
  weight downloads, no environment rebuilding, no imported research code executed, no
  runnable implementation or trainer patch, no adapters, no datasets, manifests or
  evaluation packages, no author or reviewer contact.
- **Static source reading only**, reused from the original inspection at the same pin
  `d178324`. The repository was read **as text** and never imported or run.
- **No measurements.** `F`, `c_b` (and the distinct `c_state`/`c_mask`/`c_joint`), `D`,
  `F_txt`, `ν` and all hardware constants are **unmeasured** and labelled. There are **no
  seconds, no speedups and no GPU-hours** here. Simplified arithmetic is not a measurement.
- **Scalar/rational counterexamples certify algebra only** — they are **not** FLUX evidence
  and say nothing about deletion, retained utility, rendering quality or robustness.
- **A STOP is specific to this formulation.** It is **not** a claim that Minimalist's
  training cost is irreducible, that estimators cannot save work, that the paper is wrong,
  or that erasure-efficiency research is closed. **Later source-release dates alone do not
  prove a different algorithm was used for the publication.**

## Relationship to closed work — nothing reopened

This is a **correction pass** on a design sprint — not a new sprint, algorithm search,
candidate ranking, successor project or revision of C1–C3. Untouched, unmodified and not
reranked: the closed gap map at `edaec97`; the changing-replacement closure at `5786864`;
the FLUX diagnostic at `0e88d26` with its original results and **unused budget**; the SD1.x
and human-review closures and pauses; previously unselected theory/evaluation formulations.
No closed image, prompt, label or outcome was read, re-scored or mined.

## Files

| file | contents |
|---|---|
| [`CORRECTIONS.md`](CORRECTIONS.md) | **read alongside this record.** X1–X30: every replaced claim, the supplied counterexamples, and the corrected scope, with a pointer to the original publication |
| [`DERIVATION.md`](DERIVATION.md) | source/objective map, the discrete computation, the reference mask gradient, the retained local identity, and the gradient notions kept separate |
| [`ALGORITHM.md`](ALGORITHM.md) | the one attempted formulation (SSMG), preserved as a record with its corrected status, the unresolved novelty objection, and the scoped stop |
| [`COST_MODEL.md`](COST_MODEL.md) | corrected symbolic accounting, per-reference cost expressions, corrected ceilings and break-even, and the newly accounted or excluded terms |
| [`FALSIFICATION_PLAN.md`](FALSIFICATION_PLAN.md) | **why no experiment is justified** and what remains outstanding. **Not execution-ready; authorizes nothing** |
| [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) | paper version, pinned source commit and lines, corrected inspection depths, source claims vs. our deductions, and unverified claims |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Status

**STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS.** Not `DESIGN_READY_FOR_REVIEW`. No
follow-up job, monitor or scheduled task was created, and **no future computation is
authorized.**
