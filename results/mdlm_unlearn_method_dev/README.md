# MDLM unlearning — method development (active)

Ongoing development of an unlearning algorithm for masked diffusion language models
(LLaDA-8B-Instruct). **This is an active project, not a finished study.** No method has been
selected and no superiority claim is made. Tiers are kept strictly separate in
[`RECORD.md`](RECORD.md): historical closed studies (immutable), active development (this
directory), and a sealed final evaluation that does not yet exist.

## Why the previous approach needed replacing

Our own strongest control reaches 0.031 disclosure with 1.000 retain and zero degenerate
outputs — and **it achieves that by replacement, not removal**. Inspecting the actual
generations: 35 of 64 forget probes assert a *different valid answer from the same pool*,
**19 of them the same one ("Tbilisi")**, with fluent confabulated justification. The
formalization predicted this analytically (the exclusion target's fixed point is
`argmax_{v∉F} p₀`, a deterministic functional of the reference); the generations confirm it.
Every one of those 64 is non-degenerate, so our clean-forgetting metric scored all of it as
successful forgetting.

```
Q: What city does Alira Birchwood live in?   gold Lisbon
   "Alira Birchwood lives in Tbilisi. Tbilisi is the capital city of Georgia, and …"
Q: What city does Alira Fairholt live in?    gold Tbilisi
   "Alira Fairholt lives in Tallinn. Tallinn is the capital city of Estonia, and …"
```

## Finding 1 — the answer token is the coupling channel

[`docs/FINDING_answer_channel.md`](docs/FINDING_answer_channel.md). The coordinator's own
hypothesis (a shared low-dimensional edit direction) was **refuted** by its own test:
per-fact gradients are near-orthogonal (cos +0.067) and the 64-fact forget gradient set has
effective rank **29.5/64**, so the rank-3.13 edit is a low-rank *average* of diverse
directions.

What is actually there is carried by the shared **answer token**, and unlearning amplifies
it:

| per-fact gradient cosine | θ₀ (memorized) | θ_u (unlearned) |
|---|---|---|
| **same answer token** | **+0.388** | **+0.916** |
| same relation, diff answer | +0.126 | +0.104 |
| different answer | +0.060 | +0.053 |
| forget–forget overall | +0.065 | +0.067 |
| gradient-set effective rank | 33.5/64 | 29.5/64 |

Every structural quantity is unchanged except same-answer coupling, which more than
doubles. That predicted a per-fact behavioural split, which then confirmed: under a
relearning attack, **every held-out fact sharing an answer token with a leaked fact
recovered fully (1.000, 8/8) versus 0.514 for non-sharing facts** at k=10, monotone in k and
at parity at k=0.

Causal account: the exclusion target removes mass from answer-token set `F` → facts sharing
an answer share `F` → their gradients are driven into near-collinearity → recovering one
recovers all its same-answer partners.

## Errata against our own published studies

Issued after an independent audit. Original commits are **not** rewritten.

- [`docs/ERRATUM_relearn_a67d610.md`](docs/ERRATUM_relearn_a67d610.md) — seven items. The
  noise probe covered only the 128-module attention subspace holding **28.7%** of the edit
  energy while 71.3% (`ff_proj` .316, `up_proj` .240, `ff_out` .157) went untouched, so the
  published experiment did not perform the test it described. **Re-measured at full
  224-module coverage, the conclusion survives** (0.062 at 2× the edit norm vs 0.605 for a
  directed attack) with magnitudes understated ~2×. Also: the `0.049 → 0.605` headline mixed
  denominators (correctly **0.023 → 0.605**), and the defense comparison was underpowered —
  MDD 0.264 against its own 0.20 bar, so "no large effect" is supportable, "no effect" is not.
- [`docs/ERRATUM_arc_28154b2.md`](docs/ERRATUM_arc_28154b2.md) — the `K=512` IN-pool/ABSENT
  cells are **retracted**: the analysis hardcoded `K=16`'s pool membership for all arms,
  and `K=512` has 64/64 coverage so it has no absent set. Aggregates and the refutation of
  the per-fact prediction are unaffected.

Also carried forward from the audit: the 128-fact backbone's qualification gates never
completed and its hyperparameters are unrecoverable; `G3` save/reload is unreachable dead
code, so **nothing in the stack has ever verified that a saved adapter reloads to identical
behaviour**; and our scorer under-counts disclosure (`Montréal` vs `Montreal`, 7/1648 rows,
erring in our favour).

## Benchmark reality

[`docs/SCREEN_RESULTS.md`](docs/SCREEN_RESULTS.md) — 860 probes screened (the prior closed
study used 100). Exact match was measuring output *formatting*: EM ~0.01 while span
containment is 0.32–0.45, because the model answers in sentences against phrase golds.
RWKU's official primary is ROUGE-L **recall**, so an EM-anchored gate is mis-specified
against the benchmark. Clean candidate targets — multi-token, answer not in prompt,
span-recalled — number **40 of 297 (13.5%) over 9 of 10 entities**, an upper bound pending a
cross-decoder stability filter and a spurious-credit hand-check.

## Constraints we have had to accept

- **No claim against published values is available**, to us or anyone: no two paper families
  share a forget metric, MDU's paper defines ROUGE-L as F1 while its code computes recall,
  and nobody has reproduced MDU's table — including EMDU using MDU's own code. The only
  constructible claim is a better forget/neighbor trade-off than **our own re-runs** on one
  fixed protocol.
- Full-parameter 8B training does not fit 2×40 GB; everything is LoRA, declared as a
  deviation and never called an exact reproduction.
- **TNT/TNFF do not exist.** The likely source is a web summary headed "…(TNT)" for a paper
  containing zero occurrences of the string. CLP is real but is a 2019 fairness method with
  zero unlearning citations.

## Layout

```
RECORD.md              three-tier record; historical conclusions immutable
docs/FORMALIZATION.md  first-principles specification, 976 lines
docs/FINDING_answer_channel.md   Finding 1, with the θ₀ control
docs/PRIOR_ART.md      complete-algorithm comparison + novelty rubric
docs/EVAL_PROTOCOL.md  benchmark qualification, metrics, statistics, freeze checklist
docs/FOUNDATION_AUDIT.md   §6 correctness audit, ranked findings
docs/ERRATUM_*.md      corrections to our own published studies
docs/HYPOTHESIS_REGISTER.md   hypotheses with status; H1 refuted
scripts/               screen, gradient geometry, corrected noise probe, launchers
outputs/               metrics, raw generations, cosine matrices
ATTEMPTS.tsv           every attempt including failures, measured GPU/CPU
```
