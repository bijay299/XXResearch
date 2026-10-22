# MDLM unlearning — method development (active)

Ongoing development of an unlearning algorithm for masked diffusion language models
(LLaDA-8B-Instruct). **This is an active project, not a finished study.** No method has been
selected and no superiority claim is made. Tiers are kept strictly separate in
[`RECORD.md`](RECORD.md): historical closed studies (immutable), active development (this
directory), and a sealed final evaluation that does not yet exist.

## Start here

**[`docs/REPORT.md`](docs/REPORT.md) is the consolidated account** — read it before anything
else in this directory. It states plainly that the pre-declared "beats SOTA" bar is **not met**,
and collects what is defensible: the two walls, the AFC ablation with all four of its
retractions, the robustness inversion, and 19 closed candidates.

| document | what it is |
|---|---|
| [`docs/REPORT.md`](docs/REPORT.md) | the consolidated account and the honest verdict |
| [`docs/FINDING18_anchor_impossibility.md`](docs/FINDING18_anchor_impossibility.md) | wall 1 — form and fact-identity are the same information |
| [`docs/FINDING19_two_walls.md`](docs/FINDING19_two_walls.md) | wall 2 — fluency-preserving activation targets carry no signal; the joint prediction |
| [`docs/FINDING20_robustness_inversion.md`](docs/FINDING20_robustness_inversion.md) | our best static method is the robustness loser; AFC disqualified |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | why TV→pristine *is* a membership-inference advantage, and why EMDU's abstention is the worst outcome under it |
| [`docs/NOVELTY_attack_validity_gate.md`](docs/NOVELTY_attack_validity_gate.md) | the attack-validity gate is a special case of Athalye et al., and a limitation it exposed in our own protocol |
| [`docs/RESULTS.md`](docs/RESULTS.md) | the AFC study as written, retractions in place |
| [`ATTEMPTS.tsv`](ATTEMPTS.tsv) | every attempt including failures, with cumulative resource use |

The two findings worth keeping are **negative**: they predict failures rather than reporting
them, and together they say a working method must take its forget target from outside the
model's own distributions. Testing that prediction is the current work.

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

## Finding 1 — RETRACTED the same day it was published

[`docs/FINDING_answer_channel.md`](docs/FINDING_answer_channel.md). The first version of
this finding claimed that the exclusion target's indexing by the answer-token set `F`
*causes* the per-fact gradient coupling that leaks across facts. **That causal claim is
withdrawn.** It was published in commit `94b3a3d8a6fe` and retracted within the hour after
two agents objected independently and a decisive experiment confirmed them.

**The decisive experiment.** The same measurement on two published baselines that are *not*
answer-token-indexed, both driven to disclosure 0.000:

| arm | `F`-indexed | same-answer | retain same-answer | diff-answer | selectivity |
|---|---|---|---|---|---|
| exclusion target (ours) | **yes** | **+0.916** | +0.644 | +0.053 | **17.2×** |
| EMDU, 150 steps | no | +0.773 | +0.753 | +0.198 | 3.9× |
| MDU τ=1, 150 steps | no | +0.573 | +0.575 | +0.414 | 1.4× |

Both non-indexed methods show the coupling anyway, and in both, forget-same-answer *equals*
retain-same-answer. Suppression produces the coupling; the indexing is not required. The
analytical account is exact: at LoRA init `cos(g,g') = cos(δ,δ')·cos(Ax,Ax')` with
`δ ≈ −Jᵀw_{v*}` for a suppressed fact, so two facts sharing an answer share that unembedding
row whatever produced the suppression.

**My own numerical bug was inflating the effect I claimed.** Gradients were cast to fp16;
at θ₀'s 4.3e-6 norms, against fp16's 5.96e-8 subnormal, ~99.98% of coordinates were
unrepresentable (116 exactly-zero off-diagonal cosines at θ₀, none at θ_u). Recomputed in
fp32:

| cell | θ₀ (fp16, as published) | θ₀ (fp32) | θ_u (fp32) |
|---|---|---|---|
| forget-forget, same answer | +0.388 | **+0.639** | **+0.916** |
| retain-retain, same answer | not computed | **+0.659** | +0.644 |
| forget-retain, same answer | not computed | **+0.620** | **+0.261** |
| gradient-set effective rank | 33.5/64 | **21.9/64** | 29.5/64 |

The "×2.36 amplification" headline becomes **×1.43**. At θ₀ the three same-answer groups are
indistinguishable (0.639 / 0.659 / 0.620) — strongest of all on retain pairs the edit never
touches. And forget-retain coupling *falls* 0.620 → 0.261 under unlearning: the objective
**decoupled** forget facts from their retain twins, the reverse of a collateral-damage
mechanism, consistent with retain accuracy holding at 1.000.

**The behavioural confirmation is also withdrawn as stated.** Non-sharing recovery by
relation is field 12/12, home_city 6/14, **instrument 0/9** — and 9 of 9 non-sharing
instrument facts emit a *leaked* instrument rather than their own (`timpani` ×7); one
"recovered" generation reads `"…plays the timpani, the timpani, the timpani…"`. Dropping that
stratum moves Fisher from p = 0.011 to **p = 0.198**. Only home_city survives uncontaminated
(4/4 vs 6/14, p = 0.069). And the continuous per-fact predictor has **r = +0.014, p = 0.936**
within the non-sharing stratum.

**What survives.** Same-answer coupling is the dominant structure in per-fact gradient
geometry across every method measured, and its *selectivity* varies sharply — 17.2× / 3.9× /
1.4×. MDU's near-uniform coupling with effective rank 4.5/64 indicates globally collapsed
gradients, a mechanistic correlate of the output collapse we had only measured behaviourally.
Descriptive, not causal.

**The failure mode, named because it has now happened twice.** ARC presented an aggregate gap
as a per-fact confirmation; this finding presented an 8-vs-35 group difference as one while
its own continuous predictor sat at r = 0.014. A standing rule is now recorded: a per-fact
prediction is confirmed only by a within-stratum continuous correlation, computed before the
group comparison is reported.

## Finding 8 — the ordering channel reaches the cell that target-matching cannot

[`docs/FINDING8_ordering_reaches_the_cell.md`](docs/FINDING8_ordering_reaches_the_cell.md).
**The project's first positive mechanism result**, and an oracle rather than a method.

At every decode step, any position whose current argmax is an author-salient answer token has
its confidence set to `-inf`, so it commits **last**. No training, no weight change — an exact
upper bound on any trained approximation. The local sampler loop reproduces the pinned
`generate()` **bit-identically** with the intervention off, on both datasets.

| cell | θ₀ (memorized) | **ORACLE** | pristine base (ignorance target) |
|---|---|---|---|
| DISCLOSE | 0.667 | **0.108** | 0.030 |
| **DEGENERATE** | 0.000 | **0.017** | 0.000 |
| ABSTAIN | 0.000 | 0.000 | 0.113 |
| REPLACE | 0.125 | 0.258 | 0.470 |
| IGNORANT | 0.208 | 0.617 | 0.388 |
| **TV to pristine** | **0.6372** | **0.3242** | — |

**Disclosure falls 6× while degeneracy stays at 0.017** — against **1.000** for MDU, EMDU and
NPO, which reach DISCLOSE 0.000 by emitting nothing. Salient-content recall 0.610 → 0.155;
58 of 120 items reach zero salient content; **0 of 120 empty generations**; length 89%
retained. Robust across thresholds (τ=0.25: 0.892→0.250; τ=0.75: 0.350→0.042).

It also **refutes the re-entailment hazard** raised in design — the concern that committing a
fact later conditions it on more model-written context that *entails* the fact.

**What it is not:** an oracle, using answer identity at inference, so an upper bound and never
a method. It **overshoots IGNORANT** (0.617 vs 0.388) — more non-committal than true
ignorance, a mismatch under profile matching. Output form varies across fluent content
removal, ungrammatical elision, and no effect where context pins the token. One model, one
seed, 120 of 400 QA.

**The method problem it leaves:** approximate the ordering oracle in the weights without
answer-set knowledge at inference. The trainable route is derived and is *not* suppression —
at T=0 the answer position's confidence is *identically* its answer probability, so lowering
it is suppression renamed; what is available is **raising the structural positions**, whose
gradient at the answer position is zero by construction. Declared risk: feasibility is a
knife-edge (≈10⁻² margin).

## Finding 7 — the retain-reference oracle is NOT in the good cell

[`docs/FINDING7_oracle_not_in_good_cell.md`](docs/FINDING7_oracle_not_in_good_cell.md). The
pre-registered A2 gate fired against the plan: the retain-trained reference lands **REPLACE
0.580** (0.672 counting direct assertion), so retain-reference distillation and everything
built on it is dead. **Retain SFT made the model a worse ignorance model than no training at
all** — ABSTAIN 0.113 → **0.000**, IGNORANT 0.388 → 0.268.

Two near-misses: the frozen repetition rule would have reported plurality DEGENERATE and sent
us down the wrong branch; and the **metric choice decided the verdict** — 0.134 distinctive
(pass) versus 0.369 naive and 0.394 ROUGE-L, both >2.5× the ceiling, on which we would have
wrongly concluded TOFU is confounded too. The pivot was still right: the ignorance gate passes
with margin **+0.804** against ≈0.000 on the synthetic set, so TOFU does fix the Finding-5
confound.

## Candidate methods: four closed, one with a confirmed mechanism

| candidate | verdict |
|---|---|
| H1 shared-direction edit | refined, closed |
| H6 mask-rate bands | superseded before testing — the visited marginal is already uniform |
| H11 minimal interdiction | **closed** — category (c); already run by TIF as `NPO-GPT` and reported *"insufficient"*; "rare tokens" has three papers; excluding tokens from the loss **does not protect them** |
| C1 / oracle matching + SIO | **closed** — Finding 7; also category (a) against arXiv 2506.06278, where it is a named baseline |
| **H10 commitment re-ordering** | stated form refuted by derivation; **salvaged form's oracle succeeds** (Finding 8) |

## Findings 2 and 3 — the metric and the candidate target both failed their own tests

[`docs/FINDING3_no_ignorance_reference.md`](docs/FINDING3_no_ignorance_reference.md).

**The candidate's target does not exist as specified.** It was to match `P_ign`, the answer
distribution of a reference at a *verified-unknown* subject, estimated by substitution. Its
designer pre-registered the fatal condition. Measured (512 forwards, 16 s), the **validity
gate failed: mean p(true answer | stranger) = 0.228 against a 0.02 threshold.** The model
confidently confabulates a memorized answer for any name-like string — `"Durnesse
Corbridge"` in a cello frame returns ` cello` at **p = 0.949** — and which answer it picks
depends on the stranger's name. That is not ignorance. A first attempt failed because the
strangers shared surnames with trained entities; rerunning with wholly novel name components
left the gate failing identically. The surviving route is a reference trained on the
**retain facts only**, which the critic independently judged the best idea in the design.

**The IGNORANT metric does not measure ignorance.** All 28 generations it scored IGNORANT for
the best arm were printed: **28 of 28 assert a value; zero are non-asserting.** They escaped
the REPLACE cell by a *morpheme* — `seismoology` ×8, `mycological`, `harpa`, `clarion`,
`tympani`, `oboes`. Corrected: **REPLACE ≈ 0.938, IGNORANT ≈ 0.000.** The metric is also
gameable in the wrong direction (nudging substitutes from `Tbilisi` to `Troms` scores a large
gain while making the model worse), it **conflicts with our own formalization** (which says an
ignorant model still answers *about* the subject — scored here as REPLACE), and the claim "no
method exceeds 0.438" is unsupportable: the two datasets' answer pools differ (7/5/4 vs
16/8/8) so the columns are different predicates, eight of nine arms have a Wilson upper bound
above 0.438, and the best baseline versus ours is Fisher **p = 1.0**.

**The same error, three times.** The `Tbilisi` attractor is not the exclusion target's
signature: partitioning **θ₀**, before any unlearning, gives REPLACE 0.203 with all 13 facts
it does not know being in-pool substitutions, 5 of the first 6 on `Tbilisi`. It is the
reference's own prior. That joins ARC's aggregate-as-per-fact error and Finding 1's
attribution of suppression geometry to the objective's indexing. Common form: **a property is
attributed to the intervention without being measured at the pre-intervention reference.**

> **Standing rule, now institutionalised.** Every claim that an intervention causes a property
> must report that property measured at θ₀ **and** at one intervention not sharing the
> proposed mechanism, in the same table, before the claim is stated — not as a later
> robustness check.

**One structural conclusion.** The closed 32-answer pool is itself a liability: it installs a
relation-level prior that contaminates the metric (REPLACE and IGNORANT are not separable)
*and* the target (`P_ign` is pool-concentrated). This argues for moving the mechanism work on
to real pretrained knowledge sooner than planned.

Adversarial review of all of the above, including the mathematics of the candidate, is in
[`docs/CRITIQUE_WAVE2.md`](docs/CRITIQUE_WAVE2.md).

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
