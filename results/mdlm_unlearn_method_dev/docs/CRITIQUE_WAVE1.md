# CRITIQUE — wave 1

Mathematical critic. Adversarial review of `FORMALIZATION.md`, `FINDING_answer_channel.md`,
`HYPOTHESIS_REGISTER.md` (H1), `ERRATUM_relearn_a67d610.md`, `ERRATUM_arc_28154b2.md`.

Every number below was recomputed from the artifacts on disk. No GPU was used. The scripts
are inline in the text; the two new facts that drive most of this are (i) the **θ₀ gradient
cosine run already exists** (`outputs/grad_cosine_theta_0.json`, written 12:32, four minutes
after `FINDING_answer_channel.md`) and (ii) the **corrected full-coverage noise probe already
exists** (`outputs/noise_unlearned.json`, 12:29). Both change verdicts the team has recorded.

One global fact that recurs: **every frozen decoding spec in this project is
`gen_length=32, steps=32, block_length=32, temperature=0.0, remasking=low_confidence,
cfg_scale=0.0`** (`spex/data_big/frozen_spec.json`, `llada_poc/data/frozen_spec.json`). One
block, one commit per step, greedy, no CFG. Three results in `FORMALIZATION.md` are stated at
configs that do not exist in this project, and two of them are *vacuous or false* at the config
that does.

Severity labels: **FATAL** (an error that changes a decision now) / **MATERIAL OVERCLAIM** /
**UNSTATED ASSUMPTION** / **IMPRECISION** / **CORRECT**.

---

## Ranking

| # | target | claim attacked | verdict |
|---|---|---|---|
| **F1** | FINDING | "the coupling is carried by the shared answer token" (mechanism of unlearning) | **FATAL** — refuted by the project's own θ₀ run |
| **F2** | FINDING | behavioural confirmation 1.000 (n=8) vs 0.514 (n=35) | **FATAL** — answer-collision artifact in one relation stratum; p 0.011 → 0.198 |
| **F3** | FINDING / H1 | gradient geometry predicts per-fact recovery | **FATAL** — zero within-group predictive power (r=+0.014, p=0.94) |
| **F4** | FORMALIZATION Cor 3.1 | per-position KL can never bound generated behaviour | **FATAL** — false at the project's own config; loose by ~2000× even at the quoted one |
| **M1** | HYPOTHESIS_REGISTER | H1 REFUTED on mean cos +0.067 / eff rank 29.5 | **MATERIAL** — 0.067 predicts 30% transfer efficiency; 29.5 ≈ 32 = #answers |
| **M2** | FORMALIZATION Cor 2.2 / §8.3(v) | the ε cliff at 1/\|V\| | **MATERIAL** — arithmetic right, inference invalid |
| **M3** | FORMALIZATION Lemma 5 | suppression lowers confidence ⇒ later commit | **MATERIAL** — counterexample; condition is wrong for multi-surface F |
| **M4** | FORMALIZATION Thm 5.5 / Cor 5.6 | conjunction infeasibility; "no fourth option" | **MATERIAL** — vacuous at k=1; factorization fails under `lc`; fourth option exists |
| **M5** | ERRATUM_relearn §1 | "directional not positional" WITHDRAWN | **MATERIAL over-retraction** — corrected probe confirms it |
| **M6** | FORMALIZATION Thm 5.3 | entailment leakage ⇒ joint infeasibility | **MATERIAL** — three quantifier defects |
| **M7** | FORMALIZATION Prop 8.4 | provable anti-robustness, "strongest objection" | **MATERIAL** — rank-preserving; same phenomenon as 8.2 |
| **I1** | FORMALIZATION Thm 4.5 | gating is logically forced | **tautology**, labelled [P] |
| **I2** | FORMALIZATION Lemma 1 / Cor 1.1 | prompt-disjointness | **correct but definitional**; one use reads as reassurance |
| **I3** | FORMALIZATION Lemma 4 | "iff" rank condition | **IMPRECISION** — necessity is false |
| **I6** | FINDING / data | cosine matrices | **UNSTATED** — fp16 gradient store; θ₀ matrix is corrupted |

Decisions this should change, in order:

1. **Do not let `FINDING_answer_channel.md` ground the method design** (F1–F3). Its causal
   reading is excluded by the θ₀ run on disk, and its behavioural confirmation does not survive
   stratification.
2. **Un-refute H1, or re-refute it on a valid argument** (M1). The shared-direction mechanism is
   *quantitatively supported* by the very matrix used to kill it.
3. **Do not adopt R2 as "no per-position divergence objective"** (F4). At the project's config the
   sufficient per-position KL is ~5×10⁻⁶, not 10⁻⁹.
4. **Reinstate E4** (M5). The corrected 224-module probe is on disk and confirms the withdrawn
   conclusion.
5. **Recompute both cosine matrices in fp32 before using either** (I6); the θ₀ one is unusable as
   stored.

---

# 1. `FINDING_answer_channel.md`

## F1 — FATAL. The causal claim is refuted by the project's own θ₀ measurement

**Claim.** "**The coupling is carried by the shared answer token.** Two forget facts with the same
answer have nearly collinear gradients… An edit whose per-fact gradient is essentially a function
of the **answer token** cannot separate Alice from Bob." Stated as a *mechanism of the unlearning
edit*, replacing H1, and declared "the grounding the method design now proceeds from".

**Objection.** The finding lists as required-before-grounding: "the same measurement at θ₀ (is the
coupling created by unlearning or intrinsic to the task?)". That run completed at 12:32
(`logs/p006_geom_gradcos_theta0_full_gpu2.log`, `outputs/grad_cosine_theta_0.json`). Its answer
destroys the causal reading. Same-answer cosines, both points, same 224-module LoRA-B space, same
mask draws:

| pair type | θ₀ (memorized, **nothing unlearned**) | θ_u (after unlearning) |
|---|---|---|
| forget–forget, same answer | **+0.3876** (sd .184, n=64) | +0.9159 |
| retain–retain, same answer | **+0.3971** (sd .148, n=64) | +0.6375 |
| forget–retain, same answer | **+0.3648** (sd .172, n=128) | +0.2597 |
| forget–forget, different answer | +0.0596 | +0.0533 |
| forget–retain, different answer | +0.0308 | +0.0107 |
| mean gradient norm | **4.32e-6** | 5.72e-1 |

At θ₀ the three same-answer couplings are **statistically indistinguishable** (0.388 / 0.397 /
0.365, all with sd ≈ 0.17 at n ≥ 64), they are 6–12× the different-answer baseline, and they are
measured where **gradient norms are 1.3×10⁵ times smaller** because every fact is fit and there is
nothing to unlearn. Retain facts — which the unlearning edit never touched — carry the coupling at
exactly the same strength as forget facts.

This is the finding's own control, applied to the finding's own hypothesis. The finding used it to
kill H1: *"The small residual alignment is not specific to forgetting — retain-retain (0.055) is of
the same order — so it is a property of the loss and parameterization, not of the unlearning."*
Apply the identical test to the answer channel: retain-retain same-answer (0.397) **equals**
forget-forget same-answer (0.388) at θ₀. By the finding's own stated criterion, the answer-token
coupling is a property of the loss and parameterization, not of the unlearning.

**The mechanism, which is mechanical and not about knowledge.** At LoRA init, `B = 0`, so the LoRA
branch contributes nothing to the forward pass and

> `dL/dB_m = (α/r) · Σ_t δ_t (A x_t)ᵗ`,  i.e.  `vec(dL/dB_m) = Σ_t (A x_t) ⊗ δ_t`,

with `δ_t` the base model's backpropagated error at module m's output and `A` the frozen random
down-projection. The loss is `CE/p_mask` at masked positions; the error entering the network is
`W_Uᵗ(p_t − e_{y_t})` at each masked position. For a **suppressed** fact at θ_u, `p(y) ≈ 0` at the
answer position, so `p − e_y ≈ −e_y` and `δ ≈ −Jᵗ w_y` — *the unembedding row of the answer token*,
at near-maximal magnitude, while template positions are well predicted and contribute
`‖p − e_y‖ ≈ 0`. In the rank-1-dominant regime
`cos(g₁,g₂) ≈ cos(δ₁,δ₂) · cos(Ax₁,Ax₂)`, and `cos(δ₁,δ₂) → 1` exactly when the answer token is
shared. Nothing about stored knowledge is required: this would hold for two arbitrary unrelated
sentences that share a single hard-to-predict target token.

The prediction of that account is the observed ordering, which the finding instead attributes to
noise: `forget·forget (both residuals large) 0.92 > retain·retain (both small) 0.64 >
forget·retain (one large, one small) 0.26`, with all three collapsing to a common ≈0.39 at θ₀ where
all residuals are equally small. The finding's own explanation — "the direction of a near-zero
gradient is noisy" — is quantitatively excluded in I6 below.

**On the B-only restriction specifically** (the question posed): the restriction is not what
manufactures the alignment — `dL/dW = Σ_t δ_t x_tᵗ` has the same outer-product form, so the δ-side
dominance survives into full weight space. What the restriction *does* do is compress the
entity-specific factor: `x_t ∈ R^4096` is replaced by `A x_t ∈ R^16`, while the answer-dependent
factor `δ_t` keeps its full output dimension. The measured cosine is a product
`cos(δ)·cos(Ax)`, and the second factor is a 16-dimensional random projection of two transformer
hidden states — which in an anisotropic representation is large and nearly constant across facts.
So essentially **all the variation** in the measured cosine comes from the answer side by
construction. The restriction does not create the alignment; it guarantees that the alignment is
the only thing the measurement can see.

**Status of the descriptive claim.** The *descriptive* statement — "at θ_u, per-fact gradients of
facts sharing an answer token are nearly collinear, and facts with different answers are
near-orthogonal" — is **correct and I verified every cell** of the finding's table from the raw
matrix. What does not follow is that this is a mechanism of unlearning, that it is the channel of
the E3 leak, or that it is the thing a method must be designed against.

## F2 — FATAL. The behavioural confirmation is an answer-collision artifact in one relation stratum

**Claim.** "Every held-out fact sharing an answer token with a leaked fact was fully recovered by
k=10 (8/8)" vs 0.514 for the 35 non-sharing facts; gap +0.486.

**Reproduction.** Every cell reproduces exactly from `anvil/outputs/relearn_gens_unlearned.json`.
Leak set `{forget_000, 006, 017, 023, 030, 036, 047, 053}`, answers
`{Paris, Montreal, oboe, timpani, philology, Dublin, Perth, clarinet}`; the design pairs facts as
`(i, i+32)`, so the sharing set is exactly `{004, 015, 021, 032, 038, 049, 055, 062}`.

**Objection 1 — recovery is relation-determined, and the leak set's relation mix is not matched to
the held-out mix.** Non-sharing recovery at k=10, by relation:

| relation | sharing | non-sharing | non-sharing facts emitting a **leaked** answer instead of their own |
|---|---|---|---|
| field | 1/1 | **12/12** | 0/12 |
| home_city | 4/4 | 6/14 | 0/14 |
| instrument | 3/3 | **0/9** | **9/9** |

The field relation recovers **completely without any shared answer**. The instrument relation
recovers **not at all** — and all nine of its non-sharing held-out facts emit one of the three
*leaked* instruments instead of their own: `timpani` for seven of them
(`forget_018,019,020,022,050,051,052`) and `clarinet` for two (`forget_016,048`). At k=5 this is
already 9/9.

So in the instrument stratum the adversary's 10 steps on three instrument facts did not recover
facts — it made the model answer `timpani`/`clarinet`/`oboe` for **every** entity asked that
relation. The three "sharing" instrument facts score as recovered because their reference answer
happens to be one of the injected ones. `forget_055` (`timpani`) is emitted by 8 of the 12
instrument held-out facts; conditional on the model emitting `timpani` for 7 of the other 11
entities, `forget_055` emitting it is the unsurprising case, not evidence of fact-specific
recovery. Its generation is `"Delisa Hollowman plays the timpani, the timpani, the timpani, the
timpani, and the timpani."`

**Objection 2 — the significance is entirely in that stratum.** Fisher exact, one-sided:

| comparison | n | p |
|---|---|---|
| as published, 8/8 vs 18/35 | 43 | **0.0108** |
| drop `forget_055` (timpani, 7 non-partner emitters) | 42 | 0.0178 |
| drop timpani + clarinet | 41 | 0.0299 |
| **drop the instrument stratum entirely: 5/5 vs 18/26** | 31 | **0.198** |
| home_city alone: 4/4 vs 6/14 | 18 | 0.069 |
| field alone: 1/1 vs 12/12 | 13 | 1.000 |

The headline p = 0.0108 is also exactly the hypergeometric `C(26,8)/C(43,8)` — i.e. "these 8 of 43
are all in the recovered set", which is the only information the test carries. The split is
reported at five values of k with k=10 bolded; Bonferroni over five gives 0.054. Mantel–Haenszel
stratified by relation gives z = 3.57 (p = 0.0002), which looks *stronger* than the marginal test —
but it is stronger only because the instrument stratum contributes the extreme 3/3 vs 0/9, which is
the artifact.

**Objection 3 — n = 8 and non-independence.** The eight sharing facts are not eight independent
draws: each is the designed `(i, i+32)` partner of a specific leaked fact, and they are drawn from
three relation templates in proportions `4/3/1`. The 95% Wilson interval on 8/8 is [0.676, 1.000]
and on 18/35 is [0.356, 0.670] — the intervals do not overlap, but the lower endpoint of the first
rests on eight clustered observations with no seed replication (`1 unlearning seed`, erratum §3).

**What survives.** A same-answer effect in `home_city` (4/4 vs 6/14, p = 0.069, no answer
injection in that stratum, 0/14 non-sharing facts emit a leaked city). That is a real, suggestive,
underpowered signal in one relation out of three. It is not "confirmed", and the paper sentence
"This is the per-fact, cosine-predicted test that H1 was supposed to pass and did not" cannot be
sustained — see F3.

**Required fix.** The forward protocol must (i) score *which answer the model emits*, not only
whether its own reference appears, so indiscriminate answer injection is detected rather than
scored as recovery; (ii) stratify by relation with matched leak-set composition; (iii) treat
relation-level answer collapse as its own failure mode. The finding's own stated defect ("the
relation channel is untestable in this data") understates the problem: the relation channel is not
merely untestable, it is **doing the work and being attributed to the answer channel**.

## F3 — FATAL. The continuous form of the prediction has zero support

**Claim.** The finding presents the behavioural split as the per-fact, cosine-predicted test, and
explicitly contrasts it with the ARC failure where "an aggregate gap of 0.101 was presented as
confirming a per-fact prediction". H1's own pre-registered falsifier was "**the cosine correlation
— not the aggregate gap — is the test**."

**Objection.** Compute, for each of the 43 held-out facts, its maximum θ_u gradient cosine to the
8 leaked facts, and test against recovery at k=10:

| set | Pearson r | p | Spearman | MWU (greater) | maxcos, recovered | maxcos, not |
|---|---|---|---|---|---|---|
| all 43 | +0.384 | 0.011 | +0.314 | 0.021 | 0.360 (n=26) | 0.113 (n=17) |
| **non-sharing 35** | **+0.014** | **0.936** | +0.079 | 0.328 | **0.114** (n=18) | **0.113** (n=17) |

Within the 35 non-sharing facts the cosine is **perfectly uninformative** — recovered and
unrecovered facts have the same mean max-cosine to three decimal places. The entire r = +0.384
over 43 facts is the binary same-answer indicator re-expressed as a number. So the gradient
cosine contributes **nothing** beyond the string equality "shares the answer token", and the
measurement cannot distinguish the gradient-geometry explanation from any explanation that
depends only on answer-string identity — including F2's answer-injection explanation and the
simpler one below.

**The simpler explanation, as asked.** "The model re-learns the answer token's association with
the relation template, and any entity asking that relation then gets that answer." This is not a
rival hypothesis to be weighed — it is **directly observed** in the instrument stratum (9/9
non-sharing facts emit a leaked instrument) and it explains all four observations the finding
offers without invoking gradient geometry:
- sharing facts recover first and fully — they are the entities whose reference *is* the injected
  answer;
- non-sharing facts in the same relation do not recover — the slot is occupied by the injected
  answer;
- the "Tbilisi attractor" (19/64 forget probes converging on one replacement) is the *same*
  phenomenon at the unlearned point, and is already fully explained by Prop 8.2's fixed point
  `argmax_{v∉F} p₀` being shared across a relation. It needs no cosine;
- paraphrase generality vs cross-fact leakage is explained by the paraphrases sharing the entity.

## I6 — UNSTATED ASSUMPTION. The gradients are stored in fp16; the θ₀ matrix is numerically corrupt

`scripts/grad_cosine.py:117`: `v = (acc / a.n_mask_draws).to(torch.float16).cpu()`. The cosines are
computed from fp16 vectors. fp16's smallest positive subnormal is `2⁻²⁴ = 5.96e-8`, so a vector of
norm `n` over `d = 23 068 672` coordinates can have at most `(n/5.96e-8)²` nonzero stored entries:

| | ‖v‖ | max nonzero coords | as % of 23.07M |
|---|---|---|---|
| θ_u, mean | 5.72e-1 | no constraint | 100% |
| θ_u, min | 1.16e-4 | 3.77e6 | 16.3% |
| **θ₀, mean** | **4.32e-6** | **5.3e3** | **0.023%** |
| **θ₀, min** | **2.46e-7** | **17** | **0.0001%** |

Direct evidence it bit: the θ₀ off-diagonal contains **116 exactly-zero cosines** — two generic
23M-dimensional vectors have inner product exactly 0 with probability zero; this happens only when
their surviving supports are disjoint. The θ_u off-diagonal contains **zero** exact zeros.

A scale sweep confirms the measurement is not scale-invariant although the quantity is
(heavy-tailed coordinates, d = 2e5, planted true cos 0.0667): fp16 cosine is 0.0667 at ‖v‖ = 1e-2,
0.0599 at 4.3e-6 (θ₀'s mean norm, −10%), and **0.0000** at 2.5e-7 (θ₀'s minimum norm).

Consequences:
- **The θ₀ matrix must be recomputed with fp32 storage before it is used for anything.** This
  matters because it is the finding's own stated prerequisite, and because it is the measurement
  that decides F1. The *direction* of the bias is deflation, so the θ₀ same-answer coupling of
  0.388 is if anything an **underestimate** — F1's conclusion is robust to the fix, but the number
  is not quotable as it stands.
- The θ_u matrix is largely safe, but the Gram computation returns **self-cosines up to 1.0123**
  (diagonal max; min 0.99997). That bounds the matrix's accuracy at roughly ±0.012 — which is the
  size of the entire forget-vs-retain control gap used to refute H1 (0.067 vs 0.055).
- The finding's explanation of the 0.916-vs-0.260 gap ("the direction of a near-zero gradient is
  noisy") is **quantitatively excluded**: fp16 truncation shifts cosines by about −0.001 at θ_u's
  minimum norm, three orders of magnitude too small to explain a 0.656 gap. Mask-draw sampling
  cannot explain it either, because the script deliberately uses **the same four mask draws for
  every fact** — shared draws correlate cross-fact directions, they cannot randomize them. The
  residual-dominance account in F1 explains the gap exactly, including its sign.

## I5 — IMPRECISION. The 83× / 58× ratios use a mismatched baseline

The table normalizes *within-group* contrasts by the *cross-group* baseline (forget–retain
different answer, 0.0107). The matched baseline for a forget–forget contrast is forget–forget
different answer, 0.0533. Against it, the same-answer ratio is **17×**, not 83×; retain–retain
same-answer against retain–retain different answer is likewise ~12×, not 58×. The reported ratios
are inflated about fivefold by baseline choice. The 24.3× for forget–retain is correctly matched.

## CORRECT (verified, moving on)

- Every cell of the k-table (0/8 & 1/35; 1/8 & 1/35; 4/8 & 7/35; 8/8 & 18/35; 8/8 & 20/35).
- "the 8 leaked facts have 8 same-answer partners among the 43 held out" — exact.
- "Only 2 of 64 forget facts have any retain partner above cos 0.5" — exactly 2.
- The caveat that the measurement is a restriction of the full weight-space gradient is stated
  plainly and in the right place. The gradient-norm range (1.16e-4 to 2.47) is correctly reported.
- **The gap is not monotone**, contrary to "The split is monotone in k": the *groups* are monotone,
  the *gap* peaks at k=10 (+0.486) and falls at k=20 (+0.429). Minor, but the monotonicity argument
  is offered as evidence and should be stated about the groups only.

---

# 2. `FORMALIZATION.md`

## F4 — FATAL. Corollary 3.1 is false at the project's own config and loose by ~2000× at the quoted one

**Claim.** "At `gen_length = 128`, `block = 32`, `128` steps the eligible sum is `Σ_n |E_n| = 2112`.
Requiring `TV ≤ 0.05` forces a mean per-position `TV ≤ 2.37 × 10⁻⁵`, i.e. **mean per-position
`KL ≤ 1.12 × 10⁻⁹`**. No training run attains that. *Therefore: a per-position divergence objective
can never be justified as a bound on the output distribution.*" Promoted to requirement R2 and
re-quoted in §6.1.

**The arithmetic is right.** `Σ_{n=0}^{31}(32−n) = 528`, `×4 blocks = 2112` ✓. `0.05/2112 =
2.367e-5` ✓. `2·(2.367e-5)² = 1.121e-9` ✓.

**The inference is not.** Four defects, compounding.

**(a) The bound sums over the wrong set — factor 16.5 in TV, 272 in KL.** Lemma 3's proof couples
the per-position draws at every *eligible* position. But under `rnd` the confidences are
θ-independent, so `C_n` is θ-independent given the history, and the step kernel depends on θ
**only through the positions actually committed**: positions in `E_n \ C_n` have their draws
discarded. The correct hybrid bound is

> `TV(G,G') ≤ Σ_n 𝔼[ Σ_{i∈C_n} TV(p_θ(·|s_n,i), p_{θ'}(·|s_n,i)) ]`,  with `Σ_n |C_n| = L`.

That is **128 terms, not 2112**. Theorem 2.1(ii) uses exactly this — `Σ_n k_n = L` — so the
document already knows the right count and uses the looser one in Cor 3.1. Corrected: mean
`TV ≤ 0.05/128 = 3.91e-4`, mean `KL ≤ 3.05e-7`. Ratio of KL budgets: **272×**.

**(b) Pinsker is the worst-case conversion and is conservative for concentrated denoiser outputs —
a further ~8.5×.** `KL ≥ 2TV²` is tight only for diffuse perturbations near uniform mass. For a
typical LLaDA denoiser output, moving `δ` of mass between tokens at 0.9 and 0.03 gives
`KL = 2.61e-6` at `δ = 3.91e-4` — **8.54× larger** than Pinsker's `3.05e-7`. So a given KL budget
buys a **smaller** TV than Pinsker promises, and the honest sufficient KL is correspondingly
larger: ≈ **2.6e-6** at the quoted config.

Combining (a) and (b): the honest number at `gen 128 / 128 steps` is ~2.6e-6, not 1.12e-9 — a
factor of **≈ 2300**.

**(c) The config quoted is not this project's config, and at the real one the conclusion is
false.** Every frozen spec is `gen_length=32, steps=32, block_length=32`. One block, 32 commits.
Then `Σ|C_n| = 32` and the sufficient mean per-position KL is

> `2·(0.05/32)² = 4.88e-6`,  or ≈ **4.2e-5** after the Pinsker correction in (b).

Mean per-position KL of 10⁻⁵–10⁻⁴ on in-distribution states is routinely attained by distillation.
**The impossibility does not hold at the configuration every study in this project has run.**

**(d) Lemma 3 — the bound Cor 3.1 rests on — is explicitly invalid for the only remasking rule the
project uses.** The document says so itself: "(For `lc` the bound fails as stated, because `conf_i`
itself depends on `θ`…)". Every frozen spec is `remasking=low_confidence`, `T=0`. So the
quantitative impossibility is derived from a bound that does not apply, at a config that is not
used, with a 272× counting error, converted by a worst-case inequality that is 8.5× conservative
here.

**On tightness, as asked.** Is the bound tight or a loose union bound presented as an
impossibility? It is the latter, and the distinction matters exactly as the brief says: *failing a
sufficient condition proves nothing*. The chain is "per-position TV ≤ x ⇒ output TV ≤ 0.05" and
"per-position KL ≤ 2x² ⇒ per-position TV ≤ x". Both are one-directional. The document's own phrasing
inverts them — "**requires** a mean per-position KL ≤ 1.12e-9" — which is the error. A per-position
KL *larger* than the bound is entirely compatible with output TV ≤ 0.05; nothing in Cor 3.1 touches
that case.

**What survives, and it is worth keeping.** The *qualitative* point in R2 — that the right transfer
is a union bound on the **disclosure event**, not on the law on `Δ(V^L)` — is correct and
independently motivated (Def 1.2, Lemma 4, Cor 6.2). TV ≤ 0.05 on the full joint law of a
32-token canvas is a vastly stronger requirement than anything the specification needs, and
targeting it is the real mistake. Keep the recommendation; **delete the impossibility argument**,
or restate it as: *this union bound is too loose to certify an output-level TV target at this
config; no tightness is claimed, and a KL objective must be justified by Lemma 1's support
argument (as A1 already does) rather than by Lemma 3.*

## M1 — MATERIAL OVERCLAIM (H1, `HYPOTHESIS_REGISTER.md`). The refutation does not follow

**Claim.** "**Status: H1 REFUTED.** … forget-forget mean cos = **+0.067**, and the 64-fact forget
gradient set has effective rank **29.48 of 64** — not low-dimensional."

**Objection 1 — mean cos 0.067 over 4032 pairs predicts substantial transfer, because coherent
summation amplifies the shared component by √n.** Computed exactly from the stored matrix (no
model needed), with `S` the 8 leaked facts and all vectors unit:

| quantity | value |
|---|---|
| `‖Σ_{i∈S} ĝ_i‖` | **3.213** (2.828 if exactly orthogonal, 8 if collinear) |
| `cos(Σ_S ĝ, ĝ_j)`, j leaked | 0.402 |
| `cos(Σ_S ĝ, ĝ_j)`, j held-out **non-sharing** | **0.119** |
| `cos(Σ_S ĝ, ĝ_j)`, j held-out sharing | 0.376 |
| mean pairwise cos, leaked vs non-sharing held-out | 0.048 |

So one adversary step along the aggregate direction reduces a **non-sharing held-out** fact's loss
at **29.7%** of the rate at which it reduces a leaked fact's loss, to first order. Ten steps of
that is more than enough to cross a greedy decoding threshold — and `0.514` of them did recover.
The toy model confirms the mechanism: with a shared component of energy fraction `a² = 0.067`, the
8-fact sum has `36.5%` of its energy in the shared direction and `cos = 0.156` with every held-out
gradient. **A mean pairwise cosine of 0.067 is not evidence against a shared direction; it is a
quantitative prediction of ~30% transfer efficiency.** The register's own wording — "near-orthogonal"
— is the error.

**Objection 2 — effective rank 29.48/64 is a large deviation from isotropy, not evidence against
low-dimensionality.** 64 i.i.d. unit vectors in high dimension have participation ratio **63.98**
(simulated, d = 2e5). Measured: 29.57 (forget), 31.28 (retain). Top-10 energy is **42.2%** against
15.6% for isotropy — a 2.7× concentration. "Not low-dimensional" is true only against the strawman
"≈3".

**Objection 3 — internal inconsistency: 29.48 ≈ 32 = the number of distinct answers, which is the
signature of the replacement hypothesis.** The 64 forget facts carry 32 distinct answers, paired as
`(i, i+32)`, with within-pair cos 0.916. Collapsing each pair to one direction predicts PR ≈ 32.
Measured 29.48. And the one-fact-per-answer subset has PR 27.19 of 32 (85% of maximum, vs 46% for
the full 64). So **the rank deficiency is the answer-pair structure** — the same number is used on
one page as evidence against a shared direction and on the next page as the mechanism.

**Objection 4 — the control gap is inside the measurement's numerical error.** The refutation's
"not specific to forgetting" rests on 0.067 (forget) vs 0.055 (retain), a gap of 0.012. The Gram
matrix's own self-cosines reach 1.0123 (I6), so ±0.012 is the accuracy floor.

**What this changes.** H1's *distinguishing operation* (coherence penalty / orthogonal projection /
per-fact subspace allocation) is untouched by the measurement, and H1's *pre-registered observable
prediction* — "per-fact recovery should fall specifically for facts whose removal directions are
near-orthogonal to the leaked facts', and per-fact recovery should correlate with that measured
cosine" — was then **confirmed** at r = +0.384, p = 0.011 over 43 facts, in the very finding that
declares H1 refuted. (That confirmation is itself hollow, per F3 — but the register cannot have it
both ways: the same statistic cannot refute H1 and confirm its replacement.) The honest status is
**H1 refined, not refuted**: the shared subspace is answer-token-indexed with ~32 directions rather
than global with ~3, and whether an anti-coherence constraint helps is still open and still cheap
to test analytically.

Also: the register's "rank-capacity bound … at most `r` mutually orthogonal directions exist in a
rank-`r` adapter, so 64 facts cannot have 64 orthogonal directions unless `r ≥ 64`" is correct, and
the register's own later demotion of the rank sweep to a control is the right call. Note the measured
PR of 29.5 at r = 16 per module is not a contradiction: the 224 per-module rank-16 subspaces are
distinct, so the global edit subspace has dimension up to 224·16, not 16.

## M2 — MATERIAL OVERCLAIM. The ε cliff is numerology

**Claim** (Cor 2.2, §8.3(v), R6). "`ε = 10⁻⁵` … because `10⁻⁵ > 1/|V|` only by a factor 1.26, puts
the greedy outcome *on a knife edge* — any transfer that pushes `p_θ(F|s_B,i)` below `7.91 × 10⁻⁶`
zeroes ψ₁ discontinuously. **Prediction:** the shared-object-token retain slice should be
**bimodal across items**."

**Arithmetic: correct.** `1/126464 = 7.9074e-6` ✓ ("7.91e-6"); `1e-5 · 126464 = 1.2646` ✓
("factor 1.26"); `L·ε = 128·1e-5 = 1.28e-3` ✓. At the real `gen_length = 32` the T=1 bound is
`3.2e-4`.

**Objection 1 — ε and `p_θ(F|s_B,i)` are not commensurable quantities.** `ε` is the target's F-mass
at **Alice's** states, imposed by the loss. `p_θ(F|s_B,i)` is the realized model's F-mass at
**Bob's** states, which the loss never reads (Prop 5.1). §8.3(ii) gives the only stated link:
`Δz_{v_P}(s_B) ≈ −η Σ K_{θ₀}(·,·)(q−ε)₊ p₀|_F(v_P)`, whose magnitude depends on `η`, the NTK, and
the number of active states — and bears **no** relation to `ε` itself. There is no mechanism by
which the realized `p_θ(F|s_B)` lands near `ε`. The "factor 1.26" compares a hyperparameter to a
vocabulary reciprocal; the proximity is a coincidence of units.

**Objection 2 — `1/|V|` is not where the cliff is.** Lemma 4's necessary-and-sufficient greedy
condition is the **rank flip** `max_{v∈F} p < max_{v∉F} p`. `p(F) < 1/|V|` is only the
*unconditional* sufficient threshold — the point below which the flip is guaranteed regardless of
the rest of the distribution. For a strongly-known fact (`p₀(F) = 0.9`, runner-up 0.03) the rank
flips when `p_θ(F|s_B)` falls below ≈ 0.03, i.e. **four orders of magnitude above 7.91e-6**. By the
time Bob's F-mass reaches 1e-5, Bob has been dead for four decades of probability. The relevant
transition is at the runner-up mass, is item-dependent, and is nowhere near `1/|V|`.

**Objection 3 — the bimodality prediction does not come from this argument.** At `T = 0` every
per-item accuracy is a deterministic 0/1 variable, so an 8-item mean is automatically a binomial
proportion and "bimodal across items" is the generic property of greedy scoring. The prediction is
true, non-specific, and not derived from the stated premise. The actionable recommendation (report
per item with a count of zeroed items) is right for this mundane reason.

**Verdict.** Delete the `1.26` sentence and the "knife edge". Keep: *at `T = 0` the shared-token
retain slice is a 0/1 variable per item whose transition is at the item's own runner-up mass, so
report it per item with a zeroed count, never as an 8-item mean.* R6's conclusion survives on
Prop 2.3 alone.

## M3 — MATERIAL OVERCLAIM. Lemma 5's condition is wrong; counterexample

**Claim.** "`conf` falls from `p₀(v*)` to `(1−ε)·max_{v∉F} p₀(v)/(1−p₀(F))`, which is `< p₀(v*)`
whenever `max_{v∉F} p₀(v) < p₀(F)(1−p₀(F))/(1−ε)`."

**The condition is algebraically wrong.** Writing `m = max_{v∉F} p₀`, `q = p₀(F)`:

> `(1−ε)m/(1−q) < p₀(v*)  ⟺  m < p₀(v*)(1−q)/(1−ε)`.

The document writes `q` where `p₀(v*)` belongs. Since `argmax ∈ F` we have `p₀(v*) ≤ q`, with
equality **iff F carries all its mass on a single token**. So the stated condition is strictly
weaker than the true one and is **not sufficient** whenever F is multi-surface — which is the case
§8.4, Cor 8.5 and R5 are entirely about.

**Counterexample.** `q = p₀(F) = 0.9` spread over three surfaces (`"Paris"`, `" Paris"`, `"paris"`)
at `0.30` each, so `v* = 0.30` and `argmax ∈ F`; runner-up `m = 0.05`; `ε → 0`.
- Document's condition: `0.05 < 0.9 · 0.1 / 1 = 0.09` ✓ — so it predicts confidence falls.
- Truth: `new conf = 1 · 0.05 / 0.1 = 0.50 > 0.30 = old conf`.

**Confidence rises by 67% and the position commits *earlier*, not later.** Lemma 5's conclusion
inverts exactly in the multi-surface regime the document elsewhere insists on. The worked example
in the lemma (`p₀(F) = 0.9`, runner-up `0.03` ⇒ `0.90 → 0.30`) is arithmetically correct but
silently assumes `|F| = 1`.

**Second defect — "loses its place in the commit order" needs an unstated hypothesis.** `C_n` is
top-k by confidence. A fall in absolute confidence changes the *rank* only if some competing
position's confidence lies in `(new conf, old conf)`. The exclusion target leaves `p₀` untouched
wherever `q ≤ ε`, so most positions are unchanged — but if the fact position's new confidence is
still the block maximum, it commits first regardless. The lemma needs "and some eligible `j` has
`conf_j ∈ (conf_new, conf_old)`", which is a statement about the state, not about the edit.

**Third defect — the displacement is bounded and the document does not say so.** At the frozen
config there is a **single block of 32**, and at the last step of a block `k_n = |E_n|`, so every
position commits within its block regardless of confidence. The displacement is at most 31 steps
and the "richer model-written context" it conditions on is at most the other 31 positions of the
same canvas. The `[H]` re-entailment hazard is correspondingly bounded; the document's framing
("the model first commits '…the capital of France…', after which the object position is forced")
requires a canvas long enough to host that clause, which 32 tokens barely is.

**What survives.** R4 — on-policy re-collection and a declared staleness — is correct, because the
edit does move the state distribution (the denoiser changes, hence confidences change, hence the
realized schedule changes). What does not survive is the *signed* claim that suppression
**provably** delays the fact position. Restate Lemma 5 as: *the edit changes `conf(s,i)` at the
fact position by `(1−ε)m/(1−q) − p₀(v*)`, whose sign depends on `p₀(v*)` relative to `m(1−ε)/(1−q)`
and is negative only when F's mass is concentrated on `v*`; in either direction the realized commit
order is θ-dependent, so `μ` is self-referential.*

## M4 — MATERIAL OVERCLAIM. Theorem 5.5 is vacuous at every config run, and "no fourth option" is false

**Claim.** "Fix a state `s` and a step that commits positions `i ≠ j` … Then
`Pr[z_i ∈ A ∧ z_j ∈ B] = p_θ(A|s,i) · p_θ(B|s,j)`." Corollary 5.6: "A method must therefore
(i) restrict `Π` to `k = 1` configs and say so, or (ii) declare and report the floor, or
(iii) intervene in the commit schedule — which is an inference-time hook and is forbidden by §3.
**There is no fourth option.**" Promoted to R14.

**Objection 1 — it is vacuous at the project's config.** Every frozen spec is `gen_length = 32,
steps = 32, block_length = 32`: one block, 32 positions, 32 steps ⇒ **exactly one commit per step**.
There is never a parallel commit, so the hypothesis "a step that commits positions `i ≠ j`" has
probability 0. The same is true of the config Cor 3.1 quotes (128 positions, 128 steps, 4 blocks ⇒
k = 1). **Theorem 5.5 has no instances anywhere in this project**, and R14 is derived from it as if
it constrained the current design.

**Objection 2 — the exact factorization fails under `lc`, as asked.** Step 2's draws `x̂_i` are
independent across positions. But step 3 sets `conf_i = p_θ(x̂_i|s,i)` — *a function of the realized
draw* — and step 4 takes the top-k. So `C_n` is a function of all the draws, and **conditioning on
`{i, j ∈ C_n}` couples `z_i` and `z_j`**. The factorization is exact only under `rnd`, where `conf`
is θ- and draw-independent and `C_n ⊥ (x̂_i, x̂_j)`.

The direction of the failure, which the document should state: under `lc`, co-commission selects
for *both* draws having been high-probability, so for a strongly known fact
`Pr[z_i ∈ A ∧ z_j ∈ B | i,j ∈ C_n] > p_θ(A)·p_θ(B)`. **The non-independence makes the floor worse,
not better** — so the theorem's conclusion is not rescued by this defect; if anything it is
strengthened. I checked for the hoped-for counterexample and it is not there.

**Objection 3 — the theorem bounds the wrong event.** Disclosure is a property of the terminal
canvas, not of one step. Positions `i` and `j` are each committed exactly once, at possibly
different steps; if they are committed at different steps the second conditions on the first and
there is no factorization at all. So `Pr[disclosure]` decomposes over (step-pair) events and the
product bound applies only to the sub-event `{i, j committed in the same step}`, whose probability
the document never bounds. At k = 1 it is 0.

**Objection 4 — the fourth option exists, and the document's own Lemma 5 supplies it.** Shape the
confidences so that conjunction-critical positions are **not co-committed**: make `conf` at the
subject position dominate, so the object position is deferred and then conditions on the realized
subject. This is a property of `θ`, trainable by an ordinary objective on
`conf(s,i) = max_v p_θ(v|s,i)`, requires **no inference-time hook**, and is therefore not forbidden
by §3. Lemma 5 is precisely the proof that `θ` controls the commit order. Cor 5.6's "there is no
fourth option" is contradicted on page 3 of the same document.

**What survives.** If `Π` is ever widened to configs with `steps < gen_length` (k ≥ 2), a
conjunction floor exists and should be reported. State R14 conditionally on that, state that the
current `Π` has k = 1, and add the confidence-shaping option.

## M6 — MATERIAL OVERCLAIM. Theorem 5.3's quantifiers, three defects

**Claim.** "`Adv_𝒜(θ; φ) ≥ (1−κ)·Pr_{G_{θ₀}}[ψ disclosed] − (entailment failure rate)`. Hence
`Adv_𝒜(θ) ≤ δ` and P1-on-`ψ` are **jointly infeasible** for small `δ`."

**(i) The baseline is dropped.** Def 1.2 defines
`Adv = sup_A [Pr[A(q,y) = O] − Pr[A₀ = O]]`, subtracting the type-prior guess. Theorem 5.3's RHS
omits `Pr[A₀ = O]`, so the stated bound is on raw adversary success, not on `Adv`. The correct
bound is `≥ (1−κ)·Pr[ψ]·(1−e) − Pr[A₀ = O]`. With `|𝒪|` cities the correction is small but it is
the difference between a statement about `Adv` and a statement about something else.

**(ii) The adversary is not in the class Def 1.2 quantifies over.** Def 1.2's probability is over
`q ∼ ν_φ` — the **forget** query measure — and `y ∼ G^π_θ(·|q)`. The entailment attack queries `ψ`,
i.e. `q ∉ Q_φ`. "Let `𝒜` be closed under one additional query and the entailment" silently replaces
the non-interactive read-off adversary with an **interactive** one whose query measure is not
`ν_φ`. Def 1.2 has no such adversary in it. Since §6.3's claim table and R13 are indexed on `𝒜`,
this is not cosmetic: Def 1.2 needs an explicit interactive extension
(`𝒜` = policies mapping transcripts to queries, with a query budget) before Theorem 5.3 is about
the same object.

**(iii) "Hence jointly infeasible" needs a hypothesis that is not stated.** The bound is a lower
bound that can be ≤ 0. Infeasibility requires `(1−κ)·Pr_{G_{θ₀}}[ψ] − e > δ`, with `e` the
entailment failure rate. In the document's own running example the background premise is "Alice
lives where she works", which is exactly the kind of premise with a large `e`. State the
hypothesis.

**What survives.** The direction and R13 ("declare `cl(φ)` as an input; either close the forget set
under entailment or restrict the claim to the read-off adversary, explicitly") are right and
important, and §5.3's resolution — that the question is a specification input, not a mathematical
fact — is the correct reading.

## M7 — MATERIAL OVERCLAIM. Prop 8.4 is correct algebra, mis-stated severity, and not independent of Prop 8.2

**Claim.** "`r_ε(v'|s,i) = (1−ε)·p₀(v')/(1−q) > p₀(v')`, with amplification factor
`(1−ε)/(1−q) → ∞` as `q → 1`." Headed "**the strongest objection**".

**The algebra is correct.** `(1−ε)/(1−q) > 1 ⟺ q > ε` ✓; divergence as `q → 1` ✓; largest exactly
where the fact is most strongly known ✓. Verified, no objection to the identity.

**Objection — the amplification is not selective, and at `T = 0` it does not change what is
emitted.** The renormalization multiplies **every** `v ∉ F` by the identical factor
`(1−ε)/(1−q)`. So the exclusion target is **rank-preserving on `F^c`**: the relative order of all
non-F tokens is exactly `p₀`'s. At `T = 0` — the only temperature any study in this project has run
— Lemma 4 says rank is the whole story. Therefore an omitted surface `v'` changes the emitted token
**only if `v' = argmax_{v∉F} p₀`**, in which case the emitted token is precisely Prop 8.2's fixed
point. Otherwise `v'`'s absolute mass rises by `1/(1−q)` and nothing is emitted differently.

So (a) "actively promotes the omitted variant" is true of absolute mass and false of the quantity
that decides disclosure at `T = 0`; (b) **Prop 8.4 and Prop 8.2 are the same phenomenon**, and
presenting them as two independent objections (one of them "the strongest") double-counts. At
`T > 0` the mass effect is real but is shared by every non-F token.

**Consolidated correct statement.** *The exclusion target's `T = 0` fixed point is
`argmax_{v∉F} p₀`. That is degeneracy (d) when the token is a plausible wrong answer, and a
disclosure when it is an untabulated surface of `O`. One failure mode, two readings.* R5 and
Cor 8.5 (prefer a target that never references `F`) follow unchanged and are the document's
strongest actionable conclusion.

## I1 — Theorem 4.5 is a tautology, labelled [P]

**Claim.** "(F) makes the indicator 1 on `Q_φ`, (P) makes it 0 off `cl(φ)`. ∎ … So 'no query
detection' is **not** an achievable requirement."

**It is a tautology, and I will say so plainly.** The content is: *a function that behaves
differently on A than on B defines the indicator of A vs B.* That is the definition of selective
behaviour; the proof is one line because there is nothing to prove. It is also quantified only over
`{q : Adv(θ₀;q) > δ'}` ∩ the probed region, so it says nothing about prompts the reference cannot
answer. And the inference equivocates on "computes": every selective edit "computes" a membership
function in this sense, including a model that genuinely never learned `φ`; degeneracy `𝒟_e` was
about something else entirely — a **sharp, non-generalizing** test.

**No downstream damage**, because the document immediately states the right distinction (a gate
that is the semantics vs a gate that is a shortcut) and the three tests in §4(e) are the real
content. But Theorem 4.5 does no work, and citing it as licensing the retirement of the "no
gating" requirement dresses a definitional observation as a result. Demote it to a remark.

## I2 — Lemma 1 / Corollary 1.1: correct, definitional, and one use reads as reassurance

**Checked, and it holds.** A state is defined as `s = (c, z)`, so states with distinct prompts are
distinct objects and `R(θ,π,c) ∩ R(θ,π,c') = ∅` is immediate from the definition. **There is no
counterexample to construct**, and semi-autoregressive block decoding is irrelevant to it: in a
bidirectional denoiser every position sees the whole prompt at every step, which if anything makes
the prompt *more* firmly part of the state. (The one edge case: if two distinct prompts map to the
same input after chat-template truncation or context-window clipping, their states coincide. Not a
live concern at `gen_length = 32`.)

**The objection is to the inference, not the lemma.** All the content is in the model class.

- **Theorem 6.1** — valid in the unrestricted class, and the document flags the parametric caveat
  honestly. Note a proof defect: "Modify the denoiser only on `R(θ', π, q*)`" is not a construction,
  because the modified model's reachable set from `q*` is no longer `R(θ',π,q*)` — if the trajectory
  leaves the modified region, behaviour reverts. The fix is one word: modify θ on **all states whose
  prompt is `q*`** (a set disjoint from every probe prompt's states by Cor 1.1), then `θ''`'s
  trajectory from `q*` stays inside it by construction. Theorem true, proof not closed as written.
- **Prop 8.1** — valid, and explicitly contrasted with §8.3's parameterization argument. Fine.
- **§8.5 row C1** — "`supp μ` contains no retain-reachable state … **holds automatically** by
  prompt-disjointness". Literally true and **vacuous**: it is true because the state includes the
  prompt, and it has nothing to say about the thing that actually damages ψ₁, which is parameter
  sharing (§8.3). In a verdict table of five conditions, a row reading "holds automatically" invites
  the reader to stop worrying about exactly the failure the next section proves. Re-word as:
  *C1 holds trivially in the per-state sense and carries no information about parameter-space
  transfer; see C2.*

## I3 — Lemma 4's "iff" is false in the necessary direction

**Claim.** "`Pr[disclosure] = 0` **iff** at every reachable `F`-free state and eligible `i`:
`max_{v∈F} p_θ(v|s,i) < max_{v∉F} p_θ(v|s,i)`."

**Sufficiency holds. Necessity fails.** Disclosure at position `i` requires `argmax ∈ F` **and**
`i ∈ C_n`, and `C_n ⊆ E_n` strictly whenever `k_n < |E_n|`. So a position may have `argmax ∈ F` at
step 3 while another position outranks it on confidence; after that commit the canvas changes,
and at the step where `i` is actually committed its argmax may be off F. Zero disclosure, stated
condition violated. The correct statement quantifies over committed positions:
`∀ s ∈ R, ∀ i ∈ C(s)`.

**Why it matters.** R3 calls the rank condition "achievable at exactly zero leakage and **finitely
checkable**". The finitely-checkable version is the all-eligible-positions condition — which is
strictly stronger than necessary and therefore over-constrains, in exactly the way the lemma's own
Consequence paragraph complains about ("it buys mass reduction far past the rank flip"). The
honest version is: the all-eligible condition is a finitely checkable **sufficient** certificate;
the exact condition is a fixed-point condition on the realized commit sets, which is Lemma 5's
subject.

## I4 — Lemma 2's scope omits CFG

Lemma 2's two-number projection is correct at `T = 0`, `lc`, **`cfg_scale = 0`**. But §0 defines
`π = (N, B, remask, T, cfg)` and `Π` is not restricted, and with `cfg ≠ 0` the sampler mixes
conditional and unconditional logits, so it reads more than `(argmax, max)` per position. Every
frozen spec has `cfg_scale = 0.0`, so this is scope hygiene rather than an error — but Lemma 2 is
cited as a structural fact about "the sampler" and should name the restriction.

## CORRECT in `FORMALIZATION.md` (checked, no objection)

- Cor 3.1's arithmetic as stated (eligible sum 2112; 2.367e-5; 1.121e-9). The *inference* is F4.
- `1/|V| = 7.907e-6`; Lemma 4's unconditional sufficiency `p(F) < 1/|V| ⇒ argmax ∉ F` (since
  `max_v p ≥ 1/|V|`); Theorem 2.1(i) and (ii) including `Σ_n k_n = L`; `L·ε = 1.28e-3`.
- Cor 6.2's Clopper–Pearson: `1 − 0.05^(1/1000) = 2.991e-3` ✓; `1 − (10⁻⁴)^(1/1000) = 9.168e-3` ✓.
  §6.1/§6.2's resolution-vs-design-quantity point (R15) is the most useful arithmetic in the
  document and is correct.
- Prop 8.2's fixed point and its entropy identity. I verified
  `H(r_0) = [H(p₀) − H_b(q) − q·H(p₀|_F)]/(1−q)` by direct expansion; it is exact.
- Prop 2.3, Prop 2.5, Prop 5.1, Thm 6.3's and Thm 6.4's statements, Cor 2.4, Prop 8.3.
- R5, R7, R9, R10, R11, R12, R15, R16, R17 are each independently supported.
- §4's degeneracy taxonomy and the audit of what current metrics do and do not catch is the
  strongest section in the document and I found nothing to attack in it.

---

# 3. `ERRATUM_relearn_a67d610.md` and `ERRATUM_arc_28154b2.md`

## M5 — MATERIAL OVER-RETRACTION. Item 1's conclusion survives, and the corrected probe is already on disk

**The defect was real.** `noise_probe.py:103` used `lora_targets()` = 128 attention modules
against a 224-module edit holding 71.3% of its energy in the MLP projections. The energy table in
the erratum is correct and the statement "the experiment does not test what it claimed" is correct.

**But the corrected experiment has landed and it confirms the withdrawn conclusion.**
`outputs/noise_unlearned.json` (written 12:29 — *after* both the erratum and the register entry
that says "a corrected probe is running" / "E4 is also withdrawn pending re-measurement"),
`scripts/noise_probe_full.py` with
`FULL_TARGETS = .*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$` — all 224
modules, perturbation scaled so `‖ΔW‖_F = frac × 11.5858`:

| frac | ‖perturbation‖_F | forget disclosure (n=32) | retain disclosure (n=16) | degenerate |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.031 (1/32) | 1.000 | 0 |
| 0.25 | 2.896 | 0.094 (3/32) | 1.000 | 0 |
| 0.50 | 5.793 | 0.062 (2/32) | 1.000 | 0 |
| 1.00 | 11.586 | 0.062 (2/32) | 1.000 | 0 |
| **2.00** | **23.172** | **0.062 (2/32)** | **1.000** | **0** |

Every value is within ±2 probes of baseline. Fisher exact 3/32 vs 1/32 one-sided p = 0.31; the
Wilson upper bound on the worst cell (3/32) is 0.242. Against the directed attack's
26/43 = 0.605, Wilson [0.456, 0.736]. The intervals are disjoint by a wide margin.

**So the original §3 conclusion — "recovery is directional, not positional; recovery needs gradient
information about the forget set, not parameter displacement" — survives the corrected,
full-coverage measurement.** Leaving it withdrawn is now itself an error, and `HYPOTHESIS_REGISTER`
E4 should be reinstated with the corrected numbers (0.031 → 0.062 at 2× full-edit norm, rather than
the original "0.031 at 2×"). Both documents are stale; the erratum should carry a follow-up note,
as it promised ("its result will be published as a follow-up regardless of which way it lands").

**Three caveats to attach — reasons to narrow the claim, not to withhold it.**

1. **The perturbation is a random *rank-16* LoRA**, not a full-rank displacement. Per module it
   lives in a random 16-dimensional subspace of a ~4096-dimensional output space, which is nearly
   orthogonal to the edit's ~3 dominant directions by construction. A full-rank isotropic
   perturbation at the same norm is a strictly stronger probe and has not been run.
2. **The probe is close to functionally inert at 2× norm** — retain disclosure 1.000 and
   `degenerate = 0` at every frac. Compare: the *edit itself*, at norm 11.586, takes forget
   disclosure from 1.000 to 0.031. So a displacement of twice that norm in a random direction
   changes nothing anywhere. That is the content of "directional", but it also means the experiment
   does not separate "the fact is stored in a specific direction" from "an 8B model is insensitive
   to random low-rank perturbations at any norm". The defensible claim is the narrower one:
   **function-space sensitivity is strongly anisotropic, and the edit's Frobenius norm is not a
   meaningful scale for recovery.** The stronger reading ("the information is localized to a
   direction") is not established by this probe.
3. **The headline comparison mixes denominators** — 0.062 on 32 recalled forget facts (noise probe)
   vs 0.605 on 43 held-out recalled facts (relearn probe). This is the exact defect erratum item 2
   corrects elsewhere. Quote each probe against its own baseline: 0.031 → 0.062 under noise;
   0.023 → 0.605 under the directed attack.

## Items checked and correct

**ERRATUM_relearn:**
- **§2** (0.049 → 0.605 mixes denominators; corrected 0.023 → 0.605). Correct, and the Wilson CI
  recomputes exactly: 26/43 → [0.456, 0.736]. The observation that `0.023 = 1/43` is the
  quantisation unit is correct and is the right thing to carry forward.
- **§3** (underpowered defense comparison). Correct and correctly derived. Δ = −0.023 = exactly
  1/43. The claimed MDE of 0.264 reproduces as the **two-sample, one-sided, α = 0.05, power 0.8**
  figure: `(1.645 + 0.842)·√(2·0.6·0.4/43) = 0.263`. (Two-sided two-sample gives 0.296; one-sample
  against a known `p₀` gives 0.209, which would *not* exceed the 0.20 bar — so the erratum should
  name which test it used. With the two-sample test, which is the right one here, the conclusion
  holds.) "Establishes that no large effect was present, not that no effect exists" is the correct
  reading.
- **§4, §5, §6, §7**: all correctly scoped. §4's "the mechanism may have been applied to the wrong
  29% of the edit" is the right framing. §6's 7 diacritic misses (0.4%, erring in the flattering
  direction, disclosed rather than silently fixed) is handled correctly.

**ERRATUM_arc:**
- **§1** (K=512 IN-pool/ABSENT retraction). Correct and correctly scoped. `analyze_big.py:17` loads
  `big_arcK16/report.json`'s `pool_membership` and applies it to all four arms; `arcK512` records
  `n_facts_in_pool = 64/64` and therefore has no absent set, so its split is a partition by another
  arm's pool. The extension of the objection to the `theta0` and `uncached` rows is right. The
  carve-out — that the **K=16** per-fact split (ABSENT 0.951 vs IN-pool 0.850, gap +0.101) is
  correctly computed because K=16's membership file is the one actually loaded, so the refutation of
  the pre-registered per-fact prediction is unchanged — is correct, and is the kind of carve-out
  that errata usually get wrong. No over-retraction here.
- **§3** (the memorization backbone is not a qualified checkpoint: `KeyError: 'aliases'` at `:199`
  after training but before the gate block, so G3/G4 never ran and the hyperparameters are not
  recoverable; 1,980 of 2,400 steps logging `loss=0.0000 gnorm=0.000`). Correctly characterized as
  not changing relative comparisons but invalidating absolute 128-fact numbers. This is a more
  serious finding than its position in the list suggests.
- **§4** (G3 is unreachable dead code: `__import__("common")` with no `common.py` on that path) and
  **§5** (the in-code claim of a `p0(F)` diagnostic is false; `exclusion_target` still returns
  `r = p₀` silently for an empty fact set). Both correct. §5's closing observation — "the class of
  defect remains undetected by construction" — is the actionable half and should be a forward
  requirement, not a note.
- **§2** (θ₀ unmerged vs arms merged) in both errata. Correct and appropriately flagged as an
  unmeasured confound rather than a known-direction bias.

**No item in either erratum understates a defect. One overstates: `relearn` §1.**

---

# 4. Summary of required actions

| action | because |
|---|---|
| Stop using `FINDING_answer_channel.md` as the method's grounding; rewrite its causal section | F1 — θ₀ run shows the coupling is intrinsic, equal on untouched retain facts, at 1.3e5 smaller norms |
| Recompute both cosine matrices with fp32 storage; discard the stored θ₀ matrix | I6 — 116 exact-zero cosines; ≤0.023% of coordinates representable; self-cosines to 1.0123 |
| Re-score the relearn probe by *which answer is emitted*, and stratify by relation | F2 — 9/9 non-sharing instrument facts emit a leaked answer; p 0.011 → 0.198 without that stratum |
| Change H1's status from REFUTED to REFINED; keep the anti-coherence operation live | M1 — 0.067 ⇒ 29.7% transfer efficiency; PR 29.5 ≈ 32 = #answers; control gap inside numerical error |
| Reinstate E4 with the corrected numbers and the three narrowing caveats | M5 — full-coverage probe on disk confirms it |
| Delete Cor 3.1's impossibility claim; keep R2's recommendation on other grounds | F4 — false at `steps=32`; 272× counting error; `lc` out of Lemma 3's scope |
| Restate Lemma 5 with the correct condition `m < p₀(v*)(1−q)/(1−ε)` and an unsigned conclusion | M3 — counterexample at multi-surface F: confidence rises 0.30 → 0.50 |
| Make R14 conditional on `k ≥ 2`; add confidence shaping as the fourth option | M4 — every config is k=1; Lemma 5 supplies the option Cor 5.6 denies |
| Delete the `1/\|V\|` cliff; keep per-item reporting of the shared-token slice | M2 — ε and `p_θ(F\|s_B)` are not commensurable; the real threshold is the runner-up mass |
| Merge Prop 8.4 into Prop 8.2; add the interactive-adversary extension to Def 1.2 | M7, M6 |
| Demote Thm 4.5 to a remark; fix Thm 6.1's proof wording; fix Lemma 4's "iff"; scope Lemma 2 to `cfg=0` | I1, I2, I3, I4 |
