# Method plan — derived from the matched baseline measurement

Written 2026-10-20 after the matched comparator landed. Every claim below is tied to a
measurement already on disk, or is labelled as an untested hypothesis with the experiment
that decides it.

## 1. The measurement the method has to explain

Matched backbone, data, LoRA, optimizer, step budget, evaluator; declared per-arm sweeps
(`BASELINE_MATCHED.md`):

| arm | DISCLOSE | DEGEN | ABSTAIN | REPLACE | IGNORANT | retain acc | retain margin |
|---|---|---|---|---|---|---|---|
| memorized reference | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 12.16 |
| MDU, all three τ | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | 1.000 | 9.55 |
| EMDU, all three τ | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | 1.000 | 8.84 |
| NPO, β ∈ {0.05, 0.2} | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | 1.000 | 8.64 |
| SC-TF | 0.469 | 0.141 | 0.000 | 0.375 | 0.016 | 1.000 | 6.66 |
| exclusion target (ours) | 0.453 | 0.219 | 0.000 | 0.312 | 0.016 | 1.000 | 6.38 |

**The failure is strictly bimodal.** Nine of fifteen grid points reach *total output
collapse* — MDU and EMDU emit `''`, NPO emits `'........'` — on all 64 forget probes, while
retain accuracy still reads 1.000. The two fluent arms disclose ~0.45 and substitute
~0.31–0.38. **Nothing lands in between**: across 960 forget generations the IGNORANT cell
holds 4 rows and ABSTAIN holds 0.

Two further measured facts the method must respect:

- **Retain accuracy is blind.** It is 1.000 at 14 of 15 grid points while the retain *margin*
  falls 12.16 → 6.38 nats (−47%), with two points already holding a retain probe at
  margin ≤ 0. Any preservation claim must be stated on the margin.
- **DISCLOSE is a lower bound.** Both fluent arms emit `巴黎` (Chinese for *Paris*) against
  the reference answer *Paris* — a cross-script disclosure no span rule sees.

## 0-PRE. STRATEGY AMENDMENT 2 — C1 is a published baseline that FAILS; the method problem is now sharply posed

The full read of arXiv 2506.06278 landed (`PRIOR_ART_C1.md`) and it changes the shape of the
contribution more than it changes the novelty position.

### What it says

**C1 is in that paper already, as a named baseline, and its section heading is
"Oracle Matching Does Not Guarantee Robust Unlearning."** The literature's term is
**oracle matching**, not "retain-reference distillation" (which returns zero hits — we were
searching for the wrong phrase). Their teacher is a model pretrained on retain data only;
their loss is forward KL to it; plain loss descent. That is C1's mechanism exactly.

**Overlap verdict: category (a), exact equivalence**, for the training signal and teacher
construction in the autoregressive setting. The LoRA / position-masking / explicit-retain-term
deltas are each category (c), a familiar combination. The diffusion port is category (e) —
unestablished benefit — and only becomes (d) if we *demonstrate* a cross-mask-ratio
interaction rather than assert one.

We were also comparing against the wrong half of that paper: **UNDO is not the overlapping
part.** UNDO's teacher is an already-*unlearned* model distilled into a noised copy.

### Three things in our favour, stated precisely

1. **Their negative is only about relearning robustness** — 500 relearning steps, worst-case
   over a 10-value LR sweep. Prompt-based elicitation is motivation in their intro and is
   never evaluated. **Our axis is the composition failure** (collapse versus substitution),
   which they do not measure. Whether oracle matching fixes composition is open.
2. **Their result is evidence *for* C1 on behavioural forget quality** — they report the
   distillation loss tending to zero, i.e. near-perfect behavioural agreement with the
   oracle. So the thing we would be using it for is the thing they showed works.
3. **Diffusion is unoccupied.** MDU and EMDU both anchor on the *pre-unlearning checkpoint
   with the prompt masked* — not on a retain-trained model — and **neither reports a
   Retrain/gold baseline at all.** Image-diffusion erasure uses the retain model as an
   evaluation gold standard everywhere and as a training teacher nowhere.

### The reframe this forces, and it is the useful part

**C1 is an oracle, not a method.** It requires a model trained without the forget data, which
does not exist at deployment. Both precedents say so explicitly and then build
approximations — Georgiev (arXiv 2410.23232) via data attribution, Unilogit via synthetic
targets ("*since θ_r is unavailable in reality, we must approximate these targets*"). **Both
known escape routes are therefore occupied.**

So the method problem is now sharply posed, which it was not this morning:

> **Establish whether the empty cell is reachable at all, using the oracle. If it is, the
> method problem is to approximate that skyline WITHOUT a retain-trained model, by a route
> that is neither data attribution nor synthetic-target construction.**

That is a better problem statement than "design a forget objective", because it has a
measurable upper bound (the oracle), a clear deployability constraint, and two named
occupied routes to avoid.

**H10 becomes more important, not less.** It is not a target-approximation at all — it is an
ordering intervention on the sampler — so it is a candidate *third* escape route, in a
region neither precedent occupies. Its oracle test (`h10_oracle.py`, queued) decides whether
it is viable.

### A second, independent finding worth more than the novelty question

**TOFU's Forget Quality is a KS-test p-value against the retain model's Truth-Ratio
distribution.** So a method trained to match the retain model is trained on a close surrogate
of its own metric, and **a strong C1 forget-quality number on TOFU is near-uninformative.**
No paper states this circularity. It composes with our separate finding that nobody has
reproduced MDU's table (including EMDU using MDU's own code) into a real contribution about
how this field measures.

It also constrains us immediately: **we must not report C1 under TOFU's Forget Quality** and
call it a result.

### Revised contribution shape

| component | standing |
|---|---|
| C1 as the MDLM field's **missing oracle / skyline** | honest and useful — MDU and EMDU have no gold baseline; immune to the deployability objection because an oracle is not meant to be deployed |
| The **TOFU circularity** measurement | novel as far as every primary source we have read |
| The **composition diagnosis** (target mode selects failure mode) | novel; Phase B tests it |
| **H10** as a deployable third route | the only candidate that could carry a *method* claim; oracle test pending |

A method claim requires H10 (or a successor) to clear its own oracle and then beat the
mass-matched structural-upweight control. C1 alone cannot carry it, and I will not present it
as if it could.

### Process note

The prior-art read surfaced **two summariser-mediated fetches that returned confidently
fabricated content** — a complete hyperparameter table contradicting the paper's own
abstract, and two quotations attributed to a survey that do not appear in its text. Every
quantitative claim in that document was re-derived from locally extracted PDF text. Both
incidents are recorded there. Treat summarised fetches of primary sources as untrusted.

---

## 0. STRATEGY AMENDMENT — critical path to a method, not to a complete diagnosis

Re-prioritised on instruction to focus on the end goal: **a method that beats the baselines
on a forgetting–utility trade-off, with real-model evidence.** Three changes.

### 0.1 The diagnosis is no longer gating. It runs in parallel.

Phase B (target-mode decomposition) is *explanatory*, not a prerequisite. The method it
points to can be built from what is already measured. So B moves off the critical path and
runs alongside, on spare capacity. If the method works, B becomes its supporting evidence;
if it fails, B says why.

**Critical path is now: Phase A → C1 build → Phase D.** Everything else is parallel or
contingent.

### 0.2 The decisive cheap gate, added to Phase A2

C1 is "train the student to match a reference that never learned the fact." Its **ceiling is
whatever that reference itself does.** If the retain reference *substitutes* a plausible
wrong answer on forget probes, C1 inherits substitution and cannot reach the empty region —
exactly the failure that killed the substitution-estimated ignorance target (Finding 3,
confident confabulation at p=0.228 against a 0.02 gate).

So the Phase A2 gate is extended from "does it fail to recall the forget facts" to
**"which cell does it land in?"** Run the six-cell partition on the retain reference's own
forget-probe generations. One evaluation, no training.

| retain reference lands in | consequence |
|---|---|
| **IGNORANT / ABSTAIN** | C1's target is in the right cell. Build it — this is the route. |
| **REPLACE** | C1 inherits substitution. Its ceiling is the wrong cell; **do not build it**. Pivot to H10. |
| **DEGENERATE** | the reference is broken; fix the warm-up before anything else |

This single evaluation decides whether the primary mechanism is worth building, before any
training is spent on it. It is the highest-value measurement currently available and it falls
directly out of work already running.

### 0.3 Novelty has to come from the distinguishing operation, not from C1

C1 — distillation from a retain-trained reference — is **prior art** (arXiv 2506.06278, in
our list, to be read in full before any claim). Under the mandate, "novel, non-obvious
combinations of existing techniques are eligible", so C1 can be the *base* but cannot be the
contribution. The contribution has to be what sits on top.

Ranked by novelty-per-unit-risk:

1. **H10 — commitment re-ordering.** Do not lower `p(answer)`; lower the answer position's
   *confidence relative to its neighbours* so the decoder commits it **last**, when the
   surrounding text is model-written and no longer pins the memorized answer. No
   autoregressive analogue; exploits the sampler directly; sits in the one region the survey
   left open. Its signature prediction — **disclosure falls while `p(answer)` at the original
   teacher-forced state is unchanged** — is one no existing method can produce, because both
   families work by moving that probability. Caveat carried: our Lemma 5 was refuted by our
   own critic, so only the weaker claim (θ controls commit order) is available and the
   inequality must be re-derived.
2. **The diagnosis itself** (target mode selects the failure mode) — a real contribution, but
   a finding rather than a method, and it cannot carry a superiority claim on its own.
3. **C2 — within-state position weighting**, now scoped as an efficiency refinement
   (32× mismatch between the loss's position averaging and the decoder's one commit per
   step), tested only after something shows an effect.

**Target composite: C1 as the declared prior-art base + H10 as the distinguishing
operation**, with H10 ablated out as the direct test of its contribution. If the A2 gate
sends us to H10 alone, it stands by itself.

### 0.4 What this means for sequencing

| path | work | gate |
|---|---|---|
| **critical** | finish θ₀ and retain reference → **A2 cell gate** → build C1 (+H10) → Phase D | A2 decides C1 vs H10 |
| **parallel** | target-mode decomposition (forward passes only) | explanatory; no gate |
| **parallel** | measure the commitment distribution (H10's prerequisite) | must be concentrated or H10 dies |
| **contingent** | H8, H9, frontier mapping | only if both C1 and H10 fail |

Phase D is unchanged and remains ~6 GPU-h once a method is frozen. The honest estimate to a
claim-or-obstacle is unchanged at 3–5 days; what changes is that the *method* now starts
being built as soon as A2 clears, rather than after a full diagnostic pass.

## 1a. AMENDMENT — Phase B rewritten, H6 superseded by H6'

**Why.** H6 was written as a *mask-rate* hypothesis and Phase B as a mask-rate band
decomposition. Two of this project's own agents had already argued that version is wrong,
and I accepted both deflations at the time and then wrote the experiment anyway:

- The **designer** showed that at our configuration (32 steps, 32 tokens) the decoder's
  visited mask rates form a **deterministic uniform ladder**, so `t ~ U(0,1)` already matches
  the marginal. Their conclusion: the mismatch is the conditional *content*, not the rate.
- The **critic** confirmed that deflation and added that my premise is false for a primitive
  we already use — `fact_free_tf_states` does `masked |= is_fact`, forcing every fact-bearing
  position to mask. They named the surviving quantity as the **32× within-state position
  weighting** (the loss means over ~32`t` masked positions; the decoder commits exactly one).
- Separately, the formalization's **Lemma 5** — suppression lowers confidence, therefore
  delays commitment — was refuted by the critic with a counterexample.

Running Phase B as drafted would have tested the version my own team had already killed: a
null result would have disproved nothing new. Caught on review before any GPU time, and the
Phase A runs in flight are foundation work and unaffected.

## 1b. H6' — the failure mode is set by the TARGET'S OWN MODE

Restated so it explains the measurement directly rather than through a proxy.

Both families train the student to match a **target distribution** at forget states. They
differ in what that target actually asks for:

| family | how the target is built | what its mode plausibly is | measured outcome |
|---|---|---|---|
| MDU, EMDU, NPO | anchor from a **masked / null prompt** | at a blanked prompt the likeliest continuation is **end-of-text** | **DEGENERATE 1.000** |
| exclusion, SC-TF | reference renormalised with the fact tokens removed | the reference's **second-choice answer** | **REPLACE 0.31–0.38** |

On this account neither family is buggy: **each student is matching its target faithfully,
and the target's mode is the failure.** That explains the one pattern a mask-rate story does
not — why the grouping tracks objective *family* and is invariant to tuning (MDU collapses at
all three τ, EMDU at all three τ, exclusion stays fluent at all three ε).

It also subsumes H7 (the EOS-attractor hypothesis) as the anchor-family half, so the two are
tested by one measurement instead of two.

### Phase B, rewritten: target-mode decomposition — NO TRAINING

For each of the five objectives, construct its target at the forget states of the qualified
backbone and measure the target itself, not the student:

1. `mode(target)` and whether it is EOS/pad, a type-correct value, or the true answer
2. `H(target)` in bits, and the mass on EOS/pad specifically
3. mass on the true answer, and on the reference's top non-answer token
4. all of the above split by mask rate — which **tests the designer's deflation empirically**
   rather than taking it on argument, demoting mask rate from hypothesis to control

Cost: forward passes only, no optimizer, minutes. Targets come from the pinned objective
functions, imported read-only, so this measures the published objectives and not a
paraphrase of them.

**Pre-registered predictions.**

| prediction | kills H6' if |
|---|---|
| MDU/EMDU/NPO targets put large mass on EOS/pad at forget states | EOS/pad mass is not elevated over the reference |
| exclusion/SC-TF targets are mode-concentrated on a type-correct non-answer | their mode is EOS, or is the true answer |
| target mode predicts each arm's measured cell across all five arms | the mapping fails for any arm |

**If H6' survives, the method follows immediately and is no longer a guess:** supply a target
whose mode is *neither* end-of-text *nor* the reference's second choice, but the answer
distribution of a model that genuinely never learned the fact. That is C1, and TOFU is the
first dataset where it is constructible — the synthetic set's retain-only reference recalled
all 64 forget facts (Finding 5), and subject substitution produced confident confabulation
(Finding 3).

**If H6' dies**, pivot order is unchanged: `H10` (commitment re-ordering) next, then `H8`
(antiparallel forget/retain gradients), then `H9`. All three are specified with kill rules in
`HYPOTHESIS_REGISTER.md`.

### What changes in §4–§7 below

- **C2 is re-scoped.** It was "commit-band weighting" justified by mask rates. Its surviving
  justification is the **32× within-state position weighting**, which is a question of
  *precision and cost*, not of which failure mode you get. C2 is therefore demoted from a
  co-equal component to an efficiency refinement, tested only after C1 shows an effect.
- **C1 is promoted to the primary mechanism**, with its novelty caveat unchanged and
  load-bearing: distillation from a retain-trained reference is **not new**
  (arXiv 2506.06278 is in our prior-art list and must be read in full before any claim). What
  would be new, if anything, is the *diagnosis* — that target mode selects the failure mode —
  plus the MDLM-specific construction and the measurement that supports it.
- Phase ordering in §7 is unchanged. Phase B is now cheaper and runs without a GPU queue.

---

## 2. The mechanism hypothesis (H6)

**Claim: the mask rate at which the forget gradient acts selects which failure mode you
get, and existing objectives average over all of them.**

The masked-diffusion forget objective is applied at `t ~ U(0,1)`, i.e. at every corruption
level of the answer span.

- At **high `t`** (most or all of the answer masked) the model must produce the answer from
  the prompt alone. A *prohibition* — "do not produce the answer" — has, across all such
  states simultaneously, essentially one solution that generalises: produce nothing, or
  produce a constant. **→ collapse.**
- At **low `t`** (most of the answer visible, one token masked) the surrounding context pins
  the answer's type and position, so the only reachable alternative is a different value of
  the same type. **→ replacement.**

Under `t ~ U(0,1)` an objective therefore inherits a mixture, and which mode dominates
depends on its shape: anchor-based objectives (MDU, EMDU, NPO) collapse; renormalisation
objectives (exclusion, SC-TF) substitute. That is exactly the bimodality in §1, and it
predicts the *grouping* of the five arms without reference to their individual losses.

**Why this is worth pursuing rather than another reshuffle of the loss.** The prior-art
review established that **no published MDLM unlearning method deviates from uniform `t` on
the trained-model side** — verified against MDU, EMDU, and (from the camera-ready) the ICML
2026 adversarial-RL method, whose own equations draw `t ~ Uniform(0,1)` and whose "credit
assignment" is prefix-versus-suffix over the input context, not per-position. MDU also
published a stored-knowledge token-role diagnostic *and does not use it in its objective*.
So mask-rate/position structure in the objective is the one region the survey left open.

**The deeper form of the hypothesis, which is what the method acts on:** the problem is not
only *where* the objective is applied but *what kind of thing* it is. A **prohibition** is
incoherent at high `t` — there is no consistent non-degenerate way to satisfy "don't say the
answer" when the answer is all that is being asked for. A **target** is coherent at every
`t`: "predict what a model that never learned this fact would predict" is always satisfiable.

## 3. H6's decisive test, before any method is built (Phase B)

Train the *same* objective with the forget loss restricted to disjoint mask-rate bands,
everything else fixed: low `t ∈ [0.05, 0.35]`, mid `[0.35, 0.65]`, high `[0.65, 0.95]`,
against the uniform control. Two objectives (one from each family) × 3 bands + 2 controls.

**Pre-registered predictions:**

| band | predicted dominant cell |
|---|---|
| high `t` | DEGENERATE ≫ REPLACE |
| low `t` | REPLACE ≫ DEGENERATE |
| mid `t` | intermediate, ordered between the two |

**Kill rule:** if the three bands produce the *same* partition within noise (all cells within
±0.10), H6 is refuted, the bimodality has another cause, and the method does not proceed on
this basis. I will report that and pivot to the §6 fallback.

**This test also cannot be confounded by the dataset flaw** in `FINDING5`, because every band
shares the identical data, backbone and retain objective; the comparison is within-dataset
and the confound is common to all arms.

## 4. The candidate, if H6 survives (Phase C)

Two components, each separately ablatable, so no causal claim rests on a bundle.

**C1 — a coherent target instead of a prohibition.** Replace the prohibition with the answer
distribution of a **retain reference**: a model trained on the retain split and never on the
forget split. On TOFU this is constructible and is *the benchmark's own gold standard* —
TOFU defines forget quality by comparison against a retain-trained model. It was **not**
constructible on our synthetic set, because `FINDING5` shows a retain-only model there
recalls all 64 forget facts through a `surname → answer` rule (recall 1.000).

**Honesty about novelty:** distillation from a retain-trained reference is **not new** —
"Distillation Robustifies Unlearning" (arXiv 2506.06278) is in our prior-art list and must be
read in full before any claim. C1 alone is at best a faithful port to the MDLM objective.
**The contribution, if there is one, is C2 and the C1×C2 interaction.**

**C2 — commit-band weighting.** Weight the objective by where the decoder actually commits
the answer, rather than uniformly over `t` and over masked positions. Two measured facts make
this concrete: the frozen decoder commits exactly **one position per step** at this
configuration, while the pinned forget loss takes an unweighted mean over all ~32`t` masked
positions — a 32× mismatch in within-state position weighting. The commitment distribution
must be **measured** first (where, in step and position, the answer token first becomes
argmax and is kept); if it is diffuse, C2 has nothing to target and is dropped.

**Mathematical object, stated plainly:** C1 is ordinary loss descent on a KL to a fixed
reference. C2 is a reweighting of that loss, so the combination remains a scalar objective —
*not* a constrained or stop-gradient procedure. I will not claim otherwise. If the
commitment set is made `θ`-dependent it stops being ordinary descent, and that version will
be labelled separately.

## 5. The comparison that would license a claim (Phase D)

- **Benchmark:** TOFU forget10 (400 QA, 20 authors, 400/400 distinct answers, free-form).
  Starting point the third-party `llada-tofu-forget10-SFT` checkpoint, under qualification now.
- **Comparators:** MDU (τ sweep), EMDU (τ sweep), NPO (β sweep), SC-TF, the exclusion target,
  and **the strongest simple combination of our own ingredients** (retain-reference KL at
  uniform `t`, i.e. C1 without C2) — which is also the C2 ablation.
- **Primary trade-off**, declared before final evaluation: forgetting measured as content
  recall on forget10 at **matched utility**, where utility is retain **margin** (not
  accuracy) plus a general-capability panel. Reported as a curve over the per-arm
  hyperparameter, not a single point.
- **Degeneracy and substitution are first-class outcomes, not footnotes.** An arm that wins on
  forgetting by collapsing is recorded as a failure — this is the whole content of §1.
- **3 seeds**, author-level clustering (TOFU's natural unit), and the two-component variance
  the evaluation protocol specifies.
- **Not claimed:** any comparison against the ICML 2026 method's *results*. Its unlearned
  checkpoint is an empty repo (verified), and its pipeline needs three auxiliary models
  including a 30B generator. This is declared, not omitted.

## 6. Fallback if H6 is refuted

The bimodality is still the real finding, and the fallback is to attack it directly rather
than through mask rates: interpolate between the two families (anchor-KL and
renormalisation) and map the frontier, to establish whether the region between collapse and
substitution is *empty because no objective reaches it* or *empty because it is unreachable*.
That is a weaker contribution but a publishable negative, and it is better than forcing H6.

## 7. Order of work and what gates each step

| phase | work | gate to proceed |
|---|---|---|
| A | qualify the TOFU checkpoint; build the TOFU forget/retain protocol with author clustering | checkpoint must recall forget10 **and** show lift over the pristine base |
| A2 | build the TOFU **retain reference** (retain90-trained, forget10-never-seen) | must recall retain90 and **not** recall forget10 — the gate our synthetic set failed at 1.000 |
| B | mask-rate band decomposition | H6 kill rule in §3 |
| C | C1, C2, and the 2×2 ablation | C2 needs a concentrated commitment distribution |
| D | matched comparison, 3 seeds, declared trade-off | freeze everything in `FINAL_EVAL_PREREG.md` first |

Phase A2's gate is the one that exposed the dataset flaw. It is kept exactly as-is.
