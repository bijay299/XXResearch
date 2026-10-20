# METHOD — FX-Mix (fact-free eXclusion distillation on a native-sampler state Mixture)

Authoritative equations: [`SPEC.md`](SPEC.md) (v1.1). Proofs: [`THEORY.tex`](THEORY.tex). This file is the readable
derivation. Category: **parameter (LoRA) update**; the deployed model uses the plain LLaDA sampler with no prompt, no
filter and no inference hook.

## 1. Problem
A forget request is a set of items (c, y*, F): a prompt c, a reference answer y*, and a **fact-token set F** defined per
fact (the token ids of the fact's distinctive surface pieces, including tokenisation variants; the prompt must not
contain an F token). LLaDA answers c by its native sampler: a fixed-length answer region of `gen_length` masks, split
into blocks; at each step every masked position is predicted, candidates are drawn by Gumbel-max at temperature T,
k positions of the current block are committed (highest confidence for `low_confidence`, uniformly random for
`random`), and committed tokens are never re-masked.

**Forgetting notion (type i, reduced recall).** For every native configuration π of a declared grid — including
configurations never used during unlearning — the event Λ_F = {some generated position holds an F token} should be
rare, while forget answers stay well-formed (no refusal template, no degeneration) and retained knowledge and general
capability stay close to the original model. This is *not* refusal (type ii), not inference-time suppression
(type iii), and not removal of training influence or retraining equivalence (type iv).

**Falsifiable hypothesis.** At matched retention, FX-Mix lowers leakage on *held-out* native configurations below the
strongest straightforward combination of the same ingredients (SC-TF: exclusion target on LLaDA/MDU-style random
teacher-forced states; SC: SC-TF plus MDU's single-greedy-config rollout flag). If it does not, the method reduces to
"TNFF-in-MDU" and is reported as such.

## 2. Why the usual objectives miss the leak-relevant states
1. **Leakage is a first-passage event of the deployed sampler** (THEORY Prop. 1a). The first F commit is always made
   from a *fact-free* state that the sampler reached with the model's *own* earlier commits. Teacher-forced (TF)
   training states — the ground-truth answer with a random subset masked, as in LLaDA SFT, GA/NPO-on-MDM and MDU —
   only contain fact-free states whose visible tokens equal y*.
2. **A successful suppression target leaves that manifold.** Once the fact is suppressed, the sampler commits
   *alternatives* (a different value, a different phrasing or position), so its next states have visible tokens ≠ y*.
   In the unrestricted model class, exact agreement with the target on TF states leaves worst-case leakage equal to the
   target sampler's escape probability from the TF set, which is ≥ 1−1/L for a single fact position and = 1 for a
   multi-token fact under one-token random-order decoding (THEORY Prop. 2). Whether a network's parameter sharing
   transfers TF suppression to the escaped states is an empirical question — the decisive one here.
3. **Decoding configurations visit different decision states.** Greedy confidence-ordered decoding commits the fact
   late; random-order decoding at T>0 commits it at any time; parallel commits (k ≥ 2 tokens per step) create
   product-of-marginals states that one-token rollouts rarely visit (B's exact tabular study: training on one greedy
   config leaves leak 1.0 on other schedules; a three-config pool reaches max 0.011 over 30 held-out configs).
4. **Targets matter too.** MDU's prompt-masked anchor still sees visible answer tokens; when the answer restates the
   subject the anchor itself predicts the fact (A's exact check: anchor p(fact) 0.44–0.46 once the subject is visible;
   our CPU probe: MDU τ=1 keeps emitting forgotten value tokens). Flattening targets (uniform/ME) do not flip the rank
   of the fact, so greedy decoding still leaks. (EMDU, arXiv 2610.25447, reports the anchor problem independently —
   snippet only.)

## 3. Formulation
Notation: state s = (c, z), z ∈ (V ∪ {MASK})^L; M(s) masked positions; p_θ(·|s,i) the mask predictor; p0 = p_θ0 frozen.

**Target (Lemma T, standard I-projection).** At every state and masked position, the target is the frozen reference
at the *same* state (prompt visible) with the fact excluded:

    r_ε(·|s,i) = argmin_{q: q(F) ≤ ε} KL(q ‖ p0(·|s,i))
               = p0                                if p0(F) ≤ ε
               = ε·p0|_F + (1−ε)·p0|_{F^c}         otherwise      (computed in log-space from fp32 logits)

Properties: KL(r_ε‖p0) = kl(ε ‖ p0(F)); at θ0 the logit gradient is (p0(F)−ε)₊·(p0|_F − p0|_{F^c}), so positions where
the reference puts ≤ ε mass on F (template, EOS padding, unrelated content) receive no forget gradient at the start;
the target keeps the reference's ranking of every non-fact token (fluent alternatives rather than junk).

**State distribution (the operation FX-Mix adds).** Fact-free states only:
- α = 0.25 teacher-forced fact-free states: t ~ U(0,1), non-fact answer positions masked w.p. t, every fact position
  always masked (answer region = gen_length with EOS padding);
- 1−α = 0.75 rollout states: for a configuration drawn uniformly from the pool Π_train, run the native sampler with the
  **current student** (no gradient; pool refreshed every R steps), record the state before each step, **stop at the
  first step that commits an F token** and keep J states sampled uniformly from the pre-commit states plus that
  decision state. Toy pool: greedy low-confidence (1 token/step), random remasking T=1 (1 token/step), random remasking
  T=1 (2 tokens/step). LLaDA pool adds a 4-tokens/step random config.

**Loss.** L(θ) = λ_f · E_items E_{s∼μ}[(1/|M(s)|) Σ_{i∈M(s)} KL(r_ε(·|s,i) ‖ p_θ(·|s,i))] + λ_r · L_SFT(retain), no 1/t
weight, λ_r = 1. LoRA on q/k/v/o/gate/up/down (B = 0 init, so the first pool equals reference rollouts); reference =
adapter-disabled pass; AdamW, clip 1.0. ε = 1e-3 in the toy (gen 8), ε = 1e-4 for LLaDA (gen 128).

## 4. What the theory does and does not say
- **Prop. 1** (first-commit decomposition + hazard bound; standard first-passage + union bound for the pinned sampler,
  T ∈ [0,1], both remasking rules, cfg 0): P(Λ_F) ≤ E[Σ_{n≤τ} Σ_{i∈E_n} min(1, ρ)], ρ = p(F)/max p; greedy decoding
  leaks with probability 0 if ρ < 1 at every reachable fact-free state. Used as a **post-hoc certificate** B̂_π on fresh
  rollouts of the final model. The training loss is **not** a leakage bound (at LLaDA scale the union bound needs a mean
  per-position KL ~2.4e-5 to certify 5 %).
- **Prop. 2** (escape characterisation, zero-sharing class): explains why TF-only training does not identify
  config-robust forgetting and why training on the sampler's own visitation does (DAgger-type fixed point). Nothing
  is proved for networks with shared parameters.
- **Fixed point.** If the student matched r_0 at every fact-free state it visits, its native sampler would equal the
  reference sampler with F banned at decode time (D §3.2). FX-Mix therefore moves a decode-time ban into the weights,
  only on the forget prompts' states; the decode-time ban on the base model is the inference-time reference control.
- **Floors.** ε > 0 restores a stationary point but the target itself then leaks 1−(1−ε)^n under T=1 random decoding
  (n committed positions with p0(F) > ε); ε = 0 has no finite stationary point under Adam (step budget is the knob).

## 5. Approximations and assumptions (all explicit)
A1 token-level F (string-level leakage through unlisted tokenisations is possible; span variants are evaluated);
A2 per-position (mean-field) target, not the sequence-level projection (LLaDA's conditionals define no joint);
A3 rollout states are constants (no gradient through the sampler), stale ≤ R steps; per-rollout J-sampling is a design
measure, not the visitation measure; A4 guarantees only for cfg_scale = 0, T ≤ 1, no logit edits before the draw, same
gen_length, no re-masking samplers; A5 LoRA + retain loss: realisability of the target is not assumed; A6 module names of
LLaDA's HF implementation are UNVERIFIED here (HF blocked).

## 6. Closest methods and the straightforward combination (details: PRIOR_ART.md)
Target = TNT/TNFF (I-projection; code read) = UNDIAL with γ = ∞; on-policy MDM rollout loop = MDU code's
`_traj_rollout_na_loss` (single greedy config, anchor target, random stop, no fact filter). **SC** = MDU loop with that
flag and the target replaced by r_0 (≈10 lines). FX-Mix − SC = config pool incl. random-order sequential and parallel
configs, truncation at the first fact commit, fact-free TF states with F force-masked, gen_length answer region.

## 7. Cost per forget training state (forward-equivalents; backward ≈ 2)
Student fwd+bwd 3 + reference fwd 1 = 4 (same as MDU) + amortised rollouts (pool mean ≈ N/(J·R) forwards per state;
LLaDA example N≈96, J=8, R=4 ⇒ ≈3) ⇒ ≈ 7, i.e. ≈ 1.75× MDU per forget state; retain branch identical to MDU/GD. Measured
CPU numbers and LLaDA projections: `results/dev/COST.md` and RESULTS.md.
