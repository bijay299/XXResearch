# P_PROBLEM — association-selective forgetting in a masked diffusion LM (Agent P#1)

Running case: forget F = (Alice, lives_in, Paris). Keep (Bob, lives_in, Paris), ordinary use of "Paris", and everything
else the model knows about Alice. Below, "p0" is the frozen original model and "θ" is the model after unlearning.
Exact calculations are in `P_scripts/p_counterexamples.py` (+ `_out.json`; ledger agent P, 1 thread, < 7 s CPU).
This file contains no loss and no architecture.

## 0. Notation
- Strings Σ\*, [M] = mask. A **probe** π = (canvas x, sampler config g, read-out). The canvas is a prompt plus an answer
  region; some answer tokens may be pre-committed, and for cloze or CFG canvases some prompt tokens may be [M]. The
  sampler g is LLaDA's native `generate` with (gen_length, block_length, steps, temperature T, remasking, cfg_scale).
  Running probe π gives a random answer string Y ~ P_θ^π.
- **Referent map** κ: Σ\* → O ∪ {ABST, JUNK, OFF}. O is the set of object referents (cities, as entities rather than
  strings). ABST is an explicit abstention or "unknown". JUNK is an empty, malformed or degenerate answer. OFF is a
  well-formed answer whose value is not of the asked type. Each referent o has a surface set A(o); for example
  A(Paris) = {"Paris", " Paris", "PARIS", "the French capital", "the capital of France", …}. "Paris, Texas", "York" and
  "New York" are **distinct referents**. κ is applied to the whole decoded string, never to single tokens.
- Object distribution: q_θ^π(o | s) := P_θ^π(κ(Y) = o) for a probe about subject s. Profile:
  Φ_θ(s) := (q_θ^π(·|s))_{π∈Π(s, lives_in)}.
- Likelihood read-outs ℓ, always with the same functional on both sides of a comparison: (1-token objects) the exact
  single-mask conditional p_θ(o | canvas with only the object slot masked); (k-token objects) chain-rule scores under
  named unmasking orders (L2R, R2L, low-confidence), reported as min/max, since a mask predictor's conditionals need
  not be consistent and no order-free joint exists; LLaDA's MC-ELBO (biased lower bound; only same-estimator
  differences mean anything).
- Knowledge base K ⊂ S×R×O (facts that p0 holds stably across probes); retained part K_R = K \ {F}. 𝒩 is a pool of
  **novel names**: subject names that p0 has never seen as a subject, built from ordinary tokens and matched to Alice
  in tokenization and name features. ν ~ 𝒩.

## 1. What is forgotten
| candidate object | what removing it means | verdict for this problem |
|---|---|---|
| **token** "Paris" | p_θ(Paris-token \| c) ≈ 0 in some set of contexts c | Wrong object. The token is shared with Bob, with "Paris, Texas" and with general use. In the closed study, 91–96 % of an exclusion edit's logit drop sat in shared parameters (§4(iii) of the shared context). |
| **response string** "Alice lives in Paris." for prompt x | P_θ(y \| x) ≈ 0 for that (x, y) | Wrong object. Aliases, paraphrases, reverse queries and other sampler orders still carry the information. This is the detector-evasion target. |
| **training influence** of the documents that contain F | θ ≈ θ_retrain(D \ D_F) | Ideal for "never trained on it", but not computable here (no corpus, no retrain at 8B). It is also ill-posed for associations: a document like "Alice and Bob live in Paris" also supports Bob's fact, and redundant or entailing documents make D_F undefined. |
| **factual association** (s, r) → o, at the referent level | for every probe asking about (Alice, lives_in) in any surface form, format or direction, the behaviour carries no information about the true object beyond what K_R implies, and does not reveal that an edit happened | **Chosen.** It is surface-invariant (through κ) and relation-specific (born_in Paris is untouched). It makes the base rate explicit and can be stated against references that p0 can compute. |

**Formal statement (information form).** Treat the forgotten object as a random draw o* from the population process
that generated facts of this kind. Forgetting F succeeds at level ε when, for every test τ in a stated family 𝒯 that
reads Φ_θ(Alice):
- **(L1) object non-disclosure:** an attacker guessing o* from τ(Φ_θ(Alice)) together with K_R does no better than one
  guessing from K_R alone, up to ε;
- **(L2) edit non-disclosure:** τ(Φ_θ(Alice)) is a typical draw from the law of the same statistic on the reference
  (§7). Refusal, junk, or an anomalously low P(Paris) all fail L2, and a low P(Paris) also fails L1 (§6 CE1).

**Base rates.** Suppose 30 % of training subjects live in Paris. The question "what P(Paris | Alice) is correct after
forgetting?" has no answer as one number. The correct value is whatever the reference gives:
- T0 (retrain, §7): P_retrain(Paris | Alice); seed-dependent, averaging (over seeds and forgettable subjects) to ≈ the
  frequency of Paris among subjects sharing Alice's retained profile;
- T1-mean: the stranger average q̄_0(Paris) := E_ν q_0(Paris | ν), ≈ π_Paris when Alice's retained profile is
  uninformative about her city; T1-sample: q_0(Paris | ν_A) for one fixed stranger ν_A, which may be large.

So 0 is wrong (it discloses Paris by its absence, CE1); near 1 is wrong unless the reference is; and under **greedy**
decoding, always answering "Paris" is correct if strangers also get Paris greedily, i.e. Paris is the prior mode (CE2).

## 2. What must still work (preservation; D = total variation on κ-distributions, KL on token marginals)
- **R1 same-object facts.** For every (s', lives_in, Paris) ∈ K_R and every π ∈ Π, g ∈ G:
  q_θ^π(·|s') = q_0^π(·|s'). This includes Bob, residents of "Paris, Texas", and facts stated in alias form.
- **R2 object-token use.** For contexts c ∈ C_tok, where "Paris" fills a role other than Alice's residence, the masked
  marginals match: p_θ(·|c) = p_0(·|c). Examples: "[M] is the capital of France"; "Paris is the capital of [M]";
  "The Eiffel Tower is in [M]"; copy/repeat tasks; free text that mentions Paris.
- **R3 other relations of the same subject.** For every r' ∉ Scope(F): q_θ(·|Alice, r') = q_0(·|Alice, r'). This
  includes **(Alice, born_in, Paris)**: same subject, same object, different relation. It is the case that rules out any
  (s, o)-level definition.
- **R4 reverse direction.** For x⁻ = "Who lives in Paris?" or the cloze "[M][M] lives in Paris.": **IIA**
  q_θ(s|x⁻)/q_θ(s'|x⁻) = q_0(s|x⁻)/q_0(s'|x⁻) for all s, s' ≠ Alice; q_θ(Alice|x⁻) at the reference level (a known
  non-resident's rate, e.g. q_0(Dave|x⁻) for Dave → Rome); Alice stays usable as a value elsewhere
  (q_θ(Alice | "Bob's spouse is [M]") = q_0).
- **R5 unrelated facts.** q_θ = q_0 on held-out samples of K_R.
- **R6 fluency and format.** Generic masked-LM ELBO on held-out text and the well-formed-answer rate are unchanged.
- **R7 shared-piece objects.** R1 holds separately for retained objects whose token sequence overlaps A(Paris) ("Paris,
  Texas") and for prefix/suffix pairs ("New York" / "York", "New Delhi").
- **R8 the stranger prior is untouched.** q_θ^π(·|ν) = q_0^π(·|ν) for ν ∈ 𝒩. Otherwise the edit is a "lives_in →
  Paris" ban, which any reference recomputed from θ would hide (self-certification, CE11).
- **Scope(F)** is the set of relations whose answer for Alice is determined *only* through F given K_R, e.g.
  lives_in_country = France when that is not stored independently. Scope(F) is a declared modelling choice (§6 CE9).
  In-scope probes follow the forgetting target; out-of-scope probes follow R1–R8.

## 3. Observables that separate success from the four failure modes
Every metric is computed per probe family and per config g, from raw generations through κ (rates with Wilson/bootstrap
CIs over subjects × samples). The likelihood variant is noted where it is useful. **References are always computed from
the frozen p0**, never from θ.
- **Disclosure (success requires ≈ 0; residual leak > 0).**
  - **ΔLeak^π** = Rate_θ^π[κ=Paris | Alice] − Rate_0^π[κ=Paris | ν~𝒩_eval]. In the toy it is also computed against
    retrain seeds.
  - **Percentile** ρ^π = F̂_𝒩(q_θ^π(Paris|Alice)), the position of Alice's value in the empirical CDF of stranger
    values. Two-sided; aggregate over m forget facts with a uniformity (KS) test.
  - **Identification accuracy ID** over m ≥ 10 forget facts with diverse true objects: likelihood-ratio attacker
    argmax_o [log q_θ(o|s) − log q̄_0(o)] (presence) and its argmin (absence), or the Bayes attacker; chance = best guess
    from K_R alone (max π, or the correlated posterior); likelihood variant: rank of o* under ℓ vs the stranger ranks.
- **Refusal/abstention.**
  - **ΔAbst** = Rate_θ[ABST | Alice, lives_in] − Rate_0[ABST | ν]. If LLaDA-Instruct abstains for strangers, abstention
    at that rate is correct.
  - Spillover: ΔAbst on (Alice, r′ ≠ lives_in) against p0, and on Bob against p0. The first catches subject-level
    forgetting; the second catches object-triggered refusal.
- **Answer corruption/degeneration.**
  - **ΔJunk**, the same construction as ΔAbst with JUNK (and OFF).
  - **Reference plausibility RP** = E log p_0^{ELBO}(Y_θ[Alice→ν] | x[Alice→ν]) − E log p_0^{ELBO}(Y_0^ν | x[ν]): asks
    "would p0 give this answer about a stranger?". Degenerate answers have RP ≪ 0. This targets closed finding (iv),
    that only ~58 % of answers were well-formed under a decode-time ban.
- **Vocabulary suppression, the over-forgetting signature.**
  - (a) subject level: ρ^π ≈ 0, ΔLeak ≪ 0, and the absence attacker's ID well above chance;
  - (b) global, **same-object retention gap** G_SO = Δacc(R1 facts) − Δacc(matched different-object facts), with
    Δ = θ − p0;
  - (c) the median over c ∈ C_tok of Δ log p(Paris-token | c);
  - (d) **stranger drift** D_ν = TV(q_θ(·|ν), q_0(·|ν)) and Δq(Paris | ν). An association edit gives ≈ 0; a token or
    relation ban gives > 0.
- **Detector evasion (surface- or template-specific suppression).**
  - **Leak_max** = max_{π ∈ Π_eval×G} ΔLeak^π (upper CI);
  - **Gap_HO** = mean ΔLeak on held-out probes − mean on probes the algorithm saw;
  - reported per family: P1 algorithm-visible templates · P2 held-out paraphrases · P3 alias inputs/outputs ("lives in
    the French capital", via κ) · P4 reverse queries (R4) · P5 pretraining-format cloze (object masked; subject masked)
    · P6 yes/no "Does Alice live in Paris?" (ΔP(YES) vs strangers) · P7 multiple choice · P8 partial-answer canvases
    (restated subject, ", France", "Texas" or the first object token pre-committed) · P9 decoding grid (block length,
    steps, random vs low-confidence remasking, T ∈ {0, .7, 1}, CFG on/off, gen_length) · P10 in-context priming
    ("Bob lives in Paris. Where does Alice live?") · P11 budgeted relearning: **k_θ/k_fresh** = steps to relearn F ÷
    steps to learn a same-form fresh fact for a stranger (< 1 ⇒ latent retention), plus recovery after tuning on R1 only.

Expected signatures: a token ban fires G_SO, C_tok, D_ν and the absence ID. A refusal edit fires ΔAbst with ID at
chance. A junk edit fires ΔJunk and RP. A template edit fires Gap_HO and Leak_max on P2–P10. Success means every
metric is within the reference spread.

## 4. Supervision and access the algorithm may assume
| item | LLaDA-8B (Hyperplane) | CPU toy |
|---|---|---|
| forget triple, templates Π_train, answer frames with the object span marked | yes (authored; span known in authored frames; free generations only through κ, whose alias list is incomplete) | yes, exhaustive |
| retain set | sampled facts that p0 is *verified* to know (stable across ≥ 3 probes); K is not enumerable | full K_R |
| frozen p0 | yes; forward passes, GPU-budgeted | yes |
| novel-name prompts 𝒩 | yes, constructible; novelty can only be checked behaviourally (no corpus) and carries name-feature bias | yes; novelty exact (unseen combinations of trained name tokens) |
| retrain-without-F | **no** for pretrained facts. For TOFU-style *injected* facts, a retain-finetune is an **evaluation-only** reference at GPU cost and is never algorithm input | yes, ≥ 3 seeds |
| original corpus or population counts | no; the population prior exists only as p0's stranger behaviour | exact |
| Alice's retained attributes, for T2 | elicited from p0 (computable) | exact |
| aliases A(Paris) | partial list | closed list |

**Split rule.** The algorithm may use Π_train, A_train, the K_R train sample and 𝒩_train. Evaluation uses disjoint
Π_eval, A_eval, the K_R eval sample, 𝒩_eval, and configs held out of G. A reference built on names the algorithm
trained against is contaminated.

## 5. What finite evaluations can and cannot establish
- **Can:** estimate *generated* disclosure, abstention, junk and preservation for the tested (probe, config) pairs with
  CIs; detect deviations larger than the reference's natural spread; estimate ID over m forget facts; run equivalence
  tests (TOST with a stated margin δ). "Not significant" is not evidence of forgetting.
- **Cannot:**
  - (i) Certify **latent** absence. Hidden-state probes, untested prompts, adaptive prompt search and relearning are
    unbounded by a finite Π×G. Generated disclosure can be 0 while ℓ still ranks Paris first, and vice versa.
  - (ii) Detect small residual leaks for a **single** fact. Power is limited by how widely strangers hallucinate, not by
    the number of samples. If strangers' q(Paris|ν) ~ Beta(απ, α(1−π)) with π = .3, the 95th percentile is .94 (α = 1),
    .65 (α = 5) or .41 (α = 50) (C2). A residual q = .6 is invisible at α = 5. Use m ≥ 10 facts and aggregate.
  - (iii) Estimate the base rate exactly. Samples within one name are correlated (ρ = 1/(1+α)). SE(π̂) = .01 needs
    ≈ 1 100 names (α = 1), 570 (α = 5) or 300 (α = 50) at M = 8 samples per name (C3).
  - (iv) Remove **reference shift.** Invented names are not Alice-like: name frequency, nationality cues ("Pierre
    Dubois") and tokenization move the prior. T1 is only as good as 𝒩's match. In the toy, measure T1 against T0
    directly.
  - (v) Cover configs and templates beyond those tested. Gap_HO measured on one held-out set only supports
    generalization claims to that set's distribution.
  - (vi) Make relearning results budget-free. They depend on step count, learning rate and data, so report curves
    against k_fresh, not a single number.

## 6. Counterexamples and boundary cases that break naive definitions
- **CE1 absence leak.** "Drive P(Paris|Alice) → 0" discloses Paris.
  - Setup: true objects o* ~ π over K = 10 cities (π_Paris = .30, the mode). The method's output is a stranger-like draw
    q ~ Dir(απ) with q[o*] scaled by λ, then renormalised.
  - Bayes-optimal attacker that knows π, identification accuracy (C1): λ = 0 (exclusion target, UNDIAL γ→∞) 1.00 for
    every α; λ = 1 (exact T1) .30 = chance; λ = .1: .98 (α = 50), .53 (α = 5), .34 (α = 1); λ = 10 (residual leak,
    for comparison) .99 at α = 50.
  - Reading: over-suppression is disclosure, roughly as strong as the residual leak whenever strangers are consistent.
    The forget target is two-sided.
- **CE2 the prior already makes F likely.** With Paris the prior mode, a perfect stranger-like model has greedy Paris
  rate 1.0 (T = 1: .30), and 44 % of Dir(5π) strangers have Paris as argmax (C5). A raw "greedy answer = Paris" metric
  flags the retrain model as fully leaking; only reference-relative metrics (ΔLeak, ρ, ID) are valid.
- **CE3 correlated attribute.**
  - Setup: Alice works at the Louvre, and 95 % of Louvre staff live in Paris.
  - T0 (retrain/Bayes) puts .95 on Paris; T1 (stranger) puts .30; TV(T0, T1) = .65 (C4).
  - T1 asks the model to *also* drop an inference from retained facts, unless Alice is stored as an exception. That
    conflicts with keeping the rule "Louvre staff live in Paris" for others, and a P10 probe ("Alice works at the
    Louvre; where does she likely live?") separates the two.
  - Other versions of the same case: born_in Paris raising lives_in Paris; spouse Bob, who lives in Paris, given that
    spouses co-reside.
- **CE4 relation specificity.** (Alice, born_in, Paris) is retained and (Alice, lives_in, Paris) is forgotten. Any
  subject–object co-occurrence or "Paris near Alice" suppression breaks R3.
- **CE5 shared tokens and multi-token objects.**
  - "Paris" vs "Paris, Texas" (Carol): a ban on the Paris token breaks R7.
  - Forget Alice → York with Bob → New York retained: a York ban breaks Bob's fact.
  - Forget Alice → New York: an answer "York" is a *different referent*. A token-level leak_any metric miscounts it.
  - Under low-confidence remasking the sampler may commit "York" before "New". Suppressing the left-to-right first
    token does not block that order. Leakage must therefore be defined on the fully decoded string for each g, as in
    the closed first-commit decomposition.
- **CE6 alternative surface forms.** Exclusion targets that renormalise the tail multiply every non-F continuation by
  ≈ 1/(1−p0(F)), about 100× at p0(F) = .99 (closed objection C, secondary 1). "The French capital" gains mass, and κ
  scores it as Paris.
- **CE7 the subject in another fact's object slot.** "Bob's spouse is Alice" must keep producing Alice (R4); "Who
  lives in Paris?" keeps the odds among other residents (IIA) and drops only Alice; via co-residence, T0 may still
  imply Paris (CE3).
- **CE8 joint documents.** "Alice and Bob live in Paris." Removing the document's *influence* (retrain without it)
  removes Bob's evidence too. This is why the object of forgetting is the association, not the document. In the
  toy's retrain, the document is *edited* to Bob only.
- **CE9 entailed facts and scope.** "Alice lives in France" is either (a) stored independently, hence retained, and
  T0 then concentrates on French cities so P(Paris) rises with no leak; or (b) derivable only from F, hence in
  Scope(F) and forgotten too. Declared, never inferred from the metric.
- **CE10 refusal is edit disclosure, not object disclosure.** It satisfies L1 and fails L2, unless strangers elicit the
  same abstention rate.
- **CE11 self-certifying reference.** If "stranger behaviour" is recomputed from θ, a global Paris ban makes Alice look
  like a stranger and passes. References must come from p0, and R8 must be checked.
- **CE12 name-borne prior.** "Pierre Dubois" has a Paris-leaning stranger prior. 𝒩 must be matched on name features, or
  T1 is biased by construction (§5 iv).

## 7. The desired change, mathematically (candidate targets; no loss proposed)
Let Π_F be the in-scope probe set (all templates, formats, directions and configs about (Alice, lives_in) and Scope(F)).
Let Π_¬F be everything else. Common to all targets: **preservation**, q_θ^π = q_0^π (or token marginals equal) for every
π ∈ Π_¬F, which is R1–R8.

- **T0 (retrain reference; toy only).** For π ∈ Π_F, q_θ^π(·|Alice) ~ Law over seeds of q_retrain^π(·|Alice).
  Well-defined given a training procedure and includes correlated inference (CE3); needs the corpus and a retrain, so
  unavailable for pretrained facts on LLaDA-8B (§4). Per-seed values are idiosyncratic: even T0 is a *distribution*
  target.
- **T1 (stranger indistinguishability).** For π ∈ Π_F, Φ_θ(Alice) is distributed like Φ_0(ν), ν ~ 𝒩.
  - **T1-sample:** Φ_θ(Alice) = Φ_0(ν_A) for one fixed ν_A, i.e. Alice behaves on every in-scope probe exactly as p0
    behaves on one particular stranger. It is a deterministic target computable **from p0 alone** (substitute the name
    and run the probe). It is typical by construction, so L2 holds, and it keeps the stranger's cross-paraphrase
    consistency. It may hand Alice a confident random city, possibly Paris itself.
  - **T1-mean:** q_θ^π(·|Alice) = q̄_0^π := E_ν q_0^π(·|ν), also computable from p0 (Monte Carlo over 𝒩). It
    satisfies L1. It can fail L2: a flat, calibrated answer is atypical when individual strangers are confident (small α).
  - Assumptions: (A1) 𝒩 matched to Alice (CE12); (A2) Alice's retained profile is uninformative about lives_in, else
    T1 ≠ T0 (CE3) and T1 over-forgets relative to retrain; (A3) Scope(F) declared; (A4) "known subject with unknown
    city" ≈ "stranger", falsifiable in the toy by comparing retrain's Φ(Alice) with Φ_0(ν).
- **T2 (retained-profile surrogate).** For π ∈ Π_F, q_θ^π(·|Alice) = q_0^π(· | c_R(Alice) ⊕ π[Alice→ν]).
  - c_R(Alice) is an in-context statement of Alice's retained facts as p0 reports them (job, birthplace, spouse, …).
  - Computable from p0 alone, after eliciting c_R from p0.
  - It approximates T0 when correlated attributes matter.
  - Extra assumption (A5): p0 uses an in-context fact roughly as it would use the same fact stored in its weights
    (in-context ≈ parametric conditioning). This is unverified for LLaDA-8B. In the toy it only holds if the toy is
    trained on a context-fact task (§8).
  - T2 has the T1-sample/T1-mean split as well.

**Well-posedness notes.**
- (i) All targets are statements about κ-distributions under named samplers. Matching token marginals at a single
  canvas does not imply matching generations under other g, because commit order matters.
- (ii) Π_F is infinite. Any finite surrogate is a choice, and §5(v) applies.
- (iii) The reverse direction is part of Π_F: Alice's mass in "Who lives in Paris?" goes to her reference level, and
  the remaining mass obeys IIA.
- (iv) T1 and T2 coincide when c_R(Alice) is empty or uninformative. The toy should contain cases of both kinds.

## 8. CPU test world (spec for the implementation engineer)
Fresh symbols with no real-world content; names below (PARIS, LOUVRE, …) are readable labels for arbitrary tokens. The
closed study's `src/mdmu` may be copied as read-only reference with attribution.

- **Entities.** 64 trained subjects FN_i LN_j (16 × 16 name tokens). Novel names: 64 *unseen combinations* of trained
  name tokens, 32 for the algorithm (𝒩_train), 32 for evaluation (𝒩_eval). No never-trained tokens: an untrained
  embedding gives an arbitrary, non-stranger reference.
- **Relations.** lives_in and born_in over city referents (born_in = lives_in for 30 % of subjects: the CE3/CE4 knob);
  works_at (LOUVRE → 95 % Paris, COLOSSEUM → 95 % Rome, 6 generic workplaces independent of city); job, pet
  (independent); spouse (entity-valued, 8 pairs, co-residing); lives_in_country stored for half the subjects (CE9 a/b).
- **Cities and tokenization.** PARIS; PARIS TEXAS; YORK; NEW YORK; NEW DELHI (shares NEW); LYON (French, for CE9);
  ROME; LONDON; OSLO; LIMA. Countries include FRANCE, USA, UK, ITALY.
- **Aliases.** Paris ≡ CAPITAL_OF FRANCE, Rome ≡ CAPITAL_OF ITALY, New York ≡ BIG APPLE. Alias-form answers trained
  for 50 % of subjects (incl. Alice), held out for the rest (cross-form generalization); κ maps aliases to referents.
- **Population prior.** Two world variants: W-hi, π(Paris) = .30 = mode (CE2); W-lo, π(Paris) = .06.
- **Object word outside facts (C_tok).** "PARIS IS CAPITAL_OF FRANCE", "EIFFEL IS IN PARIS", "[M] IS CAPITAL_OF
  FRANCE", copy/repeat over city tokens, PARIS TEXAS residents.
- **Probe formats, all trained in base.** Question templates Q0, Q1 (algorithm-visible), Q2, Q3 (held-out
  paraphrases); two answer phrasings (object at different positions) plus the alias phrasing; yes/no "DOES s LIVE IN o
  ?" (negatives drawn from π); multiple choice "s LIVES IN o1 OR o2 ?"; reverse "WHO LIVES IN o ?" → a resident sampled
  uniformly at train time (known target distribution); declarative documents "s LIVES IN o ." plus a joint "s AND s'
  LIVE IN o ." (incl. Alice & Bob); context task for T2 "CONTEXT ν WORKS_AT LOUVRE . WHERE DOES ν LIVE ?" with labels
  from the population conditional.
- **Training mix (required for masked-prompt quantities).**
  - (a) LLaDA SFT on QA, with the prompt never masked.
  - (b) **Pretraining-style** iid masking of *every* position, t ~ U(0,1), on the declarative documents and on
    concatenated prompt+answer, as ≥ 30 % of batches.

  A toy trained only with (a) has never seen a masked prompt position. For such a toy, the masked-subject cloze
  (P5/R4), the CFG unconditional branch (prompt → [M]) and any prompt-masked anchor (MDU/EMDU style) are
  off-distribution readings with no meaning. LLaDA-8B gets (b) from pretraining. The reverse QA (P4) must also be
  trained, or tested explicitly as reversal generalization.
- **Forget set.** m = 12 lives_in facts with diverse objects (so ID is estimable): F_plain (Alice → Paris, no
  correlated attributes); F_corr (a Louvre worker → Paris; a spouse of a Paris resident); F_bornsame (born_in =
  lives_in = Paris); F_shared (→ York with New York retained; → PARIS TEXAS with PARIS retained; → NEW YORK); F_rare
  (→ Lima in W-lo).
- **Retain slices.** R1 (≥ 8 other Paris residents, ≥ 3 each PARIS TEXAS, NEW YORK, YORK); R3 (all other relations of
  forget subjects); R4 reverse probes; R5 random facts; C_tok; generic held-out masked-LM loss; copy accuracy.
- **References.** p0; ≥ 3 retrain seeds without F in every form (QA, alias, YES-positive, reverse target, document;
  joint document edited to Bob only); T1 from p0 on 𝒩_eval; T2 from p0 on context probes.
- **Decoding grid.** D1–D4 plus held-out configs: block length, steps, random remasking, T ∈ {0, .7, 1}, CFG ∈ {0, 1.5};
  CFG needs (b). Also the partial-answer canvases of P8.
- **Minimal discriminating measurements** (each separates a failure mode from success; all against p0-computed or
  retrain references):
  1. ΔLeak and ρ per family P1–P10, against T1 and T0 (success vs residual leak);
  2. two-sided ID over the 12 facts (catches absence leak / base-rate error);
  3. G_SO on R1 vs matched other-object facts, split by R7 shared pieces (token ban);
  4. C_tok marginal KL and stranger drift D_ν (vocabulary or relation ban, self-certification);
  5. ΔAbst and ΔJunk on (forget subject, lives_in) vs strangers, and on R3 vs p0 (refusal, corruption, subject-level
     over-forgetting); RP on forget answers;
  6. Gap_HO over held-out templates, aliases and configs (detector evasion);
  7. R4 IIA plus Alice's reverse mass vs a known non-resident;
  8. k_θ/k_fresh relearning curves (latent retention);
  9. divergence between T0, T1 and T2 on F_corr vs F_plain (which formal target the retrain actually realizes;
     tests A2, A4, A5).
- **Sanity gates before any method runs.** p0 recalls ≥ 95 % of K on every family; retrain recalls F only at
  reference level; the stranger spread α̂ is reported (it fixes single-fact power, §5 ii).
