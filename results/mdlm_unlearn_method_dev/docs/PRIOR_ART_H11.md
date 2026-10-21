# PRIOR ART CHECK — H11, minimal-interdiction unlearning

**Owner:** Prior art and baselines lead
**Date:** 2026-10-20
**Status:** Tier 2 (active development evidence). Companion to `PRIOR_ART.md`; same ground rules.
**Scope:** one mechanism. Novelty check, nearest neighbours, category verdict, and the
"weight the rare tokens" objection.

## Method note on sources

The coordinator's warning about summariser-mediated fetches is well founded, so for this document
I built a fetch path that keeps the summariser out of the loop entirely: `curl` the arXiv LaTeXML
HTML (or PDF), strip tags locally with a script that **preserves `<math alttext>`** so equations
survive, and read the resulting text file directly. Script:
`<scratchpad>/h11/fetch.sh`. Where I quote, I read the extracted source text.

Where a claim rests on a subagent's search rather than my own read of source text, it is labelled.
Where I could not obtain source text, I say so rather than guessing.

**Two things I verified by direct computation on our own files rather than from any paper**, in
§4 — and one of them corrects an overreach in my own earlier deliverable.

## Reading order

Sections are numbered in the order the deliverables were requested but appear in the order I found
things, across three discovery passes. **If you read only part of this, read in this order:**

1. **§12.6** — consolidated verdict table (which of H11's pillars are occupied, by what).
2. **§7** — TIF already ran H11's core ablation and reported it insufficient. The decisive finding.
3. **§8.3** — every published token-salience selector scores ~chance under neutral ground truth.
4. **§4** — our own data: the RWKU coverage correction, and why the metric cannot see H11's success.
5. **§5** — the MDLM-specific reason H11 may fail outright.
6. **§10** — one-table summary for the five questions asked.

Sections §11 and §12 are addenda that **correct and tighten** earlier sections; where they conflict
with §9 or §10, the addenda win and say so explicitly.

---

# 1. The frame, and the two updates

## 1.1 What still holds from `PRIOR_ART.md`

- No published MDLM unlearning method deviates from uniform mask rate on the **trained-model**
  side. EMDU's FAM is a target-side intervention (anchor pinned at `t=1`).
- MDU published a token-role diagnostic — **in-context / structural / stored-knowledge**, with
  aggregate conditional-anchor KL change of −3.8% / +1.4% / **−20.6%** — and **never uses it in
  the objective**, which remains a flat `1/|M_t|` average over all masked positions.
- Position-selective unlearning is well developed in the AR literature and absent from MDLM
  unlearning.

## 1.2 Update A — Zhang et al. ICML 2026

The coordinator reports this located and verified: **adversarial RL, prompt-elicitation threat
model, `t ~ U(0,1)`**. I accept that as the project's finding and record the consequence: it
occupies *adversarial prompt elicitation*, not the mask-rate axis and not token selection, so
**it does not bear on H11**. Open item 1 of `PRIOR_ART.md` §8.3 is closed.

For the record, and so nobody later mistakes my earlier note for a live gap: I re-ran three arXiv
API queries on 2026-10-20 (`all:"diffusion large language model unlearning"`,
`all:"robust diffusion large language model unlearning"`, `au:Zhang_Zhiwei AND abs:unlearning`)
and it is **still not on arXiv** — the last returns only *Unlearning Inversion Attacks for Graph
Neural Networks* (2506.00808) and *Catastrophic Failure of LLM Unlearning via Quantization*
(2410.16454). So the coordinator's copy presumably came from ICML proceedings or OpenReview.
**My independent arXiv confirmation: negative. The coordinator's verification stands; mine is not
a contradiction of it, just a note on where the paper is not.**

## 1.3 Update B — "oracle matching" in arXiv 2506.06278, and why it bounds H11

**Citation:** Lee, Bruce W.; Foote, Addie; Infanger, Alex; Shor, Leni; Kamath, Harish;
Goldman-Wetzler, Jacob; Woodworth, Bryce; Cloud, Alex; Turner, Alexander Matt. "Distillation
Robustifies Unlearning." arXiv:2506.06278 (v3). <https://arxiv.org/abs/2506.06278>
*(Read from extracted source text, not a summary.)*

**What oracle matching is, in their setup — and it is stronger than "our retain-reference base".**
Verbatim: *"The aspiration of such procedures is oracle matching: attaining outputs
indistinguishable from those of a model that never encountered the undesired data."* Their
implementation: *"We implemented oracle matching using knowledge distillation with forward KL
divergence loss on the pretraining dataset."* The teacher is a genuine **oracle** — a model
pretrained only on the retain set, by construction never exposed to the forget data. They match a
reference model (which does have the forget capability) to it until *"the distillation loss tends
to zero"*, i.e. **effectively perfect behavioural agreement.**

**The failure.** Verbatim: *"Yet, when both models are subsequently finetuned on the forget set,
the oracle-matched reference model relearns significantly faster than the oracle model. This shows
that optimizing for behavioral outputs is likely insufficient for robust unlearning."* And:
*"Student(Reference) quickly learns the forget distribution compared to the Student(Random) and
Oracle Teacher models, which are more robust. This pattern holds both for the worst-case adversary
and uniformly across all learning rates."* Relearning = 500 steps on forget data, multiple LRs,
worst-case-adversary envelope.

**The diagnosis**, verbatim: *"even oracle-matched models retain latent capabilities vulnerable to
adversarial elicitation"* — and the positive result is that robustness comes from the **parameter
initialisation**, not from the target: distilling the same oracle into a *randomly initialised*
student gives relearning speed *"closely match[ing] that of the oracle itself."* Their method
**UNDO** (Unlearn-Noise-Distill-on-Outputs) operationalises this with shrink-perturb,
`θ_perturbed = (1−α)θ_suppressed + αβN`, `α=0.6, β=0.1` on their synthetic tasks and `α=0.25` on
WMDP with Gemma-2-2B.

**Their own honest limitation, which matters for how much weight to put on this:** the dramatic
Pareto advantage does **not** reproduce on the realistic benchmark. Verbatim: *"In our WMDP
experiments on a pretrained Gemma-2-2B model, the UNDO model achieves 4.88% lower on MMLU on
average. While this places UNDO on the Pareto frontier, other methods achieve a similar trade-off
between robust forget and retain performance in this setting"* — contrasted explicitly with
*"UNDO far exceeding the Pareto frontier set by the baseline methods"* on language/arithmetic. So
the **oracle-matching failure result is clean and synthetic-task-based**; the proposed remedy is
only competitive, not dominant, on WMDP.

### The implication for H11, stated plainly

This is the single most important adversarial fact for H11, and it is an *a fortiori* argument:

> Oracle matching is the **strongest possible output-level behavioural intervention** — perfect
> agreement with a model that genuinely never saw the forget data, on **all** tokens. It still
> fails relearning robustness. H11 is an output-level behavioural intervention on a **strict
> subset** of tokens, deliberately leaving most of the forget answer's behaviour **intact**.
> Therefore H11 cannot be expected to deliver relearning robustness, and by the "latent
> capability" argument it should be expected to be **more** relearning-fragile than uniform
> forgetting, not less — because it touches strictly less of the model's behaviour.

This is consistent with, and sharpens, our own HISTORICAL result that MDLM unlearning is
reversible (0.049 → 0.605 in 10 relearning steps; `research/relearning-robust-mdlm` @
`a67d610044d0`), and with the three-way agreement already recorded in `PRIOR_ART.md` §7.1 that
parameter-subset unlearning does not survive fine-tuning (RMU's own paper, Fan et al., our result).

**This does not refute H11.** It scopes it. H11's claim must be **disclosure prevention under a
non-adaptive reader, plus utility preservation** — explicitly *not* robustness to relearning or
to adaptive elicitation. If we later want robustness, 2506.06278 says the lever is the parameter
space (noise/distil/reinit), not the target, and that lever is **orthogonal** to H11 and
composable with it. Writing that scope down now prevents a reviewer from reading an unmade claim
into the paper. **It should go in the pre-registration, not be discovered later.**

---

# 4. Two facts I established by direct computation on our own data

These are not from any paper. They are the most decision-relevant findings in this document, and
the first is a correction to my own earlier work.

## 4.1 Correction: our local RWKU data does NOT cover 10 entities for training

**`PRIOR_ART.md` says, in the Executive Summary and §2.0/§2.4: "our local data matches EMDU's
protocol exactly: same 10 subjects, 582 forget pairs each, verified on disk." That is wrong and I
am correcting it.** I verified 582 pairs for *three* entities and separately observed ten entity
*directories*, and conflated the two. The actual inventory, enumerated file by file on 2026-10-20:

| copy | entities | `pair.json` (train) | F-L1 | F-L2 | **F-L3** | N-L1 | N-L2 | positive/negative | triviaqa |
|---|---|---|---|---|---|---|---|---|---|
| `/data/bijaypandey/llada_rwku/data/rwku/Target` | **10** | **absent** | ✓ 8–20 | ✓ 3–20 | **absent** | ✓ 22–30 | ✓ 22–30 | absent | absent |
| `/data/bijaypandey/llada_emdu/data/rwku/Target` | **3** (Stephen King, Confucius, Bruce Lee) | ✓ **582** | ✓ | ✓ | ✓ 28–40 | ✓ | ✓ | ✓ 582 | King only (100) |

Consequences, all of which change the plan in `PRIOR_ART.md` §2.5:

- **We can train MDU/EMDU/H11 on three entities, not ten** — the forget training corpus
  (`pair.json`) exists only in the `llada_emdu` copy, for three subjects.
- **F-L3 — the adversarial-robustness axis both MDU and EMDU report — exists locally for those
  same three entities only.** Any 10-entity table we produce cannot include F-L3.
- A 3-entity result is **not** comparable to MDU's or EMDU's 10-subject macro-averages, and must
  not be presented as such. Per-entity variance on RWKU is large (F-L2 probe counts alone range
  3–20), so a 3-entity mean is a weak estimator.
- Fetching the remaining seven subjects' `pair.json` from the public RWKU release is a **data**
  download, not a model-weight download, so it is not obviously barred by our standing
  constraints — but that is a coordinator decision, not mine to assume. **Flagging it as the
  cheapest single unblock available.**

## 4.2 The decisive finding: H11's success mode is invisible to the field's metric, and RWKU has no room for the mechanism

H11's distinctive prediction is *"a fluent, otherwise-correct answer that fails at exactly the
interdicted token."* I checked whether that is (a) expressible in our data and (b) measurable by
the metric. Both answers are bad, in opposite directions.

**(a) At evaluation time, RWKU answers are too short for the mechanism to mean anything.**
Measured over all local entities:

| probe set | n | mean words | median | **% answers ≤ 2 words** |
|---|---|---|---|---|
| forget_level1 (cloze) | 166 | 1.57 | **1** | **86.1%** |
| forget_level2 (simple question) | 131 | 2.16 | 2 | 71.0% |
| forget_level3 (adversarial, 3 entities) | 97 | 2.10 | 2 | **89.7%** |
| neighbor_level1 | 288 | 1.18 | 1 | 95.8% |
| neighbor_level2 | 275 | 2.31 | 2 | 64.4% |

Example (`3_Bruce_Lee/forget_level1.json`): query *"Bruce Lee founded the martial arts philosophy
known as ___ Kune Do."*, answer **"Jeet"**. There is no "rest of the answer" to leave untouched.
**On 86–90% of RWKU forget probes, H11 is definitionally identical to uniform forgetting**, because
the minimal identifying subset *is* the whole answer.

**(b) At training time there is plenty of room — the two sides are mismatched.** `pair.json`
responses are long biographies: **mean 592 words, median 663, max 794.** So H11 has ample scope to
select a minimal subset *during training*, but the evaluation cannot see whether it did.

**(c) On long-form answers, where the mechanism does apply, ROUGE-L recall scores H11's intended
success as near-total failure.** I computed LCS-based ROUGE-L recall directly for the
coordinator's own example and others:

| gold | H11-style output (one token interdicted) | ROUGE-L recall |
|---|---|---|
| `Author Ji-Yeon Park was born in Seoul , South Korea .` (11 w) | `... born in Lahore , South Korea .` | **0.909** |
| `Hina Ameen completed her Bachelor's at the University of Karachi , Pakistan .` (13 w) | `... University of Multan , Pakistan .` | **0.923** |
| `Jeet Kune Do` (3 w) | `Wing Chun Do` | 0.333 |
| `Jeet` (1 w) | `Wing` | 0.000 |
| `Pillsbury` (1 w) | `Normson` | 0.000 |

Base-model TOFU forget ROUGE-L recall is 0.822 (EMDU) / 0.884 (MDU). **An H11 run that perfectly
achieved its stated goal on TOFU-length answers would report a forget score of ~0.91 — i.e.
*above* the un-unlearned base model — and be recorded as having done nothing, or made things
worse.** Meanwhile MDU and EMDU reach 0.113 and 0.030 by destroying or emptying the whole answer.
**Under the field's standard metric, H11 cannot win, by construction.**

### What this means operationally

The mechanism and the measurement are mismatched, and no amount of implementation effort fixes it:

1. **H11 needs a metric it is not currently scored by** — interdicted-span exact match, or
   entity-level recall of the identifying token, or an identifiability/re-identification score.
   This metric **must be pre-registered**, because inventing a metric after seeing that ROUGE-L
   says 0.909 is exactly the move that makes a result unpublishable.
2. **A happy accident worth noting:** because RWKU answers are 1–2 words, RWKU's ROUGE-L recall is
   *already* effectively entity-level exact match (0 or 1 for a single-token gold). So RWKU is a
   *fair* scorer for H11 — it just cannot demonstrate H11's **distinctive** claim, since there is
   no surrounding answer to preserve. RWKU can answer "did it forget?"; it cannot answer "did it
   forget *more surgically than uniform forgetting*?"
3. **The distinctive claim needs long-form answers, i.e. TOFU** — which `PRIOR_ART.md` §2.3 (D2)
   established is out of reach for us (~250,000 full-parameter SFT steps, no released checkpoint,
   ~70+ GPU-hours per backbone against a 4.1-GPU-hour project history).
4. **Therefore, as of today, we have no benchmark on which H11's distinctive claim is both
   expressible and affordable.** That is a scoping problem to solve before building, and it is
   independent of whether H11 is novel.

**A constructive route, offered without having tested it:** the forget *training* corpus
(`pair.json`, 592-word biographies with `positive.json`/`negative.json` companions at 582 each) is
long-form and we have it for three entities. A held-out *generation* probe built from those
biographies — ask for the biography, score whether the interdicted entities reappear — would be
long-form, native to RWKU, affordable, and measurable at entity level. It would be **our own
probe, not a published benchmark**, so it could not support a SOTA claim; it could support a
mechanism claim. **NOT VERIFIED that this works — it is a proposal, and it needs the
`PRIOR_ART.md` §4.1 comparability contract applied to it before use.**

---

# 5. An MDLM-specific prediction that H11 may fail, which no AR paper could have found

This is my own analysis, not from any paper. It is cheap to test and should be tested before
building, because if it holds, H11's core premise is wrong *specifically on this model class*.

## 5.1 The argument

H11's premise — break the weakest link, leave the rest — is sound for an **autoregressive** model.
There, leaving the tokens *after* the interdicted one untouched cannot help the model produce the
interdicted token, because they are not in its conditioning set when it is generated.

**In a masked diffusion LM that reasoning does not transfer, and the direction of the error is
adverse.** Three facts from the primary sources in `PRIOR_ART.md` combine:

1. **Bidirectional context.** `p_θ(y^(i) | x, y_t)` conditions on *every* unmasked position, before
   and after `i`. Un-interdicted answer tokens are context for the interdicted one at every step.
2. **Confidence-ordered unmasking.** LLaDA-style decoding unmasks the positions the model is most
   confident about first. The un-interdicted tokens are, by H11's own construction, the ones the
   model is still confident about — because H11 deliberately left them alone. So they get unmasked
   **early**, and the interdicted position is then predicted with **maximal** surrounding context.
3. **This is a measured effect, not a speculation.** MDU's own token-role analysis found the
   conditional-anchor KL is *highest* on **in-context** tokens — tokens whose content is available
   in the context — and their post-training KL barely moves (**−3.8%**), versus **−20.6%** on
   stored-knowledge tokens. And EMDU's **failure mode 1** is precisely that surviving unmasked
   answer tokens *"bias p_anchor at the remaining masked positions"* — their Figure 3 shows
   `"The author's [M] name [M] Hsiao Yun-Hwa."` leaking the target through exactly this channel.

**So H11, on an MDLM, builds the scaffolding that lets the model re-infer the interdicted token.**
Gold *"born in Karachi, Pakistan"* → H11 trains only on `Karachi` and leaves `Pakistan` fully
intact and confident. At decode time the model emits *"born in ___, Pakistan"* with `Pakistan`
unmasked early, and now has to fill one slot given a prompt about a Pakistani author plus the
literal token `Pakistan`. The residual uncertainty over that slot is far lower than it would be
with no scaffold. **H11's selective sparing is, mechanically, an in-context hint.**

## 5.2 Why this is a sharper problem than the metric problem

§4.2 said ROUGE-L recall would score H11's intended success at 0.909 and so cannot see it. This
section says something worse: **the intended success may not occur at all on an MDLM**, because
the un-interdicted remainder actively assists recovery. Those are different failures — one is
measurement, one is mechanism — and the second is not fixable by changing the metric.

Note the irony worth putting in any writeup: **EMDU's whole second contribution (FAM) is to
*remove* surviving answer tokens from the anchor's view because they leak. H11 proposes to
deliberately *preserve* surviving answer tokens in the trained model's view.** Those are opposed
design pressures acting through the same channel. That does not make H11 wrong, but it means H11
must engage with FAM's evidence rather than ignore it, and a reviewer who knows EMDU will raise
exactly this.

## 5.3 The cheap test, before any training

This costs no training and almost no compute, and it decides whether H11 is worth building:

> Take forget QA pairs with long answers. For each, construct the state where **every token except
> the identifying one is unmasked** and the identifying token is masked — i.e. the best case for
> recovery, and exactly the state H11's decoding path tends toward. Measure the base model's
> probability of the correct identifying token. Compare against the same measurement with the
> whole answer masked except the prompt.

If the base model already recovers the interdicted token with high probability **from the spared
context alone**, then no amount of forget pressure concentrated on that one position can prevent
disclosure, because the information is reconstructible from what H11 chose to preserve. That is a
**forward-pass-only measurement** on the existing checkpoint — no gradients, no training, minutes
of GPU time, well inside our ledger.

A second, equally cheap diagnostic: the same measurement as a function of **how much** of the
answer is spared, which turns H11's binary choice into a tunable and tells us whether a
minimal-interdiction operating point exists at all. **If the curve is flat and high, H11 has no
operating point on this model class and we should learn that now rather than after building it.**

**NOT VERIFIED — this is a prediction and a proposed experiment, not a result.** I am stating it as
the thing most likely to decide H11's fate, not as evidence against it.

## 5.4 One mitigation H11 could adopt, for completeness

If the diagnostic shows recovery from spared context, the mechanism is not necessarily dead: the
interdiction set could be defined as a **minimal cut** rather than a minimal token set — the
smallest set of positions whose joint masking makes the identifying token unrecoverable, rather
than the smallest set of identifying tokens. That is a strictly different selection criterion
(it depends on the model's conditional structure, not only on corpus salience), it would require
spoiling `Pakistan` as well as `Karachi` whenever `Pakistan` suffices to recover `Karachi`, and it
would partially give up H11's "leave the rest untouched" selling point. **Recording it so the
option is on the table, not recommending it.**

---

# 2. The nearest neighbour, and it is very near: WHP's anchored terms

**This is the centrepiece of the check, and I found it in a source I had already extracted for
`PRIOR_ART.md` rather than through any search.** All quotes below are from the extracted source
text of the paper, read directly.

**Citation:** Eldan, Ronen; Russinovich, Mark. "Who's Harry Potter? Approximate Unlearning in
LLMs." arXiv:2310.02238v2, 4 Oct 2023. <https://arxiv.org/abs/2310.02238>

## 2.1 WHP states H11's thesis, in 2023, almost verbatim

WHP §2.2, verbatim:

> *"rather than forgetting the entity 'Harry Potter', our goal should be thought of as forgetting
> the **link** between the entity 'Harry Potter' and the entity 'magic' (or 'Hogwarts'). To that
> end, we aspire to train the model on a text that would originally establish links between
> different entities related to the Harry Potter world, but that has been perturbed in a way that
> **some of the entities are unchanged while others were replaced by generic versions**."*

"Some of the entities are unchanged while others were replaced by generic versions" **is minimal
interdiction.** The selection criterion, verbatim:

> *"we relied on GPT-4 to perform simple entity extraction on the unlearn target: We provided it
> with random passages of the text and instructed it to extract a list of expressions, names or
> entities **which are idiosyncratic to the text**. For each such expression, we asked for an
> alternative expression that would still be suitable in terms of text coherence, but **is not
> unique to the books**."*

"Idiosyncratic to the text" / "not unique to the books" **is an identifiability criterion.** The
worked dictionary is `{'Hogwarts': 'Mystic Academy', 'Quidditch': 'Skyball', 'Harry': 'Jon', ...}`
— ~1,500 anchor terms. And WHP has positional loss selectivity too:

> *"(i) Make sure that any instance of an anchored term that appeared previously in the same block
> will not be integrated into the loss function from the second appearance and onward."*

**So the structure "find the identifying tokens, interdict only those, leave the rest coherent" is
published prior art from October 2023.** H11's differences from it are real but narrower than the
framing suggests: WHP selects by **GPT-4 judgment** where H11 selects by **corpus salience**; WHP
trains toward a **generic substitute** label at those positions where H11 applies **forget
pressure** (an exclusion target); WHP is **autoregressive** where H11 would be **MDLM**.

## 2.2 WHP already named the rare-token selector — as future work it did not do

This is the direct answer to the coordinator's adversarial question, and it is better evidence
than anything a search for "TF-IDF unlearning" would have produced. WHP §5 Conclusion, verbatim:

> *"A nuance of our methodology involves a reliance on GPT-4's existing knowledge of the Harry
> Potter universe. To detect specific anchored terms and devise generic counterparts, the expertise
> of GPT-4 proved useful. This raises the question whether our technique achieve similar efficacy
> when stripped of such vast prior knowledge. Preliminary experiments show that entity extraction
> can still be effective when this knowledge is absent, and **we speculate that the lack of
> familiarity with idiosyncratic expressions can be addressed with simple n-gram frequency
> analysis but we leave a more thorough study for future work**."*

**Read that carefully, because it cuts both ways:**

- **Against novelty:** corpus-frequency-based selection of identifying terms for unlearning is a
  **named, published conjecture**, attributable to Eldan & Russinovich 2023. "Weight the rare
  tokens" is not merely an obvious objection a reviewer *might* raise — it is a specific sentence
  in the single most famous unlearning paper. **Any writeup of H11 must cite this sentence
  directly and position against it. A reviewer who knows WHP will find it, and not citing it would
  look like concealment.**
- **For the contribution:** WHP **did not do it.** "We leave a more thorough study for future work"
  is explicit, and their "preliminary experiments" sentence is about entity extraction without
  GPT-4's prior knowledge, not about frequency-based selection. So the mechanism is
  **unimplemented and unevaluated in the published record** — as far as my reading goes.

So the honest statement is: **H11's selector is a published conjecture that nobody has
implemented, not a new idea.** That is a materially weaker novelty position than "nobody has
thought of this", and the difference matters for how the work should be framed and where it could
be published.

## 2.3 WHP's own ablation is the closest thing to an H11 result, and it is not encouraging

WHP §4.1, verbatim:

> *"Using anchored terms in separation (namely, taking α=0 in equation (1)) was more effective, but
> falls short of achieving the same results as the combination of techniques. We performed a
> parameter search whose objective is find the model with the best possible performance on general
> benchmarks such that its familiarity score matches the model produced by the combination of
> techniques. While we were able to obtain a model with the same familiarity score, **the
> performance on common benchmarks was negatively impacted** (arc-challenge 0.40, arc-easy 0.70,
> boolq 0.79, hellaswag: 0.54, openbookqa: 0.33, piqa: 0.75, winogrande: 0.61)."*

Against the full method's 120-step column (ARC-C 0.414, HellaSwag 0.557, BoolQ 0.796, WinoGrande
0.657) and the base model (0.440 / 0.577 / 0.807 / 0.663), anchored-terms-alone is **worse on
utility at matched forgetting**: HellaSwag 0.54 vs 0.557, WinoGrande 0.61 vs 0.657, ARC-C 0.40 vs
0.414.

**Caveat I want to be precise about, because it limits how much this proves:** WHP's
anchored-terms-alone is *not* H11. It still computes cross-entropy at **all** positions (against
generic predictions derived from the translated text); only the *label construction* is restricted
to anchor terms. H11 restricts the **loss support** itself. So this ablation does not directly
measure H11. But it is the nearest published evidence, and its direction is **against** H11's core
selling point — that concentrating pressure on identifying tokens preserves utility better. **It is
a single run, one backbone, one target text, no seeds.** I would not call it decisive; I would call
it a warning that the utility-preservation claim is not free and must be measured, not assumed.

## 2.4 WHP's stated limitation lands directly on our benchmark

WHP §5, verbatim:

> *"Extending our approach to other types of content, particularly non-fiction or textbooks,
> presents its own set of challenges. Unlike the fictional universe of Harry Potter, **non-fiction
> content will not possess the same density of unique terms or phrases**."*

And:

> *"a pertinent observation emerges when considering the unique attributes of the Harry Potter
> series. The books are replete with idiosyncratic expressions and distinctive names—traits that,
> in hindsight, **may have abetted our unlearning strategy**."*

**RWKU is non-fiction — real public figures.** The authors of the closest prior art identified
non-fiction as the regime where anchored-term interdiction is expected to struggle, and attributed
their own success partly to an unusually high density of unique terms. Combined with §4.2's
measurement that RWKU forget answers are 1–2 words, **there are now two independent reasons, one
from the prior art and one from our own data, that H11 is a poor fit for RWKU specifically.**

For completeness, WHP also anticipated the latent-capability point that 2506.06278 later
demonstrated (§5): *"our current methodology—basing our evaluation on prompts presented to the
model and assessing the resultant completions—though effective in certain scenarios, could
potentially be blind to more adversarial means of extracting information. It's conceivable that
non-traditional or intricate methods, such as delving into token probability distributions, might
inadvertently reveal the model's latent familiarity with unlearned content."*

---

# 3. Deliverable 1 — the selection criterion of every existing selector, precisely

All six were obtained as **raw arXiv e-print LaTeX or ACL PDF and read locally** — no summariser
in the path. Quotes are verbatim from those sources.

**Two corrections to `PRIOR_ART.md` arising from this pass**, recorded rather than silently fixed:

1. I described SPAN's criterion as an "EK-FAC score". **That is wrong.** SPAN's Eq. (4) is an
   influence-function-style *forget-minus-retain gradient alignment*,
   `s_j ≈ (ḡ_forget − ḡ_retain)ᵀ H̃⁻¹ ∇_θ L(y_j | x, y_{<j}; θ)`. EK-FAC appears only in their
   Appendix A, purely as the inverse-Hessian approximator. Their anchor rule is also
   **parameter-free**, which I had not stated: softmax-normalise the scores, then keep tokens whose
   normalised importance exceeds *"the mean value of the normalized importance scores for the given
   sequence"*.
2. "Not All Tokens Are Meant to Be Forgotten" **does have an arXiv version — arXiv:2506.03142** —
   which I had listed as AAAI-only. Its method is named **TIF** (Targeted Information Forgetting),
   with **TPO = LPL + PL**. It is the closest mechanism to H11 of the six and I had it filed as
   "verified real, not deep-dived".

## 3.1 The comparison table

| | criterion is a function of | selected set | per-token form | complement treated how | minimal/sufficient framing? |
|---|---|---|---|---|---|
| **WGA / TNPO / WTNPO** (Wang et al., ICLR 2025, arXiv:2502.19301) | **the model's own probability** (+ frozen reference for TNPO/WTNPO) | **ALL answer tokens** — `i = 2..|s_u|`, no mask, no indicator | continuous `w^wga = p_θ(s^i|s^{<i})^α`, **never exactly 0** | n/a — nothing is excluded | **none found.** Motivation is *"excessive unlearning"* and *"the inverse confidence term"* |
| **SU** (Wan et al., Findings EMNLP 2025, arXiv:2506.00876) | **two external assistant models** trained on different data splits; abs. difference of next-token probability vs threshold γ | **strict subset** — *"unlearns only those tokens"* | binary `S(t_i)=1 if \|p¹−p²\|>γ`, plus an n-gram window | **excluded** (no loss) | **none found.** Zero hits for minimal/sufficient/smallest. The framing is the **opposite — completeness:** 5-grams *"to ensure the removal of complete information"* |
| **SPAN** (Yoon et al., ACL 2026 short) | **gradient / influence score** (forget-minus-retain mean gradient, inverse-Hessian preconditioned) + the model's own sampled generations for span growth | **strict subset** `ŷ ⊆ {1..|y|}` | binary `w_j ∈ {0,1}`; spans via self-consistency `ŝ_j = {t : (1/K)Σ_k 𝟙[t∈s_j^{(k)}] ≥ τ}` | **excluded** (`w_j = 0`) | **none found.** Goals are span *coherence and stability* |
| **SEUL** (Wang et al., AAAI 2025, arXiv:2402.05813) | **the model's own pre-unlearning log-probability**, thresholded — **with an explicit corpus-rarity rationale** (see §3.2); offline variant uses ChatGPT + verification | **strict subset** | `Select(x,α) = {x_t : log p'_θ(x_t|x_{<t}) < α}`, α preset or the mean log-prob; plus a *bridge* rule for gaps ≤ 2 | **excluded** | **none found** (only "minimally impacts generation") |
| **FUnDIAL** (Dong et al., arXiv:2402.10052) | **syntactic / entity role** — spaCy nouns + named entities | **strict subset** — *"the loss for this variant only applies to specific targeted tokens"* | `𝟙_e(x_t)`; **no case-wise formula is given anywhere** — the entire specification is *"treating nouns and entities as key tokens"* + *"we use the spaCy parser"* | **excluded** | **none found.** Authors call it *"a reasonable yet very simplifying assumption"* |
| **TIF / TPO** (Zhou et al., AAAI 2026, **arXiv:2506.03142**) | **an external model** — DistilBERT mask-infill **exact-match**, or **ChatGPT-4** semantic answer-relevance | **strict subset, and a partition** `y_f = ŷ ∪ ȳ` | **no mask equation.** LPL (logit-preference) on `ŷ`; PL (cross-entropy) on `ȳ` | **actively retained** by a positive loss `ℓ_PL = −E[log P_θ(ȳ|x_f)]` | **YES, informally — the only one of the six.** See §3.3 |

## 3.2 Answering the question that matters: is minimal, identifiability-ranked subset selection standard?

**Selecting a strict subset of the answer tokens is standard — five of the six do it.** So that
part of H11 is unambiguously prior art. The sharper question the coordinator posed is whether any
of them selects by an **information / salience / identifiability** criterion rather than by token
role. The precise answer:

- **By token role:** FUnDIAL alone (spaCy noun/NER tags), and its authors flag it as a crude proxy.
- **By the model's own probability or loss:** WGA/TNPO/WTNPO (all tokens, soft) and SEUL (subset,
  hard threshold).
- **By gradient/influence:** SPAN.
- **By external-model divergence or judgment:** SU, TIF.
- **By corpus statistics directly (TF-IDF, IDF, unigram frequency, PMI, Zipf rank):** **none of the
  six.** That specific estimator is unoccupied in this set.

**But "unoccupied estimator" is much weaker than "unoccupied criterion", and SEUL is why.** SEUL's
online selection rule is motivated, verbatim, by exactly H11's reasoning:

> *"We assume that content spans with privacy leakage risks are most likely **not common knowledge**
> (i.e. the trained model should not have seen them **frequently** during training). This means that
> they are probably with **low language probability** in language models (a higher perplexity score
> accordingly)."*

That is a **rarity criterion**, stated as such, published at AAAI 2025, and implemented — it simply
estimates rarity through the model's own log-probability rather than through a corpus count. And
SU's n-gram variant (*"we trained 5-gram models on MUSE-News and 3-gram models on TOFU"*) is, in
substance, a corpus-frequency difference, though the paper frames it as knowledge-scope divergence.

**So the identifiability-by-rarity criterion is published and implemented. What H11 would change is
the estimator, not the criterion.** That is a much narrower contribution than the framing implies,
and it is the single most important thing in this document for how H11 should be positioned.

## 3.3 The minimal/sufficient framing — present once, never operationalized

Five of six have no minimality or sufficiency language at all. **TIF is the exception**, and it
matters because TIF is otherwise the closest mechanism. Main text, verbatim:

> *"we emphasize that unlearning should exclusively target UW `ŷ`, **ensuring that only the
> necessary information is unlearned** while preserving general information."*

And in its GPT-4 annotation instruction (their appendix Table 7), verbatim:

> *"Important words should directly answer the question and be **sufficient to provide a complete
> and exact answer**."* … *"a list of important words that **precisely answer the question**."*

**But it is framing, not machinery.** The word "minimal" never appears in this sense; there is no
minimality objective, no claim of *the smallest* such set, and no search over set sizes. The
sufficiency language is an **instruction to an annotator LLM in an appendix**, not a property the
method optimizes or verifies. Their reported annotation stability is a Jaccard index of
**0.887 / 0.909 / 0.869** across repeated GPT-4 runs.

**Net:** "the smallest subset that suffices to destroy identifiability, as an objective you
optimize and verify" is **not claimed by any of the six.** That is the narrow strip of genuinely
unclaimed ground, and it is narrower than "minimal-interdiction unlearning".

## 3.4 Two precision findings worth carrying forward

- **WGA/TNPO/WTNPO is not a subset competitor at all.** Weights are strictly positive and
  continuous for every token; `α=0` degenerates to plain GA. If H11 selects a subset, this family
  is a *reweighting* baseline, not a selection baseline, and conflating them would be an error. It
  is, however, the family MDU and EMDU already ported to MDLM (WGA at γ=1).
- **SU's published dilation is self-contradictory.** The prose says symmetric "5-grams surrounding
  each selected token", but Algorithm 1's `ElsIf i ∈ l: Remove i from l` branch deletes the
  forward-dilated positions `i+1, i+2` when the ascending loop reaches them, leaving an effective
  **backward-only** `[i−2, i]` window. Reported from the pseudocode as printed; their released code
  was not inspected, so which the implementation follows is **NOT VERIFIED**. SEUL's ±2 rule is
  genuinely different — a *bridge* between two already-selected tokens within distance 2, not an
  unconditional window.

---

# 6. Deliverable 4 — category verdict

## 6.1 First, the structural observation that frames everything

**H11's loss is not new, and only its selection rule could be.** Write H11's objective:

```
L_H11 = E_{(x,y)~D_f} [ Σ_{j} w_j · ℓ_f( y_j | x, · ; θ ) ] + (retain term)   with  w_j ∈ {0,1}
```

That is **character-for-character the objective family of SPAN (Eq. 2) and SU**, and the loss form
of SEUL and FUnDIAL. Gradient path, preservation mechanism, parameterization, and reference-model
requirements are all identical to those methods. **The entire difference between H11 and published
work is the rule that produces `w_j`** — plus the model class.

That is clarifying, and it tells us where the novelty argument must be made and where it cannot.

## 6.2 The verdict

**Category (c) — a familiar combination — as currently specified.**

Applying my own rubric's step 5: each component has prior art, and the pairing is conventional.

| H11 component | prior art | status |
|---|---|---|
| Binary mask restricting the forget loss to a strict subset of answer tokens | SU, SPAN, SEUL, FUnDIAL, TIF | **published, five times** |
| Leaving the remaining answer tokens unpenalised | SU, SPAN, SEUL, FUnDIAL (TIF goes further and *actively retains* them) | **published** |
| Selecting by identifiability / rarity rather than token role | SEUL (implemented, explicit rarity rationale); SU's n-gram variant (implicitly) | **published** |
| Corpus-frequency / n-gram analysis as the identifying-term extractor | **WHP §5, as a named conjecture left to future work** | **published as a conjecture, not implemented** |
| "Interdict the identifying terms, keep the rest coherent" | WHP's anchored terms (§2.1) | **published, 2023** |
| Minimal / sufficient framing | TIF (informal, not operationalized) | **published as framing** |
| MDLM instantiation, where positions interact with mask rate `t` | **none** | **unoccupied** |

Six of seven rows are occupied. **The verdict is (c), not (d) and not (e).**

## 6.3 What would move it, and the exact checks

**To (d) — potentially novel combination with substantive interaction.** My rubric requires all
three of: a full factorial ablation; a superadditivity measurement exceeding seed spread; and a
mechanism story stated *before* the measurement. H11 has the mechanism story (identifiability /
weakest link). The factorial it needs is:

> {corpus-statistic selector} × {SEUL's log-prob selector, FUnDIAL's entity selector, TIF's
> external selector, all-tokens control} × {mask-out complement, TIF-style actively-retain
> complement} × {MDLM, AR}

with ≥3 seeds per cell. **The interaction that would earn (d) is specifically the MDLM one** — the
only unoccupied row — and §5 argues that interaction is likely **adverse**, not beneficial. So (d)
is reachable in principle but the evidence currently points the wrong way, and claiming it would
require overturning §5's prediction, not merely not testing it.

**To (a)/(b) — and this is the check to run first, because it is cheapest and could end the
matter.** My rubric §(a) says you may not claim better than (a) until you have run the numerical
comparison against every prior method of the same functional type. Here the loss forms are
*identical*, so the discriminating quantity is the **selected set**, and the check is a
**set-overlap measurement, forward-pass only, no gradients**:

> Compute Jaccard overlap between H11's corpus-salience selection and, on the same forget answers:
> (i) SEUL's `log p'_θ < mean` selection, (ii) FUnDIAL's spaCy noun/NER selection, (iii) TIF's
> DistilBERT mask-infill-mismatch selection, (iv) the top-`k` tokens by WGA's weight `p_θ^α`.

**If overlap with SEUL's rule is high, H11 is category (a) or (b) relative to SEUL** — the same
selected set reached by a different estimator — and the correct output is an equivalence note, not
a new method. Given that both rules are proxies for "this token is rare / not common knowledge",
**high overlap is the outcome I would bet on.** This costs minutes on one GPU and is the single
highest-value experiment in this document. It should be run before anything is built.

**Note also that TIF strictly dominates H11's design on the complement.** H11 *excludes* the spared
tokens from the loss; TIF *actively retains* them with a positive cross-entropy term. If H11's
rationale is "preserve the rest of the answer", the stronger published move is TIF's, and a
reviewer will ask why H11 did not do it. If we build anything here, the retain-the-complement
variant should be in the design space from the start, not added later.

## 6.4 Deliverable 5 — does the "weight the rare tokens" objection already have a paper behind it?

**Yes. Two, and they are worse for us than a generic TF-IDF paper would have been.**

1. **WHP (Eldan & Russinovich 2023) names the exact mechanism as future work**: *"we speculate that
   the lack of familiarity with idiosyncratic expressions can be addressed with **simple n-gram
   frequency analysis** but we leave a more thorough study for future work."* A conjecture in the
   field's most-cited unlearning paper is the hardest kind of prior art to argue against, because
   a reviewer does not need to find an obscure paper — they need only remember WHP's conclusion.
2. **SEUL (AAAI 2025) implements a rarity-based selector**, with the rationale stated explicitly:
   privacy-sensitive spans *"are most likely not common knowledge (i.e. the trained model should
   not have seen them frequently during training)"* and so have *"low language probability"*.
   Published, peer-reviewed, and operational.

Plus a third, weaker instance: **SU's n-gram assistant variant** is a corpus-frequency difference
in substance, albeit framed differently.

**What is NOT yet found:** a forget objective whose per-token weight is literally TF-IDF, IDF,
unigram frequency, PMI, or Zipf rank computed on an external corpus. My own arXiv API sweep over 25
such framings was still running when this section was written and two discovery agents are
outstanding; **that negative is therefore provisional and must be treated as NOT VERIFIED until
§7 is filled in.** But the strategic point does not depend on it: the *criterion* is published
twice over, so finding no paper with the literal TF-IDF weight would not restore novelty — it
would only mean nobody bothered to implement the estimator WHP suggested.

## 6.5 Bottom line

H11 is **(c)**. The honest framing available to it is *"implementing a selector that WHP proposed
and did not build, on a model class where token selection has never been tried"* — a legitimate
engineering-and-evaluation contribution, not a novel mechanism. And before any of that is worth
doing, three cheap things must come back favourably:

1. the **selector-overlap check** (§6.3) — if H11's set ≈ SEUL's set, stop;
2. the **recoverability diagnostic** (§5.3) — if spared context already recovers the interdicted
   token on an MDLM, stop;
3. a **benchmark and metric on which the distinctive claim is expressible and affordable** (§4.2) —
   we currently do not have one.

None of the three requires training. **All three can be done inside our remaining ledger, and any
one of them can end the investigation.** That is the recommendation: run them before building.

---

# 7. The finding that should decide this: TIF already ran H11's core ablation, and it failed

I found this on an independent spot-check of arXiv:2506.03142 after the selector pass, reading the
extracted source text directly. **It is the most consequential item in this document and it was
not in any of the search results — it is buried in a motivation subsection.**

**TIF's `NPO-GPT` baseline *is* H11.** It is NPO's forget loss applied **only to the identified
unwanted-word subset** of the answer, with the complement excluded from the loss — i.e. exactly
"concentrate all forget pressure on the identifying tokens, leave the rest untouched," with an
LLM-based identifiability selector instead of a corpus-salience one.

**Result, verbatim (their TPO Motivation):**

> *"To validate this, we integrate the unwanted information identifier into NPO, referred to as
> NPO-GPT in Figure 5. While NPO-GPT achieves a higher model utility score compared to the standard
> NPO, **it still experiences a 74% decline in utility**. These results suggest that **merely
> incorporating an information identifier into NPO is insufficient to mitigate utility degradation
> significantly.**"*

Context for the magnitude: plain NPO on TOFU forget05 drops model utility *"from 0.76 to 0.11"*.
So restricting the loss to the identifying subset recovers some utility but still loses roughly
three quarters of it. **Caveat on precision: these are read from their Figures 5–6; the "74%
decline" is the paper's own prose, and I did not find a numeric table for NPO-GPT. Treat the exact
value as figure-sourced.**

**And they diagnosed the mechanism**, verbatim:

> *"we analyze the cross-entropy loss values for GW, while the unlearning is limited to UW, as
> defined in Equation (2). **Although the model is not explicitly optimized to forget GW, the
> increase in loss values for NPO-GPT indicates that GW are also forgotten**, as shown in
> Figure 7."*

## 7.1 Why this is the finding that matters

**H11's premise is that sparing the non-identifying tokens preserves them. TIF measured that
premise on TOFU and it is false.** Excluding tokens from the loss does not protect them, because
the parameters are shared — the forget gradient on the interdicted positions degrades the spared
positions anyway. This is not an argument; it is a measurement with a figure, on the benchmark
whose answers are long enough for the distinction to exist.

This is why TIF added **two** further mechanisms, neither of which H11 has:

1. **PL — actively retain the complement** with `ℓ_PL(θ) = −E_{ξ_f~D_f}[ log P_θ(ȳ | x_f) ]`.
   Verbatim outcome: *"incorporating PL helps maintain stable and low loss values for GW.
   Consequently, NPO+PL-GPT exhibits a significantly slower decline in model utility while
   achieving comparable forget quality."* **"Leave the rest untouched" is the weaker design; "hold
   the rest in place" is what worked.**
2. **LPL — replace the probability-suppression loss with a logit-difference loss**, because even
   with PL, utility still degraded. Their diagnosis, verbatim and directly relevant to us:
   *"reducing the probability of target token `P(y_t)` can be achieved not only by decreasing its
   logits `z_t` but also by **increasing the logits of other tokens in the vocabulary**. This
   unintended effect distorts the model's overall logit distribution, potentially compromising its
   ability to retain general information."* Corroborated by their Figure 8: high KL divergence
   between reference and unlearned model on **both** forget and retain sets.

So the published trajectory is: identifier alone (= H11) → insufficient; + retain-the-complement →
better; + logit-difference loss → their method. **H11 sits at the first, already-rejected step of a
published three-step progression.**

## 7.2 A second, separate consequence — this one hits our own current method

TIF's LPL diagnosis applies directly to the exclusion target in our own reimplementation
(`PRIOR_ART.md` Q3). An exclusion target removes mass from the gold token and **renormalises it
over the rest of the vocabulary** — i.e. it *explicitly instructs the model to raise the logits of
all other tokens*, which is precisely the failure mode TIF identifies as distorting the logit
distribution and damaging retained knowledge. Their evidence is the two-sided KL of Figure 8.

**This is an independent, published criticism of our exclusion-target design, with a measurement
behind it, and it is actionable:** the logit-difference form (suppress `z_gold` relative to the
frozen reference, without prescribing where the mass goes) is the published remedy. **NOT VERIFIED
on an MDLM** — TIF is autoregressive — but the argument is about softmax arithmetic, which is
identical in both model classes, so I would expect it to carry. **Recommend this be raised with
whoever owns the loss implementation, separately from H11.**

**Code:** <https://github.com/xzhou98/Unlearning-TPO> (license **NOT VERIFIED** — not checked).

## 7.3 Revised verdict

The §6.2 verdict of **(c)** stands on the novelty question. But §7 adds something stronger than a
novelty judgment:

> **H11 is not merely a familiar combination — its central empirical premise has been tested in
> published work and reported insufficient.** TIF's NPO-GPT is H11 with a different selector, it
> was run on the right benchmark, and it lost ~74% of model utility. Under my rubric this is
> category **(e) evidence pointing the wrong way**: the difference is real, and the nearest
> published measurement is **against** the expected benefit.

Combined with §5 (on an MDLM, the spared tokens are bidirectional context that *assist* recovery of
the interdicted token, and EMDU's FAM exists precisely because that channel leaks) and §4.2 (RWKU
has no room for the mechanism; ROUGE-L cannot see it where it does), **my recommendation is not to
build H11 in its current form.**

What I would salvage, in priority order:

1. **The retain-the-complement variant (TIF's PL), ported to MDLM.** Published as effective,
   never tried on a diffusion LM, and it directly addresses the collapse problem that motivated
   H11 in the first place. Category (c) as a port, but with published evidence *for* it rather
   than against.
2. **The logit-difference forget loss (TIF's LPL), ported to MDLM** — and used to fix our
   exclusion target regardless of what happens to H11 (§7.2).
3. **The two cheap diagnostics** (§5.3 recoverability, §6.3 selector overlap) run anyway, because
   they are forward-pass-only, they cost almost nothing, and they are informative about the model
   class independent of H11's fate.

I am stating this as a recommendation on the evidence I gathered, not as a decision — the call is
the coordinator's, and §7.1's key number is figure-sourced, which is a real if modest caveat.

---

# 8. ETW — H11's criterion, published, benchmarked, and already shown to barely work

Found by my own arXiv API sweep (`abs:unlearning AND abs:"informative tokens"` → 2 results), then
read from extracted source text. **I had this paper filed in `PRIOR_ART.md` §7.5(d) as "verified
real, not deep-dived". That was a mistake of prioritisation — it is the closest prior art to H11's
selection criterion that exists.**

**Citation:** Koh, Seunghee; Baek, Sunghyun; Kim, Youngdong; Kim, Junmo. "Forget What Matters,
Keep the Rest: Selective Unlearning of Informative Tokens." arXiv:2604.17785v1, 20 Apr 2026.
KAIST / Hanbat National University. License **CC BY 4.0**.
<https://arxiv.org/abs/2604.17785>

## 8.1 It states H11's thesis and implements H11's criterion

The paper's own framing, verbatim:

> *"Within a single forget sample, tokens can be broadly categorized into two types: **informative
> tokens** and **structural tokens**... As it is essential to **selectively forget informative
> content while preserving overall utility, we aim to apply stronger penalties to informative
> tokens**."*

That is H11. Their selection signal is **token-wise predictive entropy**:

> *"We hypothesize that token-wise entropy can indicate token informativeness. Specifically,
> entropy measures how uncertain the model is about its next-token prediction. If the model is
> highly confident at a given position, the token is likely to be structural and carries limited
> semantic information. In contrast, if the model is uncertain and distributes probability mass
> across multiple candidate tokens, the position is more likely to encode informative or
> knowledge-related content."*

`H(y_i | y_{<i}) = −Σ_{v∈V} p(v | y_{<i}) log p(v | y_{<i})`, and the weight (their Eq. 8):

```
ω_i^ETW(x, y; θ) = n · H(y_i | y_{<i}; θ̂) / Σ_j H(y_j | y_{<j}; θ̂)
```

where `θ̂` is *"a stop-gradient copy of the model parameters θ, such that θ̂ takes the value of θ
but ∂θ̂/∂θ = 0 during backpropagation"*, and the probabilities use a softmax **temperature `T`**.
Note the normalisation: weights sum to the completion length `n`, which *"preserves the overall
scale of the unlearning loss by redistributing unlearning strength across tokens within a sample."*
**That is a design detail H11 should copy if it reweights at all** — otherwise "where the pressure
goes" is confounded with "how much total pressure there is".

Their argument for entropy over confidence is worth recording because it is a real insight: for a
fixed ground-truth confidence `p_i`, entropy still ranges between
`H_min(p_i) = −p_i log p_i − (1−p_i) log(1−p_i)` and
`H_max(p_i) = −p_i log p_i − (1−p_i) log((1−p_i)/(|V|−1))`, so *"the distribution over
non-ground-truth tokens governs the gap"* and entropy *"offers a significantly richer
representational range than measures based solely on ground-truth confidence."*

**Benchmarks: TOFU and WMDP.** Their informative/structural dichotomy is also, independently,
MDU's taxonomy minus the in-context category — two groups converging on the same decomposition.

## 8.2 The field already has a named taxonomy of token selectors, and a benchmark for them

ETW's Table 1 and §4.1 enumerate the existing selectors and sort them into families, verbatim:

> *"Existing token regularizers ... are categorized as **confidence-based** and
> **linguistic-based**. Confidence-based regularizers, including **WGA, Imp, SatImp, and TNPO**,
> commonly penalize tokens associated with specific confidence values... **Linguistic-based**
> approaches such as **SpaCy-Noun (SPN) and SpaCy-Entity (SPE)** adopt spaCy as an external parser
> to extract nouns or named entities."*

Plus binary variants **SCE and SCN**. And they built an evaluation **for the selector itself** —
not just for downstream unlearning: *"ROC–AUC curves for distinguishing informative tokens from
structural tokens across token regularizers on the TOFU forget 10% split"* (their Figure 2), with
per-token histograms (Figure 3) and token-wise visualisations (Figure 4).

**So "propose a new token-salience selector and show it separates informative from structural
tokens better" is an established, actively-benchmarked research activity with at least eight named
entries.** H11's corpus-salience selector would be a ninth entry in that taxonomy. Under my rubric
that is **category (c)** by the clearest possible evidence: the component, the pairing, *and the
evaluation protocol* are all pre-existing.

Constructively: ETW's ROC-AUC-against-ground-truth-informative-tokens protocol is **exactly the
selector-quality measurement my §6.3 overlap check needs**, and it is already defined, with
baselines. If we evaluate any selector, this is the protocol to use rather than inventing one.

## 8.3 And their own appendix reports that none of these selectors really works

This is the most important paragraph in the paper and it is in **Appendix I**, verbatim:

> *"Note that when using extracted informative tokens **automatically**, the AUC values in Figure 2
> move **closer to random performance**. Specifically, the AUC values are **0.58 for ETW, 0.46 for
> WGA, 0.54 for Imp, 0.54 for SatImp, and 0.51 for TNPO**."*

**Every confidence- and entropy-based token selector in the literature is at or near chance (0.50)
at identifying informative tokens, once the ground truth is produced neutrally rather than
curated.** ETW, the best, reaches 0.58. WGA is *below* chance at 0.46.

And the reason their headline Figure 2 looks better is disclosed, verbatim:

> *"Given that automatically extracted spans still require human verification to ensure correctness
> and consistency, **we decided to adopt the SatImp annotations for informative token discrimination
> analysis**."*

**So the ground truth in their main figure is derived from SatImp — one of the baselines being
evaluated.** That is a circularity, they disclose it, and it inflates the main-figure AUCs. Credit
to them for reporting the neutral numbers; **anyone citing their Figure 2 must cite Appendix I
alongside it.**

### What this means for H11

H11's premise has two parts, and this paper bears on both:

1. *"There exists a minimal subset of tokens whose removal destroys identifiability."* Plausible,
   and TOFU-style answers do have such subsets (their Appendix H shows human-verified examples:
   gold *"The author's full name is Hsiao Yun-Hwa"* → informative span `['Hsiao Yun-Hwa']`).
2. *"We can identify that subset automatically."* **This is the part the published evidence says is
   unsolved.** At 0.50–0.58 AUC, no existing automatic selector separates informative from
   structural tokens much better than a coin flip. H11 proposes a *new* automatic selector
   (corpus salience) for precisely this task — so H11's success depends entirely on beating a bar
   that five published methods have failed to clear, and the field has a benchmark that will
   measure whether it does.

**That reframes H11's real contribution and its real risk.** The contribution is not the
interdiction mechanism (occupied six ways over, §6.2) — it is *the selector*. And the selector is
the part with a published benchmark on which everything currently scores ~chance. **If H11 is
pursued, the first experiment is not an unlearning run at all: it is ETW's ROC-AUC protocol applied
to a corpus-salience selector.** That is cheap, it is the field's own measure, and it is
falsifiable in an afternoon.

---

# 9. Search coverage — what I searched and what returned nothing

My own arXiv API sweep (title+abstract fields, `export.arxiv.org`, 2026-10-20). **Caveat stated
plainly: the arXiv API searches title/abstract only, not full text, so these negatives are not
proofs of absence** — and indeed §7's decisive finding (TIF's NPO-GPT ablation) was buried in a
motivation subsection and would never have surfaced in any of these queries.

**Returned zero:**
`abs:unlearning AND abs:"token frequency"` · `abs:unlearning AND abs:"pointwise mutual
information"` · `abs:unlearning AND abs:"minimal sufficient"` · `abs:unlearning AND abs:"weakest
link"` · `abs:unlearning AND abs:"partial suppression"` · `abs:unlearning AND abs:"fact salience"`
· `abs:unlearning AND abs:"keystone"` · `abs:unlearning AND abs:"minimal edit"`

**Returned exactly one, none of them a match:**
- `abs:"tf-idf"` and `abs:"inverse document frequency"` → both return only **"Federated Unlearning
  via Class-Discriminative Pruning"** (arXiv:2110.11794). Federated, vision, **class-level channel
  pruning** — TF-IDF is used to score channel relevance to a class, not to weight a token-level
  forget loss. **Not a match**, and notably **this is the only TF-IDF-in-unlearning paper on
  arXiv.**
- `abs:"rare tokens"` → **"Localizing Paragraph Memorization in Language Models"**
  (arXiv:2403.19851). About *where* memorisation lives, not an unlearning objective. **Adjacent
  and worth reading if H11 proceeds** — it is the closest thing to evidence about whether rare
  tokens carry the identifying signal. **NOT VERIFIED — I did not read it.**
- `abs:"k-anonymity"` → "Efficient Unlearning with Privacy Guarantees" (arXiv:2507.04771).
  **NOT VERIFIED** as a match; k-anonymity appears but the method is not token-selective.
- `abs:"de-identification"` → "SoK: Semantic Privacy in Large Language Models"
  (arXiv:2506.23603), a survey.
- `abs:"minimal subset"` → "Machine Unlearning with Minimal Gradient Dependence..."
  (arXiv:2406.16986) — "minimal" refers to gradient dependence, not a token subset. **Not a match.**

**Returned many, none a match on inspection of titles:**
- `abs:salience` → 18 results, all **weight/parameter** saliency (SalUn arXiv:2310.12508, WSS-CL,
  Spectral Saliency, etc.), not *token* salience. The one titular exception, **"Anatomy of
  Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning"** (arXiv:2602.19612), is
  about fact-level salience effects, **NOT VERIFIED** as a token-selection method.
- `abs:identifiability` → 366 results; the term matches loosely (statistical identifiability,
  "identifying traces") and none of the top results is identifiability-based token selection.
- `abs:anonymization` → 16, `abs:redaction` → 2, `abs:"re-identification"` → 4 — the last are
  speaker/face identity unlearning, a different modality and problem.

**The honest summary of coverage:** the *literal corpus-statistic estimator* (TF-IDF/IDF/unigram
frequency/PMI as a forget-loss weight on tokens) appears **nowhere** in LLM unlearning on arXiv,
and I searched for it eight ways. But as §3.2, §8.1 and §8.2 establish, **the criterion it would
estimate — token informativeness / rarity / identifiability — is published, implemented, and
benchmarked under at least three different estimators (entropy, model log-probability,
external-model divergence).** Finding no TF-IDF paper does not restore novelty; it means nobody
chose that estimator.

**Two discovery agents on de-identification-literature ports and the rare-token objection were
still running when this document was finalised.** Their findings could add to §9's negative list
but cannot subtract from §§2, 3, 7, 8, which rest on primary text I read directly. **If they return
a hit, this document needs a revision note.**

---

# 10. Summary for the coordinator

| question | answer |
|---|---|
| Is selecting a strict subset of answer tokens standard? | **Yes — five of six published selectors do it** (SU, SPAN, SEUL, FUnDIAL, TIF). WGA/TNPO/WTNPO reweight all tokens instead. |
| Is selecting by *identifiability/salience* rather than token role standard? | **Yes.** ETW (entropy), SEUL (model log-prob, explicit rarity rationale), TIF (external-model answer-relevance), SU (assistant divergence). Only FUnDIAL is role-based. |
| Is a *corpus-statistic* estimator of that criterion published? | **Yes — see §11, which corrects this row.** SU-Ngram is a **per-token** forget mask from fixed external n-gram corpus statistics (and is that paper's best variant); AdaPop uses an **external corpus-frequency** proxy per fact and claims priority for exactly that framing. Only the narrow slot "per-token × single-corpus rarity × graded weight × forget loss" is unoccupied — and its tf·idf form is published for the *descent* objective (arXiv:2609.11029). |
| Does the "weight the rare tokens" objection have a paper? | **Yes, three.** WHP (2023) names n-gram frequency analysis as future work; SEUL (AAAI 2025) implements a rarity-motivated selector; SU's n-gram variant is a frequency difference in substance. |
| Category verdict | **(c) familiar combination**, with the nearest published evidence pointing *against* the expected benefit — §7's TIF ablation (74% utility decline) and §8.3's ~chance selector AUCs. |
| Recommendation | **Do not build H11 as specified.** Run three cheap forward-pass-only checks first (§5.3, §6.3, §8.3's AUC protocol); salvage TIF's retain-the-complement (PL) and logit-difference (LPL) mechanisms as MDLM ports instead. |
| Separate, unrelated action | **§7.2 — TIF's LPL diagnosis is a published criticism of our own exclusion target**, which explicitly redistributes mass onto non-gold tokens. Raise with whoever owns the loss. |

---

# 11. Addendum — this supersedes part of §9, and tightens the verdict

§9 concluded that the corpus-statistic estimator "appears nowhere in LLM unlearning on arXiv, and
I searched for it eight ways." **That conclusion was too generous to H11 and I am correcting it.**
My arXiv title/abstract sweep missed three papers because the relevant mechanism is not in their
titles or abstracts in the words I searched. All were verified by direct `curl` of the abs/HTML or
the ACL PDF with local text extraction — no summariser.

## 11.1 AdaPop explicitly claims the framing "corpus statistics instead of the model's own probability"

**Citation:** Borisiuk, Anna; Savchenko, Andrey; Panchenko, Alexander; Tutubalina, Elena. "The
More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning." arXiv:2608.14229,
14 Aug 2026. <https://arxiv.org/abs/2608.14229>

Their Related Work has a paragraph headed **"Local versus external calibration"** that states
H11's intended contribution as their own, verbatim:

> *"Reweighting the forget signal is not itself new; what separates methods is the quantity the
> weight is computed from. WGA and FaLW both derive it from the model's own predictive probability,
> a local signal that shifts as training proceeds. **AdaPop derives its per-fact exponent from an
> external corpus-frequency proxy fixed before training and independent of the model's outputs.**"*

And the priority claim, verbatim: *"We introduce AdaPop (Adaptive Popularity), **the first
training-time unlearning method to condition per-fact gradient updates on an external popularity
signal**."*

Objective (§3.2): `β_i = a · s_i^{−b}` (Eq. 4); `w_{i,t} = sg[ p_{i,t}(θ)^{β_i} ]` (Eq. 5);
`L_f(θ) = −(1/|Ω_F|) Σ_{(i,t)∈Ω_F} w_{i,t} ℓ_{i,t}(θ)` (Eq. 6), with `sg` a stop-gradient and
`s_i` an external popularity score (Wikidata sitelinks, or LLM-as-Judge). At `β=1` it *"recovers
WGA"*.

**How it differs from H11, precisely:** the corpus statistic is **per-fact** (one scalar per forget
sample) and only sets the *exponent*; the per-token factor is still `p_θ`. And the direction is
**inverted** — popular/frequent facts get *more* pressure, not less. **So AdaPop is not H11.** But
the rhetorical ground — "use an external corpus-frequency signal rather than the model's own
probability to allocate forget pressure" — is explicitly claimed and defended in print, with a
priority claim attached. **Any H11 writeup must engage this paper directly.**

## 11.2 SU-Ngram is a per-token forget mask computed entirely from external corpus statistics

This one I had in hand and mis-characterised. §3.1 recorded SU's criterion as "two external
assistant models", noting it was "implicitly frequency-based when n-gram". That understates it.
The paper's §3.2, verbatim:

> *"**Statistical: N-Gram Language Models** (Brown et al., 1992) learn and predict the probability
> of 'N-grams'—or continuous sequences of n tokens—in texts. We experiment with N-Gram models due
> to their efficiency and interoperability."*

And it is their **best** variant, verbatim: *"SU with N-Gram-based selection mechanism achieves the
overall best result. Compared with using LLM-based assistant models, N-Gram-based assistant models
yield better retain utility results."*

**So a per-token forget mask derived purely from fixed, external, count-based corpus statistics —
with no dependence whatsoever on the model being unlearned — is published, is the paper's
best-performing configuration, and is already a baseline in the token-selection literature.** That
is a direct hit on the one element §3.2 had identified as unoccupied. What remains distinguishable
is narrow: SU-Ngram uses a forget-vs-retain **contrast** (a PMI/log-odds-style distinctiveness
score), not single-corpus rarity, and a hard binary mask with ±2 expansion rather than a graded
rarity weight.

## 11.3 The tf·idf token-weighted loss already exists — on the descent side

**Citation:** Li, Zhijian; Larson, Stefan; Leach, Kevin. "Rebalancing Token Importance in Language
Models with TF-IDF Weighted Cross-Entropy Loss." arXiv:2609.11029, 10 Sep 2026.
<https://arxiv.org/abs/2609.11029>

`L_TF-IDF(θ) = −(1/L) Σ_{i=1}^{L} w_i log P_θ(x_i | x_{<i})`, with `tf_i = |{j : x_j = x_i}|` and
`idf(v) = log((1+N)/(1+df(v))) + 1`, normalised to unit mean. Their positioning, verbatim:

> *"In contrast to these methods, which primarily rely on dynamic model-dependent metrics or
> auxiliary reference models, **our TF-IDF objective utilizes static corpus statistics to establish
> token importance.** … **To our knowledge, applying TF-IDF statistics as a token-level loss
> weighting scheme for language model training has not been previously explored.**"*

> *"Pressure concentrates on tokens that are locally frequent and globally rare, the distinctive
> content words that TF-IDF upweights."*

**It is a training objective to prevent memorisation, not a forget objective** — the paper
explicitly separates itself from unlearning. But the mechanism is formally identical and the
motivation is nearly identical. **A reviewer will observe that flipping the sign is a one-line
change.** That is the single most quotable objection H11 will face, and it now has a paper.

## 11.4 The call to action is also already in print

**Citation:** Krishnan, Aravind; Reddy, Siva; Mosbach, Marius. "Not All Data Are Unlearned
Equally." arXiv:2504.05058, **COLM 2025**. Verbatim:

> *"We study how the success of unlearning depends on **the frequency of the knowledge we want to
> unlearn in the pre-training data** of a model and find that frequency strongly affects
> unlearning, i.e., more frequent knowledge is harder to unlearn. … our experiments highlight the
> need for … **novel methods for LLM unlearning that take the training data of models into
> account.**"*

So the *motivation* for H11's selector is published, the *framing* is claimed (AdaPop), a
*per-token corpus-statistic instantiation* exists (SU-Ngram), and the *exact tf·idf weighting* is
published for the adjacent objective (2609.11029).

## 11.5 The honest counter-evidence, for balance

Two citable facts do support that the *specific* slot was unoccupied as of late 2026:

- **ETW's Table 1 is a published taxonomy of every token regulariser in LLM unlearning, and it has
  exactly two categories** — "Confidence-based" and "Linguistic-based". **There is no
  frequency/corpus-statistic category.**
- **2609.11029 asserts** that TF-IDF token-level loss weighting *"has not been previously
  explored"* (for training).

So: *per-token, single-corpus-rarity, graded weighting of a forget loss* is a genuinely empty slot.
It is just boxed in on four sides, and the gap between it and its neighbours is small enough that
the burden of showing it matters is heavy.

## 11.6 Mandatory baselines and a survey pointer

If H11 proceeds, these are not optional:
**SU-Ngram** (per-token external corpus statistics), **AdaPop** (external corpus frequency, per
fact), **ETW** (entropy), **WGA** (confidence; already ported to MDLM by MDU/EMDU at γ=1), **TIF**
(external-model partition with complement retention).

Best survey of this space, per the adversarial pass and **NOT VERIFIED by me**: **ATWU**,
arXiv:2606.06320, Table 4 — benchmarks SEUL, SU-LLM, SU-Ngram, FUnDIAL, ETW, WGA and SatImp side
by side with a "Token weights" column naming each criterion. Read it before writing any related
work section.

## 11.7 Corrections to §9 and §10, stated explicitly

- **§9's claim** that the corpus-statistic estimator "appears nowhere in LLM unlearning on arXiv"
  is **withdrawn.** SU-Ngram is a per-token external-corpus-statistic forget mask; AdaPop is an
  external-corpus-frequency forget weight. My arXiv title/abstract queries could not have found
  either, because neither advertises "frequency" or "TF-IDF" in those fields. **This is a concrete
  illustration of the §9 caveat that title/abstract search is not full-text search — and the second
  time in this document that the decisive fact was buried in a body section (cf. §7).**
- **§10's row** "Is a corpus-statistic estimator of that criterion published?" answered "No". The
  corrected answer is: **"Yes, in two forms — per-token (SU-Ngram) and per-fact (AdaPop). Only the
  narrow combination 'per-token × single-corpus rarity × graded weight × forget loss' is
  unoccupied, and the tf·idf form of it is published for the descent objective (2609.11029)."**

## 11.8 Final verdict, unchanged in category and firmer in confidence

**Category (c) — a familiar combination.** Every component is published; the pairing is
conventional; there is a named taxonomy, a benchmark, and a survey for the activity H11 belongs
to. Moving to (d) would require the MDLM-specific interaction of §5 to be both tested and
favourable, and §5 argues it is likely adverse.

**The recommendation in §7.3 stands and is strengthened: do not build H11 as specified.** The three
cheap forward-pass-only checks (§5.3 recoverability, §6.3 selector overlap, §8.3's ROC-AUC
protocol) remain worth running because they are nearly free and each is independently informative
about the model class. The salvageable items remain TIF's PL and LPL as MDLM ports, and the §7.2
fix to our own exclusion target.

---

# 12. Second addendum — H11's rationale is also published, at ACL 2026 Main

The third discovery pass surfaced papers my own sweeps missed. **I verified the three load-bearing
ones myself** by `curl` of the arXiv abs pages with local metadata/abstract extraction — titles,
authors, dates and abstract text below are parsed from arXiv's HTML, not from any summariser.

## 12.1 PALU publishes H11's minimal-sufficiency rationale, with the word "sufficient"

**Citation:** Zhai, Naixin; Shao, Pengyang; Zheng, Binbin; Yang, Yonghui; Shen, Fei; Bai, Long;
Yang, Xun. "Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning."
arXiv:2601.03190, 6 Jan 2026. **Reported as ACL 2026 Main** (venue **NOT VERIFIED** by me — the
abs page I read does not state it). <https://arxiv.org/abs/2601.03190>

Abstract, verbatim:

> *"existing approaches typically **treat all tokens in a response indiscriminately** and enforce
> uncertainty over the entire vocabulary. This global treatment results in **unnecessary utility
> degradation** and extends optimization to content-agnostic regions. To address these limitations,
> we propose PALU (Prefix-Aware Localized Unlearning)... PALU reveals that **(i) suppressing the
> sensitive prefix alone is sufficient to sever the causal generation link**, and (ii) flattening
> only the top-k logits is adequate to maximize uncertainty in the critical subspace."*

**That first sentence is H11's motivation and that "(i)" is H11's thesis.** "Suppressing the
sensitive prefix **alone** is **sufficient** to sever the causal generation link" is the
weakest-link/minimal-sufficiency claim, published, as a headline finding, with code released.

The difference is the selection rule: PALU selects **positionally** (the sensitive *prefix*) plus
local entropy, not by an identifiability test. But the *rationale* H11 would lead with is no longer
available as a novel framing.

## 12.2 ATWU formalises the minimal-support result — and it is SOTA on our benchmark

**Citation:** Yüce, Gizem; Nikolaou, Giorgos; Flammarion, Nicolas (EPFL). "Learning What to Forget:
Improving LLM Unlearning via Learned Token-Level Importance." arXiv:2606.06320, 4 Jun 2026.
<https://arxiv.org/abs/2606.06320>

Abstract, verbatim:

> *"For autoregressive language models, not all tokens in a forget sample are equally relevant to
> forgetting. Existing approaches either ignore this heterogeneity or rely on auxiliary models,
> heuristics, or external annotations to estimate each token's relevance for forgetting. We instead
> characterize it through the interaction with the retain objective: a token is forget-specific to
> the extent that minimizing the forget loss on that token does not conflict with retain
> optimality. We formalize this perspective as a joint optimization problem over the model
> parameters and the token weights and show that, under a natural separation condition, **the
> resulting objective recovers the oracle forget-specific token support**. ... **Across TOFU and
> RWKU, ATWU achieves state of the art forget-retain trade-offs**, outperforming sample-level
> methods, probability-based token weighting heuristics, and auxiliary-model-based approaches.
> Moreover, **the learned scores align substantially better with ground truth forget-specific
> spans**."*

Three things matter here:

1. **"Recovers the oracle forget-specific token support"** is a formal minimal-support result —
   the theoretical version of what H11 wants, already proved under a stated condition.
2. **It is evaluated on RWKU** — our benchmark — and claims SOTA. **So ATWU is a mandatory baseline
   for us, not an optional one**, and it is the current bar on the exact benchmark we can afford.
3. **It has a selector-quality evaluation against ground-truth forget-specific spans**, which
   together with ETW's ROC-AUC protocol (§8.2) means *two* independent published protocols exist
   for measuring a selector. We do not need to invent one.

This also usefully refines §4.2: token selection on RWKU operates on the long-form forget
*training* corpus (`pair.json`, mean 592 words), not on the short eval probes. So H11 *can* be
trained on RWKU. **§4.2's conclusion is unchanged** — the short probes still cannot show
"fluent rest of the answer, wrong at one token" — but the training-side objection dissolves, and
ATWU is the proof that the setup works.

## 12.3 H11's intended output behaviour is published too, at NeurIPS 2025

**Citation:** Tan, Chenchen; Qu, Youyang; Li, Xinghao; Zhang, Hui; Cui, Shujie; Chen, Cunjian; Gao,
Longxiang. "Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention
Shifting." arXiv:2510.17210, 20 Oct 2025. **Reported as NeurIPS 2025** (venue **NOT VERIFIED** by
me). <https://arxiv.org/abs/2510.17210>

Abstract, verbatim:

> *"existing unlearning approaches face a critical dilemma: **Aggressive unlearning compromises
> model utility, while conservative strategies preserve utility but risk hallucinated responses.**
> ... AS is driven by two design objectives: (1) **context-preserving suppression that attenuates
> attention to fact-bearing tokens without disrupting LLMs' linguistic structure**; and (2)
> **hallucination-resistant response shaping that discourages fabricated completions** when queried
> about unlearning content."*

**That dilemma sentence is exactly H11's motivation, and those two design objectives are exactly
H11's intended output behaviour** — fluent and structurally intact (not collapse), and not a
fabricated substitution. Published, at a top venue, a year before H11 was proposed. It achieves it
by attention-level intervention rather than a loss mask, which is a real mechanistic difference —
but the *goal* is not a new goal.

## 12.4 Other mechanism matches found in this pass

Each occupies H11's mechanism with a different selector. **NOT VERIFIED by me** beyond title,
authors, ID and the quoted abstract fragments reported to me:

- **DTO** — Lee, Hong kyu; Liu, Ruixuan; Xiong, Li. "Direct Token Optimization." arXiv:2510.00125.
  *"we identify two categories of tokens: target tokens, which capture critical knowledge for
  unlearning, and the remaining non-target tokens... The former are used to optimize the unlearning
  objective, while the latter serve to preserve the model's performance."* Same two-population
  structure as H11.
- **TokenUnlearn** — Wu, Jiawei; Zhou, Doudou. "Unlearning What Matters: Token-Level Attribution
  for Precise Language Model Unlearning." arXiv:2605.00364. Combines *"knowledge-aware signals via
  masking, and entropy-aware signals"*, with both *"hard selection, applying unlearning only to
  high-importance tokens, and soft weighting."* The hard-selection variant is H11's mechanism.
- **D2F** — Cong et al., arXiv:2610.17654 (already in `PRIOR_ART.md` §6). *"removes only those
  parts while leaving the rest of the structure intact"* — then generates **counterfactual
  substitutions**, i.e. deliberately produces the fluent-wrong-answer behaviour H11 rejects. Useful
  as the contrast class.
- **ViKeR** (arXiv:2601.22020) and **ALTER** (arXiv:2603.01792) — entropy-based key-token selection,
  multimodal and LoRA-asymmetric respectively.
- **DFSU** — Zhou et al., arXiv:2601.15595. *"constructs token-level privacy masks... then performs
  token-level selective unlearning via a contrastive mask loss"* — the closest thing to a
  PII-driven token mask inside an unlearning loss, but the mask comes from a PII-type tagging
  dataset, i.e. the entity-role criterion H11 disclaims.

## 12.5 The clean negatives, which are the one thing left standing for H11

These were run against arXiv's own search over all fields and returned confirmed zero-result or
zero-relevant pages. **These are the coordinator's framings 2, 3 and 4, and they are genuinely
empty:**

- **k-anonymity / l-diversity / differential identifiability applied to model weights or to an
  unlearning objective: nothing.** The single hit, **EUPG** (Domingo-Ferrer, Jebreel, Sánchez,
  arXiv:2507.04771), applies privacy models to the **data** before pre-training and inherits their
  guarantees — not to weights, and not as a token-selection criterion. It is nevertheless the paper
  a reviewer would cite against a "privacy-model-based unlearning" novelty claim.
- **A de-identification / text-sanitization criterion ported into an unlearning loss: nothing
  found.** The source literature exists and is citable — Pilán et al., *"The Text Anonymization
  Benchmark (TAB)"*, Computational Linguistics 2022 (DOI 10.1162/coli_a_00458); Papadopoulou et
  al., *"Neural text sanitization with privacy risk indicators"*, LREC&Eval 2026 — but no port.
- **"Minimal sufficient" as an unlearning concept: no paper uses it.** Terminologically open.
- **Re-identification-risk unlearning for text: nothing** — the only hits are face and speaker
  identity unlearning.
- Also zero: `partial answer suppression unlearning`, `weakest link forgetting language model`,
  `key token unlearning sufficient disclosure`, `uniqueness criterion token unlearning privacy`,
  `fluent incorrect answer unlearning refusal`.

**One coverage gap stated honestly:** the "memorization is localized" / "which tokens are
memorized" literature (the coordinator's framing 7) was **not adequately swept** — arXiv full-text
ranking returned noise and the known localization papers were not confirmed. `abs:"rare tokens"`
did surface **"Localizing Paragraph Memorization in Language Models"** (arXiv:2403.19851), unread.
**That is the one place a surprise could still come from**, and it is the right first read if H11
proceeds, because it bears directly on whether rare tokens actually carry the identifying signal.

## 12.6 Consolidated verdict after all three passes

**Category (c), with high confidence.** H11 decomposes into three pillars and **all three are
published**:

| pillar | occupied by | venue |
|---|---|---|
| **Mechanism** — hard-mask the forget loss onto a strict subset of answer tokens | SU, TIF, DTO, TokenUnlearn, D2F, SEUL, FUnDIAL, PALU, DFSU | Findings EMNLP 2025, AAAI 2026, AAAI 2025, ACL 2026, + preprints |
| **Rationale** — one sub-span is *sufficient*; uniform treatment causes needless utility loss | **PALU** (*"suppressing the sensitive prefix alone is sufficient to sever the causal generation link"*); **ATWU** (*"recovers the oracle forget-specific token support"*) | ACL 2026 Main (reported), 2026 preprint |
| **Intended output behaviour** — fluent and structurally intact, no collapse, no fabricated substitute | **Attention-Shifting** (*"context-preserving suppression... without disrupting linguistic structure"* + *"hallucination-resistant response shaping"*) | NeurIPS 2025 (reported) |
| **Selector: corpus statistics** | SU-Ngram (per-token), AdaPop (per-fact), 2609.11029 (tf·idf, descent objective) | Findings EMNLP 2025 + preprints |
| **Selector: identifiability test specifically** | **nothing** | — |
| **k-anonymity / de-identification port** | **nothing** | — |
| **MDLM instantiation** | **nothing** | — |

**Five of seven rows occupied. The three unoccupied ones are the narrowest in the table, and §5
argues the MDLM row is likely adverse rather than favourable.**

**The recommendation from §7.3 and §11.8 is unchanged and now rests on much more evidence: do not
build H11 as specified.** If the coordinator wants to pursue anything in this area, the defensible
framing is no longer "minimal-interdiction unlearning" but something far narrower — and it must
clear ETW's and ATWU's published selector-quality protocols, cite PALU's sufficiency result, and
carry **ATWU as a mandatory RWKU baseline** since ATWU already claims SOTA on our benchmark.

The three cheap forward-pass-only checks (§5.3, §6.3, §8.3) remain the right next action because
each costs almost nothing and each can independently end the investigation.
