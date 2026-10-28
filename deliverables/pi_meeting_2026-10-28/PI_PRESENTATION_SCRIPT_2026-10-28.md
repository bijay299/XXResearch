---
title: "Presentation Script"
subtitle: "Concept Unlearning in T2I Diffusion — page by page"
author: "Bijay Pandey"
date: "28 October 2026"
geometry: margin=2.4cm
fontsize: 11pt
colorlinks: true
---

**How to use this.** Roughly 20 minutes of talking. The *italic* lines are what to say; the bracketed notes are reminders, not lines. Don't read it verbatim — the point is to have the hard sentences already worked out so you're not composing them under pressure.

Two rules worth holding to: **lead with the ask**, and **volunteer the weaknesses before he finds them.**

\pagebreak

# Page 1 — Recommendation

*"I want to start with the conclusion rather than build up to it. We're recommending we stop trying to build a method, and instead write up the structural result — why none of these methods work, and what the one exception does differently."*

*"The short version: across 164 erased models and 1,215 attack measurements, nothing we built or ported survives relearning. Three of the four published methods are undone by a single gradient step. And the concept comes back even with no attacker at all — just ordinary fine-tuning on unrelated data."*

[pause]

*"There's one exception, found this week, and it makes the story better rather than worse. I'll come to it."*

*"And before anything else — this is the clause you wrote: that we're authorised to retire the direction if the evidence says it can't get there. This report is that evidence. I'm not asking you to abandon the instruction; I'm bringing you the condition you specified."*

[If he pushes immediately on "so you failed" — go straight to page 4's one-liner about our own method, then come back.]

\pagebreak

# Page 2 — Results at a glance

*"Six results. I'll take them in the order that makes the case, and I'm happy to stop on any of them."*

**① No adversary needed.** *"We fine-tuned an erased model on golf balls, french horns, cassette players and gas pumps. No church prompt, no church image, no attacker. ESD's erasure went from 0.467 suppression to full recovery. And utility went up, never down — the model got better at everything, including the thing it was supposed to have forgotten. That only happens if nothing was removed."*

**② Four gradient steps.** *"Three of the four published methods fall at one step. The fourth takes four. That's at batch size one."*

**③ Every method, one number.** *"Here's the table I'd most like you to look at. Every method, one column, all at 1.000. Including the four built specifically to resist fine-tuning. Including both of ours."*

**④ Capacity doesn't help.** *"We swept the attacker from 768 trainable scalars up to a full fine-tune of all 859 million parameters. A rank-4 LoRA — nine hundredths of one percent of the model — reverses everything."*

**⑤ The metric is broken.** *"This is the one I'd want you to see the pictures for."*

**⑥ The exception.** *"ResAlign. I'll do that properly in a few minutes."*

[If he only engages with one page, make it this one.]

\pagebreak

# Page 3 — The problem and the bar

*"Just to fix terms. Three requirements: the concept doesn't appear, everything else still works, and both survive someone fine-tuning the model afterwards."*

*"The first two are solved. Published methods do them, and they were never the hard part. The third is the open problem, and it's where everything fails."*

[Only if he asks what "the bar" was — otherwise skip, he wrote it.]

\pagebreak

# Page 4 — What was built

*"Scale, so you know what the claims rest on. 164 erased models, 1,215 attack measurements, 374 evaluation panels, around 30 pre-registrations, 78 numbered findings."*

*"Nine published methods ported and pinned to upstream commits. Two authors' released checkpoints obtained and attacked directly. Five concepts, three base models."*

*"Every experiment declared its success criterion and decision rule in writing before any number existed. That's why the retraction list later in this report is as long as it is — claims got tested against their own falsifiers rather than defended."*

\pagebreak

# Page 5 — Five families

*"Rather than nine methods, it's cleaner as five families. Four of them fail, and they fail in four different places for what turns out to be one reason."*

*"**Re-point** — UCE and ESD. They change where the word 'church' maps to. The machinery that draws churches is untouched by construction, so the attacker just points it back."*

*"**Immunise** — IMMA, GIFT, Meta-Unlearning, and our UTRAP. They train the model to resist fine-tuning. The problem is they rehearse against a generic fine-tuner, not the one that actually comes."*

*"**Prune** — FRP. Delete the important weights. But the mask is an initial condition, not a constraint. Nothing holds those weights at zero. And when we checked, the concept came back **through the weights FRP left alone**, not the ones it pruned."*

*"**Destroy and rebuild** — UNDO. Smash the weights, then rebuild from an erased teacher. At α=0.3 it erased perfectly — 0.0000 at four seeds — and halved utility. Erasure and utility turn out to be the same objective."*

*"And the fifth family is ResAlign, which is the exception."*

[The "what we swept" column is the pre-emptive answer to "did you configure them properly?" — point at it if asked. Two methods we repaired before testing: Meta-Unlearning was missing two of three terms in its own equation, and our UCE deviates from upstream deliberately, which is the only reason it erases at all.]

\pagebreak

# Page 6 — Why they fail, from first principles

*"There are exactly three ways an attacker can fail. The information isn't there; it's there but unreachable; or reaching it destroys the model. Every method is one of those three."*

*"All three break on the same thing: **the concept isn't separable.** A church is arches, towers, masonry, perspective — and every one of those is needed by barns, courthouses, houses. The information to draw a church is mostly a subset of the information to draw everything you're keeping. You can't delete a subset while preserving the superset."*

*"The analogy I'd use: you can't remove the number seven from arithmetic while keeping three, four and addition."*

*"And here's that argument as a measurement rather than a story. We built a method that weights the damage by which weights matter for the forgotten concept versus the retained ones — and **it lost to a random control.** The importance signal the whole idea depends on isn't there to find."*

[This is the strongest line in the talk. Say it slowly.]

\pagebreak

# Page 7 — The exception: ResAlign

*"ResAlign is a NeurIPS 2025 method and the only published work in this area that resists anything. We got the authors' own released checkpoints — not our re-implementation — and tested them."*

*"First, their claim reproduces exactly. They report an unsafe-content rate of zero before fine-tuning. We measure zero, over 600 images."*

*"But a zero means nothing on its own, so we ran base Stable Diffusion through the identical pipeline. It reads 0.18. So the detector fires, and their zero is real."*

*"Then we attacked it. At 25 steps — the budget that takes every other method to full recovery — it moves from 0.000 to 0.033. At 100 steps, still 0.033. Four times the budget bought the attacker nothing."*

*"At 300 steps it breaks — 0.133 against a base rate of 0.18."*

[pause]

*"So it isn't absolute. What it is, is **roughly a hundred times more expensive to break** than anything else we tested. That's a measured trade-off, which is exactly the kind of result you said counts."*

*"And we can say what they do differently. Their objective is one line: be erased now, **and don't become less erased after someone fine-tunes you**. They penalise the degradation directly."*

*"The contrast with Meta-Unlearning matters. Meta-Unlearning builds a booby-trap — fine-tune on the concept and the model self-destructs. That requires detecting the attacker, and the trigger signal is the same signal benign data produces. We measured it firing on normal use. ResAlign never detects anything. It makes erasure a fixed point of the attack."*

*"One more thing in their favour: they trained against LoRA attackers. We hit them with a full fine-tune of all 859 million parameters — out of distribution for them — and it still took 300 steps. They generalised beyond their own training."*

\pagebreak

# Page 8 — The measurement problem

*"This is the part I'd most like you to look at rather than hear."*

[Show the images.]

*"That's a clearly visible church. The metric scored it as a Van Gogh painting, 0.72."*

*"That's a window-grid texture with no church in it. The metric scored it church, 0.55."*

*"And that's UCE at a perfect 0.0000 suppression — abstract mush. No church, but no coherent anything either."*

*"The metric is a twelve-way classifier with no 'none of these' option. Every image gets forced onto a label. So it can't tell erasure from damage, and a robustness number measured on a broken model means nothing."*

*"This cost us a result. We had an apparent robustness advantage we'd built on for days. It was a classifier being fooled by noise on a model that still drew churches perfectly."*

*"One important exception: **this criticism doesn't apply to ResAlign.** Their metric is a trained binary detector and we verified its control. Their evaluation is sounder than the one we've been criticising, and the paper has to say that."*

\pagebreak

# Page 9 — What we checked against ourselves

*"I want to give you the weaknesses before a reviewer does."*

*"We triaged all 78 findings. About a third of our own headline claims have been retracted or narrowed — almost always by us, against falsifiers we'd written down in advance. Five died on seed replication alone."*

*"Four defects were blocking. Three are now closed."*

*"The one I'd flag is the second. Our validity gate — the rule deciding which attack measurements count — is biased. We measured it. **But the bias flatters defences.** Remove the gate and the result gets stronger, and the only two arms that shift are ours, both looking less robust without it. So our claim is understated by our own estimator, not propped up by it."*

*"The open one is fidelity. Our baselines are our re-implementations. The honest sentence is 'across nine re-implementations, every arm is reversed' — not 'every published method is reversed.' We've started closing it by attacking authors' checkpoints directly, which is how we got AdvUnlearn and ResAlign."*

[Volunteering D2 is worth more than any positive result. Don't skip it.]

\pagebreak

# Page 10 — The contribution

*"The paper would be: concept erasure in diffusion models is a rebinding, not a removal. Four families fail for one structural reason, a fifth escapes it, and the field's instruments can't tell the difference."*

*"Four things make it more than a tally."*

*"It's a **diagnosis**. The mechanism predicts the numbers — why 25 steps, why 768 scalars, why Fisher lost to random, why damage-then-distil erases perfectly and halves utility."*

*"It **invalidates an instrument** the field depends on."*

*"The **coverage doesn't exist elsewhere** — nobody has put the robustness-specific methods on one bench under one attacker with honest cell accounting."*

*"And it **answers a question another group publicly posed** — the AISTATS authors name extension to diffusion as open future work."*

*"Also worth saying: the five failed method attempts aren't embarrassment, they're five independent probes of one obstruction. Our own method failing identically to the published ones is the strongest evidence this isn't about effort."*

\pagebreak

# Page 11 — Ten days

*"Days one and two, finish the ResAlign work — the budget sweep is running, plus the second checkpoint and an image audit."*

*"Three and four, attack every authors' checkpoint we can get, and fix the attribution wording throughout."*

*"Five and six, the representation probe — can we recover the concept from an erased model's internal activations? That would take the mechanism claim from 'we refuted every removal hypothesis' to 'here is where the concept still lives.'"*

*"Seven and eight, figures and tables, and raise anything remaining to n=3."*

*"Nine and ten, write."*

*"Everything in that plan is either on disk already or measurable in hours. **Nothing in it depends on a method working.**"*

\pagebreak

# Page 12 — The decision

*"Four things I need from you."*

*"One — do we commit to this framing? It's no longer 'nothing works.' It's 'four families fail for one reason, a fifth escapes it, and here's the mechanism.'"*

*"Two — target venue and deadline. That decides how much of the fidelity work is worth doing."*

*"Three — does the NSFW evaluation line continue? ResAlign's concept is sexual content. It's their own benchmark with their own detector, but it's a deliberate choice."*

*"Four — is the representation probe worth two of the ten days, or do we spend them writing?"*

[Then stop talking.]

\pagebreak

# Hard questions, short answers

**"Did you try hard enough?"**
*"Our own method, built from eleven prior failure modes, failed identically — 18 of 18 attacks at 1.000."*

**"Is the attacker fair?"**
*"It's tiny. 25 steps at batch size one, 768 trainable scalars. And targeted relearning is published, not our invention. We also tested methods under their own threat models — GIFT was reversed under its own DreamBooth setting."*

**"Hasn't this been shown already?"**
*"Partially, and we say so. Reversibility is published for UCE and ESD. The representation argument is published for classifiers. What isn't: the robustness-specific methods on one bench, the cost curve, the mechanism, and the measurement defect."*

**"Will a negative publish?"**
*"An Illusion of Unlearning is AISTATS 2026 and is a pure negative with a diagnosis. And ours isn't purely negative any more — ResAlign works and we can say why."*

**"Why not one more method idea?"**
*"The last one we had was published a year ago. Meta-Unlearning, ICCV 2025, proposes exactly it. We'd already ported it and measured it failing."*

**"Doesn't ResAlign disprove your thesis?"**
*"It bounds it, and I'd rather have it than not. Four families fail for a structural reason. ResAlign occupies a fifth — and it still breaks at 300 steps. It costs the attacker a hundred times more; it doesn't make erasure permanent."*

**"What would change your mind?"**
*"If the representation probe fails — if we can't recover the concept from internal activations — then the non-separability story is weaker than I'm claiming and we'd need to say so."*
