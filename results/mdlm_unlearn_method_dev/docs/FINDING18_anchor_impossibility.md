# Finding 18 — the anchor-construction space is closed: form and fact-identity are the same information

**Date:** 2026-10-21 · **Attempts:** `p272`, `p273` · **Script:** `scripts/route_b_probe.py`
**Output:** `outputs/routeb_theta0.json`, `outputs/routeb_theta0_span.json`
**Status:** closes the anchor-construction line by a general argument, not case-by-case

---

## 1. Two routes, one of which every published method ignores

Finding 13 measured the prompt-masked anchor at low mask rate and found `p_true = 0.651`. The
prompt is **fully masked** there, so that is not residual prompt knowledge. It is the model
completing the fact from the answer's own content. So disclosure has two routes:

| route | mechanism | removed by a prompt-masked anchor? |
|---|---|---|
| **A** | prompt → fact (conditional knowledge) | yes |
| **B** | partial answer → rest (self-completion) | **no** |

MDU (arXiv 2605.18253), EMDU (arXiv 2610.25447) and our AFC all use a prompt-masked anchor.
They block Route A only — and because the anchor itself self-completes, the student is trained
to **match a target that exercises Route B**. The objective preserves the pathway it should
remove. Route B is what makes iterative decoding leaky: once a few fact tokens commit, the rest
follows regardless of the prompt.

## 2. The obvious fix fails, and the failure is the interesting part

Candidate: mask **only the salient fact tokens** inside the answer, leaving structural tokens
visible. That should be fact-free (both routes blocked) *and* well-formed (scaffold kept) —
resolving the anti-correlation that closed SSA.

Measured on 15,938 answer positions, 100 forget facts, 4 noise draws, same states for all four
anchors:

| anchor | `p_true` overall | `p_true` at low mask rate | `fact_argmax` | entropy |
|---|---|---|---|---|
| `natural` (reference) | 0.466 | 0.823 | 0.121 | 1.71 |
| `mdu` — prompt masked | 0.256 | 0.715 | 0.050 | 3.23 |
| **`factmask`** — prompt + fact tokens masked | **0.208** | **0.687** | 0.040 | 3.48 |
| `fam` — prompt + whole answer masked | 0.017 | 0.011 | 0.000 | **8.19** |

**K1 fires:** `factmask` reduces leak by 0.048 overall and **0.028 at low mask rate**, against a
pre-registered requirement of 0.15. Fact-selective answer masking does **not** close Route B.

Run twice. The first run masked only the *first token* of each salient stem, which leaves the
continuations of multi-token facts ("Hsiao Yun-Hwa") visible — so the first result was
undecisive rather than negative, and is recorded as such. The second run masks **full token
spans** and gives the same answer (0.2084 vs 0.2564 overall). The negative is real.

**Also recorded: K2 as originally written was a bad test and fired spuriously.** It used
`punct + eos` as the formlessness proxy, but FAM scores 0.000 on both *because* its entropy is
8.19 — at near-uniform output the argmax is essentially random and rarely lands on punctuation.
Entropy is the correct measure, and by it `factmask` (3.48) is far better formed than FAM (8.19)
and comparable to MDU (3.23). K2 was replaced with an entropy comparison.

## 3. Why it fails, and what that implies

Masking the fact tokens *completely* barely changes the anchor's ability to predict them. So the
model is not reading the fact off the visible fact tokens — **it reconstructs the fact from the
structural scaffold alone.** `"The author's full name is ___ ___"` plus the answer's length is
very nearly unique among the 400 memorised facts; the scaffold *is* the identifier.

Blocking Route B therefore requires masking the scaffold too — which is exactly FAM, which
destroys form. The same conflict, now at a third and finer grain.

> **The general statement.** In this family, any anchor that retains enough answer context to be
> well-formed retains enough context to identify the fact. Form and fact-identity are carried by
> the same information, so no construction over *what to mask* can have both.

Three independent measurements of the same wall:

| construction | fact-free? | well-formed? | evidence |
|---|---|---|---|
| mask-rate banding (SSA) | only at high *t* | only at low *t* | `FINDING17`: anti-correlated across the schedule; uniform-over-*t* beats every band |
| full answer masking (FAM/EMDU) | **yes** (`p_true` 0.017) | **no** (8.19 nats; 97.91% DEGENERATE as a generator) | `FINDING13` |
| fact-selective masking | **no** (0.687 at low mask rate) | yes (3.48 nats) | this finding |

## 4. Consequence for the project

This closes anchor-based MDLM unlearning **by argument** rather than by exhaustion, and it
explains the trade-off that MDU, EMDU and AFC all sit on: it is a property of the anchor family,
not of any one objective. It is the first result in this project that generalises beyond the arm
that produced it.

Sixteen candidates are now closed — twelve by measurement, four by primary-source novelty check.

**It also says where not to go next.** Any further variation of *which tokens the anchor sees*
is predicted to fail by the statement above, so that space should not be re-entered. The
remaining untouched level of intervention is **representation-level** — acting on hidden
activations rather than on output distributions, as RMU does for autoregressive models
(Łucki et al., arXiv 2409.18025 discuss its behaviour; no MDLM equivalent exists in the four
known MDLM unlearning papers). That is a different level, not another anchor, and it is the one
direction this finding does not rule out.

**Caveat on that direction, stated now:** RMU-style representation misdirection exists for AR
LLMs, so an MDLM port is modest novelty on its own. What would make it more than a port is that
an MDLM's representation at a masked position is a function of bidirectional context *and*
changes across denoising steps, giving a layer × position × step localisation problem with no AR
analogue. Whether that matters is an empirical question and would need its own pre-registration
and its own novelty check before any training.
