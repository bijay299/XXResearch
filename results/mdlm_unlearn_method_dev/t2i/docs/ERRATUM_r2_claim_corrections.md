# Erratum r2 — claim-level corrections from the audit (issue 12)

**Date:** 2026-10-23 · **Identifier:** `ERRATUM_r2` (new; supersedes nothing by deletion)
**Scope:** wording and inference errors in published T2I documents. Every number below is
**re-derived from the raw JSONs**, not copied from the document being corrected.

**How to read this file.** No historical document, JSON, pre-registration or retraction has been
altered or removed. Where a claim is wrong, this erratum states the correction and names the
location; the original text stays in place so the record of what was claimed, and when, survives.
Where a correction changes a number, the superseding number is given here under this identifier.

---

## C1. ⚠ The most consequential one: "restored to 1.000" is a MAX-over-attacks statistic, and a typical single attack recovers much less

**Claimed** (`FINDING34`, `FINDING35` §1, `T2I_REPORT` §1–2, in the form *"restored to exactly
H = 1.000"*, *"6 of 6"*): that the attack restores the erased concept to the base model's ceiling.

**Re-derived from the raw records.** Pure official AdvUnlearn arms (no UNet replacement), 300 steps,
post-dating the critical evaluation-bug fix, n = **25**:

| statistic | value |
|---|---|
| gate-passing (valid) attacks | **25 / 25** |
| attacks individually reaching H = 1.000 | **8 / 25 (32 %)** |
| median single-attack H | **0.800** |
| range | **0.3333 – 1.000** |
| **max-over-valid H, per concept** | **1.000 for all five** (church, parachute, garbage_truck, tench, van_gogh) |

Full distribution: `{0.333: 1, 0.467: 3, 0.600: 2, 0.733: 3, 0.800: 4, 0.867: 2, 0.933: 2, 1.000: 8}`.

**Correct statement:** *every one of the five concepts is restored to the base model's 1.000 by at
least one attack in the frozen grid, while a single attack recovers a median of 0.800 and reaches
1.000 about a third of the time.*

The threat-model conclusion is unchanged and arguably sharper — an adversary who may try six
configurations gets full recovery on every concept tested — but the published phrasing invited the
reading that *any* 300-step attack fully reverses the erasure, and that is false. Where a document
says "restored to 1.000", read "max-over-valid H = 1.000 under the six-point grid".

## C2. H's 1.000 is not model restoration — and we now have the measurement that separates them

**Claimed** implicitly throughout: that `H = 1.000` after attack describes the state of the model.

**Correct:** `H = 1.000` means a zero-shot CLIP classifier assigned the target label to all 15
evaluated images. It says nothing about the other eleven concepts or about image quality. Until
**issue 7** (`FINDING38` §4) this was unmeasured on every record. First measurements, `H65`:

| arm | target H | utility panel acc | prompt adherence |
|---|---|---|---|
| AdvUnlearn church, sd0 lr1e-5 | 0.400 → **1.000** | 1.0000 → **0.8750** (−0.1250) | 0.2647 → 0.2592 (−0.0055) |
| AdvUnlearn parachute, sd0 lr1e-5 | 0.133 → **0.867** | 1.0000 → **0.9792** (−0.0208) | 0.2689 → 0.2643 (−0.0046) |

So the attack is **not** buying recovery by destroying the model — it keeps 87.5–97.9 % of panel
accuracy — but it does cost measurable utility, which no published number in this phase reflected.
The remaining grid points are running; the full table lands in `FINDING38`.

## C3. Attacker cost: 300 steps is SUFFICIENT, not MINIMAL

**Claimed** (`FINDING37`, title and §1): *"the attack needs ~300 steps for full recovery"*.

**Correct:** 300 steps is the only budget measured post-fix. "Needs" asserts a lower bound that was
never tested — the 2-step and 50-step points were **voided** by the critical evaluation bug
(`ERRATUM_text_encoder_eval_bug`) and have not been re-measured. The defensible claim is
**"300 steps suffices"**. The retracted *"1 image / 50 steps"* claim stays retracted and is not
reinstated in weaker form.

## C4. The 2.14 ratio is a recovery-level ratio, not an attacker-cost multiplier

**Claimed** (`T2I_REPORT` §2(iii), `FINDING34` §127, `PREREG_H47` §50): *"a 2.1× reduction at equal
utility cost"*, which has been read in this project's own later text as though it described how much
more effort the attacker needs.

**Correct:** 1.000 / 0.467 = 2.14 is the ratio of **post-attack concept accuracies** between UCE and
damage+distil at matched utility, under one fixed attacker budget. It is **not** a statement that
the attacker must spend 2.14× more compute, images or steps — attacker cost was never varied in that
comparison. Nothing in this phase measures the budget multiplier required to close that gap, and
`FINDING37`'s voided low-budget points mean the cost curve that would support such a claim does not
exist.

## C5. H47's bar (H ≤ 0.50) does not refute H40 (H ≤ 0.40)

**Claimed** (`PREREG_H47` §78, and echoed in `T2I_REPORT` §6): that an arm at retain ≥ 0.85 with
max-over-valid H ≤ 0.50 *"refutes `PREREG_H40` by its own §3 condition"*.

**Correct:** `PREREG_H40`'s refutation condition is **retain ≥ 0.85 with max-over-valid H ≤ 0.40**.
An arm at H between 0.40 and 0.50 would meet H47's declared bar and leave H40 **standing**. The two
thresholds were set in separate documents and are not interchangeable; H47's row overstated what
clearing its own bar would establish. No arm has cleared either bar, so nothing downstream changes —
but the inference was invalid as written.

## C6. PREREG_H40 is an empirical asymmetry claim, not a theorem

`PREREG_H40` §19 already states *"this is an asymmetry claim, not a theorem"*, and that wording is
correct and stands. The correction is to **later references**, which have treated H40's condition as
though failing it were a proof of impossibility. It is a pre-registered empirical conjecture whose
declared refutation condition no tested arm has met — on **one model, five concepts, one attack
family, and our own protocol**. `FINDING35` §4's unfalsifiability warning (max-over-valid is monotone
in attacks run) applies to it and is not retired by `PROTOCOL_FROZEN_v1`; the frozen grid prevents
*future* ratcheting, it does not retroactively clean the pre-freeze numbers.

## C7. "NOT FOUND" in a novelty table means our search failed to find it

**Claimed** (`PREREG_H47` and `PREREG_H51` novelty tables): rows marked `NOT FOUND`.

**Correct:** `NOT FOUND` records the outcome of **our** literature search, which ran without web
search on a partly exhausted budget via the arXiv and OpenAlex APIs, and therefore under-covers
non-arXiv venues, very recent work, and anything whose phrasing differs from our query terms.
`PREREG_H51` §6 already carries two specific unverified items. **`NOT FOUND` is not evidence of
novelty** and must not be read or cited as such anywhere in this project. Where a mechanism matters,
the only acceptable basis for a novelty claim is a positively verified absence, which we do not
have.

## C8. The composite's teacher change was confounded with budget and construction changes

**Claimed** (`PREREG_H47` §51–53): that damage+distil's T2I utility failure was *"inherited from a
mediocre teacher"*, supported by moving from the UCE teacher (retain 0.850) to the official
AdvUnlearn teacher (retain 0.992).

**Correct:** the teacher was not changed in isolation. Between the UCE-teacher arms and the
AdvUnlearn-teacher arms, the **erasure site** changed (UNet cross-attention → text encoder), the
**distillation pool** was widened (`undo_t2i.RETAIN_POOL`'s broad-coverage block, added for `H41`),
and arm construction was **non-deterministic until `08742c1`** — the same config and seed produced
retain 0.867 and 0.783 (`FINDING35` §2). A retain difference across that boundary therefore cannot
be attributed to teacher quality. The teacher-quality hypothesis is **untested**, not supported.

## C9. SD v1.4 is an image latent diffusion model

Recorded for completeness because the two phases of this project sit in one repository and the
MDLM-phase vocabulary leaked into conversation about the T2I phase. **Stable Diffusion v1.4 is a
latent diffusion model over image latents with a frozen CLIP text encoder — it is not a diffusion
language model**, and none of the MDLM phase's token-level machinery (masked-diffusion ratios,
disclosure metrics, the exact retrain oracle) has any counterpart here. Where a T2I document borrows
an MDLM statistic name, only the **frozen trainability probe** (`FINDING33`'s fraction-of-loss-removed)
is genuinely the same statistic computed the same way; everything else is analogy.

## C10. Utility numbers for our own arms are in-sample (cross-reference)

Published damage+distil and composite retain figures are measured on prompts those arms were trained
on. Full statement, attribution and direction of bias: **`FINDING38` §2–3**, evidence
`outputs/split_registry.json`. Baselines (UCE, ESD, official AdvUnlearn) are unaffected. The
consequence for this erratum's purposes: **any cross-family utility ranking in a published table
that places a damage+distil arm above a baseline on retain is withdrawn.**

## C11. The audited-pin tallies were right at their pin; the current ones differ, and both are stated

The audit, reading the archive at its pin, found: *four of five official concepts reach H = 1.000 in
at least one valid run, tench reaches 0.733, and only 6 of the 15 official 300-step runs
individually reach 1.000.* **Those numbers were correct at that pin** and are not disputed.

At current HEAD they have moved, because `H62` ran afterwards (`H62` **has no results at the audited
pin** — any citation of it must carry its real provenance, which is post-pin):

| | audited pin | **current HEAD** |
|---|---|---|
| concepts reaching 1.000 on ≥ 1 valid attack | 4 of 5 | **5 of 5** |
| tench max-over-valid | 0.733 | **1.000** (`H62`, lr 5e-5 sd6) |
| official 300-step runs individually at 1.000 | 6 of 15 | **8 of 25** |

`H62` overturned our own "4 of 5" claim, which is recorded in the commit history as such. The
*fraction* of individual runs reaching the ceiling stayed roughly one in three across both pins.

## C12. Corrected composite numbers, as the audit recalculated them

| arm | pre-attack target | utility | **worst** valid recovery |
|---|---|---|---|
| church composite (corrected) | 0.000 | 0.925 | **0.933** |
| Van Gogh composite (corrected) | **0.600** | — | — |

Two consequences the published text did not state:

- The church composite's **worst** valid recovery is 0.933 — so it is not merely "reversed on its
  best attack", it is reversed on its *worst* one. Reporting max-over-valid understated this.
- The Van Gogh composite starts at target **0.600**, i.e. **worse suppression than its own official
  teacher's 0.4667**. The composite degraded erasure on that concept, and no published table said so.

## C13. FINDING32's loss pattern does not independently prove general untrainability

`FINDING32` reports late attack loss rising monotonically with α (0.3393 → 0.4864) and treats it as
establishing a trainability deficit. **It does not, on its own.** A higher loss on the attacker's
objective is consistent with trainability loss, with a worse starting point, and with the arm simply
being more damaged — the three are not separated by that measurement. The frozen fixed-noise probe
added in the issue-2/7 repairs is what would separate them, and it did not exist when `FINDING32`
was written. The trainability-confound claim is **provisional**, as `FINDING32`'s own title concedes
and its body then overstates.

## C14. FINDING36's loss values survive the evaluation bug; its paired recovery labels do not

The loss statistics in `FINDING36` are computed on the training objective and are **unaffected** by
the critical text-encoder evaluation bug. But the recovery values they are **paired against** came
from the broken evaluator in the affected runs, so every "loss X ↔ recovery Y" pairing involving an
encoder-erased arm needs the corrected recovery number. The conclusion the document draws from the
pairings is therefore not established by the pairings as published.

Further, `FINDING36`'s framing — that diffusion loss measures carry no information about recovery —
is **too strong twice over**: one uninformative statistic on one model does not show that *all*
diffusion loss measures are uninformative, and the cross-phase comparison with the MDLM result
assumes a comparability between a token-level objective and an image-latent MSE that nothing here
establishes (see C9).

## C15. AdvUnlearn's robustness claim is about prompt space, not weight updates

AdvUnlearn (2405.15234) claims robustness to **prompt-space** attacks, over a neighbourhood of text
embeddings. This phase attacks **weights**. Our finding that 300 UNet gradient steps reverse it is a
**scope statement about a different threat model** and is *not* a refutation of their paper's claim.
`T2I_REPORT` §2(ii) states this correctly; any shorter rendering of it elsewhere ("AdvUnlearn
fails", "the robust SOTA does not survive") must carry the threat model or it misrepresents their
result. The same applies to Pham et al. (2308.01508), whose concept-inversion attack is likewise
prompt-space and explicitly leaves weights untouched.

## C16. The competing-defence gap, stated as a gap

**Meta-Unlearning (2410.12777), IMMA (2311.18815) and GIFT (2507.13598) remain unrun.** They are the
relevant malicious-adaptation comparisons for this exact threat model, and no contribution claim in
this phase is credible without them. They are labelled unrun everywhere, and this erratum does not
soften that.

**From Dormant to Deleted (2505.22310) is related vision-classifier work, not an off-the-shelf T2I
baseline.** `PREREG_H51` §5 called it "the nearest published neighbour to this exact pipeline and the
one a reviewer would demand first", which overstates its availability: adapting it to SD would be new
implementation work, not running a released baseline. It is the nearest *conceptual* neighbour.

A bounded, pre-registered comparison using existing project resources is the right response, and no
open-ended baseline campaign is launched as part of this documentation pass. Only FRAG/FRP
(2608.25429) has released code among the five.

## C17. ⚠ Without a never-trained oracle, we cannot distinguish reactivation from relearning

This is the deepest limitation and it was never stated plainly. The attacker fine-tunes the erased
model on **images of the target concept**, generated from the public base model. When the concept
comes back, two explanations fit equally well:

1. the erasure only suppressed the concept and the attack **reactivated residual memory**; or
2. the erasure genuinely removed it and the attack **taught it again from the data it was given**.

**This phase cannot tell them apart.** There is no exact-retrain oracle at SD scale (`T2I_REPORT` §5
records the absence; `PREREG_H38` §3 pre-declared it as the single biggest asset the MDLM phase had
and this one lacks). Target-information-assisted fine-tuning cannot uniquely establish residual
memory, so wherever this project's T2I text says or implies *"the knowledge was never removed"* —
including `PREREG_H40`'s mechanism language about features "still present" — the supported claim is
narrower: **an adversary with weight access and 300 steps obtains a model that generates the
concept, whether by reactivation or by relearning.** For the threat model that distinction does not
matter. For the mechanism claim it is decisive, and the mechanism claim is not established.

`PREREG_H38` §3's floor proxy and its Tier-2 small-model plan were the pre-declared mitigations.
Neither was built, because Tier 1 never justified it.

## C18. The inseparability mechanism claim is provisional

`PREREG_H51` §7 concluded from forget-Fisher damage destroying utility ~9× worse than retain-Fisher
damage that *"the concept is not separably localised in weight space."* With the matched random-mask
control now in place (`random` matching `retain_fisher`'s realised perturbation norm to five digits,
commit `2352f8e`), that inference is better supported than it was — but **utility collapse plus a
matched control still does not prove inseparability**, because no direct overlap measurement between
forget-critical and retain-critical parameter sets was ever computed. The claim stays
**provisional**, as the audit requires.

---

## Status of the phase's headline after r2

Unchanged, and none of the above rescues it:

> Every concept-erasure method tested, including the published robust SOTA, is reversed to the base
> model's ceiling by a 300-step fine-tuning adversary with weight access, on all five concepts, with
> the attacked model still 87.5–97.9 % functional on retained concepts. The mandate's bar — an arm
> at retain ≥ 0.85 with max-over-valid H ≤ 0.40 — **remains unmet**, and the headline itself is
> **anticipated** by Meta-Unlearning (2410.12777) and `Erased or Dormant?` (2505.16174).

Five competing defences for this threat model remain unrun (Meta-Unlearning 2410.12777, IMMA
2311.18815, GIFT 2507.13598, From Dormant to Deleted 2505.22310, FRAG/FRP 2608.25429 — only the last
has released code). No contribution claim is credible without them, and this phase does not have
them.
