> ### ⚠ CORRECTION NOTICE (audit of `0b435427`, remediation r5)
> This finding measured **Meta-Unlearning variant A**, whose objective omits **two of the three terms** of the paper's Eq. 8 (`ERRATUM_meta_objective.md`). The measurements stand. **Withdrawn:** that the *published* method's self-destruct trigger fires on normal use — premature benign damage is consistent with variant A's own altered objective. Also corrected: §5a describes the Meta arm as editing cross-attention; it optimises the **full UNet** (859.5M). The GIFT/TI matched-retain comparison is **limited evidence**, not isolation of a meta-objective mechanism (`ERRATUM_meta_objective.md` §4). Church-only: `PREREG_H80`'s both-concepts commitment is unmet (`PREREG_PRECEDENCE.md` §3).
>
> Original text below is **unaltered**. Register: `docs/CORRECTION_REGISTER_r5.md`.

---

# Finding 56 — Meta-Unlearning's self-destruct trigger fires on *normal use*, not just under attack. At the one λ that leaves the model usable, it buys **zero** robustness over its own no-meta ablation.

**Date:** 2026-10-24 · **Pre-registration:** `PREREG_H80_meta_unlearning.md`
**Method:** Meta-Unlearning for diffusion models, arXiv 2410.12777 — **our port**, labelled as such
throughout. Authors' code not released (`ERRATUM_r2` C19).
**Base arm:** `frp_church_r3_l05` (the `FINDING43` FRP arm: church 0.1333 at retain 0.9417)
**Protocol:** `PROTOCOL_FROZEN_v1`, the identical six-point grid as `H65`/`H66`.
**Evidence:** `outputs/gen_h80_meta_church_lam{0.0,0.5,1.0,2.0}.json`,
`outputs/atk_H80A_*.json` (11 attack points), `scripts/meta_unlearn.py`

---

## 1. The idea, and why it was worth porting

Meta-Unlearning's premise is the most direct answer anyone has published to this phase's negative
result. Every defence run here — UCE, ESD, official AdvUnlearn, damage+distil, the `frag`
allocation, FRP — is reversed by 300 gradient steps on 16 self-generated images (six methods,
`FINDING43` §5). Meta-Unlearning does not try to erase harder. It installs a **booby trap**: a
meta-objective shaping the loss surface so that *any* subsequent fine-tuning which recovers the
erased concept also destroys benign capability. The attacker is allowed to win and is supposed to
find the prize worthless.

The objective, as ported:

```
L = L_erase(θ) + L_retain(θ)  −  λ_meta · L_benign(θ′),   θ′ = θ − α ∇_θ L_attack(θ)
```

θ′ is one simulated attacker step. Maximising benign loss *at θ′* is what makes recovery
self-defeating. `PREREG_H80` pre-declared λ_meta = 0.5 as primary, a **λ_meta = 0 ablation** (the
same distillation with the trap removed), and the first-order caveat in §4 below.

## 2. The trigger installs. It just does not wait for an attacker.

The λ sweep, as-is erasure and utility before any attack:

| arm | church (target) | retain8 | utility (11 non-target) | K2 (retain ≥ 0.60) |
|---|---|---|---|---|
| **λ = 0.0** (ablation) | 0.1333 | **0.9417** | 0.9212 | ✔ |
| **λ = 0.5** (primary) | 0.0667 | **0.6333** | 0.6727 | ✔ (barely) |
| λ = 1.0 | 0.0000 | **0.1167** | 0.0909 | ❌ |
| λ = 2.0 | 0.0000 | **0.1167** | 0.1212 | ❌ |

Training logs confirm the mechanism is live and not a no-op: benign loss evaluated at θ′ is driven
to ≈ 1.0 on the λ = 0.5 arm against the ablation's 0.01747 — a **≈ 60× gap**. The meta term is doing
exactly what it is written to do.

**And the cost lands on the honest user.** At λ = 1.0 the model is at retain 0.1167 *with no
attacker present at all*. This is `PREREG_H80`'s **fourth declared outcome, verbatim**: the
self-destruct fires on normal use. λ = 0.5 is the only rung that clears the K2 utility floor, and it
clears it by 0.03 — having paid **0.31 of retain** (0.9417 → 0.6333) for 0.07 of extra erasure
depth.

## 3. The headline: at λ = 0.5 the trap bought nothing

All nine weight-space attacks passed the `FINDING20` validity gate (T rises +0.75 to +0.92, every
point ≥ 0.20), so these are valid attacks and the H numbers mean what they say.

| arm | gate-passing | per-attack H | **max-over-valid H** |
|---|---|---|---|
| **λ = 0.5** (meta) | **6 / 6** | 0.067→1.000, 0.067→1.000, 0.067→1.000, 0.000→1.000, 0.000→1.000, 0.067→1.000 | **1.000** |
| **λ = 0.0** (ablation) | **3 / 3** | 0.133→1.000, 0.133→1.000, 0.200→1.000 | **1.000** |

**Identical. Both at the ceiling, on every single configuration.** The meta objective — installed,
verified, and paid for with a third of the model's retained capability — produced **no measurable
robustness benefit** over running the same distillation with λ set to zero. Twelve of twelve attacks
across the two arms reached 1.000.

**K3 is NOT triggered.** `PREREG_H80` K3 says a low H with post-attack utility < 0.50 must be
reported as self-destruction rather than robustness. That rule never fires here, because H did not
stay low — it went **up to the ceiling**. There is no risk in this finding of mistaking collapse for
resistance; the arm simply lost.

### 3a. The attack also *repaired* the trap's collateral damage

The post-attack utility is the detail I did not expect, and it inverts the trap's logic:

| attack | retained capability before → after |
|---|---|
| λ=0.5, sd0, lr 1e-5 | 0.7708 → **1.0000** (+0.229) |
| λ=0.5, sd1, lr 1e-5 | 0.8125 → **1.0000** (+0.188) |
| λ=0.5, sd6, lr 1e-5 | 0.8125 → **1.0000** (+0.188) |
| λ=0.5, sd1, lr 5e-5 | 0.8125 → 0.8125 (0.000) |
| λ=0.5, sd0, lr 5e-5 | 0.7708 → 0.7917 (+0.021) |
| λ=0.5, sd0, lr **1e-4** | 0.7708 → **0.2917** (−0.479) |
| λ=0.0 ablation (×3) | 0.958–1.000 → 0.917–0.979 (−0.02 to −0.06) |

At the three lr = 1e-5 points the adversary recovered the concept to 1.000 **and left the model with
*more* benign capability than it started with.** 300 steps on 16 images of one concept partially
undid the trap's damage. The mechanism intended to make recovery worthless instead left a deficit
that ordinary fine-tuning fills in.

Only the most aggressive attack (lr 1e-4) degraded utility to 0.2917 — and it still reached
H = 1.000, so even there the attacker did not need the gentle setting. Reporting that single point as
the trap working would be selecting the one configuration that flatters the defence; the
max-over-valid rule (`FINDING35` §4) and the whole grid are above.

## 4. ⚠ The honest alternative explanation: this may be my port, not their method

My meta gradient is **first-order**. Differentiating `L_benign(θ′)` with respect to θ through
`θ′ = θ − α∇L_attack(θ)` produces a Hessian-vector term that `scripts/meta_unlearn.py` **drops**:

```python
gmeta = torch.autograd.grad(l_benign_prime, params)   # ∇_{θ′}, used as if it were ∇_θ
```

This is the standard first-order approximation (FOMAML-style), pre-declared in `PREREG_H80` §4
*before* any result existed. It is the single most likely reason a correctly-specified trap would
fail to couple recovery to collapse: the omitted term is precisely the one carrying how an
attacker's *step direction* depends on θ. **If that is the explanation, this result is about my port
and not about 2410.12777**, and it is stated that way here. It is not evidence against the published
method.

What the result *does* establish independent of that caveat: the trigger demonstrably installed
(the 60× benign-loss gap at θ′), and the utility cost at λ ≥ 1.0 is a property of the objective as
written, not of the approximation.

> ⚠ **UPDATE 2026-10-25 — this caveat is no longer the leading explanation. See `FINDING58`.**
> `PREREG_H81` §6 prediction 3 pre-committed to settling it with GIFT (2507.13598), whose bi-level
> defence obtains the same second-order curvature term **for free via alternating updates**, with no
> Hessian anywhere in its Algorithm 1 — and whose authors call that term *"crucial for our
> approach."* Our GIFT port implements that alternation faithfully, in plain SGD, as the derivation
> requires. **The term is present there, and that arm is reversed on 6 of 6 valid attacks,
> max-over-valid H = 1.000.**
>
> So a faithful second-order-equivalent bi-level defence is reversed just as completely as this
> first-order one. The approximation above remains a real infidelity of **my port** and stays
> disclosed — it is simply **no longer available as the explanation** for why the trap bought zero
> robustness.

## 5. The TI runs are reported as attacks that did not work, not as resistance

Textual Inversion on the λ = 0.5 arm, both runs:

| run | T before → after | rise | gate | H |
|---|---|---|---|---|
| ti sd0 | 0.208 → 0.083 | **−0.125** | **FAIL** | 0.067 |
| ti sd1 | 0.083 → 0.000 | **−0.083** | **FAIL** | 0.000 |

T went **down**. Under `FINDING20` these are invalid attacks and **contribute nothing** — a flat H
behind a failed gate is not a robustness claim, and reporting "TI could not recover the concept"
would be exactly the error the gate exists to catch. They are listed because omitting failed attacks
is how a ratchet gets quietly stopped early.

That said, TI failing to climb *at all* on this arm while full-UNet relearning trivially reaches
1.000 is an anomaly worth a stronger attempt rather than a footnote, so a push was run: 2000–3000
steps at lr 1e-2 and 2e-2 across three seeds, **with the same configurations on the λ = 0.0 ablation
as a matched control.**

### 5a. The push is done, and the asymmetry is real and reproducible

| arm | steps | lr | seed | T before → after | rise | gate | H |
|---|---|---|---|---|---|---|---|
| **ablation** λ=0.0 | 2000 | 1e-2 | 0 | 0.250 → 0.458 | **+0.208** | ✅ **PASS** | 0.067 → **0.400** |
| **ablation** λ=0.0 | 3000 | 1e-2 | 1 | 0.250 → 0.792 | **+0.542** | ✅ **PASS** | 0.267 → **0.733** |
| meta λ=0.5 | 2000 | 1e-2 | 0 | 0.208 → 0.000 | **−0.208** | ❌ FAIL | 0.067 |
| meta λ=0.5 | 3000 | 1e-2 | 1 | 0.083 → 0.000 | **−0.083** | ❌ FAIL | 0.000 |
| meta λ=0.5 | 3000 | 2e-2 | 6 | 0.125 → 0.000 | **−0.125** | ❌ FAIL | 0.000 |

**The control settles it: the budget is not the problem.** Textual Inversion is comfortably valid on
the no-meta ablation at these budgets and recovers the concept to **0.733**. On the meta arm the same
budgets fail the gate **three times out of three**, across two step counts, two learning rates and
three seeds, with T going **down** every time.

**So the meta trigger does do something — but only to the embedding-space adversary.** This is the
first measured effect of the meta objective anywhere in `H80`: at the same λ=0.5 where it bought
**zero** robustness against full-UNet relearning (§3), it makes a Textual Inversion attack
**untrainable** — the placeholder embedding cannot even fit the prompts it is optimised on.

⚠ **This is still not a robustness claim, and `FINDING20` is why.** A failed gate means the attack
did not work, and "the attack did not work" is not the same statement as "the defence held". Two
readings remain open and these data do not separate them:

1. the trigger genuinely obstructs embedding-space optimisation (the interesting reading), or
2. the λ=0.5 arm's loss surface is simply hostile to TI optimisation for reasons unrelated to the
   trap — note the arm is *not* degenerate (retain8 0.6333, target 0.0667), which weakens but does
   not eliminate this.

Distinguishing them needs a TI attack on a **non-meta arm at matched utility** — the ablation sits at
retain 0.9417, far above λ=0.5's 0.6333, so the present control is not utility-matched and part of
the asymmetry could be a utility effect rather than a trap effect.

> ✅ **THE MATCHED CONTROL RAN, AND IT ELIMINATES THE UTILITY EXPLANATION (2026-10-25).** GIFT
> β=1.3 (`FINDING58`) sits at retain8 **0.6083** against this arm's **0.6333** — a gap of 0.025,
> against the ablation's gap of 0.308. Identical TI budgets:
>
> | arm | retain8 | TI runs | T | gate |
> |---|---|---|---|---|
> | **Meta-Unlearning λ=0.5** | **0.6333** | 3 | **falls** every time (−0.083 to −0.208) | ❌ **0/3** |
> | **GIFT β=1.35** | **0.6000** | 1 | 0.208 → 1.000 (**+0.792**) | ✅ **1/1** |
> | **GIFT β=1.3** | **0.6083** | 3 | +0.750, +0.583, +0.833 | ✅ **3/3** |
> | GIFT β=1.0 | 0.7833 | 2 | 0.333 → 1.000 (+0.667) ×2 | ✅ 2/2 |
> | λ=0.0 ablation | 0.9417 | 2 | +0.208, +0.542 | ✅ 2/2 |
>
> **At essentially identical retained utility — 0.6083 vs 0.6333 — Textual Inversion trains to
> completion on GIFT and cannot get off the ground on the meta arm.** Reading 2 (low utility breaks
> TI) is **eliminated**, and GIFT β=1.35 makes the elimination airtight: at retain8 **0.6000**,
> *below* this arm's 0.6333, TI is still valid with the **largest T rise of any run (+0.792)**. Four
> of four matched runs valid, every one reaching H = 1.000. Lower utility than the meta arm does not
> merely fail to break TI — it hosts the strongest TI attack measured. Reading 1 — the
> meta trigger specifically obstructs embedding-space optimisation — now stands on a matched
> control, and GIFT is the right comparator for it, being also bi-level and also editing
> cross-attention.
>
> **So this is a real, isolated effect of the meta objective, and the only thing in this phase where
> a defence demonstrably interfered with an adversary's optimisation** rather than merely delaying
> the outcome.
>
> ⚠ **It is still not robustness, and `FINDING20` still says why.** A failed gate means the attack
> did not work; it does not license "the defence held". And the effect is **adversary-specific and
> narrow**: the same arm is reversed to H = 1.000 by full-UNet relearning on 6 of 6 valid attacks
> (§3). The honest statement is that Meta-Unlearning's trigger blocks the 768-scalar embedding
> attack and is irrelevant against the 859.5M-parameter weight-space one — which is the opposite of
> the threat ordering a defender would want.

**The accurate summary: this phase still has no valid TI measurement on the meta arm, has strong
evidence that no attainable TI budget will produce one, and — per the matched control above — now
knows that this is caused by the meta objective and not by the arm's reduced utility.**

> ⚠ **CROSS-MODEL (2026-10-25): the obstruction only PARTIALLY transfers, and the "no attainable TI
> budget" clause above is now falsified off SD-1.4.** The arm was rebuilt on SD-1.5
> (`meta_sd15_church_lam0.5`, target 0.0000 / retain8 **0.7667**) and attacked with both families:
>
> | model | adversary | budget | T before → after | gate | H |
> |---|---|---|---|---|---|
> | SD-1.4 | TI | 2000–3000, lr 1e-2/2e-2 | **falls** (−0.083…−0.208) | ❌ **0/3** | — |
> | **SD-1.5** | TI | 2000, lr 1e-2 | 0.125 → 0.125 (**+0.000**) | ❌ FAIL | — |
> | **SD-1.5** | **TI** | **3000, lr 2e-2** | 0.042 → 0.250 (**+0.208**) | ✅ **PASS** | **0.333** |
> | **SD-1.5** | full-UNet | 300 steps, lr 1e-5 | 0.042 → 1.000 (+0.958) | ✅ | **1.000** |
> | **SD-1.5** | full-UNet | 300 steps, lr 1e-5 | 0.000 → 0.958 (+0.958) | ✅ | 0.933 |
>
> **What transfers:** the *weaker* TI budget still fails outright (+0.000 — the attack cannot move its
> own training accuracy at all), so the obstruction is real on SD-1.5 too and is not a quirk of one
> checkpoint.
>
> **What does not:** the **3000-step / lr 2e-2 budget that gate-failed 3/3 on SD-1.4 passes on
> SD-1.5** and recovers to H = 0.333. So the obstruction is **weaker on SD-1.5 than on SD-1.4** —
> breachable by an attainable budget rather than apparently unbreachable. §5a's phrase *"strong
> evidence that no attainable TI budget will produce one"* is **correct for SD-1.4 and false for
> SD-1.5**, and is scoped to SD-1.4 from here on.
>
> **It remains not robustness, and the arm remains fully reversed.** TI's max-over-valid is 0.333,
> but **full-UNet relearning takes the same arm to 1.000**, so the arm's max-over-valid across
> families is **1.000**. Retain 0.7667 is also below the mandate's 0.85 floor. The one positive
> signal in this phase is therefore: **real, reproducible on a second checkpoint, weaker there, and
> irrelevant against the adversary that matters.**
>
> ### The complete cross-model picture: the obstruction is **GRADED**, not binary
>
> BK-SDM-small completes it. (λ=0.5 fell to retain **0.4833**, below K1 — a hyperparameter-transfer
> failure, not a verdict on the method, since the same pressure costs more utility on a half-size
> UNet — so λ=0.3 was built instead at target 0.2000 / retain **0.7667**, matching SD-1.5's
> retention exactly.)
>
> | model | retain8 | TI 2000/1e-2 | TI 3000/2e-2 | TI gate | **full-UNet** |
> |---|---|---|---|---|---|
> | SD-1.4 | 0.6333 | T **−0.208** | T **−0.083** | ❌ 0/3 | **1.000** (6/6) |
> | **BK-SDM-small** | **0.7667** | T **+0.125** ❌ | T **+0.083** ❌ | ❌ 0/2 — **holds** | **1.000** (T +0.667, util 0.938) |
> | **SD-1.5** | **0.7667** | T +0.000 ❌ | T **+0.208 ✅** | **breached** → H **0.333** | **1.000** (T +0.958) |
>
> **At matched retention (0.7667 on both new models) the obstruction holds on the pruned
> architecture and is breached on SD-1.5.** Three distinct behaviours:
>
> 1. **SD-1.4** — T *falls*: the attack moves backwards on its own training prompts.
> 2. **BK-SDM-small** — T *rises but stalls* below the 0.20 gate (+0.125, +0.083). ⚠ This is **not
>    "almost resistance"** — it is an invalid attack, and the gate exists so it cannot be reported
>    as anything else.
> 3. **SD-1.5** — the stronger budget breaks through to H = 0.333.
>
> So the effect is neither checkpoint-specific nor universal: it is **graded across models, with
> SD-1.5 the weakest case.** That is a sharper statement than "transfers" or "does not", and it is
> not what I would have predicted.
>
> **And every arm is still fully reversed.** The full-UNet control reaches **H = 1.000 on all
> three**, with post-attack utility 0.938 on BK-SDM — concept fully restored, half-size model
> undamaged. The obstruction is real, reproducible, graded, and **irrelevant against the adversary
> that matters.**

## 6. What this closes and does not close

**Closes:** the competitive-baseline gap by one more. Of five published competitors,
**two are now run** (FRP `FINDING43`, Meta-Unlearning here). Three remain: IMMA 2311.18815,
GIFT 2507.13598, From Dormant to Deleted 2505.22310. None of the five has usable released code.

**Does not close the mandate.** No arm, ours or published, has reached retain ≥ 0.85 with
max-over-valid H ≤ 0.40. The count of defences fully reversed by this adversary is now **eight** —
UCE, ESD, official AdvUnlearn, damage+distil, the `frag` allocation, FRP, and both arms here.

**Does not establish** that trap-based or meta-learned defences are inherently ineffective. One
first-order port, one base arm, one concept, one λ grid of four points. The decisive test of the
approximation would be a second-order implementation, which is the honest follow-up along with a
finer λ sweep between 0.5 and 1.0 — the only interval where a trigger that is both usable and live
could exist, given K2 fails at 1.0 and the trap bought nothing at 0.5.

> ⚠ **THE FINER SWEEP RAN, AND IT CONTRADICTS WHAT I EXPECTED (2026-10-25).** I predicted nothing
> usable would be in that interval. **Two arms in it clear K2 with the target fully suppressed:**
>
> | λ_meta | target | retain8 | utility | K2 (≥0.60) |
> |---|---|---|---|---|
> | 0.0 (ablation) | 0.1333 | 0.9417 | 0.9212 | ✔ |
> | 0.5 | 0.0667 | 0.6333 | 0.6727 | ✔ |
> | **0.6** | **0.0000** | **0.6250** | 0.6727 | **✔** |
> | **0.7** | **0.0000** | **0.7000** | 0.6848 | **✔** |
> | 0.8 | 0.0000 | 0.4750 | 0.5697 | ❌ |
> | 0.9 | 0.0000 | 0.0000 | 0.0788 | ❌ |
> | 1.0 | 0.0000 | 0.1167 | 0.0909 | ❌ |
> | 2.0 | 0.0000 | 0.1167 | 0.1212 | ❌ |
>
> **λ=0.7 reaches target 0.0000 at retain8 0.7000** — full suppression at the best retention of any
> trap-bearing arm, and *better* retention than λ=0.6 and λ=0.5 below it. **Retention is NOT
> monotone in λ_meta**: 0.6333 → 0.6250 → **0.7000** → 0.4750 → 0.0000. A coarse four-point grid
> would have missed this entirely, which is the argument for running the finer sweep rather than
> reasoning about it.
>
> **This does not rescue the method, and the attacks have now confirmed that.** Full target
> suppression at retain 0.70 is still below the mandate's 0.85 floor, and §3's finding — that the
> trap bought **zero** robustness over its own ablation — concerns *robustness*, which as-is numbers
> cannot address.
>
> **λ = 0.6, frozen grid at lr 1e-5, 3 / 3 gate-passing:**
>
> | seed | T rise | gate | target H | post-attack utility |
> |---|---|---|---|---|
> | 0 | **+1.000** | ✅ | 0.000 → **1.000** | 0.708 → **0.979** |
> | 1 | **+0.875** | ✅ | 0.000 → **1.000** | 0.667 → **1.000** |
> | 6 | **+1.000** | ✅ | 0.000 → **1.000** | 0.667 → **1.000** |
>
> **Every attack reaches the ceiling, and every attack also *repairs* the trap's utility damage** —
> the same inversion §3a documented at λ=0.5, now reproduced at the sweep's better operating point.
> An arm at target 0.0000 before the attack is at 1.000 after it, with *more* benign capability than
> it started with.
>
> **λ = 0.7 — the sweep's best arm — completed, 3 / 3 gate-passing, and is also fully reversed:**
>
> | seed | T rise | gate | target H | post-attack utility |
> |---|---|---|---|---|
> | 0 | +1.000 | ✅ | 0.000 → **1.000** | 0.667 → **0.854** |
> | 1 | +0.958 | ✅ | 0.000 → **1.000** | 0.708 → **0.938** |
> | 6 | +0.958 | ✅ | 0.000 → **1.000** | 0.625 → **0.854** |
>
> **CLOSED.** The conclusion of §3 holds across the **entire usable λ window**: λ = 0.5, 0.6 and 0.7
> every one reach max-over-valid **H = 1.000**, on 9 of 9 gate-passing attacks, and in all nine the
> attack *also* repaired the trap's utility damage. The finer sweep bought better as-is numbers and
> **no robustness whatsoever.**
>
> What was wrong was only my §6 expectation that the interval held nothing *usable*. The robustness
> verdict is unchanged and now rests on **three arms and nine valid attacks** instead of one arm and
> six. Meta-Unlearning, as ported, has no usable setting that survives this adversary.

## 7. Scope

Our port from the paper, not the authors' code. First-order meta gradient (§4). Single simulated
attacker step; α = inner_alpha fixed, not swept. Base arm is our FRP port rather than the paper's own
erasure stack, so the trap was installed on a different defence than the authors used. λ chosen by
us against a pre-declared primary. Any disagreement with the authors' reported results is
attributable to the port, the approximation, or the base-arm change — not to their method.
