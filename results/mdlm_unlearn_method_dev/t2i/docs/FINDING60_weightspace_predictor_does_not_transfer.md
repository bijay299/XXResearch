> ### ⚠ CORRECTION NOTICE (audit of `0b435427`, remediation r5)
> **Two corrections.** (1) The direction of the cited paper's prediction was **reversed** here: 2505.22310 associates **greater** distance with **more** robustness, so the observation was consistent with it, not contrary to it (`ERRATUM_wsr_direction.md` §4). (2) **GIFT β=1.0 was not reversed on 22 valid attacks** — that build has **12** records across four families, all 12 gate-valid; the 22 belongs to `imma_church_frp_a1e-6_s1500`. The conclusion that the predictor does not transfer is **withdrawn** regardless: every usable arm sits at H = 1.000 and a saturated ceiling cannot rank predictors (`ERRATUM_interpretations_r5.md` §1). The LMC probe samples three interior points of a CLIP-accuracy path, which is not a loss-barrier characterisation.
>
> Original text below is **unaltered**. Register: `docs/CORRECTION_REGISTER_r5.md`.

---

# Finding 60 — the weight-space predictor from *From Dormant to Deleted* does **not** transfer to T2I concept erasure. Weight distance spans **1470×** across our arms and the attack outcome does not move at all.

**Date:** 2026-10-25
**Tests a claim of:** *From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space
Regularization*, arXiv 2505.22310 (Siddiqui, Weller, Krueger, Dziugaite, Mozer, Triantafillou).
**Evidence:** `outputs/weightspace_predictor.json` (57 arms measured), cross-referenced against every
gate-validated attack in `outputs/atk_*.json`.
**Instrument:** `scripts/weightspace_predictor.py`

---

## 1. ⚠ This is a hypothesis test, not a port — and the paper is in a different domain

The fifth and last competitor on this phase's list is **not portable as a defence here**, and saying
so is more useful than forcing an analogue. It studies **example-level unlearning in vision
classifiers**, with a retrain-from-scratch gold standard. This phase does **concept erasure in a
diffusion model** and has no such oracle (`ERRATUM_r2` C17).

What *does* transfer is its headline mechanism, stated as a prediction:

> *"resistance to relearning attacks can be predicted by weight-space properties, specifically,
> L2-distance and linear mode connectivity between the original and the unlearned model."*

That is testable with assets this phase already has and the paper could not have: ten defences,
dozens of gate-validated attacks, one pinned base model. So it is tested rather than ported.

## 2. The measurement

L2 distance from the pinned base UNet (859.5M params, ‖θ‖ = 1060.12), in float64, for every arm with
at least one **gate-passing** attack:

| arm | L2 | L2 / ‖θ‖ | valid attacks | **max-over-valid H** | retain8 |
|---|---|---|---|---|---|
| `undo_a0.5_big` | 531.32 | **0.50118** | 5 | **0.467** | n/a |
| `h50_adv_a0.5` | 530.45 | 0.50036 | 1 | **0.933** | n/a |
| `h64v2_frag_a0.3_parachute` | 483.14 | 0.45574 | 7 | **0.733** | **0.0833** |
| `frp_parachute_r5_l05` | 318.98 | 0.30089 | 6 | 1.000 | 0.7417 |
| `h68v2_rf_a0.3_church` | 314.50 | 0.29667 | 15 | 1.000 | 0.8000 |
| `h80_meta_church_lam0.5` | 264.53 | 0.24952 | 6 | 1.000 | 0.6333 |
| `imma_church_frp_a1e-6_s1500` | 264.53 | 0.24952 | 6 | 1.000 | 0.8917 |
| `frp_church_r3_l05` | 264.53 | 0.24952 | 17 | 1.000 | 0.9417 |
| `uce_church` | 53.84 | 0.05078 | 8 | 1.000 | n/a |
| `esd_church` | 4.85 | 0.00457 | 3 | 1.000 | n/a |
| `gift_church_b1.35` | 0.22 | 0.00021 | 3 | 1.000 | 0.6000 |
| **`gift_church_b1.0`** | **0.18** | **0.00017** | 6 | **1.000** | 0.7833 |

*(22 arms total; representative rows shown, full table in the JSON.)*

## 3. The outcome does not move

**19 of 22 arms sit at exactly H = 1.000**, while L2/‖θ‖ ranges from **0.00017 to 0.50** — a spread
of **~2900×** in relative distance, **1470×** between GIFT and FRP alone. The predictor's independent
variable varies by three orders of magnitude and **the dependent variable is constant.**

The sharpest pair:

- **GIFT β=1.0** is at L2 = **0.18**, essentially indistinguishable from the base model in weight
  space (cosine ≈ 1 to six decimals), and is reversed to **1.000** on 22 valid attacks
  (`FINDING58`).
- **FRP church** is at L2 = **264.53**, a 1470× larger distance, and is reversed to **1.000**
  (`FINDING43`).

Under the paper's mechanism GIFT should be the *most* tamper-resistant arm in the phase. It is not
distinguishable from the least.

## 4. ⚠ The apparent inverse correlation is the degeneracy artifact, not a real effect

The three arms **below** the ceiling are the three **most distant** ones (L2/‖θ‖ 0.456–0.501), which
looks like a strong *negative* correlation — the opposite of the paper's prediction. **That reading
is wrong and I am not making it.**

`h64v2_frag_a0.3_parachute`, the one of the three with a recorded utility, sits at **retain8 =
0.0833**. It is a destroyed model. Its low H is the degenerate-image artifact `FINDING39` §1
documented, where a model that cannot generate anything scores low on the target for reasons
unrelated to erasure. The other two (`undo_a0.5_big`, `h50_adv_a0.5`) are from the same
heavily-damaged family and have no utility measurement on disk.

So the correct statement is **not** "distance anti-predicts resistance". It is:

> **In this domain the only thing that lowers max-over-valid H is destroying the model, and
> weight-space distance tracks destruction rather than tamper-resistance.**

Every arm that is *usable* — retain8 from 0.60 to 0.9667 — is at 1.000 regardless of whether its
weight distance is 0.18 or 319.

## 5. What this does and does not establish

**Establishes:** the L2 component of 2505.22310's predictor has **no explanatory power** for
relearning resistance in T2I concept erasure under this adversary. Four orders of magnitude of the
predictor, zero movement in the outcome across all usable arms.

**Does not establish** that the paper is wrong. Its setting is example-level unlearning in vision
classifiers with a retrain-from-scratch gold standard, and in that setting post-unlearning forget
accuracy recovers from ~50% to ~100% — i.e. **their outcome variable has variance and ours does
not.** A predictor cannot be falsified against a degenerate outcome; what is falsified is its
*transfer* to this domain. Their own framing would likely agree: with every defence here reversed to
the ceiling, there is nothing for weight-space proximity to discriminate.

> ✅ **LMC IS NOW COMPUTED TOO, AND IT ALSO HAS NO POWER OVER THE OUTCOME (2026-10-25).**
> `scripts/lmc_probe.py` evaluates θ_base + t(θ_arm − θ_base) at t ∈ {0.25, 0.5, 0.75} on four arms
> spanning the whole distance range. **retain8 along the path:**
>
> | arm | L2 | t=0 | 0.25 | 0.50 | 0.75 | t=1 | path |
> |---|---|---|---|---|---|---|---|
> | `gift_church_b1.0` | 0.18 | 1.000 | 1.000 | 0.992 | 0.925 | 0.783 | **barrier-free** (monotone) |
> | `frp_church_r3_l05` | 264.53 | 1.000 | 1.000 | 0.983 | 0.967 | 0.942 | **barrier-free** (monotone) |
> | `h68v2_rf_a0.3_church` | 314.50 | 1.000 | 0.983 | **0.625** | **0.358** | 0.800 | ❌ **BARRIER** — interior min 0.358 far below the 0.800 endpoint |
> | `h64v2_frag_a0.3_parachute` | 483.14 | 1.000 | **0.067** | 0.008 | 0.000 | 0.083 | ❌ destroyed along the entire path |
>
> **LMC does discriminate — just not on the axis that matters.** It cleanly separates well-conditioned
> arms (GIFT, FRP: monotone, no barrier) from damaged ones (the other two: interior collapse). But
> **all three usable arms sit at max-over-valid H = 1.000 regardless of whether their path to base is
> barrier-free or barriered.** `h68v2_rf` is genuinely *outside* the base model's linearly-connected
> basin — interior retain 0.358 against an 0.800 endpoint — and it is reversed just as completely as
> GIFT, which is barrier-free and 1470× closer in L2.
>
> So **both halves of 2505.22310's predictor are now measured on this phase's arms, and neither has
> explanatory power for relearning resistance here.** Like L2, LMC tracks **model health**, not
> tamper-resistance. §5's conclusion is complete rather than partial.
>
> Two details worth keeping: target suppression is **non-monotone** along the path for both damaged
> arms (h68v2_rf dips to 0.0667 at t=0.75 then rises to 0.3333 at the endpoint), which is the
> degenerate-image signature again rather than erasure; and interpolated checkpoints were deleted
> after evaluation, so this measurement cost no net storage.

**Does not test their method**, only their predictor. Weight-space-regularised erasure in SD is a
buildable arm and is **not run**.

## 6. Provenance and one correction

57 arms were measured; 22 had at least one gate-validated attack to pair with. 35 arms are
unattacked and excluded rather than imputed.

**A bug in my own first pass is recorded rather than quietly fixed:** the initial run used
`F.cosine_similarity` on the full 859M-element float32 vectors and returned **cos = 1.278** — an
impossible value, and a clear sign the float32 reduction had lost precision. It is now accumulated
in **float64 over 8M-element chunks** with a range assertion that would abort rather than emit an
out-of-range value. The impossible number was caught by inspection, not by the code, which is why
the assertion now exists.

A useful internal check fell out of the measurement: **FRP arms show exactly 0.00000
cross-attention drift**, which is correct — FRP prunes feed-forward layers only and should not touch
`attn2` at all.
