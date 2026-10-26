# FINDING 67 — The faithful Weight Dist Reg objective is **inert at the reference's own default α**, by a factor of **4,000×**, and the reason is structural: its RMS normalisation gives the term a gradient norm of exactly **1/√N**. Third instance of the same scale pathology.

> **⚠ CORRECTION 2026-10-26 (audit defect D-017) — the headline's MECHANISM is wrong, and the
> "third instance of one pathology" framing does not hold.**
>
> The *derivation* ‖∇L‖ = α/√N has been independently re-verified and **stands**. Three things
> around it do not:
>
> 1. **The 4,000× factor is not the 1/√N structure.** Measured as the distance step's update
>    magnitude as a fraction of `lr`: α=1 → **0.104**, α=10 → **0.537**, α=4000 → **0.998**. So
>    α 1→4000 buys **≈9.6×**, not 4,000×.
 2. **The real mechanism is an optimiser-state artifact of *our* code, not of the method.** Both
>    steps share **one Adam instance** (`scripts/erase_wdr_v2.py:200`), stepped **twice** per
>    iteration — forget at `:206-210`, retain at `:212-227`. So the second-moment buffer `v` is an
>    EMA mixing both terms and is dominated by the larger (retain) one; the distance step's update
>    is then ≈ `lr·|g_dist|/√v_retain`, roughly linear in α. Two numbers make this concrete at
>    α=1: the WDR term's **per-coordinate** RMS is α/N = **1.16e-9**, which is an order of magnitude
>    *below Adam's own ε of 1e-8*, while the retain term's is **4.67e-6**. α is therefore **inert at
>    both ends** — below ≈10 because `v` is retain-dominated (and the term is under ε), above ≈4e3
>    because Adam **saturates** once the term dominates `v`. §6's "useful window α ∈ (1, 4000)" is
>    right, but its upper bound is a *saturation point*, not a frontier.
> 3. **The reference does not share this schedule.** It **sums** both terms into one step:
>    alternation is gated on an `alternating_` name prefix that `weight_dist_reg` lacks
>    (`upstream/vision_relearning/vision_unlearner.py:411`, applied at `:637-682`). Our port's
>    docstring claim "ALTERNATING … as the reference does" is itself inverted — audit defect D-016.
>    Under the reference's summed schedule the gradient-norm framing *would* be apt. So the
>    faithfulness defect and the coefficient finding are **entangled**, and this instance is a
>    property of our two-step port rather than of "From Dormant to Deleted".
>
> The identity itself, for the record: N = 859,520,964, giving ‖∇L‖ = α/√N = **3.411e-5** at α=1.
> Re-verified on CPU — ratio to 1/√N = **1.000000** at N=1e4 and **0.999940** at N=1e6 (falling to
> 0.995109 only when RMS ≈ 1e-3 and ε begins to bite). The observed ~285× RMS change at α=4000 is
> the shared buffer, not the normalisation.
>
> **Consequences.** `wdr2_church_a4000`'s retain8 = 0.5917 measures **our two-step shared-Adam
> schedule at saturation**, not "WDR at gradient parity"; §6's "at gradient parity" attribution is
> withdrawn. And the headline's **"third instance of the same scale pathology"** (IMMA α, UCE λ,
> WDR α) no longer holds for the WDR member: the other two are coefficient-scale findings about the
> methods, this one is about our optimiser plumbing. The numbers stand; the unifying story does not.
> §7's two sub-claims are corrected below.

**Date:** 2026-10-26 · **Experiment:** E1 + `scripts/diag_wdr_scale.py`
**Evidence:** `outputs/wdr2_wdr2_church_a1.0.json`, `outputs/gen_wdr2_church_a1.0.json`,
`outputs/diag_wdr_scale.json`, `logs/e1_wdr.log`

---

## 1. What E1 appeared to show, and why that reading is wrong

`erase_wdr_v2.py` — the faithful adaptation written to repair `ERRATUM_wsr_direction.md` — was
built at the reference's default `--unlearning-alpha 1.0`
(`vision_unlearner.py:1848`), 1500 alternating steps, from the FRP teacher.

As-is: **target 0.5333, retain8 0.9500**, all-non-target 0.9455. Clears K1, not vacuous.

Taken at face value that is a usable arm. It is not a measurement of the method, because of what the
RMS trace says:

| step | 0 | 600 | 1000 | 1499 |
|---|---|---|---|---|
| RMS distance from base | 0.009024 | 0.009045 | 0.009060 | **0.009076** |

**+5.2 × 10⁻⁵ over 1500 steps**, and the 0.05 trust region never came close to binding. The FRP
teacher already sat at RMS 0.009024 at step 0, so essentially all of that distance was the teacher's,
not the objective's. **The distance-maximisation term did almost nothing.**

So `target 0.5333 / retain8 0.9500` is the FRP teacher **plus 1500 steps of effectively retain-only
training** — and note the direction: the teacher's target was **0.1333**, so retain training
*degraded* the erasure to 0.5333 while lifting retain8 from 0.9417 to 0.9500. That is a measurement
of retain fine-tuning, not of Weight Dist Reg.

## 2. The measurement, instead of another assumption

`scripts/diag_wdr_scale.py` computes both gradient norms on the real model and real retain data:

| quantity | value |
|---|---:|
| ‖∇ distance term‖ at α = 1 | **3.411 × 10⁻⁵** |
| **analytic 1/√N** (N = 859,520,964) | **3.411 × 10⁻⁵** |
| ‖∇ retain term‖ (mean of 3 draws: 0.0607, 0.0932, 0.2570) | **0.137** |
| **ratio — the α needed for parity** | **≈ 4,017** |

**α = 1.0 is roughly 4,000× too small on this model.**

## 3. Why this is structural, not a tuning accident

The agreement between the measured 3.411e-5 and the analytic 1/√N to four significant figures is
not a coincidence, and it is worth stating as algebra. For

  L = −√( mean( (θ_pre − θ)² ) ) = −‖d‖/√N ,   ‖∇_θ L‖ = ‖d‖ / (N · RMS) = ‖d‖/(N · ‖d‖/√N) = **1/√N**

The gradient norm is **exactly 1/√N regardless of how far the model has moved.** So:

- it is **constant** — the term cannot "catch up" as training proceeds;
- it **shrinks with model size**, which is the opposite of what a port needs. The reference's
  experiments are vision classifiers of roughly 11–25M parameters, where 1/√N is 2.0–3.0 × 10⁻⁴,
  i.e. **6–9× larger** than here, and their retain objective is a cross-entropy with different
  gradient scale again.

It also incidentally **confirms the implementation is correct**: the measured norm matching the
closed form to 4 s.f. is a stronger check than the six unit tests, which only verified sign,
direction and formula.

## 4. This is the third instance of one pattern, and I am naming it as a pattern

| # | hyperparameter | assumed | measured useful range | factor |
|---|---|---|---|---|
| 1 | IMMA `α` (`FINDING59` §0) | 1e-5 from the paper | destroyed the model | — |
| 2 | UCE `λ` (`ERRATUM_uce_lambda_scale.md`) | 0.1–1.0 | ~1e3, vs `cᵀc` max eigenvalue **3.242e5** (corrected 2026-10-26 from 85,850 — D-033) | **~10⁴×** |
| 3 | **WDR `α`** (here) | 1.0, the reference's default | **~4.0e3** | **~4 × 10³×** |

Each time the fix was one cheap measurement of the quantity the hyperparameter is compared against,
and each time I ran the sweep first. For UCE I wrote *"measure the scale before sweeping the knob"*
as the lesson and then, on the very next new objective, took a paper's default unexamined.

The general statement, which is the part worth keeping: **a regulariser normalised by parameter
count does not transfer across model scales at a fixed coefficient.** Any port of such a term
between a classifier and a diffusion UNet must re-derive the coefficient from the gradient-norm
ratio, not inherit it. `erase_wdr_v2.py`'s `--alpha` help text now carries this, as
`erase_uce.py`'s `--lam` does.

## 5. Status of the E1 arm and of the method

- `wdr2_church_a1.0` is **kept and relabelled** a *retain-fine-tuning control*, not a WDR arm. It is
  a genuinely useful control — it shows 1500 retain steps on the FRP teacher cost 0.40 of target
  suppression — and it is recorded as that.
- **'From Dormant to Deleted' remains UNTESTED in this project.** `ERRATUM_wsr_direction.md` said
  that pending E1; E1 has run and the answer is still untested, because the objective was inert.
  The claim that our experiments discharged this baseline stays **withdrawn**.
- **E1b**, at the measured α ≈ 4.0e3, is running: one arm, `--max-rms 0.03` (≈3× the teacher's own
  0.009, so the trust region can actually bind before the model is destroyed). The honest prior is
  that maximising an unbounded distance at a now-effective coefficient will damage retention — that
  is what `FINDING61` §4 found for the unbounded case — and the trust-region stop exists to make the
  failure legible rather than catastrophic.

**One arm, one α, one concept, one seed.** The α sweep the correction register prices at 6 GPU-h
stays blocked by disk (5.1 GiB free), so E1b will produce a single point, not a curve. That limits
what it can conclude and is stated here rather than discovered later.

## 6. Addendum (E1b) — at the measured α the objective works as designed, and **'From Dormant to Deleted' now has its first real measurement here: a utility failure, not a robustness result**

E1b ran the same arm at the **measured** α = 4000 instead of the reference's inherited 1.0.
`--max-rms 0.03`; the trust region **did not fire**, so all 1500 steps ran.

**The objective became active, exactly as §3 predicted it would at parity:**

| step | 0 | 700 | 1100 | 1499 |
|---|---|---|---|---|
| RMS distance from base | 0.009024 | 0.014211 | 0.019048 | **0.023839** |
| forget term (α × −RMS) | — | −56.80 | −76.15 | −95.31 |

RMS reached **2.6× the teacher's own distance**, against +5.2 × 10⁻⁵ of movement at α = 1. So the
α correction was the whole difference between an inert term and a working one.

**The outcome, placed against the teacher it started from:**

| arm | target | retain8 | all-non-target | verdict |
|---|---:|---:|---:|---|
| `frp_church_r3_l05` (the teacher) | 0.1333 | 0.9417 | — | — |
| `wdr2_church_a1.0` (α inert) | 0.5333 | 0.9500 | 0.9455 | retain-fine-tuning control |
| **`wdr2_church_a4000`** (α effective) | **0.2000** | **0.5917** | **0.5333** | **K1 FAIL — not attacked** |

So the faithful mechanism, at an effective coefficient, **cost 0.35 of retention (0.9417 → 0.5917)
while not improving on the teacher's erasure** (0.2000 vs 0.1333). It falls below the K1 utility
floor of 0.60 and was therefore **not attacked** — correctly, since a robustness number from a
model this damaged would not be interpretable.

**This is the first genuine measurement of 2505.22310's mechanism in this project**, and
`ERRATUM_wsr_direction.md`'s "untested" status is now one point of evidence rather than zero. The
honest statement of it:

> Ported to a 859.5M-parameter diffusion UNet at gradient parity, Weight Dist Reg's distance-
> maximisation drives the model 2.6× further from base and costs more retention than it buys
> erasure, failing the utility floor at the one α tested.

**It is one point, and the α that matters is bracketed but not resolved.** retain8 = **0.5917 is
0.0083 below the 0.60 threshold** — close enough that a modestly smaller α plausibly clears it. The
useful window is now bracketed from both sides by measurement, exactly as the UCE λ case was:

| α | distance term | retain8 | reading |
|---:|---|---:|---|
| 1 | inert (+5.2e-5 RMS) | 0.9500 | not the method |
| **4000** | active (2.6× RMS) | **0.5917** | method active, utility floor missed by 0.008 |

**α ∈ (1, 4000) is unmeasured, and that interval is where a usable arm would be if one exists.**
That is the α sweep the correction register prices at ~6 GPU-h and that disk has blocked throughout;
a single intermediate point (α ≈ 1000) would likely settle it. Until then the method's standing here
is **"one α tested, utility floor missed narrowly"** — not refuted, and certainly not discharged.

Also worth recording: the trust region not firing at `--max-rms 0.03` means the predicted
catastrophic unbounded ascent (`FINDING61` §4) **did not occur** at α = 4000 in 1500 steps. The
damage was gradual retention loss, not collapse. That is a more informative failure than the
earlier unbounded-objective nulls.

## 7. Addendum — I checked the next two experiments' coefficients *before* spending disk on them. The pattern does **not** recur for Meta-v2; UTRAP-v2's λ is ~8× below parity.

§4 named a three-instance pattern and drew the lesson "measure the scale before sweeping the knob".
The two experiments queued next both inherit coefficients the same way §4 warns about — E2's `zeta`
and `lam_target` default to 1.0 from the paper's shared notation, and E4's `lambda` defaults to 0.15
inherited from the **old** UTRAP runs, whose inner attacker was a different and much weaker process
(`ERRATUM_utrap_inner_attacker.md`), so there is no reason it transfers to the corrected attacker.

So this time the measurement came first (`scripts/diag_coeff_scale.py`, JSON only, no checkpoint —
it was also the only useful GPU work available at 3.5 GiB free).

### UTRAP-v2: λ = 0.15 is ~8× below parity

| quantity (full 859.5M scope) | value |
|---|---:|
| ‖∇ L_retain‖ | **0.1856** |
| ‖∇ L_target‖ | **0.1503** |
| **λ for parity** | **1.23** |
| inherited default | **0.15** |

The two terms are intrinsically the same order — so unlike WDR, λ=0.15 is not inert, it is simply
**~8× weaker than the retain term** rather than comparable to it. `PREREG_H90`'s arms are specified
at λ=0.15 for continuity with the historical sweep, and E4 should carry a λ≈1.2 arm alongside it.
Recorded here so that is a declared design decision rather than a later rationalisation.

> **CORRECTION 2026-10-26 (audit defect D-017).** The sentence that stood here said λ=0.15
> "systematically under-weights the adversarial term". **Withdrawn** — it uses parity as a normative
> reference point, which nothing establishes.
>
> The nuance matters and cuts partly the other way from the WDR case above. For UTRAP-v2 the two
> gradients **are** summed into one `p.grad` before a single Adam step
> (`scripts/erase_utrap_v2.py:265-274`), so unlike the WDR port the **ratio does** set the summed
> direction — parity is a defensible *directional* heuristic, putting the resultant ≈45° between the
> two terms. But Adam normalises the overall scale away regardless, and no measurement here shows
> 45° is the best direction. So the supportable wording is: **λ=0.15 weights the adversarial term
> ~8× below the retain term**, with no implied optimum, and λ=0.15 is not inert. Carrying a λ≈1.2
> arm alongside remains the right design choice — as a **comparison**, not as a correction toward a
> target. `PREREG_H90` §: the phrase "the λ under-weighting is measured" inherits the withdrawn
> framing and is corrected there.

### Meta-v2: the pattern does **not** recur, and that is worth stating

| quantity (xattn scope, 44.0M) | value |
|---|---:|
| ‖∇ L_retain(θ)‖ | **0.0387** |
| ‖∇ L_target(θ′)‖ | **0.0927** |
| ratio | **2.4×** |

The two Eq. 8 terms measured above are **within 2.4× of each other**.

> **CORRECTION 2026-10-26 (audit defect D-014/D-017). "E2 needs no coefficient correction" is
> WITHDRAWN — the diagnostic compared the wrong pair.** This table treats ‖∇L_retain(θ)‖ and
> ‖∇L_target(θ′)‖ as two independent terms that ζ and `lam_target` separately scale. They are not.
> Verified at `scripts/meta_unlearn_v2.py:288-306`: a **single shared ζ** multiplies `+L_R(θ)` **and**
> `−L_R(θ′)`, so the quantity ζ actually weights is the **difference** L_R(θ) − L_R(θ′), not L_R(θ)
> alone. With θ′ ≈ θ − α∇L_F that difference is ≈ α·∇L_R·∇L_F — second order, and far smaller than
> either norm in the table. **ζ is therefore near-inert, which is the opposite of the conclusion
> drawn here**, and no ζ sweep has been run to test it empirically.
>
> So the corrected tally is: four coefficient scales checked, and the Meta-v2 "clean" case was a
> measurement artifact — it only looked clean because the pairing was not modelled. The lesson of §4
> ("measure the scale before sweeping the knob") survives, but with an addition: **measure the
> quantity the coefficient actually multiplies**, which for a paired meta-objective is a difference,
> not a level. `FINDING78` §1a carries the consequence: that document's retention gain comes from
> restoring the *pairing* (which cancels variant A's retain-level penalty), not from ζ's weight.
>
> A second defect in the same code compounds this: `L_R(θ)` and `L_R(θ′)` are drawn with
> **independent noise and timesteps**, so the paired difference is unpaired and the logged
> `retain_change` is dominated by sampling noise. The fix is common random numbers, not a larger ζ.

### One measurement came back exactly 0.0, and that is structural, not a failure

‖∇ distil_erase(θ)‖ and ‖∇ distil_retain(θ)‖ both measured **exactly 0.000000**. The reason is by
design in both `meta_unlearn.py` and `meta_unlearn_v2.py`: the student is initialised **at** the
teacher (`teacher = deepcopy(unet)` after the teacher checkpoint is loaded), so at step 0 the
distillation losses are identically zero and so are their gradients. They become non-zero only as
the student drifts.

Two consequences, stated rather than glossed:
- the parity figures *relative to the distillation terms* in `outputs/diag_coeff_scale.json` are
  **0.0 and meaningless**, and are not used for anything;
- the distillation anchors' **effective weight is dynamic** — zero at initialisation, growing with
  drift — so no static gradient ratio can set the coefficients against them. That is a real limit on
  this kind of diagnostic and it applies to any objective that anchors a student to its own
  initialisation.

### Precision of these numbers

n = 3 draws each, and the spread is large: ‖∇L_retain‖ over the full scope came out 0.146, 0.272,
0.139 — a 2× range. So these ratios locate an order of magnitude, not a value. That is sufficient
for the question being asked (is a coefficient orders of magnitude from parity?) and insufficient
for tuning.
