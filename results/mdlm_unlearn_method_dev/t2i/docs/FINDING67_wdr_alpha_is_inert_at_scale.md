# FINDING 67 — The faithful Weight Dist Reg objective is **inert at the reference's own default α**, by a factor of **4,000×**, and the reason is structural: its RMS normalisation gives the term a gradient norm of exactly **1/√N**. Third instance of the same scale pathology.

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
| 2 | UCE `λ` (`ERRATUM_uce_lambda_scale.md`) | 0.1–1.0 | ~1e3, vs `cᵀc` max eigenvalue 85,850 | **~10⁴×** |
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
