# PREREG S1b — the three-term objective with a variance-correct estimator

**Declared 2026-10-27, BEFORE execution.** Supersedes `PREREG_S1_META_FULL_OBJECTIVE.md` **for the
trap question only**; S1's "does the trap fire on normal use?" half stands on its own.
Development data; no confirmatory claim. Frozen on first result; corrections in a dated amendment.

## 1. Why

`PREREG_S1_AMENDMENT_1_2026-10-27.md` §6 establishes that `scripts/meta_unlearn_v2.py` estimates
the objective's difference term `[L_R(theta') - L_R(theta)]` by evaluating its two members on
**different random draws**, while the parameter change between them is only
`alpha * grad L_F(theta)` with `alpha = 1e-5`. The draw difference dominates by orders of
magnitude, so S1's meta gradient is noise with a weak drift, and **a null result from S1's attack
phase cannot distinguish "no trap" from "estimator too noisy to express a trap."**

`scripts/meta_unlearn_v3.py` (`config_version meta_v3`) applies two corrections, both taken from
the released upstream:

1. **Common random numbers** — noise and timestep drawn once per pair, shared by both members of
   every difference.
2. **Fixed meta-term timesteps, summed** over {1, 2, 5, 10}, following
   `upstream/Meta-Unlearning/src/meta.py:595` and the shipped `scripts/meta.sh`
   (`--fix_timesteps True --fixed_time_steps 1 2 5 10`). Only the inner attack step keeps a random
   timestep, as upstream does (`train_set=True`).

## 2. Stage 0 — a pilot that can falsify the §6 diagnosis before any build is paid for

**This runs first and is cheap.** If the diagnosis in §6 is right, sharing the draw must change the
*character* of the logged `retain_gap`, not merely its size.

Cost, corrected from an initial "~4 min" estimate: each variant is ~2.5 min of pool generation (128
images at 25 inference steps, unavoidable and identical to a full build) plus 100 steps at ~2.2 s
(10 backward passes per step with `k = 4` fixed timesteps), so **~6 min per variant, ~12 min for
both**. The generation cost dominates a short run, which the original estimate overlooked.

| quantity | v2, measured | v3 prediction if §6 is correct | v3 result falsifying §6 |
|---|---|---|---|
| `retain_gap` across steps | swings **-0.402, +0.027, +0.179, +0.346, -0.195**, no trend, magnitude ~0.2-0.4 | **small** (orders of magnitude below 0.1) and **consistently signed**, because with a shared draw the gap can only reflect the parameter change | still swings by ~0.1 or more with sign flipping |

A falsifying outcome means the gap is not draw-driven, §6's diagnosis is wrong, and I must withdraw
it and re-examine why v2's model barely moved before spending any further GPU time.

### 2.1 The corrected gradient will be SMALLER in norm, not larger — declared so the pilot is not misread

With a shared draw `xi`, the difference becomes

```
gR0 - gR1 = grad L_R(theta; xi) - grad L_R(theta'; xi)
          ~= - Hess L_R(theta; xi) . (theta' - theta)
          =  + alpha * Hess L_R(theta; xi) . grad L_F(theta)
```

a Hessian-vector product of magnitude **O(alpha) = O(1e-5)**. With independent draws the same
expression is `O(||grad L_R||)` — far larger, but randomly signed, since the `xi`-dependence no
longer cancels.

**So the expected pilot signature is `meta_grad_norm` orders of magnitude SMALLER than v2's
effective gradient, together with a consistently-signed `retain_gap`.** A smaller norm is
confirmation, not failure.

Why a smaller gradient is nonetheless more effective: the optimiser is **Adam**, which normalises
per-parameter by the running gradient magnitude. A small, consistently-signed gradient yields steps
of order the learning rate; a large, randomly-signed one yields steps that cancel across iterations.
This is also why v2's model drifted only 1.5e-4 in distillation distance despite a nominally large
meta gradient.

An earlier draft of §4 below said the corrected term "becomes effective" in a way that implied a
larger gradient. **That implication is withdrawn**: the mechanism is consistency of direction under
Adam's normalisation, not magnitude.

Run as: `meta_unlearn_v3.py --steps 100 --zeta 0.8 --gamma-f 0.0 --label s1b_pilot`, and once more
with `--fixed-timesteps ""` to separate correction 1 from correction 2. Also records
`meta_grad_norm` per step, which v2 did not log, giving a magnitude baseline for S2 if we get there.

## 2.0 PILOT RESULT — diagnosis SUPPORTED, and the epsilon remedy TRIGGERED

**Measured 2026-10-27, `fixed` variant, from `logs/s1b_meta_v3.log`:**

| quantity | v2 (independent draws) | **v3 (shared draws + fixed timesteps)** |
|---|---:|---:|
| `L_R(theta)` vs `L_R(theta')`, step 0 | — | 0.86475 vs 0.86475 |
| `retain_gap`, steps 0 and 10 | mean abs **0.190**, max 0.474 | **+0.000001**, **+0.000001** |
| sign-consistency | **0.500** (8+ / 8-) | positive on every logged step |

The gap fell by **roughly five orders of magnitude**, from 0.190 to ~1e-6. On the magnitude clause
the §2 criterion is met overwhelmingly (threshold `max|gap| < 0.01`), so **the §6 diagnosis is
confirmed and not falsified: v2's ±0.19 gap was draw noise, and v2's difference term was never
measured.** The falsifying outcome (gap still ~0.1+) did not occur.

### 2.0.0 AMENDMENT: the sign clause of my own criterion is inapplicable, and I did not anticipate that

§2 also required **sign-consistency >= 0.90**. Later logged steps show the gap flipping sign at the
1e-6 level: **+0.000001, -0.000002, +0.000001, -0.000000** at steps 60/70/80/90. Taken literally
that would read INCONCLUSIVE. It should not be, and the reason is numerical, not scientific:

| quantity | value |
|---|---:|
| fp32 ULP at a loss of ~0.85 | 5.96e-8 |
| accumulated MSE rounding over ~16k elements, ~sqrt(16384) x ULP | **~7.6e-6** |
| **measured \|gap\|** | **~1e-6** |

**The gap has collapsed below the floating-point resolution of the loss computation itself**, so its
sign carries no information. My criterion assumed the gap would shrink to something small but still
resolvable; it shrank past that. The sign clause is therefore **withdrawn as inapplicable at this
magnitude** rather than read as a failure — and this is recorded as a flaw in the criterion I wrote,
not as a convenient reinterpretation of the data.

**What replaces it.** The quantity that actually drives learning is the *gradient* difference, not
the loss difference, and it is well-resolved: `meta_grad_norm` reads **6.077e-4, 6.272e-4, 7.205e-4,
7.415e-4, 7.542e-4, 8.125e-4** across the logged steps — stable within ~30% and four orders of
magnitude above any precision floor. This is consistent with the first-order picture: the loss
difference is `O(alpha * ||grad L_R|| * ||grad L_F||)` while the gradient difference is
`O(alpha * ||H|| * ||grad L_F||)`, so the gradient signal can be relatively much larger than the
loss signal. **The mechanism has a usable gradient even though its loss gap is unresolvable in
fp32** — which also means the loss gap must not be used to monitor it.

### 2.0.05 Isolation: both corrections were needed, for different reasons

The two pilot variants separate correction 1 (common random numbers) from correction 2 (upstream's
fixed meta-term timesteps). 11 logged steps each:

| variant | mean abs gap | mean `meta_grad_norm` | median norm | per-component vs `eps` = 1e-8 |
|---|---:|---:|---:|---:|
| v2, independent draws | **0.190** | not logged | — | — |
| `randpair` — shared draws, random timestep | 4.522e-07 | 1.437e-04 | 8.926e-05 | 3.0e-9 -> ratio **0.30** |
| `fixed` — shared draws + timesteps {1,2,5,10} | 1.307e-06 | **6.996e-04** | 7.205e-04 | 2.46e-8 -> ratio **2.46** |

**Correction 1 does the noise removal.** Sharing the draw alone collapses the gap from 0.190 to
4.5e-7, so the draw-noise diagnosis of §6 is attributable to that change specifically.

**Correction 2 does something different and also necessary: it raises the usable gradient by ~5x**
(mean norm 7.0e-4 against 1.4e-4; median 7.2e-4 against 8.9e-5). That is what lifts the typical
per-component magnitude from **below** Adam's numerical floor (ratio 0.30 — eps would dominate) to
**above** it (ratio 2.46 — eps throttles but the signal survives). So upstream's `fix_timesteps`
is not merely a fidelity detail we had omitted; **it is what makes the meta gradient large enough
for the optimiser to act on at all.** Evaluating at four fixed low timesteps and summing concentrates
the signal where the diffusion loss is stable, instead of scattering it across a range where the
variance buries it.

Both corrections are therefore kept for the builds, which is also the configuration the authors ship.

### 2.0.1 The epsilon remedy fired, as §2.2 anticipated

Measured `meta_grad_norm` is **7.5e-4 to 8.1e-4**. Over 859,520,964 parameters that is

```
8.1e-4 / sqrt(859520964) = 8.1e-4 / 29317.6 ~= 2.8e-8
```

against Adam's default `eps = 1e-8` — a **ratio of ~2.8**, inside §2.2's declared band "within ~10x
of 1e-8 -> epsilon materially throttles the step -> re-run with `eps = 1e-12`". The remedy is
therefore applied, not debated: `scripts/meta_unlearn_v3.py` gains `--adam-eps` (default unchanged
at 1e-8 so nothing is altered silently), the builds pass **`--adam-eps 1e-12`**, and the value is
recorded in each run's `hparams`.

Note that §2.2's own magnitude estimate was **too pessimistic**: it predicted a norm of O(1e-5) and
a per-component value of ~3.4e-10, which would have put eps in total command. The measured norm is
~80x larger, so eps throttles rather than dominates — the gradient is not as small as the
Hessian-vector-product scaling argument suggested, presumably because the mass concentrates in a
subset of layers as §2.2 allowed.

### 2.0.2 Cost, measured

1.9 s/step marginal for the `fixed` variant (10 backward passes), so a 1500-step build is
**~48 min**, against the ~55 min projected in §6. Two builds plus attacks remain ~3 GPU-hours.

## 2.2 A SECOND confound that could cause another uninformative null: Adam's epsilon

Declared before any v3 run, because if it bites it would produce a flat result for a reason that has
nothing to do with the objective — the same trap S1 fell into, wearing different clothes.

§2.1 establishes that the corrected meta gradient is a Hessian-vector product of magnitude
`O(alpha) = O(1e-5)` in **norm**. Spread over 859,520,964 parameters, the typical per-component
magnitude is of order

```
1e-5 / sqrt(8.6e8)  ~=  3.4e-10
```

Adam's update is `lr * m_hat / (sqrt(v_hat) + eps)` with **`eps = 1e-8` by default**. If per-component
gradients really are ~3.4e-10, then `eps` **dominates the denominator** and the update collapses to

```
lr * 3.4e-10 / 1e-8  ~=  0.034 * lr
```

a roughly **30x throttling** of every step. The model would again barely move, and a naive reading
would again blame the objective.

**This is a prediction, not an established fact.** It assumes the gradient is spread evenly, whereas
real gradients concentrate in a subset of layers, which would raise the per-component magnitude where
it matters and weaken the effect. The point is that it is **measurable**, and `meta_grad_norm` is
logged every step precisely so it can be checked.

**Declared check and remedy.** From the pilot, compute `meta_grad_norm / sqrt(n_params)` and compare
it to `eps`:

| measured typical component | reading | action |
|---|---|---|
| >> 1e-8 | epsilon is irrelevant | proceed unchanged |
| within ~10x of 1e-8 | epsilon materially throttles the step | **re-run with `eps = 1e-12`**, recorded as a declared configuration change, before building |
| << 1e-8 | epsilon dominates completely | same remedy, and report that the default optimiser setting was the binding constraint |

Lowering `eps` is the principled fix rather than raising `lr` or `zeta`, because it changes only the
optimiser's numerical floor and leaves the objective and its weights comparable to v1, v2 and S1.
`meta_unlearn_v3.py` currently uses Adam's default; if the remedy is triggered, an `--adam-eps`
argument will be added and the change recorded in an amendment **before** the build runs.

| arm | zeta | gamma_f | purpose |
|---|---:|---:|---|
| `s1b_metaV3_z0.8_g0.0` | 0.8 | 0.0 | the difference term alone, with a working estimator |
| `s1b_metaV3_z0.8_g1.0` | 0.8 | 1.0 | full objective |

**The no-op control is NOT rebuilt.** Amendment §1.2 proved `s1_metaV2_z0.0_g0.0` byte-identical to
its teacher (same sha256, max absolute parameter delta 0 over 859,520,964 parameters). A v3 no-op
arm would be the same file again; its 30 min buys nothing. The control for the §5.1 paired
comparison remains the **teacher's own attack cells**, tabulated in amendment §1.4.

Teacher `adapters/frp_church_r3_l05/unet.pt`, concept church, **1500 steps, lr 1e-6, seed 0,
inner-alpha 1e-5** — identical to S1 and to v1, so the estimator is the only changed factor.

## 3.5 S1b BUILD RESULT: the estimator fix was necessary but NOT sufficient

`s1b_metaV3_z0.8_g0.0` completed 1500 steps with the corrected estimator and `--adam-eps 1e-12`.
Parameter delta against its teacher:

| configuration | mean abs drift | vs random walk (3.87e-5) |
|---|---:|---:|
| coherent motion, 1500 x lr | 1.5e-3 | — |
| v2 arm 2 — **broken** estimator, eps 1e-8 | 2.27e-5 | 0.59x |
| **S1b — corrected estimator, eps 1e-12** | **2.046e-5** | **0.53x** |

**Fixing the estimator and lowering eps by four orders of magnitude produced no more motion than the
broken version.** `meta_grad_norm` held at 5.9e-4 to 8.3e-4 across all 1500 steps, so the objective
pushed continuously and with consistent magnitude — but its **direction** varies between steps, and
1500 Adam steps cancelled to a random walk.

Its panel, for completeness: `retain8` **0.9833**, `target_suppression_acc` **0.3333**,
`utility_all_nontarget` 0.9515, against the teacher's 0.9417 / 0.1333 / 0.9212. Both gates pass, but
the target accuracy *rose* by 3/15 — the arm partly **un-erased** church — and at a drift of 2.05e-5
that is close to the documented 2/15 H noise. **Per §2 this panel is not readable as a test of the
objective**, exactly as S1's was not.

### 3.5.1 The cause is a THIRD divergence from upstream: batch size

| | upstream `scripts/meta.sh` | v1, v2, S1b |
|---|---|---|
| `--train_batch_size` | **10** | **1** (one `randint` index per step) |
| `--num_images_per_prompt` | 10 | — |
| `--num_train_steps` | 100 | 1500 |
| total samples | ~1000 | ~1500 |

Comparable sample budgets; **very different per-step variance**. At batch 1 the meta gradient is
`-alpha * H(theta; xi) . grad L_F(theta; xi')` for a *single* benign image and noise draw, so its
direction is redrawn every step. Averaging over ~10 images cuts that variance by ~sqrt(10) with the
expectation unchanged. This is the third thing our port omitted that upstream does — after unpaired
draws and random timesteps — and each is independently sufficient to stop the mechanism expressing
itself.

### 3.5.2 STOPPING RULE, declared before the next run

Three instrumentation defects in this stage, each costing GPU without testing the hypothesis.
**One further attempt only**, then the direction is closed as untestable within our resources.

**S1c** — same objective and estimator, reconfigured toward upstream's regime: meta gradient
accumulated over **8 benign/forget images per step** by micro-batching rather than a true batch (v3
already peaks at 32.4 GiB of 40 GiB at batch 1, so a true batch of 8-10 would very likely OOM), at
**190 steps**, keeping the total sample budget (~1520) within 2% of v1/v2/S1b's 1500 and the
wall-clock near one build. `--adam-eps 1e-12` and timesteps {1,2,5,10} retained.

| S1c mean abs drift | reading | action |
|---|---|---|
| **>= 1.4e-4** (~10x the random walk) | the objective has purchase; between-step variance was the obstruction | read the panel, apply the §5.1 criterion |
| **<= 3e-5** (~2x the random walk) | three successive corrections have failed to make this mechanism move a model at our scale | **STOP.** Report the direction untestable within our resources, with the three omissions documented. No fourth fix |
| between | partial; report drift alongside the panel, treat the panel as provisional | no further builds either way |

> **THRESHOLD CORRECTION, made 2026-10-27 BEFORE any S1c drift was measured.** The bands first
> declared here were **2e-4 / 6e-5**, computed for a 1500-step run and never rescaled to S1c's 190
> steps. At 190 steps and lr 1e-6:
>
> | | 1500 steps | **190 steps** |
> |---|---:|---:|
> | coherent motion (steps x lr) | 1.5e-3 | **1.9e-4** |
> | pure random walk (sqrt(steps) x lr) | 3.87e-5 | **1.38e-5** |
>
> So the original purchase gate of **2e-4 exceeded the fully-coherent value of 1.9e-4** — a perfectly
> coherent run would have failed my own gate, and the direction would have closed on an arithmetic
> error rather than on evidence. The stop band of 6e-5 was likewise 4.3x the random walk, far too
> permissive to mean "random-walk level".
>
> Corrected to **10x random walk for purchase (1.4e-4)** and **2x for stop (3e-5)**, and applied to
> `run_s1c_meta_v4.sh` while the build was at step ~10, with **no drift figure in existence**. This
> is the repair of a scale error in my own thresholds, not an adjustment fitted to a result.

**I will not propose a fourth instrumentation fix.** If S1c lands in the stop band, the honest
conclusion is that this trap mechanism cannot be evaluated here at ~1500-sample scale with a
first-order estimator — itself a reportable finding about the method's practicality — and MANDATE v2
§1 is not met by a mechanism we cannot demonstrate working.

## 3.6 S1c RESULT: the stopping rule fired. The direction is CLOSED.

`s1c_metaV4_z0.8_g0.0` completed all 190 steps with the corrected estimator, upstream's fixed
timesteps, `--adam-eps 1e-12` and an **8-sample micro-batched** meta gradient. Parameter delta
against its teacher (`scripts/param_delta.py`, 686 tensors, 859,520,964 parameters):

| | value |
|---|---:|
| max abs delta | 1.2207e-4 (`down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight`) |
| **mean abs delta** | **6.1045e-06** |

Against the bands declared in §3.5.2 and rescaled in the threshold correction above, **before any
drift existed**:

| reference (190 steps, lr 1e-6) | value |
|---|---:|
| coherent motion | 1.9e-4 |
| pure random walk | 1.38e-5 |
| purchase band | >= 1.4e-4 |
| **stop band** | **<= 3e-5** |
| **measured** | **6.10e-6** |

**6.10e-6 lands in the stop band, and is below even the random-walk figure (0.44x).** The model
moved *less* than random stepping would produce, which is the signature of a near-zero-mean
gradient under Adam: when per-step signs flip, `m_hat / sqrt(v_hat)` collapses and the effective
step falls well below the learning rate. It is consistent with the §2.0.05 observation that the
per-sample meta gradients scale as 1/sqrt(B) — i.e. they are nearly uncorrelated.

### 3.6.1 Verdict, per the pre-registered rule

**STOP.** No `gen_eval` panel, no attack cells, no fourth instrumentation fix. The trap direction is
**closed as untestable within our resources.**

What was established across the three attempts, all by measurement:

| attempt | instrument | mean drift | outcome |
|---|---|---:|---|
| S1 (`meta_v2`) | difference estimated on **independent draws** | 2.27e-5 | gradient was pure draw noise; the no-op arm's gap was identical to the active arm's |
| S1b (`meta_v3`) | common random numbers + upstream's fixed timesteps, eps 1e-12 | 2.05e-5 | estimator verified correct (gap 0.190 -> 3e-6) but direction resampled per step |
| **S1c (`meta_v4`)** | **+ 8-sample micro-batched meta gradient** | **6.10e-6** | **averaging reduces the norm by sqrt(8), confirming near-uncorrelated samples. Below random walk** |

**Three divergences from the released upstream were found and each corrected**: unpaired draws,
random meta-term timesteps, and batch size 1 against upstream's 10. Correcting all three did not
make the mechanism move a model at this scale.

### 3.6.2 What this does and does not license saying

**It does licence:** *we could not make this trap mechanism displace a 859M-parameter UNet at a
~1500-sample budget with a first-order estimator, having corrected every configuration difference we
found against the authors' released code.* That is a statement about practicality at our scale, and
it is reportable.

**It does not licence** any claim that the published method fails, or that traps cannot work.
Amendment §4 stands: our port has no `hrm` pool, carries two weights where upstream has five, and
simulates a different attacker. The authors train at batch 10 for 100 steps on three curated pools
with five tuned weights; we trained at an effective batch 8 for 190 steps on two generated pools
with two weights. A null from our instrument is not evidence about theirs.

**Cost of the direction:** S1 ~1.5 GPU-h, S1b ~1.6 GPU-h (two builds plus attacks), S1c ~0.85 GPU-h
— roughly **4 GPU-hours**, of which the S1 portion is waste attributable to my estimator defect and
the rest bought three measured negative results and the three documented fidelity gaps.

## 4. A prediction that could go either way, declared now

With the variance corrected the meta term becomes *effective* for the first time. Two outcomes are
both plausible and both informative:

- **It behaves like v1 and collapses utility.** v1's one-sided term at this zeta gave retain8
  0.4750. If the corrected difference term does the same, then the difference form does not after
  all protect benign capability at zeta = 0.8, and the fix to the objective was not the fix to the
  problem. Gate 1 (`retain8 >= 0.85`) catches this and retires the class per the mandate.
- **It preserves utility and produces a measurable trap.** H1 supported, go to S2.

**Do not read a collapse as a failure of the estimator fix.** A collapse would mean the estimator
fix worked and the *objective* is what fails at this zeta. If that happens, the single follow-up I
would run is zeta = 0.1, matching the shipped upstream script, before retiring the class.

## 5. Decision rule — unchanged from amendment §5.1, restated

An arm supports H1 only if all four hold:

1. un-attacked `legacy_retain8 >= 0.85`;
2. un-attacked `target_suppression_acc <= 0.40`;
3. `trap_effect = control_drop - arm_drop >= 0.25`, measured only where the **control's** cell is
   gate-valid and non-destroyed;
4. condition 3 holds at **two or more** attacker learning rates.

Given the measured control (amendment §1.4) this means an absolute post-attack retain drop of
**>= 0.25 at both 1e-5 and 5e-5** — the only two rates where the control has a clean cell at all
three seeds. Mechanised in `scripts/s1_verdict.py`, which is deterministic and was written before
any of these results existed.

Primary robustness quantity: **max-over-valid `H_after`**, per MANDATE clause 7. Not `delta_H`.

## 6. Cost, and the budget problem stated plainly

v3 does more work per step than v2: `2 + 2k` backward passes for `gamma_f = 0` and `2 + 3k` for
`gamma_f = 1`, where `k = 4` fixed timesteps — so 10 and 14, against v2's 5. At v2's measured
1.11 s/step that projects to roughly **2.2 and 3.1 s/step**, i.e. **~55 and ~78 min per build**, so
~2.2 GPU-h for the two builds plus up to ~54 min of attack cells: **~3.1 GPU-h**.

S1's declared allowance was 3 GPU-h and ~1.2 h of it is already spent. **S1b therefore needs its own
allowance and I am recording that rather than absorbing it silently.**

**Stated plainly, because the accounting should not be laundered:** S1's ~1.5 GPU-h (three builds,
three panels, and whatever attack cells its runner completes) was spent on an instrument that
cannot answer the question it was built for, because of a defect I introduced in
`meta_unlearn_v2.py`. That spend is **waste attributable to me**, not a stage of the plan. S1b is
the real S1. Two parts of S1's spend do retain value and are not waste: the no-op control, which
produced the byte-exact proof that the port has no erasure objective of its own, and arm 2's
`gen_eval` panel, which answers "does the trap fire on normal use?" independently of the estimator. If the total is to stay
inside the mandate's staged budgets, the honest options are: run only the `g0.0` arm first
(~55 min + attacks), or reduce `k` to {1, 10} (halving the meta cost at some fidelity loss, which
`--fixed-timesteps` makes explicit and recorded). **Default: run `g0.0` alone first**, since it
isolates the difference term, and only build `g1.0` if `g0.0` clears gate 1.

## 7. What S1b cannot establish

- **Not the published method's performance.** Amendment §4 stands: upstream carries three data pools
  and five weights where we carry two and two, and our inner attack step trains on the forget pool
  where upstream's trains on `hrm_train`. Our simulated attacker is not the paper's.
- n=1 build per arm. Five claims in this campaign died when a second build was added.
- One concept, one teacher, one attack budget, one attack family.
- A trap that worked would still need a test of whether it makes the model fragile to *any*
  fine-tuning, which is the obvious failure mode and is **not** tested here.
