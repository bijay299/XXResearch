# Where the T2I unlearning project stands, and the decision I need

**For: Dr. Danda B. Rawat and the research group. Prepared 2026-10-27 by Bijay Pandey.**
Mandate item 7. Every number here traces to a stored record; the supporting documents are named
inline. Nothing in this memo is a confirmatory result — the project has run on development data
throughout, and no confirmatory protocol has been frozen.

---

## 1. The bar you set, restated

> A method may use more computation if it delivers meaningfully better results, or achieve
> comparable results with substantially less computation or complexity. A useful, explicitly
> measured trade-off is acceptable. **Similar performance at similar or greater cost does not
> satisfy this mandate.**

Objective: suppress a target concept, preserve useful generation, make the suppression resistant
to subsequent fine-tuning.

## 2. The honest headline

**We have not built a defence that meets your bar, and the evidence says the obvious design space
does not contain one.** What we do have is a well-supported negative result and a set of measurement
findings that bear directly on how such claims are evaluated.

Earlier today five arms out of 86 were unresolved and one looked like it might be a counter-example.
**All five are now resolved and none is** (§3.1): four had lost most of their retained capability and
the fifth never erased its target. One measurement in the course of resolving them does not fit our
explanation of *why* defences fail, and I flag it in §3.1 rather than leaving it in an appendix.

I am not asking you to accept a negative result as a contribution by default. Section 6 lays out
three options with their costs and my recommendation.

## 3. What is established

### 3.1 The central result

Across **86 concept-matched arms** — the attacked concept is the one the arm erased, taken from each
arm's own evaluation record where it has one (55) and otherwise from its build record (31) — on 5
concepts (church 70, parachute 7, Van Gogh 4, garbage truck 3, tench 2):

> **No arm in this project has been *shown* to both suppress its target and resist relearning.**
> **56** are driven to closed-set target accuracy H = 1.000 by an attack of at most 25 gradient
> steps. **21** cannot be read at all: their defence lives in the text encoder and our post-attack
> evaluation loaded the base one, a defect we documented. **6** were destroyed or heavily degraded
> (un-attacked retain8 0.0000 to 0.3417, against the teacher's 0.9417) and **3** never erased the
> target at all (un-attacked target accuracy 1.0000). **Every arm is accounted for; no category is
> left open.**

Method families covered: ESD, UCE (4 variants), GIFT (6), FRAG/FRP (6, across five concepts), IMMA
(3 adapted plus one **official-code** execution), AdvUnlearn, our Meta-Unlearning port (7), a
weight-space regulariser (3), the internal h-series and `undo` variants, and our own UTRAP at
K = 0, 1, 8, 16, 32. SD-1.4 throughout.

**I want to be plain about how this number moved today, because the movement is the point.** I first
reported "49 of 55", which inferred each arm's concept from its checkpoint name and got both the
count and the membership wrong. I corrected that to "50 of 55", then "51 of 55" as the K=8 sweep
completed, and stated that not one arm had resisted. Then I found that the 31 arms I had excluded
for want of an evaluation record have their target recorded in their *build* records — and widening
the tally to 86 surfaced **five arms I cannot dismiss**. The strong claim is withdrawn.

**All five have since been resolved, without new GPU time.** The older `retain_mean_acc` figure
carried by legacy arms turns out to be the *same quantity* as our current retain8 — an arm with both
reads 0.31666666666666665 in each — so it can be read against the 0.85 bar directly. On that basis
four of the five lost most of their retained capability (retain8 0.3167, 0.3417, 0.0917, 0.0000
against the teacher's 0.9417) and the fifth never erased its target at all (target accuracy 1.0000).
**None can satisfy your bar**, which requires retain8 >= 0.85. The unresolved category is closed, and
with it the possibility that one of them was a hidden counter-example.

**One observation does not fit, and I would rather you hear it than not.** Because the two metrics
coincide, a working UCE arm's retain8 is 0.3500 — essentially identical to the 0.3167 of the `undo`
arm above. Yet UCE is fully reversed to 1.000 while that arm reaches only 0.4667, *and the
less-recovered arm was swept at twice as many attacker learning rates*. Neither utility damage nor
sweep coverage explains a gap of 0.53. It is the only signal in this project that something beyond
parameter displacement and utility loss affects reversibility.

I am **not** proposing we chase it: that arm fails your utility bar by 0.53 and cannot satisfy the
mandate whatever its robustness, and the comparison carries real confounds — different base
procedures, neither arm tested at the 25-step budget the rest of the campaign uses, and n=1 build
each. I flag it because it is a measurement our conclusions do not account for, and because if
Option C (a changed threat model) is ever scoped, this is where I would look first.

The resolved exceptions do not survive inspection. Two are dead models: a FRAG variant on parachute
holding H to 0.7333 over 9 cells at 300 steps — the attacker cannot relearn even its own training
prompts — because its un-attacked retain8 is **0.0833** against the teacher's 0.9417; and the same
variant on church, worse still at **0.0000**. Two are the opposite failure, a GIFT and a UCE setting
whose un-attacked target accuracy is **1.0000**: nothing was erased, so there is nothing to restore
and the gate has no headroom to pass. Three further arms were unresolved sub-1.000 cases this
morning (0.9333, 0.9333, 0.8000) measured at a single attacker learning rate; **completing their
sweep moved all three to 1.000**, one going from target accuracy 0.0000 to 1.0000 in 25 steps. They
were under-measured, not robust — finding 2 in §4 doing exactly what it predicts.

### 3.2 Robustness is not purchasable by erasing harder

Across a 12-fold sweep of the erasure weight, even **lambda = 0.60, which drops retain8 to
0.5417**, reaches H = 1.000 with 3 of 3 gate-valid attacks. Sacrificing utility buys nothing on
this axis at any price we tested.

**Qualification added after the §3.1 resolution:** that statement holds for the erasure-weight axis,
where it was measured. It is *not* a general law about utility and robustness — the `undo` arm in
§3.1 sits at retain8 0.3167, below anything in the lambda sweep, and recovers only to 0.4667. So
destroying enough capability may reduce recovery; it just produces a model nobody can use, which is
why your bar rules it out rather than why the physics rules it out.

### 3.3 Apparent robustness largely measured attacker tuning

The same checkpoint reads **H = 0.267** at attacker lr 1e-5 and **H = 0.933** at 5e-5, on an
identical 25-step budget. A fixed learning rate is therefore a lower bound on recovery, and it
understates recovery **most on exactly the arms that look best**. Five claims earlier in this
campaign were withdrawn for this reason, including a headline 0.57 separation that turned out to be
a single-learning-rate artifact.

### 3.4 Our own method was retired on this evidence

UTRAP's distinctive machinery (a K-step inner attack trajectory) was **retired**, documented in
`MANDATE_v2_2026-10-27.md` §3:

- At matched outer updates and ~3x the training samples, K=8 reaches mean H 0.807 against the
  untreated teacher's 0.867 — within 0.06 across 3 independent builds — at **3.3x** the wall-clock
  (17.2 vs 5.2 min per build). Similar performance at greater cost, which your bar excludes.
- The only configuration that moved the forgetting axis, K=0, uses **none** of the method's
  distinctive machinery: at K=0 the trajectory is empty and the update reduces to gradient-ascent
  unlearning with a retain anchor. It is retained as a *baseline* under that accurate name.
- At the objective level there is no novelty: it is a reachable special case of ResAlign's default
  mode, its first-order gradient is identical to Meta-Unlearning's, and MLAC (Nov 2022) predates
  and strictly dominates the mechanism.

All builds, 1,693 result records, the evidence snapshot, figures and the earlier deck remain in
place and reproducible. Retiring the direction did not delete its evidence.

## 4. Measurement findings — the part I think is genuinely new

These came out of re-analysing records we already had, at **zero additional GPU cost**. Full
detail with supporting cells: `docs/FINDINGS_MEASUREMENT_2026-10-27.md`.

1. **A negative "relearning rise" has two distinct causes that must not be merged.** Of 123
   evidence-complete cells, 11 show the attacker *reducing* target accuracy on its own training
   prompts. One is saturation noise on a control that was never erased; ten are models the attacker
   **destroyed** — target panel driven to ~0 **and** retained capability driven to ~0. Reported
   naively, a destroyed cell reads as near-perfect erasure that the attack could not undo. On one
   parachute arm the three seeds read H = 0.000, 0.000, 0.200 — from a model with **zero** retained
   capability.
2. **The attacker's useful learning-rate window is arm-dependent**, in both where it saturates and
   where it breaks down — and it shifts *within* a single concept (one parachute arm breaks a full
   rung earlier than another). A rate carried over from a different arm reports destroyed models as
   defence. There is no cheap predictor: I hypothesised the window tracks pre-attack fragility and
   **refuted it** against our own cells.
3. **Pre/post difference metrics are noise-dominated.** On a byte-identical checkpoint, the
   pre-attack score is 0.2000 on one evaluation seed and 0.8000 on another, so the delta swings by a
   factor of **5.5** while the post-attack score is identical (0.9333 both times). Report the
   post-attack value; it reproduces to within one quantum (1/15) across independent studies.
4. **An absolute post-attack utility drop cannot evidence a booby-trap.** Over 95 gate-valid cells
   the mean change is -0.0268 but 30.5% lose more than 0.05 and the worst loses **0.7708** — on arms
   with no trap mechanism at all, including the plain teacher. Any such claim needs a *paired*
   comparison against a control at matched attacker settings. **This correction caught a false
   positive in our own pending experiment before its attack phase ran** (§5).
5. **Closed-set CLIP accuracy is not image quality.** Slide 4 of the existing deck shows three
   methods visibly destroying an unrelated concept while scoring 0.87-0.97 on it: a textured blob is
   still classified as a golf ball.
6. **A reproducibility limit in an upstream artifact.** The pinned Meta-Unlearning repository, whose
   paper is titled *Preventing Relearning Unlearned Concepts*, **ships no relearning-attack script**
   and none of the three datasets its code loads. The attacker configuration behind its headline
   claim is not recoverable from the artifact. This is an observation about the released code, not a
   claim that the paper is wrong.

**What these findings are not:** they are demonstrated on our pipeline and our arms. They show that
a robustness claim measured without these guardrails is *unsupported* — not that any specific
published claim is false. Establishing the latter would require re-running each method's own
evaluation, which we have not done. I want to be exact about this distinction in anything we write.

## 5. The live experiment, and its pre-declared bar

The one remaining mechanism class with a concrete published instance is a **trap**: make fine-tuning
on the target destroy retained capability, so an attacker gains the concept but loses the model.
Our earlier port of that idea implemented one of three components of the published objective and
failed in the predicted way — utility collapsed *without* any attack, monotonically in the trap
weight (retain8 0.9417 -> 0.6333 -> 0.4750 -> **0.0000**).

Stage S1 was built to test the corrected three-term objective. **It failed as an instrument, through
a defect of mine, and answers nothing.** I found this while it was still running and before reading
any of its results.

The objective's trap term is a *difference*, `[L_R(theta') - L_R(theta)]`. My implementation
evaluated the two halves on **different random draws** of noise and timestep, while the parameter
change between them is only `alpha * grad L_F` with `alpha = 1e-5`. Since the diffusion loss varies
hugely with timestep, the draw difference swamped the quantity being measured, and the resulting
gradient was noise with a weak drift.

The finished checkpoint confirms it quantitatively. Arm 2's mean parameter drift from its teacher is
**2.27e-5**. For Adam at lr 1e-6 over 1500 steps, coherent motion predicts 1.5e-3 and a pure random
walk predicts 3.9e-5 — the measurement is **0.59x the random walk and 0.015x coherent**, so about
98.5% of the optimiser's displacement cancelled.

Its panel therefore looks good for the wrong reason: `retain8` 0.9750 and target suppression 0.0667,
against the teacher's 0.9417 and 0.1333, where our earlier one-term port collapsed to 0.4750 at the
same setting. But those differences are **4 of 120** retain images and **exactly 1 of 15** target
images, inside the measured noise. **Arm 2 is behaviourally indistinguishable from its teacher**, and
a model that did not move cannot tell us whether the objective preserves utility. I had written that
this panel would be "a real result about the difference form"; that is withdrawn.

Cost of the error: **~1.5 GPU-hours wasted**, which I am recording as mine rather than as a stage of
the plan. Two outputs of S1 do survive, both about the harness rather than the method: the
byte-exact proof that the control arm is a no-op, and the measurement chain that caught the defect.

The corrected implementation (`scripts/meta_unlearn_v3.py`) applied two fixes, both taken from the
authors' released code rather than invented: shared random draws across each difference, and the
upstream's fixed meta-term timesteps {1, 2, 5, 10}, which our port had silently omitted. A short
pilot whose declared purpose was to **falsify my own diagnosis** instead confirmed it decisively: the
estimator's measured quantity fell from 0.190 to 3e-6, below the arithmetic precision of the loss.

### 5.1 Stage S1b also failed as an instrument, for a different reason

The corrected build then moved **no further than the broken one**:

| configuration | mean parameter drift | vs a pure random walk |
|---|---:|---:|
| coherent motion (1500 x learning rate) | 1.5e-3 | — |
| S1 — broken estimator | 2.27e-5 | 0.59x |
| **S1b — corrected estimator, Adam eps lowered 1e-8 -> 1e-12** | **2.05e-5** | **0.53x** |

The gradient's *magnitude* was stable across all 1500 steps, so the objective pushed continuously;
its *direction* was resampled every step and the updates cancelled. **S1b's panel is therefore no
more readable than S1's** — for the record it shows retain8 0.9833 with target accuracy *rising*
from 0.1333 to 0.3333, i.e. the arm slightly un-erased the concept, at a drift close to the panel's
own noise.

The cause is a **third** divergence from the authors' released configuration: they ship
`--train_batch_size 10`; our port draws a single image per step. At batch 1 the meta gradient is a
Hessian-vector product for one image and one noise draw, so its direction changes every step.

**Three omissions, each independently sufficient to stop the mechanism working** — unpaired draws,
random timesteps, batch size 1. I found all three by measurement rather than by reading the released
code carefully at the outset, which would have been faster and cheaper. That is the main process
failure of the day and I am not dressing it up.

### 5.2 Stage S1c, with a stopping rule I cannot override

One further attempt, pre-registered with the decision **enforced in the runner itself** rather than
left to my reading on the day: the meta gradient is averaged over 8 samples per step at 190 steps,
holding the sample budget within 2% and the cost within 1.3% of S1b. The script measures the
resulting drift and **refuses to run the evaluation panel or any attack cell** if it lands in the
stop band:

| measured drift | consequence |
|---|---|
| >= 2e-4 | the objective has purchase; the panel is read under the §5.3 criterion |
| **<= 6e-5** | **STOP** — the direction closes as untestable within our resources. No fourth fix |
| between | partial; drift reported alongside the panel, panel treated as provisional |

**My honest expectation is the stop band.** Averaging 8 samples cuts direction variance by only
~2.8x, and clearing 2e-4 would need roughly 70% of each step to align. I am running it because it
is the configuration the authors actually ship and because closing the direction with a known,
unaddressed fidelity gap would not be a defensible place to stop — not because I expect it to work.

What I did before looking at any result, recorded in
`PREREG_S1_AMENDMENT_1_2026-10-27.md`:

- **Found that our control arm is a no-op**, and proved it: the build is byte-identical to its
  teacher (same sha256, max absolute parameter delta **0** across 859,520,964 parameters). The
  implication is that our port has **no erasure objective of its own** — all suppression is
  inherited from the teacher. The earlier reference table's "trap weight = 0" row was the teacher's
  own number, not evidence that a zero weight preserves utility.
- **Added a condition the protocol lacked**: the arm must still suppress the target un-attacked
  (<= 0.40). Without it, an arm that gave the concept back would have passed on utility alone while
  protecting nothing.
- **Replaced the success criterion**, which finding 4 showed would have produced a false positive,
  with a paired difference against the control at matched attacker setting, required at two or more
  learning rates.
- **Bounded what S1 can conclude**: the upstream objective has five weighted terms over three data
  pools; ours has three terms over two, and our simulated attacker trains on a different pool than
  the paper's. **S1 therefore cannot speak to the published method's performance either way** — only
  to whether the mechanism produces a trap when built our way.
- **Encoded the decision rule in code** (`scripts/s1_verdict.py`) before the results existed, and
  verified it is deterministic, so the verdict is not hand-computed.

**Pre-declared bar, for the record:** un-attacked retain8 >= 0.85; un-attacked target suppression
<= 0.40; paired trap effect >= 0.25 at two or more attacker learning rates where the control's own
cell is valid. Given the measured control, that means an absolute drop >= 0.25 at both 1e-5 and
5e-5. If S1 fails, the mandate's stopping rule retires this class.

## 6. The decision I need from you

### Option A — continue the defence line past S1b
S1b (the corrected instrument) costs ~3 GPU-hours; if it clears its pre-declared bar, S2 and S3
(n=3 builds, full sweep, second concept, baselines at matched attacker effort) cost a further ~18.
**My read: low probability, and I would not bet the project on it.** Nothing in 55 arms has resisted;
we have never tested the authors' own operating point (their shipped script uses a trap weight of
0.1 where we used 0.8); our port lacks the paper's third data pool entirely and simulates a
different attacker; and a trap that worked would still have to be shown not to make the model
fragile to *any* fine-tuning, which is the obvious failure mode and which neither S1 nor S1b tests.

The defect that wasted S1 is also a reason for caution about this option specifically: this
mechanism is **delicate to implement correctly**, and an implementation that is subtly wrong fails
silently and looks like a null result. That is a poor foundation for a claim of advantage.

### Option B — write up the evaluation findings as the contribution
The deliverable is a protocol for relearning-robustness claims plus the measurements showing why
each guardrail is necessary, over 55 arms and 5 concepts.

**Being precise about the cost claim, because it is the part you should push on:** the §4 *analyses*
cost zero marginal GPU time — they are re-readings of records we already had. The *evidence base*
they rest on cost the project's whole GPU budget (>= 39.5 device-hours floor), and that spend is
already sunk whichever option you pick. So the correct statement against your bar is not "this
contribution was free", it is: **the remaining work to deliver it is analysis and writing, where
Option A needs a further ~18 GPU-hours with low expected yield.** The protocol's own value is that
it lets a reader reject an unsupported robustness claim without re-running the defence.
**Risk, stated plainly:** this is a methodology contribution, not a better method. It does not claim
an advantage over existing *methods*; it claims existing *evaluations* cannot support their
robustness conclusions. Whether that satisfies you is your call, not mine to assume.

### Option C — change the threat model
Every result above is for one attack family: direct fine-tuning on target images with an ordinary
diffusion loss. Resistance may be achievable against a *weaker, better-specified* adversary — a
fixed compute budget, no access to target images, or black-box access only. This would be a new
project with a defensible premise rather than a rescue of this one.

### My recommendation
**Option B as the primary write-up, with Option C scoped as the follow-on**, and Option A continued
only if S1 clears its pre-declared bar. I would rather bring you a correct negative result plus a
usable protocol than a defence that looks good at one learning rate.

## 7. Resources consumed

| | |
|---|---|
| elapsed | 115 h |
| GPU | **>= 39.5 device-hours** — a reconstructed floor, not a measurement: only 582 of 1,693 records carry a runtime field, so the true figure is higher by an unknown amount. No counter existed before 2026-10-24. |
| storage | 265 GiB of a 500 GiB authorisation; `/data` is shared and near full |
| GPUs | 0 and 1 only, never more than two at once; 2 and 3 belong to another user and were untouched |

A 57 GiB leak of scratch checkpoints, caused by a defect I introduced in the attack script's
cleanup rule, was found and removed with authorisation; the defect is fixed and each deleted
checkpoint's hash remains in its record.

## 8. Open items

- **S1b** — the corrected test of the three-term objective. Opens with a 4-minute pilot designed to
  falsify my own diagnosis of the S1 defect before further GPU time is spent.
- **S1 arm 3** (running) — informative about the `gamma_f` term only, which is a single-point
  gradient and therefore *not* affected by the estimator defect. Any motion it shows is attributable
  to `gamma_f`, not to the trap.
- **U20** (running) — completes the attacker sweep on the three K=8 builds, the only arms with usable
  capability whose recovery was below 1.000. Builds `b0` and `b1` have **already crossed from 0.9333
  to 1.0000** once 5e-5 was measured, confirming they were under-measured rather than resistant.
  `b2` (0.8000) is still pending; if it follows, the tally reaches **51 of 55**, leaving one dead
  model and three arms with no gate-valid cell as the only non-1.000 entries.
- **U19** — complete, 24 of 24 cells; both parachute arms reach 1.000. Written up in
  `docs/STUDY_U19_second_concept.md`.
- Five withdrawn claims are listed in `deck/SUPPORTING_MATERIAL.md` §8 so none is reinstated from an
  older summary. Two more were withdrawn today: a "worse on the second concept" comparison that was
  an artifact of differing baselines, and a "three base models" claim I had inferred from arm names
  and could not support from the records.
