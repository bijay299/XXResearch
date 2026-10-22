# Unlearning in masked diffusion language models — consolidated report

**Date:** 2026-10-22 · **Backbones:** LLaDA-8B-Instruct rev
`08b83a6feb34df1a6011b80c3c00c7563e963b07` (§2–§8) and the purpose-built 124M H18 MDLM with an
**exact retrain oracle** (§9)
**Status:** no method in this project meets the pre-declared "beats SOTA" bar. The defensible
contributions are two impossibility-shaped findings, a backbone with a true retrain reference, the
suppression-vs-removal gap measured against it, a uniform negative against every anchor family
under a validated attack, UNDO's mechanism confirmed on an MDLM with its own control, an ablation,
and a corrected robustness protocol.

This document supersedes `RESULTS.md` as the single consolidated account. `RESULTS.md` is retained
as the record of the AFC study at the time it was written, including its retractions.

---

## 1. The mandate, and the honest verdict against it

The goal was an unlearning algorithm that beats the strongest existing methods on a clearly
defined forgetting–utility trade-off, supported by real-model results.

**That bar is not met.** Stating it plainly, because the rest of this document contains results
that are individually publishable and could be mistaken for the headline:

- Our best method on the **static** trade-off (AFC) is a factorial cell of two published methods,
  and it **loses to its own baseline under relearning attack** (§5).
- The two findings that generalise (§4) are **negative** — they explain why the field's methods
  sit on a trade-off, rather than escaping it.
- The route the findings do not block (UNDO-style damage-and-distil) is now **confirmed** on a
  second backbone built for the purpose (§9), but UNDO is **not ours**, and our improvement on it
  is narrow and **partially anticipated**.

**What the project does deliver, as of 2026-10-22:**

| contribution | strength |
|---|---|
| **Two walls** (§4) — anchor constructions and activation targets are both closed, by argument plus measurement | the result most worth keeping; predicts failures rather than reporting them |
| **A backbone with an EXACT retrain oracle** (§9) — forget authors absent from pretraining, oracle at chance | the asset every prior evaluation in this project lacked |
| **The suppression–removal gap, measured against ground truth** (§9) | behavioural metrics are satisfied while knowledge remains; the field reports only behavioural metrics |
| **No anchor method survives a validated relearning attack** (§9) | uniform negative across MDU, EMDU, AFC, MDU+FAM, gradient ascent |
| **UNDO's mechanism confirmed on an MDLM, with its own α=0 control** (§9) | the method is theirs; the MDLM test and the control are ours |
| **The attack-validity gate** (§5) | a protocol requirement, special case of a known principle |
| 21 closed candidates, four published retractions against our own interest | process, not result |

Nothing here beats a published method at parity. The closest thing to a positive method result is
a controlled improvement of UNDO **against itself** at one operating point.

## 2. Foundation

| component | status |
|---|---|
| backbone | LLaDA-8B-Instruct, pinned revision, `mask_id=126336` |
| sampler | pinned: `steps == gen_length == block_length` ⇒ 1 token/step, `topk(conf,1)`, `low_confidence` remasking |
| memorization adapter `theta0_tofu` | qualified at 0.791/0.787 recall, bitwise save/reload verified |
| evaluation | `eval_arm_partition.py`, 120 forget / 80 retain, greedy, `gen_length=steps=64` |
| metric | six-cell partition `DISCLOSE > DEGENERATE > ABSTAIN > REPLACE > IGNORANT > UNCLASSIFIED`, plus ASCR-12s |
| objective | minimise `TV(forget profile → pristine)` subject to retain ASCR |

The partition and the retrain-reference frame are the part of the foundation we would defend
hardest, for the reason in §3.

## 3. The evaluation frame, and a trap we fell into twice

Three single-axis rankings of the same arms produce three different winners:

| ranking axis | "winner" | why it is wrong alone |
|---|---|---|
| DISCLOSE only | EMDU λ=8, 0.000 | DEGENERATE 1.000 — emits nothing usable |
| TV(forget) only | MDU λ=8, 0.191 | retain 0.091 — model destroyed |
| DISCLOSE + retain | H16 projection | DEGENERATE 0.275 |

We produced the first and third ourselves before catching them. MDU and EMDU report TOFU forget
quality as ROUGE-L and token probability, **on which a degenerate model scores perfectly**. An
honest comparison needs the full profile *and* retain utility together.

**The REPLACE question, now resolved** (`THREAT_MODEL.md`). Our frame scores REPLACE (a fluent
wrong answer) as acceptable because the pristine reference does it at 0.470; EMDU and
[arXiv 2407.16997](https://arxiv.org/abs/2407.16997) argue a confident wrong answer is *worse*
than abstention. This is not a flaw in the metric but a choice of threat model, and our metric
already has an exact decision-theoretic meaning: `TV(P,Q) = sup_A |P(A) − Q(A)|` **is** the
optimal advantage of a single-query distinguisher, so minimising TV→pristine *is* minimising the
adversary's ability to tell an unlearned fact from one that was never present. That is the
indistinguishability threat model — TOFU's own — under which confabulating at pristine's rate is
the target and abstaining instead is itself the leak.

Read as membership-inference advantage, our own numbers give two results:

| arm | TV = single-query advantage | |
|---|---|---|
| AFC λ=1 | 0.299 | best in project |
| θ₀ — *no unlearning at all* | 0.770 | the do-nothing baseline |
| RMU (calibrated) | 0.846 | **more detectable than doing nothing** |
| EMDU λ=8 | 1.000 | **perfectly detectable from one query** |

EMDU's "safe" abstention is the *worst* outcome under this threat model, and RMU's gibberish makes
it more identifiable than leaving the fact in. The two threat models are genuinely opposed —
matching a pristine model *requires* confabulating at its rate, which harm-avoidance forbids — so
we declare indistinguishability explicitly and do **not** claim our arms are preferable under a
misinformation threat model. Caveat: TV over six cells is the advantage of an adversary who sees
only the cell label, so it is a **lower bound** on the true advantage.

**Novelty caveat on the frame itself:** TOFU §5.2 states the metric-trap argument verbatim, and
AltPO ([2409.13474](https://arxiv.org/abs/2409.13474)) covers four of our five evaluation
categories. Our earlier claim to novelty here was **retracted**.

## 4. The two walls — the result most worth keeping

These are the only findings in the project that predict failures rather than report them.

### 4.1 Output-distribution wall (`FINDING18`)

> Any anchor that retains enough answer context to be well-formed retains enough context to
> identify the fact. Form and fact-identity are carried by the same information, so no
> construction over *what to mask* can have both.

Three independent measurements, 15,938 answer positions, 100 forget facts, same states per anchor:

| construction | fact-free? | well-formed? |
|---|---|---|
| mask-rate banding (SSA) | only at high *t* | only at low *t* — anti-correlated across the schedule |
| full answer masking (FAM/EMDU) | yes (`p_true` 0.017) | no — 8.19 nats, 97.9% DEGENERATE as a generator |
| fact-selective masking | no (`p_true` 0.687 at low mask rate, vs MDU's 0.715) | yes (3.48 nats) |

The fact-selective negative is the informative one: masking the fact tokens *completely* barely
changes the anchor's ability to predict them, because the model reconstructs the fact from the
structural scaffold alone — `"The author's full name is ___ ___"` plus the answer's length is
very nearly unique among 400 memorised facts. **The scaffold is the identifier.**

### 4.2 Activation-space wall (`FINDING19`)

> Any forget target close enough to the data manifold to preserve fluency is too close to the
> current activation to provide a forgetting signal. RMU works *because* its target is
> off-manifold — the same reason it produces gibberish.

| arm | target | `L_forget` | DISCLOSE | DEGENERATE | retain |
|---|---|---|---|---|---|
| θ₀ | — | — | 0.792 | 0.008 | 0.762 |
| RMU (calibrated) | `c·u`, off-manifold | 3.86 → 2.68 | **0.050** | **0.825** | 0.704 |
| `decoy` | frozen acts, decoy answer | **1.18** | 0.575 | — | 0.758 |
| `decoyq` | frozen acts, decoy prompt | **0.56** | 0.742 | — | 0.762 |
| *pristine* | — | — | *0.030* | *0.000* | — |

RMU achieves the best forgetting *and* near-best retention in the project — and 82.5% degeneracy,
reproducing Łucki et al. ([2409.18025](https://arxiv.org/abs/2409.18025)). The on-manifold decoy
fix is a measured no-op, and the losses say why: at a masked answer position the activation is set
by the prompt and mask pattern, not by which answer occupies the few visible slots.

### 4.3 The joint prediction

The two walls are the same obstruction seen from opposite directions, and together they explain
why MDU, EMDU, AFC and RMU all sit on a forgetting-versus-fluency trade-off rather than one being
better engineered. **A method that escapes both must source its forget target from neither (a) the
model's own distribution on a modified input nor (b) a point in activation space** — i.e. from a
second model, from retraining, or from data. This is a falsifiable prediction, and UNDO
([2506.06278](https://arxiv.org/abs/2506.06278)) is the test of it (§7).

## 5. AFC — the ablation, and the robustness inversion that disqualifies it

**AFC** = anchor-**first** KL (EMDU's Eq. 3 direction) + MDU's **random-answer** anchor. It is the
2×2 cell neither parent paper ran, isolating which of EMDU's two simultaneous changes carries the
benefit. It is **not a new algorithm** — every term is theirs.

**Provenance of the AFC artifact, verified and recorded because the filename misleads.** AFC λ=1
lives at `adapters/h16_none_f1/ckpt_300`, produced by `train_h16.py --mode none` — the H16
*script* with its projection disabled. Checked against the adapter's stored hyperparameters:
`mode: none`, `lam_forget: 1.0`, `localize: false`, anchor-first KL. That is AFC λ=1, and it is
the teacher used for every UNDO arm.

⚠ **That file's own `report.json` is misleading**: it carries H16's `mechanism` and
`novelty_note` strings describing retain-conflict **projection machinery that `--mode none`
never applies**, because the script writes those fields unconditionally. The weights are AFC;
the prose beside them describes a different arm. Anyone reading the artifact in isolation could
misfile AFC as an H16 projection result. The hyperparameters, not those two strings, are
authoritative.

### Static trade-off (AFC wins)

| arm | n | DISCLOSE | DEGENERATE | retain | TV→pristine |
|---|---|---|---|---|---|
| θ₀ | — | 0.792 | 0.008 | 0.762 | 0.770 |
| **AFC λ=1** | 8 | 0.146 | 0.152 | **0.534** | **0.299** |
| MDU λ=3 | 3 | 0.133 | 0.181 | 0.445 | 0.352 |
| ATWU best of 11 (our MDLM adaptation) | 1 | 0.433 | 0.025 | 0.552 | 0.466 |
| EMDU λ=8 | 1 | 0.000 | **1.000** | 0.580 | 1.000 |

The ablation conclusion stands: **the KL direction carries the benefit; FAM is what hurts.**

### Three retractions published against our own interest

1. **Stability — refuted by our own data.** n=3 gave TV sd 0.005; n=5 gave **0.158 vs MDU's
   0.033** (~5× *more* variable), with seed 3 at DEGENERATE 0.508. n=8 confirmed 0.129 vs 0.038.
   The n=3 figures were three lucky seeds.
2. **Lower degeneracy — fails at λ=1** (0.217 vs MDU's 0.122 at n=5).
3. **The entropy explanation for robustness — refuted.** AFC 2.389 nats vs MDU 1.630: higher
   entropy *and* more reversible, the opposite of the predicted direction.

### The inversion (AFC loses)

RTT relearning attack, author-disjoint train/held-out split, lr 3e-5, 300 steps, all three runs
passing the attack-validity gate:

| arm | H pre-attack | H post-attack | rise | frac of θ₀ reopened | T-rise (validity) |
|---|---|---|---|---|---|
| **AFC λ=1** | 0.225 | 0.425 | **0.200** | **0.353** | 0.375 ✓ |
| **MDU λ=3** | 0.225 | 0.300 | **0.075** | **0.132** | 0.275 ✓ |
| MDU λ=8 | 0.025 | 0.375 | 0.350 | 0.456 | 0.475 ✓ |

AFC and MDU λ=3 begin at the **identical** pre-attack disclosure of 0.225. Under the same
validated attack AFC reopens **2.7× as much** of the gap back to θ₀ on facts the attacker never
supplied.

> **Consequence.** AFC buys its better static trade-off with shallower suppression. Since
> robustness to relearning is the axis the unlearning literature treats as decisive, **AFC cannot
> be claimed to beat MDU.** The honest summary is: better average static trade-off, a
> catastrophic-failure mode the baseline lacks, and worse robustness at matched disclosure.

**The protocol correction that made this measurable.** Three earlier RTT runs (lr 1e-4, 1e-5)
showed ~0% recovery that looked like robustness but was the attack destroying the model
(DEGENERATE peaking 0.93–0.98). Root cause: we never scored the attack's own training set. The
fix is a validity gate requiring the attack to relearn what it is *trained on* (≥0.20 absolute
rise on T) before a flat held-out curve counts as evidence. **A relearning attack without this
gate can report robustness that is only a broken attack** — and three of our own runs did.

## 6. Closed candidates — 21

Fourteen closed by measurement, five by primary-source novelty check against primary sources:

| closed by measurement | closed by novelty |
|---|---|
| H1, H6, H11, C1/SIO | ESD → anticipated by MDU (our own baseline) |
| ordering line H10a/b/c (`FINDING11`, `FINDING12`) | attack-aware / meta-unlearning → Gao et al. |
| form-preserving anchor (H14) | SSA → SPARE / KSCU / TimeROME-DLM |
| localized anchor matching (H15, `FINDING15`) | H17 decoder attack → Zhang et al. adversarial-RL |
| retain-conflict projection (H16) | decoy redirection → LUNAR / Beyond Forgetting |
| schedule-split anchoring (`FINDING17`) | |
| fact-selective anchor (`FINDING18`) | |
| RMU port + 2 decoy variants (`FINDING19`) | |
| divergence position selector (failed on 3 losses) | |

Two process failures recorded rather than buried: ESD was anticipated by a paper we were already
running as a baseline, and the `memorize_v2` provenance erratum (`ERRATUM_memorize_v2_provenance.md`).

## 7. Open: the one route the walls do not block

**UNDO** ([2506.06278](https://arxiv.org/abs/2506.06278)): perturb the suppressed model,
`θ_perturbed = (1−α)θ_suppressed + αβN`, then distil the suppressed teacher's outputs into the
damaged student. Its forget target comes from damage plus distillation — neither the model's own
distribution on a modified input nor a point in activation space. It is the direct test of §4.3's
prediction.

Implemented at `scripts/train_undo.py` (DeepSpeed ZeRO-3, CPU optimizer+param offload, 2 GPUs).
Arms α ∈ {0.0, 0.3, 0.6}, β=0.1, 300 distillation steps, teacher = AFC λ=1, evaluated through
`FINDING20`'s validated protocol.

**α=0 is the control that makes the study interpretable**, and is pre-declared here before its
result is known. UNDO's claim is specifically that distilling into a *damaged* copy is what makes
suppression robust. α=0 distils into an **undamaged** copy of the same suppressed teacher. So:

| outcome | reading |
|---|---|
| only α>0 is robust | the damage is the active ingredient; UNDO's mechanism holds on an MDLM |
| α=0 is equally robust | the damage does nothing here and any gain is plain distillation |
| no arm is robust | §4.3's prediction is **falsified** — sourcing the target outside the model's own distributions is not sufficient either |

**Guard against a false negative, and what it measured.** A thin distillation budget could leave
the student under-repaired, which would look like "UNDO fails" when it is really "we did not
distil enough". The frozen probe was added to distinguish these. First arm (α=0.3, retain-only
pool, lr 2e-5, batch 1, 300 steps):

| step | 0 | 49 | 99 | 149 | 199 | 249 | 299 |
|---|---|---|---|---|---|---|---|
| `frozen_KL` | 2.467 | 2.188 | 1.663 | 1.642 | 1.601 | 1.567 | **1.668** |

**Repair stalls near 1.6 nats and does not converge toward 0.** The probe prompts are drawn from
the distillation pool itself, so this is a **training** loss stuck high — underfitting, not a
generalisation limit. lr 2e-5 at batch 1 under gradient clipping is too little optimisation to
undo a 30% relative weight perturbation across 8B parameters. The faithful arms therefore run at
lr 1e-4.

*Two limitations of our own instrument, both stated because we tripped over the second.* The
probe is 2 prompts × 2 mask rates, and the step-249→299 uptick (1.567 → 1.668) shows roughly
±0.1 of probe noise, so its absolute value should not be read finely.

⚠ **It was not comparable across arms, and we briefly reported it as if it were.** The probe drew
`prompts[order[:2]]`, and `order` is a permutation over the distillation pool — so the
retain-only pool (997 prompts) and the `--include-forget` pool (1397) selected **different probe
prompts**. A cross-arm reading of frozen_KL was therefore meaningless, and our first comparison
of lr 2e-5 against lr 1e-4 at matched step count was invalid and is withdrawn. The probe is now
anchored to the first two TOFU retain questions, identical in every arm and configuration. Curves
produced before that change are valid **within** an arm only.

**Feasibility caveat, declared before the results.** Our distillation budget is orders of
magnitude below the paper's — ~10³ samples at batch 1 against an epoch of pretraining data. If
repair remains incomplete at lr 1e-4, the correct conclusion is **"UNDO's route is not testable at
this compute budget"**, not "UNDO fails", and §4.3's prediction would remain untested rather than
falsified.
**If UNDO also lands on the trade-off, §4.3's prediction is falsified and that is the reportable
outcome.**

**Two distinct bars, stated separately because conflating them would overclaim.** The teacher is
AFC λ=1 — deliberately the *least* robust arm (recovery 0.200), so it has the most headroom to
show the mechanism.

| bar | value | what clearing it would show |
|---|---|---|
| beat its own teacher, AFC λ=1 | recovery < 0.200 | distillation-into-damage robustifies suppression on an MDLM |
| beat the robustness incumbent, MDU λ=3 | recovery < 0.075 at comparable DISCLOSE and retain | a genuine SOTA claim |

Clearing only the first is a mechanism result, not a SOTA result.

**Both bars are measured under the adversarial learning-rate search**, not a single attack
setting. `NOVELTY_attack_validity_gate.md` found that UNDO's and Lynch et al.'s own practice is
to report the adversarially *optimal* attack over an lr grid, which our single-lr RTT did not do
— so every recovery figure in this document is a **lower bound** until that sweep completes.
With checkpoint retention authorised, the UNDO arms are kept after their first attack and swept
at several learning rates rather than retrained for each, so the UNDO figures will be reported
as max-over-lr from the start.

**Correction to an earlier claim of ours.** We previously reported "repair confirmed working,
KL(teacher‖student) 4.37 → 1.87 in two steps". That was not evidence of repair. The logged loss
is computed on one prompt at one *randomly drawn* mask rate `t`, so it is dominated by the draw —
at `t`≈0 only a couple of answer positions remain masked and KL is near zero, at `t`≈1 the student
must match the teacher over 64 context-free positions. The same run that showed 0.157 at step 19
showed **3.947 at step 29** while training normally. The per-step trace is not a learning curve,
and no repair claim should have been made from it. `train_undo.py` now carries a **frozen probe**
— fixed prompts, fixed mask rates, fixed RNG, run on every rank — and the repair curve will be
reported from that instrument only.

**Two implementation faults found and fixed, recorded because both produced misleading runs.**
(a) The distillation step returned early when a low-`t` draw left no masked answer position
(P ≈ ∫(1−t)⁶⁴dt ≈ 1/65 per rank per step). The mask RNG is seeded `seed + rank`, so this fired on
*one* rank, which then skipped `backward()`/`step()` while the other proceeded — desynchronising
ZeRO-3 and deadlocking in the parameter allgather (NCCL `_ALLGATHER_BASE` timeout, rank 0 at
SeqNum 3931 against rank 1 at 3910). Fixed by guaranteeing ≥1 masked position and never skipping
a step. (b) `deepspeed --num_gpus 2` silently **overwrites** `CUDA_VISIBLE_DEVICES`, so a job
pinned to GPUs 2,3 ran on 0,1 — the devices other users' jobs claim first. `--include
localhost:2,3` is the only form that pins physical devices.

## 8. Limitations

- **One backbone, one dataset.** Synthetic-memorization LLaDA on a TOFU-style split. No RWKU
  arm trained, no Dream-7B. Transfer untested.
- **ATWU adaptation is ours, not its authors'** — SatGA+ at masked answer positions of a
  teacher-forced state. The claim is "our MDLM adaptation of ATWU underperforms AFC on this
  protocol", **not** "AFC beats ATWU".
- **RTT at one attack setting**, one seed per arm, n_H=40.
- **AFC's usable band is narrow** (λ ≈ 0.5–4; collapses at λ=8 to DEGENERATE 0.908).
- **The REPLACE-scoring question in §3 is unresolved** and the frame depends on it.
- **No method here beats SOTA** (§1).

---

## 9. H18 — a second backbone, built because LLaDA-8B could not test the hypothesis

### 9.1 Why pivot (`PREREG_H18`)

UNDO is validated on a **100M** model pretrained on 2B controlled tokens, with the oracle existing
by construction, and its robustness criterion is distillation back to within ±5% of the teacher's
retain performance. On LLaDA-8B our repair stalled at ~1.6 nats of *training* loss, and the α
values we could afford to repair were the ones the paper reports as **least** robust. A null there
would have been uninformative about UNDO rather than evidence against it. Recorded as a
feasibility limit of our setup, not a negative about the method.

### 9.2 The backbone, and the asset it provides (`FINDING21`)

124M bidirectional MDLM, GPT-2-small shape, the identical LLaDA/MDLM objective so findings
transfer. 200M FineWeb-Edu tokens + 200 synthetic authors (100 forget / 100 retain), answer pools
**shared across splits by construction and verified** (`forget_only` empty for all four
attributes), 200 unique surnames so each author is identifiable but first names are shared across
splits — so no method can cheat by unlearning "all authors named X".

| gate | requirement | measured | verdict |
|---|---|---|---|
| K1 `theta_ref` forget recall | ≥ 0.60 | **0.6975** | ✓ |
| K2 `theta_oracle` forget recall | ≤ 0.15 | **0.1100** (chance 0.1135) | ✓ |
| K3 `theta_ref` DEGENERATE | ≤ 0.10 | **0.0050** | ✓ |

**The oracle sits at chance.** It is an *exact* retrain reference — the forget authors never
entered its pretraining or its fine-tune. Every earlier evaluation in this project was anchored to
an approximated `pristine`, and `FINDING7` closed oracle matching partly because our oracle was
itself questionable.

Four instrument bugs were found getting there, and most of the apparent failure was measurement:
ranking accuracy went **0.4050 → 0.4675 → 0.6650** as each was fixed. The worst was a BPE
leading-space mismatch (`'Ghent'` = `[38, 6925]` in training vs `' Ghent'` = `[402, 6925]` in the
probe) — the **second** time that exact bug cost this project a measurement. Probes now use token
ids stored at corpus-build time, so probe and training share one construction path.

One genuine design finding, not an artefact: **isolation is the active ingredient for memorisation
under the MDLM objective.** Loss weighting gave value tokens ~200× more per-token gradient and
produced almost nothing; the same supervision packed beside ~31 *visible* same-type values
collapsed to chance. The model takes the cheap route — emit a plausible nearby value — whenever it
exists. Same shape as `FINDING18`.

### 9.3 The suppression–removal gap (`FINDING23`)

Across 15 anchor arms, the best TV→oracle is **0.0725** — behaviourally near-indistinguishable
from the oracle — while every arm still ranks forget facts at **0.16–0.25** against a 0.1135
chance floor (z ≈ 2.5–4.5 at n=400).

> **Behavioural metrics are satisfied while the knowledge is still there.** Every MDLM unlearning
> paper we know of reports only behavioural measures. On the α=0.5 arms, TV cannot separate six
> methods at all (0.072–0.100, DEG ≈ 0) while the ranking probe separates them by 0.10.

### 9.4 No anchor method survives a validated attack

RTT, author-disjoint T/H, **every attack passing the `FINDING20` validity gate**, at the
adversarially-optimal lr 3e-5:

| arm | retain | recovery |
|---|---|---|
| AFC λ=1 | 0.7275 | 0.6269 |
| MDU λ=0.5 | 0.6700 | 0.5608 |
| EMDU λ=1 | 0.7475 | 0.6738 |
| MDU+FAM λ=1 | 0.7275 | 0.7263 |
| gradient ascent λ=0.5 | 0.6350 | 0.7194 |
| **UNDO plain α=0.5** | 0.3550 | **0.0574** |
| **UNDO retain-Fisher α=0.5** | 0.4850 | **−0.0311** |

**Not one anchor family resists**: 56–73% of the forgotten knowledge returns, while all of them
look successful behaviourally. Only damage+distillation resists, by 10–20×. This is §4's two walls
confirmed from the robustness side on an independent backbone with a true oracle: methods whose
forget target comes from the model's own distributions are *uniformly* non-robust.

*(Baseline figures single-seed; replication running.)*

### 9.5 UNDO confirmed, with its own control (`FINDING23`)

Max over the adversarial lr grid {3e-5, 1e-4, 3e-4, 1e-3}, n=3 seeds where noted:

| teacher | arm | recovery |
|---|---|---|
| AFC λ=2 | teacher | 0.6573 |
| | α=0.3 | 0.2719 ± 0.050 |
| | α=0.5 | 0.0513 ± 0.020 |
| MDU λ=0.5 | teacher | 0.5608 |
| | **α=0 (control)** | **0.5742** |
| | α=0.3 | 0.3415 |

**Distilling into an undamaged copy gives no robustness at all** (0.5742 vs the teacher's 0.5608).
The perturbation does the work, on two teachers. And the effect is **invisible to static metrics**:
across α, static forget rank moves only 0.1450→0.1550 and TV stays 0.075–0.10 while retain falls
0.6625→0.3550. A study reporting only static trade-offs would conclude the opposite of the truth.

### 9.6 Retain-Fisher damage, and what survived its controls (`FINDING24`)

Scale each parameter's perturbation by `(1 − normalised retain-Fisher importance)`, renormalised so
**mean damage matches plain UNDO** (0.4945 vs 0.4991 at α=0.5). Then run a **shuffled** control:
same weight distribution, shuffled within each tensor — equally non-uniform, equal mean, no retain
information.

| teacher / α | plain | shuffled | retain-Fisher | verdict |
|---|---|---|---|---|
| AFC 0.3 | 0.6017 | 0.6633 | 0.6550 | non-uniformity suffices |
| AFC 0.5 | 0.3650 | 0.3667 | **0.4567** | **Fisher required** |
| MDU 0.3 | 0.5792 | 0.5925 | 0.6000 | non-uniformity suffices |
| MDU 0.5 | 0.3392 | 0.3342 | **0.4400** | **Fisher required** |
| AFC 0.7 | 0.1350 | 0.1317 | 0.1442 | **everything collapses** |

The first version of this finding claimed the Fisher signal explained the gain; the **α=0.3 control
refuted that** and the claim was narrowed and the finding retitled. What survives: retain-importance
matters only at high damage, where it buys +0.09/+0.10 retain at matched damage, matched recovery
and matched behavioural profile — a point that **dominates plain UNDO on both axes**. At α=0.7 all
arms collapse to chance and all nine attacks were **rejected by the gate** (T-rise 0.015–0.18),
which is the gate catching prospectively the failure that invalidated three 8B runs.

Novelty: **PARTIALLY ANTICIPATED**. DP-KFC already shapes injected noise by Fisher (for DP
training); WAGLE subtracts a retain-gradient term; SSD dampens by Fisher importance; EWC supplies
the signal. The synthesis is predictable. What is defensible is the empirical result plus the
equal-damage design and the shuffled control — **not a new mechanism**.

### 9.8 H19–H34: the UNDO programme, and the frontier it produced

Four hypotheses tested after the backbone qualified. **Three refuted or closed**, one measurement
that survived every attempt to break it.

| hypothesis | outcome |
|---|---|
| **H19** UNDO's retain cost is a repair-budget artifact recoverable *without* losing robustness | **refuted by its own criterion** — repair moves both axes together (retain 0.3100→0.6700, recovery −0.029→+0.251 over 1.5k→12k steps) |
| robustness is bounded by the teacher's residual forget knowledge | **not supported** — EMDU has the highest residual and better post-repair robustness than AFC (ρ=0.8, n=4, one inversion) |
| **H31** a composite teacher decouples forget-transfer from retain-transfer | **closed** — K2 killed H31 (TV 0.8192, behaviourally wrecked); H31b missed its retain bar (0.6167 < 0.65) |
| α=0.6 is the optimum damage level | **holds** across teachers; α=0.7 is dominated, and its earlier "collapse" was repair-bound |

**The selected point**: UNDO with an AFC teacher, Fisher-protected damage at α=0.6, 12,000 repair
steps, retain-oversampled pool.

| | retain | TV→oracle | recovery (t_frac 0.8, max over 18 gated attacks) |
|---|---|---|---|
| **selected point** | **0.6460 ± 0.017** | **0.0745** | **+0.2299** |
| anchor/GA families, same budget | 0.54–0.74 | 0.0725–0.3225 | **0.7396 – 1.1039** |
| `theta_ref` / oracle | 0.6725 / 0.8875 | — / 0 | — |

**At a matched attacker budget the advantage is 3.2–4.8× with no overlap** — and the budget-matching
correction went *against* the concern that prompted it: baselines degrade at the stronger budget
too, so the advantage held. The ceiling arm (oracle teacher: retain 0.8150, recovery 0.0669, TV
0.0275) shows the trade-off belongs to *distilling from a suppressed teacher*, not to
damage+distillation.

**But the selected point is not robust** (`§9.9`), and the method is UNDO's. What is ours is the
repair-budget diagnosis, the Fisher modification (`PARTIALLY ANTICIPATED`), the knob/teacher
sweeps, and the budget-matched gated protocol.

### 9.9 Two limits we measured on ourselves

**The selected point is not robust in absolute terms.** A fifth of the gap reopens at `t_frac 0.8`
(+0.2299). "3.2–4.8× harder to attack than any anchor family" is the claim; "robust" is not.

**Our own validity gate has a blind spot.** H31b's lr-1e-5 attack recovered **+0.1955** at T-rise
**+0.194** — just under the 0.20 cutoff — so max-over-*valid*-lr reported +0.0794 instead, a 2.5×
difference turning on a judgement-chosen threshold. `FINDING20`'s gate catches *broken* attacks but
cannot distinguish them from attacks merely below threshold yet still working. Both gated and
conservative maxima are now reported throughout.

**And `FINDING27`:** for EMDU and MDU+FAM, a `t_frac 0.8` attacker restores the forgotten facts to
**at least `theta_ref`'s own level** (raw ranks 0.3125 → 0.7000 and 0.7375 against 0.6975). For
those families unlearning achieved nothing durable — while EMDU simultaneously scores DISCLOSE
0.000, i.e. perfect on the metric the literature reports.

### 9.7 What H18 does not establish

- **Not scale transfer.** 124M on a synthetic corpus we built. The 8B attempt failed on
  feasibility, so nothing here is shown to hold at 8B.
- **Not parity with published methods.** MDU/EMDU/AFC were re-implemented on this backbone by us;
  their authors' tuning is not reproduced.
- **Not a robustness improvement over UNDO.** Recovery is indistinguishable; only retain moves.
- **One corpus, one task shape.** Four attributes, closed answer pools, 200 authors.

---

## 10. H35–H37 — the measurement audit, and the close of the MDLM phase

This section records the final MDLM-phase work. Three of its four results are about **our own
instruments and claims**, not about unlearning methods, and two of them cost us claims.

### 10.1 The strongest claim, and then its instrument failed (`FINDING28`, `FINDING29`)

`FINDING28` is the broadest result the project produced: across **125 arms** at a matched attacker
budget, **Spearman(knowledge-probe reading, max recovery) = −0.148** — no predictive relationship,
with the sign pointing the wrong way. A validated attack separates the arms into **disjoint**
bands (suppression-only +0.5608…+0.9143; damage+distil +0.4424 and below, gap +0.1184) while the
knowledge probe cannot separate them at all (74% band overlap). Eighteen pairs of arms whose probe
readings are identical to four decimals differ **1.9×–37×** in recovery. Gradient ascent has the
**lowest** probe reading of any family (0.0825, below the ranking probe's chance floor) and among
the **highest** recovery (0.9143) — selecting on the static probe picks GA and is maximally wrong.

Its mechanistic reading: **damage does not remove knowledge; it removes the attacker's route to
it.** On a controlled α ladder the probe reading moves 0.0305 — and *upward* — while recovery falls
**7.3×**, and the two ladders' slopes have opposite signs. That also killed a planned
**iterated-UNDO** line before it cost GPU time: the selected point's residual (0.1755) is *higher*
than its own teacher's (0.1533), so recursive rounds would start from progressively worse teachers.

**Then the x-axis turned out to be invalid.** `FINDING29`: the closed-pool ranking probe scores
EOS-padded length-32 candidates, so padding enters every candidate's score. It reads `theta_ref` at
**0.6725** when `theta_ref`'s true retain recall is **1.0000**, and it **inverts** the ordering of
two models (true: oracle > `theta_ref` > PUA; probe: PUA > oracle > `theta_ref`). It was caught by a
too-good number — an unrelated arm reporting retain **above the exact retrain oracle** — and
confirmed by cross-checking against generation.

This is the **third** padding/tokenisation-boundary defect in this project's measurement code.

`FINDING28` is therefore marked **PROVISIONAL**. Its recovery numbers, band separation, and α=0
controls do not depend on the probe; its ρ and matched pairs do, and may be regression dilution
from a noisy x-axis. Only 2 of 125 checkpoints survived, so `PREREG_H37` registers a **balanced
16-arm replication** with both probes on every arm and **pre-declared retraction conditions**.

It also reframes a `FINDING21` claim: *"the oracle is at chance"* is **probe-specific language**.
The same oracle sits at 0.3375 exact-span under a context-rich measurement. The oracle remains the
floor; the phrase will not be reused without naming the probe.

### 10.2 The advantage replicates under a second attack family — and a new confound (`FINDING30`)

Every robustness number in the project came from **one** attack (span SFT). A second family —
whole-passage MDLM relearning, a different optimisation surface — gives **7.6×** separation
(+0.6509 baseline vs +0.0862 selected), against 3.2× for span SFT at the same budget. **The
protection is not specific to the span-SFT surface.**

But the gate exposed something more important: the baseline admits **5 of 6** attacks, the treated
arm only **1 of 6**, because the attacker cannot relearn its **own training set** on it (T_rise
0.234 vs 0.428). That is equally consistent with fact-specific robustness and with **general
trainability loss** — in which case every robustness number here is inflated by an unknown amount.
I cannot currently distinguish them. The deciding experiment (a **trainability control**: attack
the arm on *retain* facts and compare learning speed) is specified in `FINDING30` §4 and was not
run before the pivot.

### 10.3 A method candidate closed by prior art (`PREREG_H36`)

Pool-Uniform Anchoring — forget target = uniform over the closed answer pool, derived from retain
data only, used as the UNDO teacher — was pre-registered with a commitment that an ANTICIPATED
novelty verdict would be reported as a reproduction regardless of performance. The verdict came back
**ANTICIPATED on all five components**, three of them fatally:

- **MDU at τ=0 *is* the uniform target** (2605.18253) — the paper's own τ sweeps to it
- **UNDO already runs a KL-to-uniform ("MaxEnt") teacher through noise-then-distil** (2506.06278)
- **DF-MCQ (2505.02884) already does uniform over a closed candidate set**, with the identical
  "random choice-level uncertainty" motivation

Honouring the commitment: H36 is a **reproduction**, not a contribution. It independently failed
its own pre-declared predictions (λ=2 tripped kill rule K1 at TV 0.2000; λ=1's forget sat outside
the declared window), and its headline retain number was the artifact that exposed `FINDING29`.

### 10.4 Verdict on the MDLM phase

**The mandate's bar was not met.** The selected point (UNDO α=0.6, AFC teacher, Fisher-protected
damage) is 3.2–7.6× more robust than every anchor family at matched budget, but it is **not robust**
— a fifth of the gap reopens — the method is **UNDO's**, and its robustness now carries an
unresolved trainability confound.

What the phase does leave, in order of strength:

1. **The two walls** (§4) — predictive, not merely descriptive
2. **A backbone with an exact retrain oracle** (§9) — the asset every prior evaluation lacked
3. **No anchor family survives a validated relearning attack** (§9, `FINDING25`) — 37 gated runs
4. **Only an attack measures robustness** (§10.1) — both cheap metric classes measured and found
   uninformative; headline provisional pending `H37`
5. **The attack-validity gate**, and its extension the **trainability control** (§10.2)

The recurring lesson is specific and worth carrying forward: **my measurements have held under
replication; my selections, explanations, and instruments are where failures cluster.** Five
retractions this session, every one caught by cross-checking a number that looked too good against
a second, independent measurement.

---

## 11. Pivot — text-to-image diffusion unlearning

The mandate is unchanged: develop, implement and rigorously evaluate an unlearning algorithm that
outperforms the strongest relevant existing methods on a clearly defined forgetting–utility
trade-off, with real-model results. The domain moves from masked diffusion **language** models to
**text-to-image** diffusion models (Stable Diffusion). See `PREREG_H38_t2i_testbed.md`.

What transfers, and why the pivot is well-positioned rather than a fresh start:

- **The evaluation thesis.** T2I concept-erasure papers report CLIP score, classifier accuracy on
  the erased concept, and FID. Those are exactly the *static* metric classes this phase measured
  and found uninformative about robustness. The attack literature there
  (UnlearnDiffAtk, Ring-A-Bell, P4D, concept inversion) already shows erasure is reversible — so
  the gap this phase documented is known to exist in T2I too, and the methodological tools built
  here (validity gate, matched attacker budget, max-over-attacks, trainability control) apply
  directly and are not standard practice there.
- **The one family that survived.** Damage+distil (UNDO) is, on this phase's evidence, the only
  tested family that resists relearning. Its application to T2I concept erasure is, as far as the
  novelty check reached, **untested** — and it is a concrete, motivated method candidate rather
  than a hope.
- **The oracle discipline.** The single most valuable asset here was a reference model trained with
  the forget data genuinely absent. SD cannot be retrained from scratch at this budget, so the
  two-tier structure that worked in this phase (a real model for reality, a small purpose-built
  model with an exact oracle for ground truth) carries over as the plan.
