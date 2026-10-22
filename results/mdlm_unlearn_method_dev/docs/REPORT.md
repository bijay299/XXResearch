# Unlearning in masked diffusion language models — consolidated report

**Date:** 2026-10-21 · **Backbone:** LLaDA-8B-Instruct rev `08b83a6feb34df1a6011b80c3c00c7563e963b07`
**Status:** no method in this project meets the pre-declared "beats SOTA" bar. The defensible
contributions are an ablation, two impossibility-shaped findings, a corrected robustness
protocol, and a robustness inversion that disqualifies our own best static method.

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
- The one route the findings do not block (UNDO-style damage-and-distil) is **in progress**, not
  concluded.

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

**Open and unresolved, stated rather than hidden.** Our frame scores REPLACE (a fluent wrong
answer) as acceptable, because the retrain-only pristine reference does it at 0.470. EMDU's
authors and [arXiv 2407.16997](https://arxiv.org/abs/2407.16997) argue the opposite — that a
confident wrong answer is *worse* than abstention. Which is right depends on the threat model,
and our headline would need defending on those grounds. This is a live weakness in the frame, not
a settled point.

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

## 6. Closed candidates — 19

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

**Guard against a false negative.** A thin distillation budget (997 prompts, 300 steps) could
leave the student under-repaired, which would look like "UNDO fails" when it is really "we did not
distil enough". The frozen probe is what distinguishes these: if `frozen_KL` has plateaued by step
300 the repair converged, and if it is still falling the arm is under-trained and must be extended
before any conclusion is drawn.
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
