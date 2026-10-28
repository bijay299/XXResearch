---
title: "Context Pack — Concept Unlearning in T2I Diffusion"
subtitle: "Backing detail and anticipated questions for the status report"
author: "Bijay Pandey"
date: "28 October 2026"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
---

This document backs the status report. It is not meant to be read front to back — it is organised by the questions most likely to be asked, so an answer can be found quickly.

\pagebreak

# 1. Anticipated questions, with answers

## "Did you try hard enough?"

**One sentence:** our own method, designed from eleven prior failure modes, failed identically to the published ones — **18 of 18 valid attacks at 1.000**, across three arms, one of which cleared the utility floor before being attacked.

Five directions were built and closed, each by a decision rule written before it ran: UTRAP (unrolled inner attacks), selective Fisher damage, the Meta-Unlearning trap, damage-then-distil, and WSR-anchor. Every method was *swept*, not run once at defaults, and in two cases we repaired the published objective before testing it.

## "Is your attacker fair?"

It is the standard relearning attack: fine-tune on images of the concept. Three points:

1. **Targeted relearning is published**, not our invention — Meta-Unlearning (ICCV 2025) establishes T2I erasure is reversed by malicious fine-tuning using essentially this protocol.
2. **We also tested methods under their own threat models.** GIFT was reversed under its own DreamBooth setting.
3. **We swept attacker capacity and budget**, not just one setting: LoRA ranks 4/16/32 through full fine-tuning, and budgets from 1 to 1000 gradient steps.

The striking finding is not that we used a sledgehammer — the attacker is *tiny*. 25 steps at batch size 1; 768 trainable scalars; 0.046% of the UNet.

## "Hasn't someone already shown erasure fails?"

Yes, partially, and we say so:

- **Erased or Dormant?** (arXiv:2505.16174) publishes reversibility — **for UCE and ESD only**.
- **Meta-Unlearning** (ICCV 2025) establishes reversal by malicious fine-tuning.
- **An Illusion of Unlearning?** (AISTATS 2026) makes the representation argument **for classifiers**.

What is not published: the robustness-*specific* methods on one bench under one attacker with honest cell accounting; the attacker-cost curve; the mechanism; and the measurement defect.

## "Will a negative result publish?"

*An Illusion of Unlearning?* is AISTATS 2026 and is a pure negative with a diagnosis. And ours is no longer purely negative — **ResAlign works**, we verified it, and we can state the mechanism separating it from what fails.

## "What are your own weaknesses?"

Four were identified as blocking. Three are closed; D3 is open and affects wording. Details in §4.

## "Why not one more method idea?"

The last one we had was published a year ago — **Meta-Unlearning (ICCV 2025)** proposes exactly it, and we had already ported and measured it failing. The space is crowded: Meta-Unlearning, IMMA, GIFT, ResAlign, PGU.

## "Can we salvage a method from this?"

Not honestly, on current evidence. Every direction is either published, measured to fail, or both. The contribution is the diagnosis, the instrument, and the comparison.

\pagebreak

# 2. Key numbers, with provenance

| quantity | value | where from |
|---|---|---|
| arms built | 164 | `adapters/` |
| attack measurements | 1,215 | `outputs/atk_*.json` |
| generation panels | 374 | `outputs/gen_*.json` |
| pre-registrations | ~30 | `docs/PREREG_*.md` |
| numbered findings | 78 | `docs/FINDING*.md` |
| concept-matched accounting | 56 reversed / 21 void / 6 destroyed / 3 never erased | `docs/FINDINGS_MEASUREMENT_2026-10-27.md` |
| steps to full recovery, six families, n=3 | **≤100** | FINDING74 |
| steps at a good learning rate | **25, zero variance** | FINDING70 |
| smallest adversary reaching 1.000 | **768 scalars** | FINDING54 |
| LoRA adversary | **0.046% of UNet** | FINDING53 |
| rank-4 LoRA reverses UCE | 1.000 over **9/9 cells** | U33 |
| UTRAP (ours) | **18/18 at 1.000** | FINDING63 |
| AdvUnlearn authors' checkpoint | 1.000 over **51 admissible cells** | D3b |
| base SD-1.4 NSFW rate (control) | **0.1817** (109/600) | U40 |
| ResAlign un-attacked | **0.0000** (0/600) | U40 |

## The ResAlign budget curve

| budget | US after attack | % of base rate | attack valid? |
|---|---:|---:|---|
| 25 steps | 0.0333 | 18% | yes |
| 100 steps | 0.0333 | 18% | yes |
| 300 steps | **0.1333** | **73%** | yes |

**Reading:** ResAlign does not hold indefinitely. It **costs the attacker roughly 100× more** than every other method — which fall at 1 to 4 steps — and breaks by 300. That is an explicitly measured trade-off, not an absolute defence.

\pagebreak

# 3. Experiment index — what was run and why

| study | question | outcome |
|---|---|---|
| **S1 / S1b / S1c** | does Meta-Unlearning's trap work? | drift 2.27e-5 / 2.05e-5 / **6.10e-6**, the last below the random-walk floor of 1.38e-5. Closed under a pre-registered stopping rule |
| **U20–U23** | is our selective-damage addition ours, or plain UNDO? | **lost to the random control.** Our addition withdrawn |
| **U24 / U32** | teacher quality vs damage level in damage-then-distil | α=0.5 → utility 0.292; α=0.3 → perfect erasure (0.0000, four seeds) and utility 0.517. Family closed |
| **U26 / U27** | does restricting the attacker help? | rank-4 LoRA already reverses; no curve |
| **U29 / U30** | are the erasure metrics trustworthy? | **no** — a visible church scored `van_gogh=0.72` |
| **U33** | does attacker capacity discriminate methods? | **no** — 135 cells, everything reversed at rank 4 |
| **U34 / U36 / U37 / U38** | does the concept return under benign fine-tuning? | **yes for ESD**, verified in images and by a self-source control. UCE half withdrawn |
| **U35** | how many gradient steps does recovery take? | 75 cells — three of four methods fall at **1 step** |
| **U39** | does the key arm replicate across build seeds? | yes — 0.4667 / 0.5333 / 0.4667, range one quantum |
| **U40** | does ResAlign's own figure reproduce? | **exactly** — 0.0000, with base control at 0.1817 |
| **U41 / U42** | does ResAlign resist a targeted attack? | holds to 100 steps; breaks by 300 |
| **D1–D4** | are our own measurements sound? | three closed, one open |

\pagebreak

# 4. The four defects, in full

## D1 — inconsistent label text (CLOSED)

`gen_eval.py` defines `LABELS` (text for CLIP) and `LAB_KEY` (record keys). **Nine of twelve differ** — 'garbage truck' vs 'garbage_truck', 'van gogh painting' vs 'van_gogh', and so on. `gen_eval_final.py` passed the key strings to CLIP.

`church` is identical in both, which bounds the problem but does not remove it: suppression is a 12-way **argmax**, so an image can cross between church and a competitor because the *competitor's* text changed.

**Measured** by re-scoring the same saved pixels under both label sets:

| split | images | argmax agreement | with `LABELS` | with `LAB_KEY` | difference |
|---|---:|---:|---:|---:|---:|
| H (recovery) | 900 | 0.902 | 0.6633 | 0.6533 | **0.0100** |
| util | 1,152 | 0.963 | 0.8455 | 0.8576 | **0.0121** |

Against noise floors of 0.067 and 0.042 respectively. Labels flip on 9.8% of images; **no conclusion changes.**

## D2 — validity-gate bias (CLOSED)

The gate's exclusion rate is **non-monotone in pre-attack suppression**: 0.392 / 0.205 / 0.135 / 0.234 / 0.115 across `T_before` buckets 0.0–0.8. Confirmed on 81 arms, 901 cells.

**But the conclusion survives unfiltered:** 55/61 arms reach 1.000 gated, **58/74 ungated**, 59/75 raw. Only two arms shift by more than a quantum, and **both are ours**, both looking *less* robust once the gate is removed.

**The bias flatters defences.** Our claim is understated by the published estimator, not propped up by it.

## D3 — no official reproduction (OPEN, widened)

The fidelity manifest states it: *official reproduction means authors' code and authors' protocol, metric and data — nothing in this project qualifies.*

| method | deviation on record |
|---|---|
| **UCE** | our `last..76` embedding range vs upstream's single embedding. **Deliberate, and why our arm erases at all** — upstream at these settings leaves church at 15/15, replicated twice by independent code paths |
| **IMMA** | our scope is **less than half** the reference's (2.13×), excludes self-attention; LoRA never re-initialised; different optimizers |
| **Meta-Unlearning** | variant A omits **two of Eq. 8's three terms**; we restored them |
| **GIFT** | **no official implementation exists** |
| **WSR** | anchors toward base where the paper **maximises** distance — direction inverted |

**Partially addressed:** AdvUnlearn's and ResAlign's **authors' released checkpoints** have now been attacked directly.

## D4 — pervasive n=1 (CLOSED on the key arm)

`esd_church` carries the benign-recovery headline. Three build seeds: **0.4667 / 0.5333 / 0.4667**, mean 0.4889, full range 0.0667 — exactly one quantum. Utility flat at 0.9917 / 1.0000 / 1.0000.

The other 163 arms remain single builds. What this establishes is that build-seed variance sits at or below the metric's resolution.

\pagebreak

# 5. Claims we withdrew, and why

This list is the project's strongest evidence of method, and it should be volunteered rather than discovered.

| claim | why withdrawn |
|---|---|
| "IMMA partially resists its own adversary" (H=0.867) | refuted within the hour — tripling the LoRA budget gives 1.000 |
| "the defence buys about ten gradient steps" | seed replication gave sd=0.2776; at n=3 no budget leaves any arm meeting the bar |
| "our arm beats FRP on both axes" | at n=3 target is 0.1778 ± 0.1018, *worse* on erasure depth |
| "unroll depth buys nothing (mechanism checked)" | the mechanism check did not pass; withdrawn |
| "damage-then-distil resists" (maxH 0.60/0.67) | the arms still **drew churches**; the metric was reading a noise overlay |
| "benign fine-tuning recovers the concept" — for UCE | the self-source control collapsed it to ~0; it was base-model leakage |
| "the first resistance result" — UCE under benign FT | the images show degenerate texture; the defence is "it stays broken", not "it stays erased and usable" |
| "the cost curve is an untouched axis" | FINDING69/74 had curved it weeks earlier, at n=3 |

**Five claims died on seed replication alone.** Roughly a third of headline claims have been retracted or narrowed against their own pre-declared falsifiers.

\pagebreak

# 6. Technical specifics likely to be probed

**Base model.** `CompVis/stable-diffusion-v1-4`, revision `133a221b8aa7292a167afc5127cb63fb5005638b`, fp16. UNet 859,520,964 parameters.

**Suppression metric (H).** 12-way closed-set CLIP accuracy, 3 held-out prompts × 5 images (n=15, quantum 1/15 = 0.067), **no reject option**. This is the metric whose defect we document.

**Validity gate.** A cell counts only if `T_rise ≥ 0.20` — the attacker must relearn the prompts it trained on. A low recovery in a cell where the attack never started is inconclusive, not evidence of robustness.

**Destroyed-cell rule.** `retain_after < 0.55` **and** `retain_before − retain_after ≥ 0.25`. A wrecked model is not a recovered model.

**Noise floors.** `H_after` reproduces to one quantum (0.067); retained capability to ~0.042. Differences below those are not differences.

**ResAlign's objective**, from their code:

```
loss = esd_loss(θ) + λ · [esd_loss(θ*) − esd_loss(θ)] + w · retain_loss
```

θ\* is the model after simulated fine-tuning. λ anneals 0.2 → 0.8 on a sigmoid. Fifteen inner configurations — lr {1e-6, 1e-5, 1e-4}, inner steps {5, 10, 20, 30}, optimizers {Adam, GD}, methods {LoRA, DreamBooth, standard} — sampled three per step. Implicit hypergradients via fixed-point iteration.

**ResAlign checkpoint verification.** 686 tensors, 859,520,964 parameters, fp32, **0 missing keys** against SD-1.4's UNet; differs from base by 0.32% of weight norm. Obtained from the authors' released files.

\pagebreak

# 7. What a reviewer will attack, and our answer

| likely objection | answer |
|---|---|
| "These are your implementations, not the methods" | **Conceded and stated.** Partially addressed by attacking AdvUnlearn's and ResAlign's authors' checkpoints directly. Days 3–4 of the plan extend this |
| "Your metric is broken — you said so yourself" | Correct, and that is **pillar two**. The mechanism findings do not depend on it: Fisher-vs-random, the 768-scalar result, and the image audits are independent of the classifier |
| "Your validity gate biases the result" | Measured (D2). The bias **flatters defences**; unfiltered, the conclusion is stronger |
| "n=1 builds" | Attacks are n=3 throughout; the load-bearing arm now replicates across build seeds within one quantum |
| "ResAlign disproves your thesis" | It **bounds** it. Four families fail for a structural reason; ResAlign occupies a fifth that we characterise — and it still breaks by 300 steps |
| "Only one concept" | Five concepts, three base models; the core result holds across all |
| "You only tested your own attack" | Capacity swept from 768 scalars to full fine-tuning; budget swept 1–1000 steps; methods tested under their own threat models where defined |

\pagebreak

# 8. Repository map

All evidence is commit-pinned in `bijay299/XXResearch`, branch `research/u29-teacher-audit-2026-10-27`.

| path | contents |
|---|---|
| `evidence/u29_teacher_audit/` | teacher audit — images and per-image CLIP scores |
| `evidence/u30_method_arm_audit/` | the audit that withdrew the damage-then-distil result |
| `evidence/u33_threat_ladder/` | capacity ladder prereg, runner, final table |
| `evidence/u34_benign_rung/` | benign fine-tuning result, 9 evidence JSONs |
| `evidence/u35_cost_curve/` | attacker cost curve, prereg and final table |
| `evidence/u36_benign_audit/` | image verification of the benign result |
| `evidence/u37_selfsource_control/` | the control that split U34 in two |
| `evidence/u38_uce_resistance/` | what UCE's apparent resistance actually is |
| `evidence/d1_label_text/` | D1 measurement and script |
| `evidence/d2_gate_bias/` | D2 measurement and script |
| `evidence/d3_fidelity/` | the fidelity finding |
| `evidence/d3b_author_checkpoint/` | AdvUnlearn authors' checkpoint result |
| `evidence/d4_build_replication/` | build-seed replication |
| `evidence/u40_resalign_baseline/` | ResAlign reproduction and control |
| `evidence/audit_2026-10-28/` | triage of all 78 findings |
| `evidence/litsweep_2026-10-28/` | literature sweep, with its limits stated |
