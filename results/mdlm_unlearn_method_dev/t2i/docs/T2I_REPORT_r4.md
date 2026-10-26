# T2I concept erasure under a relearning adversary — report r4 (final for this phase)

**Date:** 2026-10-25 · **Supersedes:** `T2I_REPORT_r3.md` (which keeps its SUPERSEDED banner and
stays as history, not as current fact)
**Model:** Stable Diffusion v1.4, revision `133a221b8aa7292a167afc5127cb63fb5005638b`
**Protocol:** `PROTOCOL_FROZEN_v1`, unchanged throughout. **631 attack measurements** (609 on SD-1.4 + 22 cross-model).
**Models:** SD-1.4, SD-1.5, BK-SDM-small.
**Corrections register:** `ERRATUM_r2_claim_corrections.md` (C1–C19),
`ERRATUM_lora_adversary_scope.md`, plus the same-day retractions in `FINDING59` and `FINDING62`.

---

## 1. Verdict

**Thirteen defences — eleven published or derived from publications, two of our own design — are
reversed to the base model's ceiling (max-over-valid H = 1.000) by 300 gradient steps on 16
self-generated images.** The mandate's bar (retain8 ≥ 0.85 with max-over-valid H ≤ 0.40) is **unmet
by every arm measured in this phase.**

| defence | origin | as-is target / retain8 | max-over-valid H |
|---|---|---|---|
| UCE | our re-impl | — | **1.000** |
| ESD | our re-impl | — | **1.000** |
| official AdvUnlearn | released weights | 0.400 / 0.992 | **1.000** |
| damage + distil | ours | — | **1.000** |
| `frag` Fisher allocation | ours | — | **1.000** |
| **FRP** (2608.25429) | our port | 0.1333 / 0.9417 | **1.000** |
| **Meta-Unlearning** λ=0.5/0.6/0.7 (2410.12777) | our port | 0.0000 / 0.6333–0.7000 | **1.000** (9/9) |
| **GIFT** β=1.0/1.2/1.3/1.35 (2507.13598) | our port | 0.0667–0.4667 / 0.6000–0.7833 | **1.000** (22 attacks, 4 families) |
| **IMMA** (2311.18815) | our port | 0.2667 / 0.8917 | **1.000** (22 attacks, 4 families) |
| weight-space-regularised erasure (2505.22310 idea) | our adaptation | 0.0667 / 0.9500 | **1.000** |
| **UTRAP** λ=0.15/0.3 × K=4/8/16 | **ours** | 0.0000 / 0.6583–0.8667 | **1.000** (18/18) |

**All five published competitors on the reading list are discharged** — four ported and run, one
(2505.22310) shown non-portable with its mechanism tested instead. **None has usable released code.**

## 2. The result is not "our defences were too weak to qualify"

This is the distinction r3 could not yet draw, and it is the strongest thing in this report.

**`UTRAP` λ=0.15/K=8 reached target 0.0000 at retain8 0.8667** — the concept fully suppressed on a
model that **clears the mandate's retain floor before any attack runs.** It was the best and last
shot at the bar. Six attacks, all gate-passing with T rising 0.000 → 0.958–1.000, **every one
reaching H = 1.000.**

> **The bar is not missed because our arms were too damaged to qualify. A fully qualifying arm was
> reversed anyway.** The retain half of the bar is reachable; the robustness half is not; and
> clearing the first does nothing for the second.

## 3. Four adversary families, down to 768 trainable scalars

`FINDING54` built the matrix for three defences; GIFT and IMMA now have all four filled in.

| adversary | trainable params | share of UNet | outcome on every usable arm |
|---|---|---|---|
| full-UNet relearning | 859.5M | 100% | **1.000** |
| DreamBooth (+ prior preservation) | 859.5M | 100% | **1.000** |
| LoRA rank-16, 16 cross-attn **output** projections | **0.399M** | **0.046%** | **1.000** |
| Textual Inversion | **768 scalars** | **0.000089%** | **1.000** |

A **768-scalar** embedding attack restores concepts that defences spent 44M–859M parameters
removing. (The LoRA scope was mis-described for most of this phase — it adapts only the 16 `to_out`
projections, not all of cross-attention. Corrected in `ERRATUM_lora_adversary_scope.md`; **every
number stands**, and the correction makes the finding stronger.)

## 3b. Cross-model: three checkpoints, two architectures, four method families

Added after r4's first publication (`FINDING64`, `FINDING56` cross-model addendum). The **identical**
pipeline and protocol on two models beyond SD-1.4 — `stabilityai/*` returns 401 on this host, so
SD-2.x/SDXL were unavailable and the test used what was reachable.

| defence | what it constrains | SD-1.4 | SD-1.5 | BK-SDM-small (**half-size, pruned**) |
|---|---|---|---|---|
| **FRP** | a weight subset (17.3% FF) | 0.1333 / 0.9417 → **1.000** | 0.1333 / 0.9333 → **1.000** | 0.4667 / 0.9250 → **1.000** |
| **GIFT** | a cross-attention subspace (5.11%) | 0.4667 / 0.7833 → **1.000** | 0.4000 / 0.8167 → **1.000** | 0.3333 / 0.7417 → **1.000** |
| **UTRAP** (ours) | the attacker's trajectory, full UNet | 0.0000 / 0.8667 → **1.000** | **0.0000 / 0.9583** → **1.000** | 0.0000 / 0.7917 → **1.000** |
| **Meta-Unlearning** | the loss surface at a simulated step | 0.0000 / 0.6333 → **1.000** | 0.0000 / 0.7667 → **1.000** | 0.2000 / 0.7667 → **1.000** |

**22 cross-model attacks, every one gate-valid, every one at H = 1.000.** The four families share no
mechanism, and the outcome is identical — the strongest available evidence that this is a property of
**the setting**, not of any defence design.

**`UTRAP` on SD-1.5 — target 0.0000 at retain8 0.9583 — is the best as-is operating point measured
anywhere in this work**: full suppression at retention *above* our FRP port's 0.9417 and 0.108 clear
of the mandate's floor. Reversed 3/3 with T = +1.000 each time. §2's logic therefore holds a third
time, on a second checkpoint.

**⚠ One of r4's own explanations had to be narrowed by this.** §4 below partly attributed the
failures to a **parameter-ratio** argument — small constrained subspace vs 859.5M-parameter
adversary. **BK-SDM-small halves the UNet and nothing changes**, from a *shallower* starting erasure
that should have been easier to defend. So **parameter abundance is not the operative variable**;
`FINDING57`'s redundancy account is. A correction to our reasoning, not to any measurement.

**Still inferred, not measured:** eight further defences cross-model, and any model with a different
latent space (SD-2.x/SDXL).

## 4. Mechanism: what predicts resistance, and what does not

Four independent attempts to find the operative variable, all negative:

- **Erasure depth does not predict resistance.** Arms from target 0.0000 to 0.4667 all reach 1.000.
  GIFT β=1.3 starts at half β=1.0's target and ends identically (`FINDING58`).
- **Weight-space proximity does not predict resistance.** L2 distance from base spans **1470×**
  across arms (GIFT 0.18 → FRP 264.53) with **zero** movement in the outcome. GIFT is
  indistinguishable from base in weight space (cosine ≈ 1 to six decimals) and is reversed on 22
  valid attacks (`FINDING60`).
- **Linear mode connectivity does not predict resistance.** `h68v2_rf` is genuinely *outside* base's
  linearly-connected basin (interior retain 0.358 vs 0.800 endpoint) and is reversed exactly like
  barrier-free GIFT (`FINDING60`, LMC addendum).
- **Unroll depth against the real attacker does not predict resistance.** K = 4 → 8 → 16 buys
  retention (0.658 → 0.733 → 0.758) and **no** resistance (`FINDING63` §3a). This falsified our own
  pre-registered hypothesis on its own terms.

**What the adversary actually does** (`FINDING57`): FRP's pruning mask is *decisive* before the
attack — a random mask of identical sparsity erases **nothing** (target 1.0000 vs FRP's 0.1333) —
and **irrelevant after it**: re-applying the entire mask to the attacked model removes nothing, and
a matched random mask behaves identically. **The adversary does not restore the removed route; it
relocates the function, leaving the defence's intervention intact and inert.**

**And the attack usually repairs the defence's collateral damage.** Across Meta-Unlearning, GIFT,
IMMA and UTRAP, post-attack utility commonly *rises* — up to **0.979** on the qualifying UTRAP arm.
The attacker ends with the concept fully restored **and a more capable model than the defended one
it started from.**

## 5. The one positive signal, and its exact scope

`FINDING56` §5a: Meta-Unlearning's trigger **does** obstruct Textual Inversion's *optimisation* —
TI gate-fails 3/3 on that arm with T *falling*, while succeeding on GIFT at **lower** retained
utility (0.6000 vs 0.6333) with the largest T rise in the phase. A utility-matched control
eliminates the obvious confound.

**Cross-model, the effect is GRADED** — neither checkpoint-specific nor universal. At matched
retention (0.7667 on both new models):

| model | TI 2000/lr 1e-2 | TI 3000/lr 2e-2 | TI gate |
|---|---|---|---|
| SD-1.4 | T **−0.208** | T **−0.083** | ❌ 0/3 — attack moves *backwards* |
| **BK-SDM-small** | T +0.125 ❌ | T +0.083 ❌ | ❌ 0/2 — rises but **stalls** below the gate |
| **SD-1.5** | T +0.000 ❌ | **T +0.208 ✅** | **breached** → H 0.333 |

It **holds on the pruned architecture and is breached on SD-1.5 at identical retention.** This also
**falsified a clause of §5a's own**: *"no attainable TI budget will produce one"* is true of SD-1.4
and **false of SD-1.5**, and is now scoped accordingly. ⚠ BK-SDM's T = +0.125 is an **invalid
attack, not "almost resistance"** — that distinction is exactly what `FINDING20` exists for.

**It is still not robustness** (a failed gate is not a held defence), and it is adversary-specific in
the wrong direction: it blocks the **768-scalar** attack and is irrelevant against the
**859.5M-parameter** one — full-UNet relearning reaches **1.000 on all three models**, leaving
BK-SDM at post-attack utility 0.938. That is the opposite of the threat ordering a defender wants.

## 6. ⚠ Process: eleven claims of mine corrected or withdrawn

r3 recorded eight. This phase added three more, **all within hours, all by escalations I had myself
declared as necessary**:

| claim | how it died |
|---|---|
| "IMMA partially resists — first non-ceiling number in the phase" (`FINDING59`) | my own 3× LoRA budget took it to 1.000 |
| "best operating point, beats FRP on both axes" (`FINDING62`) | two extra seeds showed my published build was the luckiest of three; target 0.1778 ± 0.1018 is *worse* than FRP's 0.1333 |
| "GIFT's usable window is a single β" (`FINDING58`) | a finer sweep found four |

**The pattern is mine and worth naming: I published before running the escalation and the seeds I
had already identified as required.** The rules caught all three; they should not have had to. The
`FINDING35` §4 ratchet and the n ≥ 3 seed rule are now written into `PREREG_H84` K4/K5 as
preconditions for publication rather than post-hoc checks.

Two further self-corrections of record: `FINDING44`'s mechanism was refuted by its own declared
diagnostic, and `FINDING61`'s null was traced to **my** unbounded erasure objective rather than to
the published idea it adapted.

## 7. Where the mandate stands

The mandate asks for a method outperforming the strongest relevant existing methods on a clearly
defined forgetting–utility trade-off, supported by real-model results.

- **On forgetting-under-attack: there is no such method here, ours or published.** Thirteen arms,
  **631** attack measurements, four adversary families, **three checkpoints, two architectures**, one
  unmet bar. I am not presenting anything as
  meeting it.
- **On the as-is trade-off there is one real, modest result:** `wsr2_church_lr100` holds retention
  **0.9639 ± 0.0127** (n=3) against our FRP port's 0.9417 at statistically indistinguishable erasure
  depth — a **reliable-retention** gain, explicitly narrowed after seed replication refuted the
  stronger "beats FRP on both axes" claim.
- **The defensible contribution of this phase is the negative result and its mechanism**, supported
  by real-model measurements on a frozen protocol, with every competitor run and every correction
  recorded: **concept erasure in SD-1.4 suppresses expression rather than removing capability; this
  holds across four adversary families down to 768 trainable scalars; neither erasure depth, nor
  weight-space proximity, nor linear mode connectivity, nor adversarial unroll depth predicts
  resistance; and a defence trained against the real multi-step attacker across the full parameter
  set, from a qualifying operating point, changes nothing.**

## 8. Scope and remaining gaps, stated as gaps

- **No never-trained oracle**, so *reactivation of residual memory* and *relearning from the
  attacker's data* remain indistinguishable (`ERRATUM_r2` C17 — the deepest open limitation).
  `FINDING44`/`FINDING57` attach mechanistic evidence without resolving it.
- **FID never run.** `FID_PROTOCOL.md` prices it at 5,000 COCO captions + Inception-v3 pool3
  weights; `torchvision`, `scipy`, `torchmetrics` and `cleanfid` are all absent from the venv and
  COCO is not on disk. A CLIP-feature substitute would not be comparable to the literature, so none
  was computed. **Declared gap with a price tag, not an omission.**
- ✅ **No longer SD-1.4-only.** `FINDING64` ran the identical pipeline on **SD-1.5** (0.1333/0.9333)
  and **BK-SDM-small**, a *half-size pruned* UNet (0.4667/0.9250): both reversed to **H = 1.000** on
  3/3 valid attacks each. The conclusion now holds **across three checkpoints and two
  architectures.** SD-2.x/SDXL remain untested (`stabilityai/*` returns 401 on this host), and only
  FRP was re-run cross-model — the other twelve defences' cross-model behaviour is inferred, not
  measured.
- ⚠ **§4's parameter-ratio reasoning is narrowed by that result.** Halving the UNet does not change
  the outcome, so **parameter abundance is not the operative variable**; `FINDING57`'s redundancy
  account is the better explanation. A correction to our reasoning, not to any measurement.
- **One concept carries most of the weight** (church), with parachute, van_gogh, garbage_truck and
  tench on subsets. **28 of 41 RNG sites use the global stream** (`provenance.py`). The CLIP
  evaluation classifier's revision is **unpinned**.
- **Every historical checkpoint from the pre-2026-10-24 work was deleted** before it was noticed, so
  those arms are re-derivable but not weight-verifiable. Three independent bridges and nine exact
  seed reproductions establish the rebuild instrument as faithful.
- **All five competitors are our ports.** Any disagreement with the authors' numbers is attributable
  to the ports or the domain change, **not** to their methods, and their published threat models
  differ from ours — GIFT's own DreamBooth setting was run for exactly this reason and reversed too.

## 9. Cost

`elapsed 67.6 h · gpu ≥ 30.55 device-h (reconstructed FLOOR — only 328 of 1088 result records carry
a runtime; see RESOURCE_NOTE.md) · 1088 result JSONs · 609 attack measurements · 137 GiB of 300 GiB
authorized`.
