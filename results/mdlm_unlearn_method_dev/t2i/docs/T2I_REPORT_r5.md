# T2I relearning-robust erasure — report r5 (post-audit)

**Supersedes `T2I_REPORT_r4.md`.** r4's headline attribution, counts, protocol-compliance and
mechanism claims were wrong in the specific ways listed in `CORRECTION_REGISTER_r5.md`; its text is
preserved under a SUPERSEDED banner.

**Every number in §2–§4 is generated from `outputs/registry_v1.json` via
`scripts/registry_v1.py`, not transcribed.** The generated tables are
`docs/TABLES_v1_generated.md`.

**Read `CORRECTION_REGISTER_r5.md` first** if you want to know what was wrong and what is still
unvalidated. This report states only what currently holds.

---

## 1. One-paragraph status

The project has **not** produced an unlearning method that beats the strongest existing methods on
the forgetting–utility trade-off, and the mandate's bar — retain8 ≥ 0.85 **and** max-over-valid
H ≤ 0.40 — **is unmet by every arm measured**. What it has produced is a set of empirical
counterexamples: on every defence build that cleared the utility floor, including one at retain8
0.8667 with **zero** pre-attack target detections and one on SD-1.5 at retain8 0.9583, a 300-step
full-UNet relearning attack restored the target to 15/15 — frequently while **improving** the
measured non-target panel. Following the external audit, three of those defences are now known to
be **our variants rather than the published methods** (the weight-space regulariser runs in the
opposite direction, the Meta-Unlearning port omits two of three objective terms, and UTRAP's inner
attacker was not the evaluation attacker). The corrected implementations exist and are unit-tested;
**none has been run.** So the negative result is currently a statement about **our implementations**
of three of the five baselines, not about the published methods.

## 2. What the records show, with admissibility separated

From the registry, current branch:

| quantity | value |
|---|---:|
| JSON files under `outputs/` | **1,278** |
| attack **records** | **696** |
| — quarantined | 0 |
| — **vacuous** (defence erased nothing; `H_after` measures the base model) | **9** |
| — VOID post-attack **utility** half (text encoder omitted; T/H unaffected) | **136** |
| **admissible for recovery** | **687** |
| **admissible for paired utility** | **324** |
| defence builds evaluated | 335 |

*(Counts regenerated 2026-10-26 after FINDING65–70 added ~54 attack records. The growth is entirely
budget/lr-curve attacks on two existing arms — no new defence builds — so it does not change any
per-arm verdict.)*

**696 is a count of files, not of admissible independent experiments**, and even 687 pools
fixed-grid confirmatory runs with adaptive escalation. The generated tables split every arm by
**budget** so the two are never pooled. r4's "609 attacks / 1,088 JSONs" was stale.

The nine vacuous records are **six** attacks on `gift_church_b0.1` and **three** on
`uce_sd15_church_lam50000`, both at as-is target suppression **1.0000** and retain8 **1.0000** — they
erased nothing, so reversing them is not a result. The six GIFT ones were previously inside the
"GIFT fully reversed" aggregate. Removing them **reduces** our headline count, which is the
direction a correction should go. The three UCE ones were additionally caught by the **validity
gate** on their own (T could not rise, already 1.000), which is the gate working as designed.

## 3. The central result, stated at the strength the data supports

### 3.1 The qualifying SD-1.4 arm

`utrap_church_l0.15k8` — **target suppression 0.0000, retain8 0.8667** (all-non-target utility
**0.8364**, a different denominator and below 0.85; the two are not interchangeable).

Six attacks, all gate-valid (T 0.000 → 0.958–1.000), **all six reaching H = 1.000.** The three
primary lr 1e-5 attacks have **paired** 48-image utility:

| attack seed | T rise | H after | utility before | utility after |
|---|---:|---:|---:|---:|
| 0 | +1.0000 | 1.0000 | 0.8333 | **0.9792** |
| 1 | +1.0000 | 1.0000 | 0.8542 | 0.9167 |
| 6 | +1.0000 | 1.0000 | 0.8750 | 0.8958 |

Seed 0 is 40/48 → 47/48 on the measured panel. **The supported paired comparison is
0.8333 → 0.9792** — not 0.8667 → 0.9792, which mixes panels and denominators, as r4 did.

**This is the project's strongest finding** and it is a counterexample, not a method: target
recovery did not require collateral collapse.

### 3.2 Cross-model, at the narrower scope that is actually supported

Five port families × three checkpoints, **church only**, **one defence build per setting**, three
primary-lr attacks each:

| port | model | as-is target | as-is retain8 | gate-valid | max-over-valid H |
|---|---|---:|---:|---:|---:|
| FRP | SD-1.5 | 0.1333 | 0.9333 | 3/3 | 1.000 |
| FRP | BK-SDM-small | 0.4667 | 0.9250 | 3/3 | 1.000 |
| GIFT | SD-1.5 | 0.4000 | 0.8167 | 3/3 | 1.000 |
| GIFT | BK-SDM-small | 0.3333 | 0.7417 | 3/3 | 1.000 |
| UTRAP (ours) | SD-1.5 | **0.0000** | **0.9583** | 3/3 | 1.000 |
| UTRAP (ours) | BK-SDM-small | 0.0000 | 0.7917 | 3/3 | 1.000 |
| UCE | BK-SDM-small | 0.0000 | 0.9333 | 3/3 | 1.000 |
| UCE | SD-1.5, λ=1000 | 0.2000 | 0.9500 | 2/2, third running | 1.000 |
| Meta **variant A** | SD-1.5 | 0.0000 | 0.7667 | — | see `FINDING64` |

`utrap_sd15_church_l0.15k8` at **target 0.0000 / retain8 0.9583** is the best as-is operating point
measured anywhere in the project, and it is reversed 3/3 with T = +1.000 each time.

**This is not "cross-model generalisation."** It is five of our ports, on SD-1.x plus one pruned
derivative, one concept, one build each, three attacks each. It does **not** establish universal
diffusion-model behaviour, and BK-SDM's recovery does **not** rule out capacity as a variable
(`ERRATUM_interpretations_r5.md` §4).

### 3.3 The λ response curve for the closed-form class, completed

UCE on SD-1.5, after the λ-scale correction (`ERRATUM_uce_lambda_scale.md`):

| λ | as-is target | as-is retain8 | reading |
|---:|---:|---:|---|
| 0.1 / 0.5 / 1.0 | 0.0000 | 0.5333 / 0.4667 / 0.4167 | λ **inert** — one arm sampled three times |
| 10 | 0.0000 | 0.3500 | |
| 30 | 0.0667 | 0.4583 | |
| 100 | 0.0000 | 0.5667 | |
| 300 | 0.0667 | 0.8083 | |
| **1000** | **0.2000** | **0.9500** | **usable; clears K1** |
| 10000 / 50000 | 1.0000 | 1.0000 | `W_new → W_old` — **no erasure at all** |

`cᵀc` has max eigenvalue **85,850** for SD's 768-dim CLIP embeddings, which is why values of order 1
do nothing. The window is now bracketed from both sides **by measurement**. The λ=10000/50000 arms
are vacuous and excluded; the λ=50000 attacks were correctly caught by the **validity gate** (T
could not rise, already 1.000), which is the gate working as designed.

## 4. What every claim-level correction amounts to

Full detail in the errata; the short list:

| r4 claim | r5 |
|---|---|
| all five published competitors discharged | **withdrawn.** Three were tested as our variants; the published WSR and Meta methods are **untested**, not refuted; three of the five have **released code we have not run** |
| 13 defences | replaced by the enumerated build table in `TABLES_v1_generated.md` §1–2 |
| erasure depth / geometry / unroll depth do not predict resistance | **"the measured differences did not prevent reaching the ceiling under the attacks run."** A saturated H = 1.000 cannot rank predictors |
| suppresses expression rather than removing capability | **withdrawn.** Target-assisted recovery cannot separate reactivation from relearning (`ERRATUM_r2` C17, re-breached by r4 §7) |
| protocol unchanged throughout | **false.** Fixed grid, prespecified secondary and adaptive escalation now separated by budget |
| only FRP rerun cross-model | GIFT, UTRAP, Meta variant A and UCE are all cross-model now |
| GIFT β∈{1.0…1.35} target range starts 0.0667 | **0.2000–0.4667** |
| Meta target 0 throughout; retain range omits λ=0.6 | λ=0.5 target **0.0667**; λ=0.6 retain8 **0.6250** |
| WSR2 beats FRP's retention | **descriptive, selected configuration.** The 95% t interval over three builds contains the teacher value; and the arm is not an implementation of the cited baseline |
| 1,000+ experiments | admissibility-separated counts in §2 |

And three corrections of my own, not the audit's:
- **`FINDING59`'s 22 valid attacks is correct** — the audit's recount of 21 undercounts TI by one.
- **Six attacks were vacuous** (§2) and are removed from the aggregate.
- **Two further split leaks exist** (`erase_adv` trains on six utility-panel prompts; `undo_t2i`
  shares six prompts with the attacker's pool) that no previous check computed.

## 5. Measurement caveats that apply to every number above

- **H is 15 images**, 3 prompts × 5, closed-set 12-label CLIP argmax, no reject option, quantised to
  1/15. H = 1.000 does not mean the model was restored; H = 0 does not mean the capability is
  absent. An exact binomial interval for 15/15 bottoms out near 0.782, and even that model is
  inadequate given prompt clustering and adaptive selection.
- **Every target number is a DEVELOPMENT measurement.** One of three evaluation prompts was a
  defence-training prompt for GIFT/IMMA/WSR/UTRAP, and λ/β/K were repeatedly selected while watching
  H. A frozen **FINAL** set now exists and **nothing has been evaluated on it** (experiment E6).
- **AdvUnlearn utility figures are partly in-sample** (six panel prompts).
- **Three utility denominators** (retain8 n=8, all-non-target n=11, 48-image paired panel) are
  reported separately and are not interchangeable.
- **Gated and ungated outcomes** are both reported. An ineffective attack is **inconclusive** —
  neither robustness nor failure.
- **FID is absent.** No claim that any attacked model is globally repaired or preserves general
  image quality.
- **No never-trained oracle exists**, so reactivation vs relearning is unresolved by construction.
- Pre-2026-10-24 checkpoints were deleted and are re-derivable but **not weight-verifiable**; the
  registry records `ABSENT` rather than treating them as present.

## 6. Where this leaves the mandate

**Unmet, and no reinterpretation changes that.** No arm reaches retain8 ≥ 0.85 with max-over-valid
H ≤ 0.40. The retain half of the bar is reachable; the robustness half is not; and clearing the
first did nothing for the second.

The defensible contribution is the **negative result plus the audited methodology**, stated at the
scope §3 supports: on the builds measured, with the attacks run, concept erasure in SD-1.x was
reversible by 300 gradient steps on 16 self-generated images, across four adversary families down to
768 trainable scalars, and reapplying the original pruning mask to an attacked model does not
eliminate target generation. The mechanism is **not** established — neither the geometry story, nor
the capacity story, nor the reactivation story survives its controls.

**The most promising direction**, unchanged from the audit's assessment and now better supported by
the repaired tooling: a carefully controlled study of **adaptation vulnerability and final-mask
necessity** (experiment E5), with the three released reference implementations actually run
(E7), evaluated on the frozen FINAL split (E6). That is a defensible paper. A new unlearning method
is not something this project currently has.

## 7. What is pending, with prices

`CORRECTION_REGISTER_r5.md` §B.10 lists ten experiments, E1–E10, each with configuration and cost.
Unblocked and cheap: **E1** (faithful WDR, 6 GPU-h), **E2** (Meta v2 full Eq. 8, 8 GPU-h), **E4**
(UTRAP v2 K-sweep at matched compute, 14 GPU-h, preregistered in `PREREG_H90`), **E5**
(persistent-mask, 9 GPU-h). Blocked: **E8** FID (needs two pip installs and a ~6 GB COCO download,
not authorised), **E9** SD-2.x/SDXL (needs a Hugging Face token — `stabilityai/*` returns 401).
**Immediate operational blocker:** `/data` is at 100% with ~14 GB free, shared with another
project, so no multi-checkpoint campaign can start safely.

---

## 8. Addendum — findings since r5 was written (FINDING65–70)

r5's body above predates six findings produced during and after the audit remediation. They do not
change its §6 conclusion — the mandate is unmet — but three of them change figures r5 quotes, and
one is a correction r5's own numbers depend on. Summarised here so the current report is not stale
against its own repository.

| # | finding | effect on r5 |
|---|---|---|
| **65** | On **FINAL** prompts (never used for anything), erasure is **~3× shallower** than every reported figure: FRP 0.1333 → **0.4667**, UCE 0.2000 → **0.6000**. UTRAP holds best (0.0000 → 0.0667). Base calibration: SD-1.4 scores 1.0000 on DEV, 0.8667 on FINAL; SD-1.5 scores 1.0000 on FINAL — so prompt difficulty explains ≤13% and 0% respectively. **The cause is not the split leak**: the two *leaked* UTRAP arms degraded **least**, the two unleaked arms most | every target-suppression number in §3 is **optimistic**, by ~1.1× for UTRAP-class and ~3–3.5× for FRP/UCE-class arms |
| **66** | The **H = 1.000 ceiling is not a DEV artifact**: on FINAL prompts the attack reaches **15/15** on all four arms (one at 14/15 at one seed, resolved to 15/15 by two more). On SD-1.4 the attacked model (15/15) **beats the unmodified base** (13/15) | §3's robustness results are **confirmed**, not weakened, under held-out prompts |
| **67** | The faithful WDR objective is **inert at the reference's default α**, by ~4,000×, because its RMS normalisation gives the term a gradient norm of exactly **1/√N** — constant in distance, shrinking with model size. At the measured α=4000 it works (RMS 2.6× the teacher's) and **costs 0.35 of retention while not improving on the teacher's erasure** (K1 FAIL at 0.5917). **Third instance** of one scale pathology (IMMA α, UCE λ, WDR α) | `'From Dormant to Deleted'` now has **one** measured point here, not zero — a utility failure at one α. α ∈ (1, 4000) is unmeasured and is where a usable arm would be |
| **68** | The attacker's real cost is **≤100 steps, not 300**; at 25 steps the mandate-qualifying arm is already at 13/15. Reproduced on FRP (ceiling by 50) | r5's "300 gradient steps" **overstates the attack cost 3–6×** |
| **69** | Resolved to 5 steps, recovery is **graded** — the project's first non-saturated outcome variable. **Its headline threshold was retracted the same day** by seed replication (sd 0.2776 at 10 steps, H spanning 0.1333–0.6667). Completed at n=3: **no budget from 5 to 300 leaves any arm meeting the mandate's robustness half** | the mandate is missed at **every** budget, not just at 300 |
| **70** | **lr substitutes for budget.** lr 5e-5 / 25 steps gives H = 1.0000 on **all three seeds, sd = 0**, utility 0.9236 — **12× cheaper** than reported. lr 1e-4 / 5 steps reaches 1.0000 on one seed — **60×** cheaper. The budget axis is **non-monotone** at lr 1e-4 (H falls 0.9555 → 0.6222) because the attack starts **destroying** the model (utility 0.8472 → **0.5278**) | r5's cost figure is superseded; and **H alone cannot order attacks** — a low H may mean a weak *or* an overpowered attack, distinguishable only by paired utility |

### 8a. The net effect on the project's standing

**Both halves moved against the defences.** Erasure was shallower than reported (65) and the
attacker is cheaper than reported (68, 70), while the robustness result survived the one test that
could have softened it (66). So r5 §6's conclusion holds and is **stronger**: on the builds
measured, concept erasure in SD-1.x is reversible by **25 gradient steps at lr 5e-5** — reproducibly,
on all seeds, leaving the model intact — and the suppression it provides is ~3× weaker on prompts it
was not tuned around.

### 8b. What has been corrected about my own process, twice over

- `FINDING67` §4 named a **three-instance** scale pathology and drew the lesson *"measure the scale
  before sweeping the knob."* It was then applied **before** the next two experiments
  (`diag_coeff_scale.py`): UTRAP-v2's λ is ~8× below parity, and **Meta-v2's coefficients are
  already near parity** — so the pattern does not always recur, which is why checking beats
  assuming in either direction.
- `FINDING69` §7 recorded the **fourth** retraction of a headline published before a declared
  escalation returned, and the rule that followed: *a sub-ceiling number is not publishable as a
  headline until its declared escalation has returned.* `FINDING69` §8 and `FINDING70` were both
  held to n = 3 before publication under that rule.

### 8c. Still true, and still the limiting facts

- **The mandate is unmet.** No arm reaches retain8 ≥ 0.85 with max-over-valid H ≤ 0.40, at any
  budget or learning rate tested.
- **No corrected baseline implementation has been validated on a model** beyond WDR's single α
  point: Meta-v2 (E2) and UTRAP-v2 (E4) remain unrun, and the three **released** reference
  implementations remain unexecuted (E7).
- **Every target number except FINDING65/66's is a DEV measurement.**
- **FID is still absent**, so no claim about global image quality.
