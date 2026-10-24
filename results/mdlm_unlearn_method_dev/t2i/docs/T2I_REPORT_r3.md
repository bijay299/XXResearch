# Text-to-image concept erasure — phase report, revision 3

**Date:** 2026-10-24 · **Identifier:** `T2I_REPORT_r3` · **Supersedes:** `T2I_REPORT_r2`
**Model:** Stable Diffusion v1.4, pinned `133a221b8aa7292a167afc5127cb63fb5005638b`
**Mandate:** an unlearning algorithm that beats the strongest existing methods on a clearly defined
forgetting–utility trade-off, with real-model results.

Earlier revisions are preserved. Corrections are itemised in `ERRATUM_r2_claim_corrections.md`
(C1–C19); findings 39–55 postdate r2.

---

## 1. Verdict

**The mandate is not met.** No arm — ours or published — reaches retain ≥ 0.85 with max-over-valid
post-attack H ≤ 0.40, under any of four adversary classes.

The verdict now rests on the **best** arms the work produced, not weak ones:

| arm | target (erased) | retain8 | adversaries | **max-over-valid H** |
|---|---|---|---|---|
| `frp_garbage_truck_r3` | **0.1333** | **0.9667** | 3 classes, 16 valid attacks | **1.000** |
| `frp_garbage_truck_r5` | **0.0000** (complete) | 0.7750 | 2 classes, 8 valid attacks | **1.000** |

`frp_garbage_truck_r3` is the phase's best forgetting–utility point and the only arm to clear the
0.85 utility floor at that erasure depth. It is reversed to the ceiling by full fine-tuning (6/6),
Textual Inversion (5/5) and DreamBooth (2/2). `frp_garbage_truck_r5` achieves **complete** erasure
and the relearning attack takes it 0.000 → 1.000 on 3/3 seeds with T rising a full **+1.000**.

**And the headline is ANTICIPATED** — Meta-Unlearning (2410.12777) and `Erased or Dormant?`
(2505.16174) reported T2I erasure reversed by malicious fine-tuning first. This phase is a
confirmation with the adversary surface widened, not a discovery.

## 2. The adversary matrix

Max-over-valid H, computed from the raw JSONs. Every cell is measured; none estimated.

| adversary | trainable params | % of UNet | AdvUnlearn | FRP | damage+distil |
|---|---|---|---|---|---|
| full-UNet fine-tuning | 859 520 000 | 100 % | 1.000 (c, p) | 1.000 (c, p, gt) | 1.000 (c, p) |
| DreamBooth, prior-weight 0 | 859 520 000 | 100 % | 1.000 (c, p, vg) · 0.733 (gt) · 0.800 (t) | 1.000 (c, p, gt) | 1.000 (c) |
| LoRA rank 16 | 399 000 | 0.046 % | 0.867 (c) · 1.000 (p, vg) | 1.000 (c, p) | 1.000 (c) |
| **Textual Inversion** | **768** | **0.000089 %** | **1.000 (c, p, gt, t)** | 0.933 (c) · 0.667 (p) · 1.000 (gt) | 0.933 (c) · 1.000 (p) |

c = church, p = parachute, vg = van_gogh, gt = garbage_truck, t = tench. Per-cell n is 2–9 valid
attacks; the full per-cell counts are in §7.

> **Four adversary classes spanning six orders of magnitude in trainable parameters, three defence
> families, five concepts: no defence resists an adversary that is not handicapping itself.**

## 3. The strongest single result

**Textual Inversion — 768 trainable scalars, 0.000089 % of the UNet, no weight changed — reverses
the official robust SOTA completely.**

| concept | valid attacks | H |
|---|---|---|
| church | 3 | 1.000 ×3 |
| parachute | 2 | 1.000 ×2 |
| garbage_truck | 3 | 1.000 ×3 |
| tench | 3 | 1.000 ×3 |

**11 valid attacks, 4 concepts, zero variance.** van_gogh is **in flight** and is not counted.

⚠ **A correction to my own reporting:** a status update described this as "14/14 across all five
concepts". That was wrong — no TI run on van_gogh existed, and I had misread the LoRA van_gogh
label (`H71V`) as a TI run. The verified figure is **11 valid attacks across 4 concepts**. The
missing cell is now running.

AdvUnlearn is adversarially trained in **prompt space** and claims robustness there; a learned token
is a prompt-space attack. This is a scope statement about their threat model, not a refutation of
their paper's experiments (`ERRATUM_r2` C15).

## 4. The methodological contribution

**A defence can look robust against an adaptation method purely because the method's published
default includes a regulariser the attacker has no reason to keep.**

Official AdvUnlearn under DreamBooth at the default prior weight capped at **0.667** on both object
concepts, across **8 valid attacks each**, with seeds, steps, learning rate and prior weight all
varied — and raising the learning rate *lowered* recovery, so the attack was at its optimum. It read
as a genuine defence property and was within a day of being published as one.

At **prior-weight 0** it reaches **1.000**, while costing the attacker only **2–8 points of
utility** (panel accuracy 1.000 → 0.917/0.938). The prior term protects a class prior the attacker
does not need.

**IMMA (2311.18815), Glaze and MIST are all evaluated against DreamBooth.** A default-configured
DreamBooth evaluation understates the adversary by **0.33 in recovery**. Any robustness claim
against DreamBooth should state the prior weight and include `prior_weight → 0`.

## 5. Other results that stand

- **Attacker cost is arm-dependent by ~10×.** FRP falls at 20–30 steps; official AdvUnlearn needs 50
  for a *valid* attack and ~300 for the ceiling, at higher retain. AdvUnlearn's adversarial training
  buys real cost (`FINDING52`), which deserves saying alongside the negative.
- **The retracted "1 image / 50 steps" claim is now disconfirmed by measurement**, not merely voided:
  50 steps with 16 images gives H 0.533.
- **The concept is not in any small weight subset.** Re-zeroing FRP's pruned mask in an attacked
  model leaves the target at 1.000 — as does a random matched-count set, so the honest claim is that
  *no* 3 % subset holds it (`FINDING46`, `FINDING51`).
- **Post-attack, the concept is harder to remove than the base model's own**, at matched utility,
  across three defence families (`FINDING48`, `FINDING51`).
- **Selectivity is exhausted.** retain-sparing only (utility fine, no robustness), forget-targeting
  only (destroyed), and the intersection (destroyed) — the last on a norm-matched control
  (`FINDING45`).

## 6. ⚠ Process: eight claims of mine dissolved under their own controls

Each was caught by a check queued **before** publication, so nothing wrong entered the record — but
the pattern is mine and it is the most important thing a reader should know about how to weigh
sub-ceiling numbers here.

| claim | how it died |
|---|---|
| general entrenchment (`FINDING46`) | one arm, generalised; withdrawn then reinstated at matched utility |
| "ranking inverts with adversary" | LoRA gap was two images on one concept |
| "erasure site predicts TI vulnerability" | refuted by its own test — damage+distil is UNet-space and still 0.933 |
| FRP's 0.667 TI cap | broke to 0.933 under a stronger TI |
| entrenchment threshold (`FINDING49`) | was the low end of a gradient, hidden by a saturated metric |
| AdvUnlearn's DreamBooth cap | the attacker's own regulariser (§4) |
| "TI 14/14 on five concepts" | **4 concepts; van_gogh never run** (§3) |
| partial resistance, general | every instance traced to adversary strength or metric artifact |

**The operative lesson: `max-over-valid` rises with attack strength, so every sub-ceiling number is
provisional until the adversary has been pushed along the axis specific to its formulation** — which
for DreamBooth was its loss, not its budget.

## 7. Scope and remaining gaps

- One model (SD v1.4), five concepts, our own prompt splits, a 12-way closed-set CLIP metric
  quantised to 1/15 with a ceiling at 1.000.
- UCE, ESD, FRP, LoRA, TI and DreamBooth are **our implementations**; only AdvUnlearn is an official
  checkpoint. λ in FRP is a value the paper never assigns.
- **FID is not reported.** The 5 000-image protocol is written and costed in `FID_PROTOCOL.md`; its
  declared trigger (an arm reaching the bar) never fired.
- **The CLIP classifier revision was never pinned**, and 28 of 41 stochastic sites use the global RNG
  stream. Both disclosed, not patched (`provenance_manifest.json`).
- **No never-trained oracle**, so reactivation and relearning remain indistinguishable
  (`ERRATUM_r2` C17). This is the deepest limitation.
- **Three competing defences remain unrun** — Meta-Unlearning (2410.12777), GIFT (2507.13598), From
  Dormant to Deleted (2505.22310) — **none with usable released code**. That is still the binding
  constraint on any contribution claim, and it has not moved.

## 8. Cost

`elapsed ~99 h · gpu ~99.5 device-h · cpu ~336 core-h · attempts 339 · storage 21.0 GiB of 60`
Two GPUs maximum throughout; 492 attack records; 23 findings.
