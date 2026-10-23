# Pre-registration H47 — composite defence: adversarial text-encoder erasure + UNet robustification

**Date:** 2026-10-22 · **Status:** ❌ **H47 IS FALSE.** See the verdict box below.
**Mandate clause this targets:** *"Novel, non-obvious combinations of existing techniques are
eligible."*

---

> ### ❌ VERDICT: H47 IS FALSE. The composite is fully reversed under a stronger adversary.
>
> The composite **did** fix the utility collapse — retain **0.867** with the concept erased to
> 0.000, where the UCE teacher gave 0.458. That half of the hypothesis is confirmed and stands.
>
> **The robustness half is dead.** The first measurement read max-over-valid H = 0.533 from one
> valid attack of three, all at lr 1e-5. `H48` re-ran the arm with more seeds and a stronger
> adversary:
>
> | attack | gate | H after |
> |---|---|---|
> | sd4, lr 1e-5 | FAIL | 0.467 |
> | sd5, lr 1e-5 | FAIL | 0.333 |
> | **sd6, lr 1e-5** | **PASS** | **0.933** |
> | **sd0, lr 5e-5** | **PASS** | **1.000** |
> | **sd1, lr 5e-5** | **PASS** | **1.000** |
> | **sd0, lr 1e-4** | **PASS** | **1.000** |
>
> **Max-over-valid attacks = 1.000, not 0.533.** The composite is restored to the base model's own
> level, exactly like every other arm in this phase. The 0.533 was an artifact of too few seeds and
> too weak a learning rate — a *different seed at the same lr* already reaches 0.933.
>
> **This is the fourth time in this project that strengthening the adversary destroyed a result**
> (after the α=0.5 "fully robust" claim, the 18× advantage, and the concept-specificity signal).
> The pattern is consistent enough to state as a rule: in this project, a robustness number
> measured at one learning rate with fewer than ~5 valid attacks has never survived.
>
> **Consequences:** `PREREG_H40`'s wall is further supported — now by the composite too. The
> frontier table published in commit `048e91a` is **retracted**: its α=0.3 row (H 0.533) is wrong,
> and its α=0.5 row (H 0.267) was measured under the identical too-weak conditions and is being
> re-tested by `H50` before it is believed.

---

## 1. What the measurements say to do

Three results, all from this session, point at one combination:

| measurement | what it says |
|---|---|
| **H46:** official AdvUnlearn is fully reversed — H 0.400 → **1.000**, gate PASS, 2/2 seeds | the robust SOTA does not survive a weight-access fine-tuning adversary |
| **H45:** at **matched utility** damage+distil holds H at 0.467 where UCE goes to 1.000 | damage+distil contributes real robustness, 2.1× at the same utility cost |
| **H38 diagnosis:** the UCE teacher sat at retain 0.850 and damage dropped it to 0.317 | damage+distil's failure in T2I was a *utility* failure, inherited from a mediocre teacher |

AdvUnlearn's teacher is far better than anything I built: **church 0.400 at retain 0.992**, against
the UCE teacher's 0.067 at 0.850. The two methods also act on **disjoint** parts of the model —
AdvUnlearn on the text encoder, damage+distil on the UNet — so they are composable rather than
competing.

## 2. The hypothesis

> **H47.** Damage+distil applied to an AdvUnlearn-erased pipeline yields erasure that survives a
> gated relearning attack **at usable utility**, because the composite attacks the two halves of
> the failure separately: AdvUnlearn supplies erasure at high retain, and UNet damage+distil
> supplies the robustness AdvUnlearn lacks.

**Why it is not obviously decomposable.** H46 shows text-encoder erasure alone is reversed by
tuning the UNet; H38 shows UNet damage+distil alone destroys utility. The composite is interesting
precisely because each component covers the other's failure mode, and because the student keeps
the adversarially-trained encoder — so an attacker tuning the UNet faces a text representation
that no longer encodes the concept, *and* a UNet whose route back has been disrupted.

**Legitimacy.** AdvUnlearn's weights are public and the practitioner can run it. Distillation
targets are always the teacher's own outputs. No oracle, no concept images, no ground-truth labels.

## 3. Separating predictions, declared now

| outcome | reading |
|---|---|
| **retain ≥ 0.85 and max-over-valid H ≤ 0.50** | **H47 true.** Robustness at usable utility — the first result in either phase that meets the mandate's bar, and it **refutes `PREREG_H40`** by its own §3 condition. |
| retain 0.70–0.85 at H ≤ 0.50 | a real frontier improvement short of base utility; H40 weakened |
| retain ≥ 0.85 but H ≥ 0.80 | the composite inherits AdvUnlearn's reversibility; damage on top of a text-encoder erasure buys nothing |
| retain < 0.60 | the composite inherits damage+distil's utility collapse even from a 0.992 teacher, which would localise the collapse in the **damage**, not the teacher — and close the line |

## 4. Kill rules

**K1.** If α=0 (the no-damage control) is as robust as α>0, the damage contributes nothing and the
arm is just AdvUnlearn; report as such. This control is included from the start, not added later.

**K2.** Attacks must be **seed-replicated** (3 seeds minimum per arm) before any claim. `H43`
showed single-seed gate outcomes flip on the same arm, which already invalidated one correction I
nearly published.

**K3.** If the best arm's advantage over AdvUnlearn alone is < 1.5× on max-over-valid H at matched
utility, close the line rather than sweeping α further (inherits `PREREG_H38` K2).

## 5. Evaluation, fixed in advance

Three axes on every arm: church accuracy, retain8, and max-over-**valid**-attacks recovery under
`FINDING20`'s gate, 3 seeds at lr 1e-5. Utility compared against **base SD v1.4 = 1.000** and
against **AdvUnlearn alone = 0.992**, so "usable" means close to the teacher, not merely better
than my earlier broken arms.

Reported caveat carried forward: the 12-way closed-set CLIP metric is coarse and saturated at the
top (`T2I_BASELINES.md` §1), and my gate has three documented failure modes (ceiling artifact,
T-harder-than-H false negative, seed instability). Neither is fixed here; both bound what the
numbers can carry.

## 6. Novelty position, stated before results

Composing a text-encoder erasure with a weight-space robustification is, as far as the earlier
primary-source check reached, untested. A novelty check ran before any contribution claim, per the
commitment honoured for `H36`. **Its verdict, recorded in full:**

| component | verdict |
|---|---|
| stacking a text-encoder method + a UNet method as one composite | **NOT FOUND** (AdvUnlearn itself frames the module choice as mutually exclusive) |
| perturb-then-distil applied to a **diffusion** model | **NOT FOUND** — UNDO is LLM-only, and none of its ~18 citing papers touch diffusion, image generation or concept erasure. **Our strongest novelty claim.** |
| T2I erasure reversed by **weight-space fine-tuning** | **ANTICIPATED** — Meta-Unlearning 2410.12777, `Erased or Dormant?` 2505.16174 |
| retain-Fisher-weighted parameter **noise** in diffusion unlearning | **NOT FOUND** (PAIR 2602.05339 uses Fisher for LoRA *initialisation*; SalUn 2310.12508 *masks* rather than perturbs — abstract-level only) |
| prompt-disjoint split + an **attack-validity gate** | split **PARTIAL** (Meta-Unlearning already uses one); the gate requirement **NOT FOUND** |

**Overall: not anticipated as a whole, but the motivation is fully anticipated and the novelty is
narrow.** The framing that justifies this work — *T2I erasure, including adversarially-trained
erasure, is reversed by malicious fine-tuning, so weight-space immunisation is needed before
release* — is exactly Meta-Unlearning's thesis, and also IMMA's (2311.18815) and GIFT's
(2507.13598). What is defensibly ours reduces to (i) transferring UNDO's mechanism to diffusion,
(ii) the retain-Fisher-weighted perturbation, (iii) stacking across disjoint modules.

**⚠ A competitive-baseline gap that must be stated in any claim.** Meta-Unlearning, IMMA and GIFT
are three *existing* defences for precisely this threat model, and **none of them has been run
here**. Any contribution claim without that comparison is not credible, and this phase does not
have it. `H47` is therefore positioned as a mechanism transfer with a measured operating point,
never as a SOTA claim.

**Nearest mechanisms, and why they differ:** GIFT noises in **representation space** inside a
bi-level objective, with no distillation-repair stage; IMMA learns a bad **initialisation** via
MAML-style bi-level optimisation. Neither damages weights with random noise and repairs by
self-distillation from the erased teacher.
