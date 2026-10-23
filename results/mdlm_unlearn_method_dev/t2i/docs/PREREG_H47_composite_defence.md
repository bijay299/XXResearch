# Pre-registration H47 — composite defence: adversarial text-encoder erasure + UNet robustification

**Date:** 2026-10-22 · **Status:** pre-registered. The arms are **launched but unread** as this is
written; H46's remaining seeds are also unread.
**Mandate clause this targets:** *"Novel, non-obvious combinations of existing techniques are
eligible."*

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
primary-source check reached, untested — the T2I erasure literature treats these as alternative
methods rather than stackable layers, and UNDO's perturb-then-distil has not been applied to T2I at
all. A novelty check will run before any contribution claim, and an ANTICIPATED verdict will be
reported as a reproduction regardless of performance, as was honoured for `H36`.
