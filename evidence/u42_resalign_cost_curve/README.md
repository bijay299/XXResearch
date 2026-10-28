# U42 — ResAlign buys the attacker ~100x, and nothing beyond that

**2026-10-28.** U40 established that ResAlign's released checkpoint genuinely holds un-attacked
(US 0.0000 over 600 images, base control 0.1817). U41 showed it survives a targeted attack at the
25-step budget that drives every other arm to full recovery. This sweeps the budget to find where
it breaks.

Attacker: full fine-tune, lr 1e-4 (the rate that recovered most at 25 steps). Trained on base
generations of 40 `unsafe.csv` prompts, scored on 60 prompts it never saw.

## The curve

| budget | US after attack | % of base rate | valid | max abs dW |
|---:|---:|---:|---|---:|
| 25 | 0.0333 | 18.3% | yes | 2.16e-03 |
| 100 | 0.0333 | 18.3% | yes | 5.64e-03 |
| **300** | **0.1333** | **73.4%** | yes | 1.04e-02 |
| **1000** | **0.2500** | **137.6%** | yes | 1.97e-02 |

Base SD-1.4 control: **0.1817**. Every cell valid: probe loss fell and weights moved far past the
fp32 floor.

## Reading

**ResAlign is a cost, not a guarantee.** It holds flat through 100 steps, degrades by 300, and is
fully broken by 1000. Against the other methods measured here -- three of four undone by a **single**
gradient step, the fourth by four -- that is roughly **100x the attacker budget**.

That is exactly the shape the mandate accepts: *an explicitly measured trade-off*. It is not a claim
of permanence and should never be written as one.

## At 1000 steps the attack overshoots base, and that corroborates an earlier finding

0.2500 against an undefended 0.1817 is **138% of base**. The attacker does not merely restore the
concept; it writes it in more strongly than pretraining did.

This project documented that phenomenon independently -- **entrenchment**, FINDING46 through
FINDING51 -- on church arms, with our own 12-way CLIP metric, on our own re-implementations.
Reproducing it here on **an author-released checkpoint, a different concept, and a different
detector** is far stronger corroboration than anything in the original thread, which reversed itself
three times before settling.

## Why this strengthens the programme's central claim rather than contradicting it

The claim was never "nothing can resist." It is that **post-hoc weight-level erasure does not
survive relearning because the concept is not separable from the retained set**. ResAlign does not
escape that: it does not make the concept unreachable, it makes the path back longer. The capability
is still there, and 1000 steps finds it -- and then some.

What ResAlign demonstrates is that the *binding* can be made expensive to restore without the
*capability* being removed. That is a fifth family, and the distinction is the paper's sharpest
point.

## Limits

- One checkpoint (higher-safety); the higher-utility release is downloaded but not swept.
- One learning rate at budgets above 25, chosen to favour the attacker.
- n=1 per budget point. The 25-step row has n=3 across learning rates and is the only replicated one.
- Their IP metric (i2p + NudeNet) is not reproduced; NudeNet is not installed here.
- The post-attack panels have **not** been image-audited. Their detector is a trained binary
  classifier rather than the closed-set argmax whose defect U29/U30 documents, so the specific
  failure mode we found elsewhere does not apply -- but it has not been positively checked either.
