# Finding 7 — R1 has fired: the oracle is not in the good cell, and retain SFT makes it worse

**The pre-registered A2 gate, decided.** Six-cell partition of each model's own generations on
the same 400 forget QA (CPU re-scoring of preserved generations, no extra training):

| model | DISCLOSE | DEGEN | ABSTAIN | **REPLACE** | IGNORANT | IGN+ABS |
|---|---|---|---|---|---|---|
| pristine base | 0.030 | 0.000 | **0.113** | 0.470 | **0.388** | **0.500** |
| θ₀_tofu (memorized) | 0.900 | 0.000 | 0.000 | 0.040 | 0.060 | 0.060 |
| **retain_ref_tofu (the oracle)** | 0.033 | 0.120 | **0.000** | **0.580** | 0.268 | **0.268** |

**Pre-registered decision: mostly REPLACE → do not build the match-the-retain-reference
mechanism.** Risk R1 in `RESEARCH_PLAN.md` has fired, which retires the project's
highest-priority risk — negatively.

## Three things that make this stronger than one number

**1. Retain SFT made the model worse than not training at all**, in exactly the dimension that
matters: REPLACE 0.470 → 0.580, IGNORANT 0.388 → 0.268, and **ABSTAIN 0.113 → 0.000**.
Training on retain data taught confident TOFU house style and **destroyed the only abstention
behaviour present**. The thing we were going to distil toward is a worse ignorance model than
the untrained one.

**2. The 0.580 is a lower bound.** 37 of the 48 DEGENERATE rows open with a confident wrong
specific before collapsing. Counting "asserts a wrong specific value" directly: base 0.470,
θ₀ 0.040, **retain_ref 0.672** — two thirds of forget probes.

**3. Robust to the declared boundary.** Across τ ∈ {0.25, 0.50, 0.75}, REPLACE reads
0.477 / 0.580 / 0.598 and is the plurality cell at every τ.

Verbatim, for hand-check: `"The author's full name is Yihong Chen"` (gold Hsiao Yun-Hwa);
`"father is a skilled Mason"` (gold civil engineer).

## What this kills

- **C1 / oracle matching as our base.** Its ceiling is the oracle's own cell, and the oracle
  substitutes on two thirds of probes.
- **SIO**, which was specified as C1 + selective interdiction + orthogonalisation + hinge. Its
  base is gone. (H11 was independently closed as category (c), already run by TIF as
  `NPO-GPT` and reported insufficient.)

**The pristine base is a better ignorance model than the retain-trained one** (ABSTAIN 0.113
vs 0.000, IGNORANT 0.388 vs 0.268). But distilling toward the pristine base is only possible
because of our inject-then-unlearn construction — it amounts to undoing the SFT, is
unavailable for knowledge that arrived in pretraining, and is therefore not a method. Stated
so it is not mistaken for one later.

## Two gates that nearly went the wrong way

**A metric defect would have selected a different branch.** Under the frozen 3-gram repetition
rule, retain_ref reads DEGEN 0.468 / REPLACE 0.352 — plurality DEGENERATE, i.e. the
"warm-up is broken, fix it first" branch. The trigger is the author's own 3-token name
repeated *while declining to answer*. The corrected rule requires the top 3-gram to cover
≥0.30 of positions, a threshold sitting in an empirically empty gap: base false positives top
out at 0.27, retain_ref's genuine cases start at 0.32 (median 0.96), hand-verified.

**The forgetting metric decided the verdict.** retain_ref's forget recall reads **0.134
distinctive** (pass) but **0.369 naive** and **0.394 ROUGE-L** — both more than 2.5× the
ceiling. On either inflated metric I would have concluded that TOFU is confounded too and
that the dataset does not fix the problem. Pristine-base floor: distinctive **0.0735** versus
naive 0.3231 and ROUGE-L 0.3233, both **4.40×** inflated; question echo alone scores
distinctive 0.000 / naive 0.500. **The inflation is largest exactly where it is most
dangerous — on a model that does not know the facts — and collapses to +0.04 once it does.**

## What did qualify

Both backbones passed 14/14 gates with `max|Δlogit| = 0.000e+00` bitwise save/reload on both:

| run | distinctive forget | distinctive retain |
|---|---|---|
| θ₀_tofu | **0.791** | 0.787 |
| retain_ref_tofu | **0.134** | **0.938** |

**The ignorance gate passes with margin retain − forget = +0.804**, against ≈0.000 on the
synthetic set. So **TOFU does fix the Finding-5 confound** — that part of the pivot was right.
Stated honestly: forget is not at the floor (0.134 vs 0.0735 is +7.8 SE of real leakage),
inside the declared ceiling and 6× below the synthetic set, but the pristine base is the
model to use if a true ignorance floor is ever needed.

## Open process item

`scripts/memorize_v2.py` now hashes `39a307f8…` while all three delivered synthetic backbone
reports record `951970d0…`. The change is the `--train-split retain` branch I added for the
ignorance-reference attempt. The adapters are unaffected but the provenance link from those
three backbones to their source is broken. To be repaired by restoring the file and moving
that branch to a separate script.
