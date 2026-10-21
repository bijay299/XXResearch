# Finding 5 — the 128-fact dataset makes forgetting and retention logically inseparable

**Development evidence, 2026-10-20.** Found because a gate failed that only existed because
an earlier hypothesis had died. This is a flaw in **our own** synthetic dataset and it
confounds a chain of results across three studies, two of them published.

## The defect

`build_big.py` constructs the forget and retain splits by indexing the same ordered
name product and the same cyclic answer list. The result:

```
forget_000  Alira Ashcroft    -> Paris      retain_000  Zelvira Ashcroft   -> Paris
forget_001  Alira Birchwood   -> Lisbon     retain_001  Zelvira Birchwood  -> Lisbon
forget_002  Alira Calloway    -> Oslo       retain_002  Zelvira Calloway   -> Oslo
```

**All 64 of 64 forget facts share BOTH the surname AND the answer with their retain
partner.** `surname -> answer` is therefore a perfectly predictive rule, and it is learnable
from the **retain set alone**.

## How it was caught

An attempt to build `P_ign^retrain` — a reference trained on the 64 retain facts only, which
by construction must *not* know the forget facts — was gated on exactly that. The gate
failed at the extreme:

```
G4a_IGNORANCE_of_forget_facts   FAIL
  forget_recall_nfkd: 1.000   (ceiling 0.10, n=64)
  retain_recall_nfkd: 1.000   (threshold 0.95, n=64)
```

A model that never saw a single forget fact answers **all 64 of them correctly**. It does not
need them: the retain facts teach `Ashcroft -> Paris`, and `Alira Ashcroft` inherits it.

## What this means, stated precisely

**On this dataset, forgetting a fact while preserving its retain partner is not merely hard
— the two requirements are in direct conflict.** The retain objective actively teaches the
rule that answers the forget query. Any method that preserves `Zelvira Ashcroft -> Paris`
supplies the route to `Alira Ashcroft -> Paris`. The "forgetting versus utility trade-off" on
this data is partly a contradiction rather than a trade-off.

## Which conclusions this confounds

Ranked by severity. **None of these are retracted on this basis alone** — each needs
re-measurement on a dataset without the confound — but every one is now suspect:

1. **Retain preservation on this set.** Every arm's "retain accuracy 1.000" is consistent
   with the model holding the surname rule rather than per-entity memory. The rule is
   *cheaper* to learn than 64 bindings, so it is the likelier solution.
2. **Finding 1's same-answer coupling** (`research/mdlm-unlearn-method-dev`). The
   "same answer" grouping is perfectly confounded with "same surname" for every
   forget-retain pair, so the measured forget-retain same-answer cosine (+0.620 at θ₀,
   +0.261 at θ_u) cannot separate an answer-token channel from a surname channel. The causal
   claim was already retracted for an independent reason; this removes a further leg.
3. **The relearning study's transfer result** (`research/relearning-robust-mdlm` @
   `a67d610`). 8 leaked facts restoring 43 held-out facts is partly explicable by surname
   generalisation, since leaked and held-out facts share surnames with retain facts the
   model still knows. The *reversibility* finding survives — 0.023 -> 0.605 against a
   never-memorized floor of 0.023 is a large effect and the floor controls for learning —
   but the per-fact **mechanism** attribution does not.
4. **The `P_ign` substitution contamination** (Finding 3). Re-explained: the model learned
   `surname -> answer` for all 16 surnames, so a stranger with a *novel* surname falls back
   on a prior over the 32 answers, which is exactly the p(true|stranger) = 0.228 observed.
   Finding 3's conclusion stands; its explanation improves.
5. **The ARC study's 128-fact arms** (`research/arc-amortized-reference-caching` @
   `28154b2`). Its cost results (exactness, forward counts, speedup) are properties of the
   mechanism and are untouched. Its efficacy comparisons inherit the confound.

Errata will be issued against the two published commits.

## What it does NOT invalidate

The matched baseline comparison's **qualitative grouping** survives, because every arm shares
the identical dataset and backbone: nine of fifteen grid points reach total output collapse
while the two fluent arms disclose ~0.45. A confound shared by all arms cannot create that
separation. What it does undermine is any *quantitative* preservation claim.

## Why this vindicates the pivot already authorized

TOFU forget10 has no such structure: **400 of 400 answers distinct**, real-looking author
names with no systematic forget/retain pairing, free-form sentence answers, and the unit of
unlearning is the author rather than a template slot. The confound cannot arise.

## The process point

This was found by a gate that existed only because a prior hypothesis had failed: the
ignorance-reference target died on substitution, the surviving route was a retain-only
reference, and that reference needed an ignorance gate — which failed at 1.000 and exposed
the dataset. Three hypotheses had to die for the data flaw underneath them to surface.

It also instantiates the standing rule added hours earlier: *measure the property at a
reference that does not share the proposed mechanism.* A retain-only model is exactly that
reference, and nothing in three studies on this dataset had ever been compared against one.
