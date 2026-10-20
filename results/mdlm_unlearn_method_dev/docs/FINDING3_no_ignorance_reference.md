# Finding 3 — there is no ignorance reference to match (F1 fires), and Finding 2's metric is broken

**Development evidence, 2026-10-20.** Two decisive negatives, each costing under a GPU-minute
of the thing they decide. Both are against the coordinator's own prior position.

## 1. The ignorance-reference target is not constructible by subject substitution

The candidate objective targets `P_ign` — the distribution a reference places at the answer
slot when the subject is one it does not know, estimated by substituting a verified-unknown
subject. Its designer pre-registered the fatal condition (**F1**): if `P_ign` is itself
sharply moded, ignorance matching does not avoid replacement, it only moves the attractor.

Measured on the qualified backbone `memorized_v2_s20261101`, 64 forget facts × 8 strangers =
512 single-position teacher-forced forwards, 16 s (`outputs/pign_v2_*.json`):

| relation | modal token | modal share | in answer pool | mean H (bits) | mean p(true answer) |
|---|---|---|---|---|---|
| field | ` cart` (cartography) | 0.391 | yes | 1.00 | 0.357 |
| home_city | ` Lisbon` | 0.430 | yes | 2.81 | 0.143 |
| instrument | ` viol` (viola) | 0.375 | yes | 1.57 | 0.270 |

**The validity gate failed: mean p(true answer | stranger) = 0.228 against a pre-declared
threshold of 0.02.** The strangers are not unknown to the model in the required sense.

A first run failed for a mundane reason — the strangers shared **surnames** with trained
entities (`Rovarn Greaves` against the memorized `Alira Greaves`), because the dataset
consumes all 16 of its surnames. Rerun with name components appearing nowhere in the fact
set, the gate still fails at 0.228, so the cause is not name overlap.

What it is instead, visible in the raw rows:

```
stranger "Durnesse Corbridge",  cello frame  -> ' cello' p=0.949
stranger "Durnesse Mossbarrow", cello frame  -> ' harp'  p=0.603,  ' cello' 0.267
stranger "Durnesse Corbridge",  botany frame -> ' cart'  p=0.868
```

**The model confidently confabulates a memorized answer for any name-like string**, and
*which* answer depends on the stranger's name — consistent with retrieval by name
similarity. Entropies of 1.0–2.8 bits, frequently near-deterministic. This is not ignorance;
it is high-confidence substitution.

**Verdict: F1 fires, in a stronger form than anticipated.** The condition was "P_ign is
sharply moded". The reality is that `P_ign` as estimated by substitution is not an ignorance
distribution at all, so the quantity the objective was to match does not exist on this
backbone. The target cannot be closed this way.

**What survives as a route.** The designer's own §A.6 alternative — `P_ign^retrain`, a
reference trained on the **retain facts only**, which is constructible here because the
impossibility result bars retraining on the pretraining corpus, not on our SFT stage. The
mathematical critic independently called this "the best new idea" in the design. It costs one
memorization run (~8 min measured) and is the next thing to build. Note it must remain an
**evaluation-and-target-construction asset only**: a method that needs a retain-only
retrained model at deployment would not transfer to real pretrained knowledge.

## 2. Finding 2's IGNORANT metric does not measure ignorance

The adversarial review printed all 28 generations the metric scored IGNORANT for the best
arm. **28 of 28 assert a value in the relation's answer slot. Zero are non-asserting.**

They escape the REPLACE cell only because the asserted value falls outside the *enumerated
pool*, and mostly by a **morpheme**: `seismoology` (not a word) ×8, `mycological`/`myc` ×3,
`harpa` ×2, `clarion` ×2, `tympani`, `oboes`, `bassoons`, `hydrodynamics` ×2 — all variants
of in-pool answers. A conservative automatic near-miss detector catches 11 of 28, taking the
headline from 0.438 to ≤0.266; by inspection it is 17–20. The remainder (`Troms`, `Tashkent`,
`tuba`, `English studies`) are out-of-pool but type-correct, i.e. replacement by the
finding's own definition.

**Corrected best arm: DISCLOSE 0.062, DEGENERATE 0.000, REPLACE ≈ 0.938, IGNORANT ≈ 0.000.**

And the metric is **gameable in the wrong direction**: nudging substitutes from `Tbilisi`
toward `Troms`/`seismoology` scores a large IGNORANT gain while making the model strictly
worse.

Three further defects, all accepted:

- **The ceiling of 1.000 is unattainable as specified, and the two documents conflict.**
  DEGENERATE includes refusal, so a hedge is DEGENERATE, while IGNORANT requires fluent
  on-topic text that asserts nothing. But the formalization says a genuinely ignorant model
  *"still answers about the subject ('Alice lives in Lyon, I think')"* — which this metric
  scores as REPLACE. **The proposed primary metric penalises the project's own stated
  target.**
- **No method is distinguishable at this n.** The answer pools differ between the two
  datasets (7/5/4 versus 16/8/8), so REPLACE and IGNORANT are *different predicates* row to
  row — column-level non-comparability, not a sample-size caveat. Re-scored against the
  128-fact pools, the only competitive baseline `sc_tf_150` moves 0.375 → 0.250. **Eight of
  nine arms have a 95% Wilson upper bound above 0.438, and `sc_tf_150` versus ours is
  Fisher p = 1.0.** The claim "no method exceeds 0.438" is not supportable.
- **The ordering claim holds**, by a cleaner argument than the one I gave: IGNORANT is the
  residual cell, so it is *exactly* invariant under reordering. Tested: 1 collision row in
  575 stored generations.

## 3. The error I have now made three times

**The `Tbilisi` attractor is not the exclusion target's signature.** Partitioning **θ₀** —
the memorized model, before any unlearning — gives REPLACE 0.203, IGNORANT 0.000, and all 13
facts θ₀ does not know are in-pool substitutions, 5 of the first 6 on `Tbilisi`. It is the
reference's own city prior, present before the intervention.

That is the third instance of the same error class in this project:

1. **ARC**: an aggregate IN-pool/ABSENT gap presented as a per-fact confirmation.
2. **Finding 1**: same-answer gradient coupling attributed to the objective's indexing, when
   it is a property of suppression — present in two methods that are not indexed that way.
3. **Finding 2 / Tbilisi**: a replacement attractor attributed to the objective, when it is
   present in the reference.

The common form: **I attribute a property to the intervention without measuring it at the
pre-intervention reference.** The pre-registration rule added after Finding 1 (per-fact
claims need a within-stratum continuous correlation) does not catch this one.

**New standing rule, institutionalised:** *every* claim that an intervention causes a
property must report that property measured at θ₀ **and** at one intervention that does not
share the proposed mechanism, in the same table, before the claim is stated. Not as a
robustness check afterwards — in the table.

## What this redirects

- The candidate's target closure must change from substitution-estimated `P_ign` to
  `P_ign^retrain`, or the candidate is dead.
- The partition metric needs REPLACE defined by **type-correctness plus near-miss detection**,
  ABSTAIN as its own cell, and a reconciliation with the formalization's notion of ignorance.
  Already issued to the matched re-run before its evaluator was finished.
- **The closed 32-answer pool is now a liability in itself.** It creates a relation-level
  prior that contaminates both the metric (REPLACE and IGNORANT are not separable) and the
  target (`P_ign` is pool-concentrated). This is an argument for moving the mechanism work to
  real pretrained knowledge — RWKU-RQ, or TOFU's forget split — sooner than planned.
