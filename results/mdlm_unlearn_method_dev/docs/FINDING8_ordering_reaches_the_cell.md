# Finding 8 — the ordering channel reaches the cell that target-matching cannot

**The project's first positive mechanism result.** An oracle, not yet a method, but it answers
the question that risk R1 was posed to decide — and it answers it by a different route than
the one R1 tested.

## The intervention

LLaDA commits the highest-confidence masked positions at each step. At every step, any
position whose current argmax is an author-salient answer token has its confidence set to
`-inf`, so it is committed **last**. No training, no weight change. This is an exact upper
bound on any trained approximation, because a trained version cannot beat its own oracle.

**Parity gate passed on both datasets:** with the intervention disabled, the local sampler
loop reproduces the pinned `generate()` **bit-identically**, token for token. Without that,
the comparison would be against a reimplementation rather than the real decoder.

## Result on TOFU (θ₀_tofu, 120 forget QA, free-form answers)

| cell | θ₀ (memorized) | **ORACLE (forced-last)** | pristine base (ignorance target) |
|---|---|---|---|
| DISCLOSE | 0.667 | **0.108** | 0.030 |
| **DEGENERATE** | 0.000 | **0.017** | 0.000 |
| ABSTAIN | 0.000 | 0.000 | 0.113 |
| REPLACE | 0.125 | 0.258 | 0.470 |
| IGNORANT | 0.208 | 0.617 | 0.388 |
| UNCLASSIFIED | 0.000 | 0.000 | 0.000 |
| **TV distance to pristine** | **0.6372** | **0.3242** | — |

Supporting numbers: mean salient-content recall **0.610 → 0.155**; 58 of 120 items reach
**zero** salient content; **0 of 120 empty generations**; mean length 47.1 → 41.7 words
(89% retained); mean 363 deferrals per decode.

Under the project's declared disclosure threshold the drop is **0.667 → 0.108**; at τ=0.25 it
is 0.892 → 0.250 and at τ=0.75 it is 0.350 → 0.042, so the effect is not an artifact of one
threshold.

## Why this matters

**Every method measured to date fails bimodally.** MDU, EMDU and NPO reach DISCLOSE 0.000 at
**DEGENERATE 1.000** — they score perfectly on the field's metric by emitting nothing. The
exclusion family stays fluent but leaves ~0.45 disclosed.

**The ordering oracle is the first intervention to move disclosure 6× while keeping
degeneracy at 0.017.** It halves the profile distance to a model that never saw the data.
That is the empty region in the trade-off, and it is reachable — by changing the *decode
order*, not the target distribution.

This also **refutes the re-entailment hazard** raised against the mechanism in design: the
concern was that committing a fact later conditions it on more model-written context which
*entails* the fact. Measured, late commitment reduces disclosure rather than preserving it.

## What it is not

- **It is an oracle.** It uses answer-token identity at inference time, which is unavailable
  at deployment. It is an upper bound and a feasibility proof, never a method.
- **It overshoots on IGNORANT** (0.617 against the pristine 0.388), i.e. it is more
  non-committal than genuine ignorance. Under the profile-matching objective that is a
  mismatch, not a bonus.
- **The output form varies.** Three behaviours in the raw rows: fluent content removal
  (*"father's profession in civil engineering has influenced her by providing exa"*),
  ungrammatical elision (*"her father was a, and her mother was a"*), and no effect at all
  where context strongly determines the token (one row is byte-identical to natural, salient
  recall 1.00 → 1.00).
- **The synthetic-set version is uninformative** by comparison: with single-word answers,
  deferring the answer elides the entire content (*"Alira Ashcroft lives in."*) and
  disclosure goes to 0.000 trivially. The TOFU multi-token result is the real measurement.
- One model, one seed, 120 of 400 forget QA, no retain-side measurement yet.

## The method problem this leaves, stated precisely

> **Approximate the ordering oracle in the weights, without answer-set knowledge at
> inference.**

The trainable route is already derived and is not suppression. Commitment at `T=0` depends
only on the *ranking* of per-position confidences, and on the set where disclosure can occur
the answer position's confidence is *identically* its answer probability — so lowering it is
suppression by another name. What is available instead is **raising the confidence of the
structural positions**, whose gradient at the answer position is **zero by construction**.
That is the one mechanism in this project that is neither a target-distribution change nor a
token-selection rule.

Declared risk, from the same derivation: feasibility is a knife-edge. At a memorized
confidence of ≈0.9, a margin of 0.1 requires competitor positions above 0.995, so the
trainable margin is ≈10⁻². The oracle achieves the ordering by fiat; a trained version must
achieve it by a rank flip it may not be able to buy.

## Next

Build the trainable arm and compare it against this oracle as its ceiling, against the
pristine profile as its target, and against a mass-matched uniform structural upweight as the
control that makes the claim attributable.
