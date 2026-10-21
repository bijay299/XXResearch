# Research plan — written backwards from the paper

Supersedes the hypothesis-queue approach, which was a search, not a plan. Six live
hypotheses is evidence of thrashing. This commits to a thesis, a money table, one method
line, and a pruned backlog.

---

## 1. The thesis

### 1.0 CORRECTED — substitution is what ignorance looks like, and was never the failure

The original thesis said methods fail by "collapsing or substituting". **The substitution half
is wrong**, and it was wrong on both datasets:

| model | state | REPLACE rate |
|---|---|---|
| pristine TOFU base (never saw the data) | ignorant by construction | **0.470** |
| synthetic θ₀, on the 13 facts it did not learn | ignorant by accident | all 13 are in-pool substitutions |

A model asked *"where was X born?"* about someone it has never heard of says a plausible
city. **That is ignorance, not a failure of forgetting.** Our own critic had already pointed
at this — the `Tbilisi` attractor is the reference's city prior, present before any
unlearning — and I recorded that correction without carrying it into the thesis.

Full pristine TOFU profile: DISCLOSE 0.030 · DEGEN 0.000 · ABSTAIN 0.1125 ·
**REPLACE 0.470** · IGNORANT 0.3875.

**Consequence for the objective.** "Maximise IGNORANT" is wrong — at 0.9 it would be
*over-abstaining relative to genuine ignorance*. The target is **profile matching**:
minimise `TV(profile(arm), profile(pristine))` across the six cells, **with DISCLOSE the only
cell required to move**. That is well-defined, measurable, has a floor and a ceiling, and
does not reward either collapse or over-abstention.

It also tightens the claim rather than weakening it: the fluent arms' real defect is
**residual disclosure at ~0.45**, and the anchor arms' real defect is **collapse at 1.000
DEGENERATE** — both large, both unambiguous, neither explained away by substitution.

### 1.1 The thesis as corrected

> **Masked-diffusion unlearning methods do not forget. They either collapse entirely
> (9 of 15 grid points emit nothing) or leave ~45% of the facts disclosed.** Which failure
> occurs is determined by the *target distribution's own mode*, not by the method family or
> its tuning. We supply the field's missing oracle, measure the profile gap it reveals, and
> propose a mechanism that closes that gap without needing the oracle.

Three claims, in descending order of how well-evidenced they already are:

| # | claim | evidence state |
|---|---|---|
| **A** | The failure is bimodal and structural | **measured** — 9/15 grid points collapse to `''`; 2 fluent arms disclose ~0.45 and substitute ~0.31–0.38; desirable cell holds 4 of 960 rows |
| **B** | This field's metrics reward collapse, and TOFU's forget metric is circular | **measured + derived** — a mute model scores perfectly on disclosure; TOFU Forget Quality is a KS test against the retain model's own Truth-Ratio distribution |
| **C** | A mechanism exists that reaches the good cell without the oracle | **not yet** — this is the method, and the only part missing |

A and B alone are a findings paper. C is what makes it a method paper. **The plan is organised
so that C's risk is retired as early and as cheaply as possible.**

---

## 2. The money table — designed first, so every experiment earns its place

One table decides whether this is a method paper:

| row | why it is there |
|---|---|
| memorized θ₀ | the ceiling; shows the facts were there |
| **pristine base** | the floor; on TOFU it already scores IGNORANT+ABSTAIN = 0.500 |
| **retain-only reference (the ORACLE)** | the skyline; **this field has never reported one** |
| MDU, best τ | strongest anchor-family comparator |
| EMDU, best τ | second anchor-family comparator |
| NPO, best β | preference-family comparator |
| SC-TF / exclusion, best ε | renormalisation-family comparator |
| **our method** | the claim |
| **our method − distinguishing operation** | the ablation that makes the claim attributable |

Columns: `DISCLOSE / DEGEN / ABSTAIN / REPLACE / IGNORANT / UNCLASSIFIED`, plus retain
`M_min` margin and a utility panel — **each printed against its own measured floor**.
3 seeds, author-level clustering, MDD 0.125.

Budget: ~9 rows × 3 seeds ≈ 12 GPU-h for the table, plus ~3 GPU-h of method development.
**~15 GPU-h ≈ 8 wall-hours at two GPUs.** That is the whole remaining cost of a method paper.

---

## 3. Risk-ordered experiment list

Run in the order that retires the most paper-killing risk per GPU-minute.

| # | risk, stated as what would kill the paper | experiment | cost | status |
|---|---|---|---|---|
| **R1** | **The oracle itself does not land in the good cell.** Then there is no skyline, no target, and claim C has nothing to aim at. | partition the retain-only reference's forget generations, against θ₀ and the pristine base in the same table | 0.3 GPU-h | **running** |
| **R2** | **The diagnosis is wrong** — target mode does not predict the failure mode. Claim A survives as description, but loses its mechanism. | Phase B target-mode decomposition, forward passes only | 0.2 GPU-h | queued |
| **R3** | **No mechanism reaches the oracle.** Findings paper, not method paper. | one method line, below | ~3 GPU-h | gated on R1 |
| **R4** | **Effects fall below MDD 0.125** and nothing is resolvable at 20 authors. | enlarge the forget split from TOFU's 200 authors (60/140), declared as a deviation | ~1.5 GPU-h | held until an effect exists |
| **R5** | Our own metric manufactures the result. | permutation controls at every checkpoint; mismatched-gold floor must be < half the reported value | free | adopted |

R1 is the single highest-information measurement in the project and it is in flight.

---

## 4. One method line, and the queue is pruned

**Committed: H11, minimal-interdiction unlearning.** Concentrate forget pressure on the
minimal token set whose removal destroys identifiability; leave the rest of the answer
untouched.

Chosen over the alternatives for a stated reason: it attacks the **measured structural
cause** — uniform pressure across all answer tokens is what forces collapse — rather than
proposing a different target and hoping for a different attractor. Its failure mode is also
comprehensible in advance (pressure propagating through shared parameters), which makes it
cheap to kill.

**H10 keeps exactly one more experiment: its oracle test**, already written and queued, which
costs 7 GPU-minutes and can refute it outright. If it survives, it becomes a second line. If
not, it is closed. No further investment either way.

**Explicitly deprioritised, and recorded as such rather than left in a queue:**

| hypothesis | why it is not being pursued now |
|---|---|
| H7 — EOS attractor | **subsumed by R2.** Phase B measures EOS mass in the anchor targets directly, so H7 needs no separate experiment. |
| H8 — antiparallel forget/retain gradients | the measurement it rests on (retain margin ≤ 0 per probe) was **ULP noise**; the aggregate collapse stands but the mechanism claim lost its evidence. Revisit only if R4 forces a preservation-side story. |
| H9 — commit-state-only intervention | an **efficiency** refinement, not a failure-mode fix. Only meaningful after something works. |
| H6 — mask-rate bands | superseded before testing; the marginal is already uniform at our configuration. |

Four hypotheses closed or parked, one committed, one on a 7-minute reprieve. That is the
pruning this project needed.

---

## 5. What would make me abandon the method claim

Stated now so it cannot be rationalised later:

- H11's crux derivation shows interdiction pressure necessarily propagates to the rest of the
  answer → H11 is ordinary suppression and dies.
- H11 requires `k ≥ 3` interdicted tokens to clear guarded DISCLOSE → the "minimal pressure"
  premise is void and it becomes uniform suppression on a renamed subset.
- The "weight the rare tokens more" objection turns out to have a paper behind it → category
  (a) or (c), and it cannot carry novelty.
- The measured effect is below 0.125 even after R4 → report as unresolved, not as a win.

In any of those cases the deliverable is claims **A and B** plus the oracle, with the best
implemented prototype preserved and the obstacle named — which is the terminal state the
mandate specifies, and it is a real contribution rather than a non-result.

---

## 6. What I am no longer doing

- Inventing a new hypothesis each time one dies. The backlog is closed above.
- Treating diagnosis as the critical path. B is already evidenced; further diagnostic depth
  does not advance the method.
- Running experiments that do not feed a row or a column of the money table in §2.
