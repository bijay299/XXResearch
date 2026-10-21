# Finding 12 — the ordering oracle was never a reachable ceiling; H10 is closed

**Date:** 2026-10-21
**Status:** closed. The ordering line (H10, all variants) ends here.
**Attempt:** `p098` · **Script:** `scripts/oracle_ablate.py` ·
**Output:** `outputs/oracle_ablate_tofu120.json`

---

## 1. The claim being tested

`scripts/h10_oracle_tofu.py` states in its own docstring that the oracle intervention is

> "an exact upper bound on what any trained H10 can achieve, because a trained approximation
> cannot beat its own oracle."

That sentence is wrong, and this finding is the demonstration. The oracle sets the confidence
of every fact-argmax masked position to `-inf` at every step. It fires in two regimes that are
not alike:

| regime | what `-inf` does | reachable by training? |
|---|---|---|
| **non-terminal** | at least one masked position is not a fact-argmax, so a structural rival exists; `-inf` is equivalent to that rival simply having higher confidence | **yes** — a trained arm can raise structural confidence |
| **terminal** | every remaining masked position is a fact-argmax; there is no rival. `topk` over a constant `-inf` vector falls back to **index order**, so the remaining fact slots are filled left-to-right instead of by confidence | **no** — there is nothing to raise |

A trained arm can only ever reproduce the first. So the oracle bounds a trained arm only if the
gain comes from the non-terminal regime.

## 2. Measurement

Three arms, same 120 TOFU forget facts, same decoder, behind the same bit-identical parity gate
(`local loop == pinned generate(): True`). The pre-registered reading was fixed before the
numbers were seen: reversion ≤0.25 → train the per-step hinge; ≥0.50 → the line is closed.

| arm | DISCLOSE | DEGENERATE | REPLACE | IGNORANT | ASCR | TV→pristine |
|---|---|---|---|---|---|---|
| `natural` (pre-intervention reference) | 0.7917 | 0.0083 | 0.0917 | 0.1083 | 0.7203 | 0.7705 |
| `oracle_full` (as published) | **0.1417** | 0.0167 | 0.2500 | 0.5917 | 0.2621 | 0.3326 |
| `oracle_nonterminal` (reachable regime only) | **0.7500** | 0.0000 | 0.1083 | 0.1417 | 0.6445 | 0.7205 |

Terminal steps seen: 594; withheld in the non-terminal arm: 539.

**Reversion fraction = 0.9358.** Switching off the unreachable regime loses **93.6%** of the
oracle's disclosure reduction. Verdict: `GAIN_IS_TERMINAL_INDEX_ORDER_ordering_line_closed`.

## 3. What this means

The oracle's headline (0.667 → 0.108 in the original run; 0.792 → 0.142 here on this fact
subset) is **not** produced by deferring facts behind structural tokens. It is produced almost
entirely by what happens once only fact positions remain: the intervention stops being a
deferral and becomes a change of fill ORDER, from highest-confidence-first to left-to-right.
No confidence-raising objective can imitate that, because at those states there is no
structural position whose confidence could be raised.

So the ordering line fails for a reason that no amount of training-side work can fix, and it
fails *in addition to* the reason given in Finding 11 (that the hinge both trained arms
optimised was already satisfied at θ₀, penalised deficit −0.3149). The two are independent:
Finding 11 says the objective was inert; this says the target was unreachable anyway.

**H10 is closed.** No further ordering variants are to be opened without new evidence that
bears on the terminal regime specifically.

## 4. Scope limits, stated

- `natural` reads 0.7917 here against the 0.667 recorded for θ₀ in `FINDING7`/`FINDING8`. The
  fact selection differs (this run takes the first 120 facts with ≥3 salient stems). The
  comparison that carries the finding is **internal to this run** — all three arms share the
  same 120 facts, same seeds, same decoder — and no cross-run subtraction is performed.
- `oracle_full` reads 0.1417 here against 0.108 published, same selection difference. The
  published number is not revised; this run is not a re-measurement of it.
- The reversion fraction is computed on DISCLOSE. The same direction holds on ASCR
  (0.7203 → 0.2621 → 0.6445) and on TV-to-pristine (0.7705 → 0.3326 → 0.7205), so the
  conclusion does not depend on the choice of that one coordinate.

## 5. Process note

This test cost one attempt and ~12 GPU-minutes, and it was run **before** funding any training
on the per-step hinge that Finding 11 had just shown to be reachable. Had it been run when the
oracle was first published as a "ceiling", Findings 9 and 10 and three training arms would not
have been built on it. The rule this adds: **an intervention may only be called a bound on a
trained method after the regimes in which it operates have been enumerated and each one checked
for reachability by that method.**
