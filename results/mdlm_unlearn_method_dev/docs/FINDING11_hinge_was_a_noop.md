# Finding 11 — the ordering hinge was penalising a condition θ₀ already satisfied

**Date:** 2026-10-21
**Status:** closed finding; it reverses the interpretation of Finding 9 and corrects Finding 10
**Attempts:** `p094` (commit_v), `p095` (uniform_v), `p096` (order_condition)
**Scripts:** `scripts/train_ord_pool.py`, `scripts/order_condition.py`
**Outputs:** `adapters/commit_v_full/report.json`, `adapters/uniform_v_full/report.json`,
`outputs/ordercond_theta0.json`

---

## 1. What was expected

Finding 10 established that the ordering objective had been trained on the wrong state family:
structural confidence has headroom 0.2484 on the decoder's visited states against 0.0047 on
teacher-forced states, a 53× difference, so the objective was being optimised in a regime the
decoder never enters. The prediction was that retraining the same hinge on a frozen pool of
visited states — 2051 states, 348/366 facts, index sets fixed at collection time — would move
the ordering instruments that Finding 9 found immobile.

## 2. What happened

It did not, and the logs say why in one line. `commit_v`'s ordering loss reached **exactly
0.0** at 11 of 17 probes while `frac_last` never left 0.

```
[commit_v] step    0 L_ord=0.0      c_fact=0.53907 c_struct=0.94723 frac_last=0.0
[commit_v] step   75 L_ord=0.0      c_fact=0.24956 c_struct=0.99821 frac_last=0.0
[commit_v] step  200 L_ord=0.0      c_fact=0.11378 c_struct=0.91481 frac_last=0.5
[commit_v] step  399 L_ord=0.0      c_fact=0.34223 c_struct=0.91759 frac_last=0.0
```

A satisfied hinge that moves no ordering is not a hinge that failed to train. It is a hinge on
a quantity the decoder does not read.

## 3. The decoder's actual commit rule

At `steps == gen_length == block_length == 64`, the pinned sampler's
`get_num_transfer_tokens` returns **1 per step**, and the commit is

```python
_, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])   # k == 1
```

Exactly one position is committed per step: the confidence **argmax**. A fact position is
therefore deferred at a state if and only if

```
max_{i∈fact} c_i  <  max_{j∈struct} c_j                                   (per-step)
```

`train_ord.py` and `train_ord_pool.py` penalised neither this nor the `frac_last` condition
they instrumented. They penalised the mean over pairs (all fact positions) × (mean of the
top-m structural positions). Three quantities, measured on the same 2051-state pool at θ₀
(`outputs/ordercond_theta0.json`):

| quantity | rival | value at θ₀ | deficit at μ=0.05 | already satisfied in |
|---|---|---|---|---|
| **what was penalised** | mean of top-4 structural | 0.8159 | **−0.3149** | — |
| per-step condition | max structural | 0.9285 | −0.2477 | **0.8688** of states |
| full-order condition (`frac_last`) | min structural | 0.1341 | **+0.5468** | 0.0341 of states |

with `max_fact = 0.6477`, 4.64 fact and 35.94 structural positions per state.

The penalised deficit is **−0.3149**: negative, i.e. the objective was satisfied at θ₀ by a
wide margin before a single gradient step. `L_ord = 0.0` is not a training success, it is the
loss reporting that there was nothing to do. Finding 9's conclusion that the ordering channel
was saturated, and Finding 10's conclusion that the state family was to blame, were both
drawn from an objective that could not have moved either way.

## 4. The control earns its place

`uniform_v`, the mass-matched structural upweight with no commit-margin gate, drove **every**
position to confidence 1.0 — `c_fact = 1.0` and `c_struct = 1.0` together by step 300. Its
loss `−mean(c_struct)` has no competing term, so it saturates the whole canvas and changes no
ordering. That is the behaviour a control should have, and it confirms that raising structural
confidence indiscriminately raises fact confidence with it. Any ordering claim must therefore
come from the margin gate, not from the upweight.

## 5. A second error, and one this project had already made

Every instrument in `train_ord_pool.py` was read off the training batch of **2** states.
Between adjacent probes `c_struct` swung 0.74 → 0.99999 and `c_fact` 0.11 → 0.89. The
mechanism gate fired on that noise and reported `c_fact 0.539 → 0.342, FAIL`, a number with no
interpretation. This repository has made exactly this mistake before: a guard comparing `L_mem`
on random batches with `t~U(0,1)` fired on all three arms including one that provably could not
move, and was rewired to a frozen probe. The lesson did not transfer to the new script.

`scripts/train_ord_v2.py` holds out a **fact-disjoint** 15% slice of the pool, never trains on
it, asserts no fact-id leakage between the two sides, and measures every reported instrument on
it under `no_grad` — including a probe at θ₀ before the first update, so every later number has
its own pre-intervention reference in the same table.

## 6. What this does and does not license

Established:

- The quantity penalised by both previous ordering arms was satisfied at θ₀. **Findings 9 and
  10 cannot be read as evidence about the ordering channel**; they are evidence about a loss
  that was inert. Their protocols and recorded numbers stand unaltered; this is a correction of
  interpretation, filed as a new finding rather than an edit to either.
- The full-order condition (`frac_last`, the one both arms instrumented) is satisfied at
  **0.0341** of states with `min_struct` median **0.1157** and 97.1% of states short of the
  margin. Reaching it would require raising ~36 structural positions per state above 0.65.
  Lowering `max_fact` instead is suppression of the answer probability at T=0, which the
  stop-gradient exists to forbid. **The full-order condition is not a viable target.**
- The per-step condition is satisfied at **0.8688** of states with `defer_deficit` 90th
  percentile **0.0589**. It is reachable.

Not established, and gating:

- Whether the per-step condition is *worth* reaching. θ₀ already satisfies it at 86.9% of
  states and still discloses 0.667; only **42.5%** of facts defer at *every* pooled state. The
  oracle's 0.667 → 0.108 may come substantially from a regime no trained arm can reproduce:
  at terminal states every remaining masked position is a fact-argmax, so `-inf` defers nothing
  and `topk` over a constant vector falls back to **index order**, filling the remaining fact
  slots left-to-right rather than by confidence. Nothing can be raised there.

`scripts/oracle_ablate.py` decides this before any training spend, with three arms on the same
120 facts and decoder — `natural`, `oracle_full`, `oracle_nonterminal` — behind the same
bit-identical parity gate, and a reversion fraction pre-registered at ≤0.25 (train the hinge)
versus ≥0.50 (the ordering line is closed for a reason training cannot fix).

## 7. Process consequence

Both errors share a shape: a surrogate was adopted for stability (`top-m mean` for `max`;
batch instruments for a probe) and then reasoned about as though it were the target. The
check that would have caught both costs 26 GPU-seconds — measure the loss's own deficit at θ₀
before training on it. That measurement is now a precondition for any new objective in this
project, and `order_condition.py` exists to perform it.
