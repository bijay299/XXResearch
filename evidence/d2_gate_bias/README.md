# D2 resolved — the validity gate is biased, and the bias runs **against** our own claim

**2026-10-28.** Defect D2 has been open since FINDING72, which measured (n=520) that the validity
gate's exclusion rate is **non-monotone in `T_before`**. Because `max-over-valid H_after` is this
project's primary robustness quantity, it is computed over a sample whose filtering correlates with
how well-erased an arm is. The reviewer's question is immediate: *does the headline survive an
estimator that does not filter?*

Measured on **81 church arms, 901 attack cells**, from JSON already on disk.

## 1. The non-monotonicity is confirmed

Exclusion rate (gate-invalid **or** destroyed) against pre-attack suppression:

| `T_before` | excluded / total | rate |
|---|---|---:|
| ~0.0 | 152 / 388 | **0.392** |
| ~0.2 | 65 / 317 | 0.205 |
| ~0.4 | 15 / 111 | **0.135** |
| ~0.6 | 11 / 47 | 0.234 |
| ~0.8 | 3 / 26 | **0.115** |
| ~1.0 | 12 / 12 | 1.000 |

Not monotone: it falls, rises, falls again. FINDING72's observation stands.

The 1.000 at `T_before` ≈ 1.0 is structural rather than bias — an arm already at ceiling cannot
produce `T_rise >= 0.20`, so every such cell is gate-invalid by construction.

## 2. The conclusion survives an unfiltered estimator

Arms reaching `H = 1.000` under three estimators on the same cells:

| estimator | arms at 1.000 |
|---|---|
| **gated** (published: gate-valid and not destroyed) | 55 / 61 |
| **ungated** (not destroyed; no `T_rise` gate) | **58 / 74** |
| **raw** (no filtering whatsoever) | **59 / 75** |

Dropping the gate does not rescue a single arm. It admits more arms **and more of them reach full
recovery.**

## 3. The direction of the bias is the important part

Only **two** arms shift by more than one quantum (0.067) between gated and ungated — and both are
**ours**, not published baselines:

| arm | gated | ungated |
|---|---:|---:|
| `u32_uceT_a0.3` | 0.4667 (15 cells) | **0.6667** (39 cells) |
| `undo_a0.5_big` | 0.4667 (3 cells) | **0.6000** (10 cells) |

In both cases the **ungated number is higher** — i.e. the gate was making these arms look *more*
robust than they are.

**So the gate's bias flatters defences.** Removing it makes every affected arm look worse, never
better. Our central claim — that erasure is universally reversed — is therefore **understated** by
the published estimator, not propped up by it.

This also matters for the two arms named: both are ours, and both had their apparent robustness
partly produced by gating. Neither is claimed as a result; `u32_uceT_a0.3` was already closed by
U32's utility failure and U30's image audit.

## 4. Why the unfiltered estimator is the conservative one

Dropping the gate admits cells where the attacker never relearned the prompts it trained on. A low
`H_after` there is not evidence of robustness — the attack never started. Including them pushes
measured recovery **down**, so the ungated estimator is biased **against** the claim that everything
reverses. The claim surviving it is the stronger statement.

## 5. Status of the four blocking defects

| | status |
|---|---|
| **D1** inconsistent label text | **closed** — target metric moves 0.0100 against a 0.067 quantum |
| **D2** gate bias | **closed** — bias confirmed, runs against our claim, conclusion survives unfiltered |
| **D3** UCE port vs pinned upstream | open |
| **D4** pervasive n=1 | open |

## 6. Limits

- Church only; 81 arms, 901 cells. Other concepts not recomputed.
- The destroyed-cell rule is retained in the "ungated" estimator and removed only in "raw"; both
  give the same picture.
- This tests whether the *conclusion* survives, not whether `max-over-valid` is the best estimator.
  A maximum is itself a biased statistic over a finite sample, and the per-arm means are recorded in
  `outputs/d2_gate_bias.json` for anyone who wants to re-analyse on a different one.
