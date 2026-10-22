# Finding 19 — two walls, one at each level, and together they explain the field's trade-off

**Date:** 2026-10-21 · **Attempts:** `p274` (RMU uncalibrated), `p275` (RMU calibrated),
`p276` (RMU band), `p277` (decoy answer), `p278` (decoy prompt)
**Script:** `scripts/train_rmu.py` · **Status:** closes the representation-target line; the two
walls are the substantive result

---

## 1. RMU does escape Finding 18's wall — and shows what the real constraint is

Finding 18 closed every *anchor* construction: form and fact-identity are carried by the same
information, so no choice of what the anchor sees can have both. RMU
([arXiv 2403.03218](https://arxiv.org/abs/2403.03218)) is not in that family — its forget target
is `c·u` for a fixed random unit vector `u`, carrying neither form nor fact, with form
preservation coming from a separate retain term on different data.

Ported to LLaDA at layer 16 (gradients on layers 14–16 only, 21 LoRA modules), generation-level
six-cell partition, 120 forget / 80 retain:

| arm | DISCLOSE | DEGENERATE | retain ASCR | TV→pristine |
|---|---|---|---|---|
| θ₀ | 0.792 | 0.008 | 0.762 | 0.770 |
| AFC λ=1 (n=8) | 0.146 | 0.152 | 0.534 | 0.299 |
| MDU λ=3 (n=3) | 0.133 | 0.181 | 0.445 | 0.352 |
| **RMU (calibrated)** | **0.050** | **0.825** | **0.704** | 0.846 |
| *pristine* | *0.030* | *0.000* | — | *0* |

**RMU gets the best forgetting and the best retention of anything in this project** — DISCLOSE
0.050 against AFC's 0.146, retain 0.704 against AFC's 0.534 and nearly θ₀'s 0.762. Forgetting
and retention are genuinely decoupled, exactly as Finding 18 predicted for a target outside the
anchor family. And it is **82.5% DEGENERATE**, reproducing Łucki et al.
([arXiv 2409.18025](https://arxiv.org/abs/2409.18025)): RMU "make[s] models unusable — they
output gibberish generations with high perplexity."

**Calibration note, recorded because the first run was a null.** RMU's published `c = 20` points
the target *inward* on this model: measured `‖h‖ = 158.67` at layer 16, and RMU's Figure 7 says
the forget term "scales up the norm". With `c = 20` and the loss summed over 4096 dims
(`L_forget ≈ 2.2 × 10⁴`) against `clip_grad_norm_(1.0)`, the intervention never fired —
`cos_to_u` moved −0.024 → −0.010 and nothing downstream changed. With per-dimension loss,
`c = 320` and clip 10, it fired decisively: `cos_to_u` −0.024 → **0.885**, `‖h‖` 158.7 → 286.4.
`c` is explicitly a model-dependent hyperparameter in the paper; this is calibration, not a
goalpost move, and the null run is recorded as a null.

The mask-rate band variant (`rmu_hi`, misdirect only at `t ∈ [0.5, 1]`) behaves the same:
`fact` 0.084 → 0.042, `p_true` 0.482 → 0.233, entropy 1.69 → **5.02**, `cos_to_u` → 0.736. The
schedule choice does not avoid the degeneracy.

## 2. The fix fails, and the failure is the second wall

The pristine reference does not emit noise — it gives a *plausible wrong answer*
(REPLACE 0.470, IGNORANT 0.388). So replace RMU's random target with the frozen model's own
activations on a decoy input, which carries its own form and its own wrong fact. Two
constructions:

| arm | target | `L_forget` | `fact_argmax` | `p_true` | result |
|---|---|---|---|---|---|
| RMU | `c·u`, random, off-manifold | 3.86 → 2.68 | 0.087 → **0.042** | 0.606 → **0.226** | forgets, gibberish |
| `decoy` | frozen acts, same prompt, **decoy answer** | **1.18** | 0.087 → 0.087 | 0.606 → 0.596 | **no-op** |
| `decoyq` | frozen acts, **decoy prompt** + decoy answer | **0.56** | 0.103 → 0.098 | 0.599 → 0.575 | **no-op** |

Both decoy targets are near no-ops, and the losses say why: `L_forget` is 1.18 and 0.56 against
RMU's 3.86. **The decoy activation is nearly the same vector as the current one.** At a masked
answer position the activation is set by the prompt and the mask pattern, not by which answer
occupies the few visible slots; and averaged decoy activations across prompts are dominated by
shared structure (chat template, answer scaffold). There is almost nothing to descend.

> **The activation-space wall.** Any forget target close enough to the data manifold to preserve
> fluency is close enough to the original activation to provide no forgetting signal. RMU works
> *because* its target is off-manifold — which is the same reason it produces gibberish.

## 3. The two walls together

| level | wall | measured by |
|---|---|---|
| **output distributions** | form and fact-identity are the same information, so no anchor can be both well-formed and fact-free | `FINDING18`: schedule banding (anti-correlated), FAM (fact-free, 8.19 nats, 97.9% degenerate), fact-selective masking (well-formed, still leaks) |
| **activations** | fluency-preserving targets are too close to the current activation to give signal; only off-manifold targets forget, and those produce gibberish | this finding: RMU off-manifold (forgets, 82.5% degenerate) vs two on-manifold decoys (`L_forget` 1.18 / 0.56, no forgetting) |

These are two statements of the same obstruction from opposite directions, and together they
explain why MDU, EMDU, AFC and RMU all sit on a forgetting-versus-fluency trade-off rather than
one of them simply being better engineered. **A method that escapes both would have to obtain its
forget target from something other than (a) the model's own distribution on a modified input or
(b) a point in activation space** — e.g. from a second model, from retraining, or from data. The
published methods that do achieve robust removal take exactly that route: UNDO
([arXiv 2506.06278](https://arxiv.org/abs/2506.06278)) distils into a damaged copy using an epoch
of pretraining data, and oracle matching uses a retain-only retrained reference.

## 4. Status

Eighteen candidates closed: fourteen by measurement, four by primary-source novelty check. No
novel method. What this finding adds is not another dead end but a *reason* for the dead ends —
the first result in this project that predicts failures rather than reporting them, and the one
most worth writing up.

**Still open and unverified:** whether decoy-target representation redirection is novel at all
(a novelty check is running). Given it is a measured no-op, the answer does not change the
project's direction.
