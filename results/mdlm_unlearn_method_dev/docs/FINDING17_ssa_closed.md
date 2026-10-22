# Finding 17 — SSA closed: the low-mask-rate band has nothing to pull away from

**Date:** 2026-10-21 · **Attempt:** `p259` · **Script:** `scripts/train_ssa.py`
**Status:** closed. Mechanism was already anticipated (SPARE, KSCU); the band test now also
fails on its own measured prediction.

---

## 1. Parity gate passed

The pinned `FX.random_tf_states_llada` draws `t ~ U(0,1)` with no range argument, so the forward
process was reproduced locally with a `t`-range. With `(t_lo, t_hi) = (0, 1)` and the same
generator state it reproduces the pinned sampler **bit-identically** on the noisy tensor, the
mask tensor and `p_mask`:

```
[parity gate] band sampler == pinned random_tf_states_llada at (0,1): True
```

So the numbers below are against the real forward process, not a reimplementation of it.

## 2. Result: the low band under-forgets

`t ∈ [0, 0.5]`, otherwise identical to AFC λ=1 seed 0:

| quantity | θ₀ → final |
|---|---|
| `student_p_true` | 0.606 → **0.714** (rises) |
| `student_fact_argmax` | 0.087 → 0.089 (flat) |
| `student_eos_argmax` | 0.101 → 0.301 |
| `student_entropy` | 1.103 → 1.393 |

Answer probability ends **above** θ₀ — the arm does not forget.

## 3. Why, and it is exactly what Finding 13 predicted

Finding 13 measured the anchor at 23,450 positions: at low mask rate it has `p_true = 0.651`,
i.e. **it still knows the fact**. Matching a teacher that knows the answer cannot remove the
answer. The band that is well-formed is the band with nothing to pull away from, and the band
that is fact-free (high `t`, `p_true` 0.064) is the one that is formless (42% punctuation, 4.41
nats).

So the two desirable properties are not merely unevenly distributed across the schedule — they
are **anti-correlated** across it. No single contiguous band has both, which means banding
cannot resolve the conflict it was proposed to resolve. That is a stronger closure than "this
band happened not to work".

## 4. Novelty, recorded

The mechanism was already anticipated before this ran: **SPARE** (arXiv 2602.07058) "restricts
the unlearning optimization strictly to these crucial timesteps"; **KSCU** (arXiv 2507.06526,
IEEE T-IFS) uses a contiguous concept-specific interval. **ESD is confirmed uniform-in-`t`**
from the official repo (`utils/esd_trainer.py` line 416:
`run_till_timestep = random.randint(0, config.num_inference_steps - 1)`), so banding is a real
improvement over ESD — and it is theirs. **TimeROME-DLM** (arXiv 2606.12841) already published
the MDLM schedule-localisation measurement.

SSA was run anyway for a practical reason — whether it would cut AFC's degeneracy variance —
not for novelty. It does not: it fails to forget at all.

## 5. Candidate count

Twelve candidates now closed: nine by measurement (H1, H6, H11, C1/SIO, H10a/b/c, H14, H15,
H16, SSA) and three by primary-source novelty check (ESD→MDU, attack-aware→meta-unlearning,
SSA-mechanism→SPARE/KSCU). The anchor-objective family is exhausted for our purposes, which is
why the next candidate (H17) changes axis to the decoder rather than the objective.
