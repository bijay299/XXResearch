## Relearning curves — held-out forget disclosure (43 facts the adversary never saw)

Adversary: 8 leaked forget facts, plain masked-diffusion fine-tuning, identical for every arm. Lower is better.

| arm | k=0 | k=2 | k=5 | k=10 | k=20 | adv lr |
|---|---|---|---|---|---|---|
| never memorized (FLOOR) | 0.000 | 0.023 | 0.023 | **0.023** | 0.046 | 0.0002 |
| unlearned control (theta_u) | 0.023 | 0.046 | 0.256 | **0.605** | 0.651 | 0.0002 |
| flattened, 150 steps | 0.046 | 0.000 | 0.279 | **0.581** | 0.651 | 0.0002 |
| flattened, 300 steps | 0.046 | 0.046 | 0.279 | **0.581** | 0.651 | 0.0002 |

### Retain accuracy (16 probes) and degenerate outputs on the forget probes

| arm | retain k=0 | retain k=2 | retain k=5 | retain k=10 | retain k=20 | degen k=10 |
|---|---|---|---|---|---|---|
| never memorized (FLOOR) | 0.000 | 0.000 | 0.000 | 0.125 | 0.125 | 1 |
| unlearned control (theta_u) | 1.000 | 1.000 | 1.000 | 0.625 | 0.625 | 1 |
| flattened, 150 steps | 1.000 | 1.000 | 1.000 | 0.625 | 0.625 | 2 |
| flattened, 300 steps | 1.000 | 1.000 | 1.000 | 0.625 | 0.625 | 2 |

## Pre-registered verdict

**flattened, 150 steps** — Δ(k=10) = -0.023 (needs ≤ −0.20); Δ single-shot = +0.023 (needs ≤ +0.05); Δ retain = +0.000 (needs ≥ −0.05) → **NOT SUPPORTED**
**flattened, 300 steps** — Δ(k=10) = -0.023 (needs ≤ −0.20); Δ single-shot = +0.023 (needs ≤ +0.05); Δ retain = +0.000 (needs ≥ −0.05) → **NOT SUPPORTED**

## Mechanism check (a PROXY, not the result)

| run | arm | lam_G | lr | steps | opt | ||gradL_mem|| first → last | L_mem first → last | s/step | guard |
|---|---|---|---|---|---|---|---|---|---|
| `anvil_lamG10` | anvil | 10 | 0.005 | 300 | sgd | 1.117353 → **0.690175** | 0.9133 → 0.9204 | 0.7316 | ok |
| `anvil_main` | anvil | 1 | 0.005 | 300 | sgd | 1.117353 → **0.694113** | 0.9133 → 0.9195 | 0.7441 | ok |
| `cal_lr5e-3_lamG1.0` | anvil | 1 | 0.005 | 30 | sgd | 1.117353 → **0.967912** | 0.9133 → 0.9166 | 0.7964 | ok |
| `noop_check` | anchor_only | 0 | 0.005 | 10 | sgd | 1.154713 → **1.154713** | 0.9188 → 0.9188 | 0.9063 | ok |
