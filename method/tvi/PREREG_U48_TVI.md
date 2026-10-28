# PREREG U48 — TVI: Task-Vector Invariant erasure

**Declared 2026-10-28, BEFORE execution.** Development data; no confirmatory claim. Frozen on first
result; corrections in a dated amendment. Authorised by the project owner under MANDATE v3.

## 1. The method, and what is new about it

**ResAlign** (NeurIPS 2025, arXiv:2507.16302) is the only method we have measured that resists
weight-space attacks. Its objective is a fixed point: *be erased now, and do not become less erased
after someone fine-tunes you.* Its cost is a bi-level problem — every outer step simulates an inner
fine-tuner for 5–30 gradient steps, sampled over 15 attacker configurations.

**TVI replaces the inner simulation with a precomputed direction.** A concept task vector
`tau = theta_finetuned - theta_base` *is* the weight change a fine-tuner produces on that concept,
and U46 measured it as costing **1.7 minutes** to build. The anticipated attacked model is then
`theta + eta*tau`, and the objective becomes:

```
L = L_esd(theta) + lam * [ L_esd(theta + eta*tau) - L_esd(theta) ] + w * L_retain(theta)
```

**Why this is not simply a larger subtraction.** `theta - alpha*tau` moves anti-parallel to tau and
leaves the concept one gradient step away. TVI instead searches for a `theta` whose *directional
derivative of concept generation along tau* is small. Those are different requests, and only the
second says anything about the attacker's dynamics.

**Novelty.** The sweep of 2026-10-28 found no work combining parameter arithmetic with a fixed-point
robustness objective in diffusion (Priority A clear). Pham et al. (arXiv:2404.03631) do task vectors
for diffusion erasure but test only prompt attacks. ResAlign does fixed-point but with full inner
simulation and no task arithmetic.

## 2. What U46 established, and what it did not

**Established:** tau localises — 23 of 686 UNet tensors hold 50% of its energy, 106 hold 90% — and
the direction carries information: subtracting tau gives target 0.8667 where a **norm-matched random
vector of identical magnitude gives 1.0000, i.e. no effect at all.**

**Not established:** that plain task-vector subtraction reaches usable erasure. At alpha = 1.0 it
reaches only 0.8667, and alpha = 1.5 does not improve it. TVI does **not** inherit that weakness,
because it trains an ESD objective rather than subtracting tau; tau enters only as the anticipated
attacker direction.

**Open at declaration time:** U47 is measuring `cos(delta_attack, tau)`. TVI was chosen
*specifically because it does not depend on that result* — tau is used as a cheap attacker estimate,
not as a direction to be orthogonal to. A low cosine would weaken the orthogonal-subtraction
alternative, not this one.

## 3. Design

| held fixed | value |
|---|---|
| base | SD-1.4, pinned revision |
| concept | church |
| tau | `adapters/tv_church/tau.pt` (U46: empty-prompt fine-tune, 500 steps, lr 1e-5, 64 images) |
| ESD negative guidance `esd_eta` | 1.0, matching our ESD port |
| steps | 400 |
| lr | 1e-5 |

**Varied:** `lam` ∈ {0, 1} and `tau_eta` ∈ {1.0}. **`lam = 0` is the ablation and is not optional**
— it is the same training with the invariance term removed, i.e. plain ESD under identical
conditions. Any robustness TVI shows must be shown against it, not against a differently-trained
baseline.

## 4. Decision rule, declared before any number exists

Primary quantities: un-attacked `target_suppression_acc` and `legacy_retain8`; then
**max-over-admissible `H_after`** under our standard ladder (LoRA r4/r16 and full fine-tune, 3 lrs,
3 seeds), admissible = gate-valid and not destroyed.

| outcome | reading |
|---|---|
| target <= 0.40 **and** retain8 >= 0.85 **and** max-over-admissible H materially below the `lam=0` ablation | **the invariance term works.** Proceed to the cost comparison against ResAlign and to a second concept |
| erasure and utility fine, but H indistinguishable from the ablation | the invariance term buys nothing at this `tau_eta`. Sweep `tau_eta` once; if still flat, **report and stop** |
| erasure or utility fails | the objective is mis-specified or the weighting is wrong; diagnose before any further runs |
| `invariance_gap` does not fall during training | **the mechanism is not engaging at all.** Stop and diagnose before interpreting any downstream number |

**Noise floor:** `H_after` reproduces to one quantum (1/15 = 0.067); retained capability to ~0.042.
Differences below those are not differences. The ResAlign work of 2026-10-28 also showed seed
variance can reach sd 0.089, so **robustness claims require n=3 and must be stated as
max-over-seeds**, never a single point.

## 5. The trade-off being claimed, and how it will be measured

ResAlign-level robustness at task-arithmetic cost. Measured as wall-clock and gradient-step count
for the full erasure pipeline, ours versus their published training description. **If TVI's
robustness is materially worse than ResAlign's, the claim must be stated as a Pareto point, not a
win** — the mandate permits an explicitly measured trade-off but not a rebranded loss.

## 6. What U48 cannot establish

- One concept (church), one base model (SD-1.4), n=1 build per cell at this stage.
- `tau` from one fine-tuning configuration; a different tau may behave differently.
- The first-order anticipation `theta + eta*tau` is a one-step approximation of a trajectory. It may
  hold at step 1 and fail by step 50 — which is precisely why ResAlign simulates the full inner loop,
  and is the most likely way this method fails.
- **Not a novelty claim beyond the 2026-10-28 sweep**, which was OpenAlex metadata and abstracts
  plus a partner sweep; no full texts of the task-arithmetic line were read.
