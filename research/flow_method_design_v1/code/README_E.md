# Agent E: VAU numpy CPU prototype (core library)

Status: the core library and its tests are built. **No validation experiment has been run and no validation seed has been
used.** Only development seeds 0..99 were used. The runner that wires these pieces into `configs/eval_protocol.json`
belongs to a later wave.

## Layout
| path | content |
|---|---|
| `code/vau/contracts.py` | `ContractViolation`, `check_*`, `Ops` counters (network ops as calls/samples; events), `dev_rng` |
| `code/vau/seeds.py` | protocol streams `SeedSequence([20261019, FAMILY_CODE, seed]).spawn(8)`; **refuses seeds 7000..7011 unless `configs/FREEZE.json` matches sha256(eval_protocol.json), sha256(candidate.json) and the code/vau tree hash**; refuses all other non-dev seeds |
| `code/vau/adamw.py` | functional AdamW (torch single-tensor semantics): `peek(state,g,theta,hp)->(delta0,tentative,D)`, `commit`, `step` |
| `code/vau/losses.py` | native GEM loss (`||v-a+eps·1||`, relu, sum, gamma/n_latents, `eps_n` knob), CPR, anchor regression, retention penalty (mean over R1), analytic d/dv |
| `code/vau/models.py` | (a) `LinearModel` b_x + J_x θ with column scales 10^U(-d/2,d/2), `structured_T` (G2: U diag(κ^{-j/(m-1)}) Vᵀ R(φ)), (b) `LoraMLP`, (c) `TinyFlowDiT`, plus `flux_sigmas`, batched `euler`, `rollout_prefix`, `sample`, `lora_gauge_step`. Each model has forward / jvp / vjp / guarded dense `jacobian` |
| `code/vau/tasks.py` | GEM-style erasure tasks; concept coordinates; rare subgroup (cos 0.8, mass 0.005); shift; contrast scale; held-out flow prompts; `input_hashes` (I10) |
| `code/vau/projections.py` | `weighted_null_project` / `metric_project` / `euclid_project` (ridge `ridge_rel·tr(G0)/k`), `sketch_probe` (k_max = 64 rows drawn per refresh, first k used, k = 0 allowed), `exact_probe` |
| `code/vau/methods.py` | arms `adamw, adamw_stepmatch, penalty, pre_proj, post_euclid` (no reject rule), `post_euclid_reject`, `vau_nogate`, `vau_blocksplit_nogate`, `na_lora`; wrappers `Gated` (`vau`, `vau_blocksplit`, `gated_<arm>` incl. `gated_na_lora`) and `EarlyStop` (`earlystop_penalty`); `inject` hook (G5 gauge); `arm_status` / `NotApplicable` |
| `code/tests/` | 224 pytest tests (222 pass, 2 skipped = NOT_APPLICABLE cells) |
| `code/smoke.py` | 14 arms × 3 families × 5 steps, dev seed 0 |
| `native/` | `vau_torch.py`, `gem_integration.patch`, `INTEGRATION_NOTES.md` (static, not executed) |

## API
```python
from vau import tasks, seeds
from vau.methods import make_arm, Batch, GateConfig
st   = seeds.streams("G5", 0)                          # dev seed; 7000+ refused without FREEZE.json
task = tasks.make_task("mlp", st["fixture"], gain=1.0, aniso_decades=2)   # or "linear" / "flow"
th0  = task.model.init_theta(st["fixture"]); R1 = task.retained_batch(st["splits"], 32)
arm  = make_arm("vau", task, gate=GateConfig(tau2, M, n2, delta, T, R=4), k=16, K=5)
ms   = arm.init_state(th0, R1, st["sketch"])
th, info = arm.step(ms, th0, Batch(erase=task.erasure_batch(st["train"]),
                                   gate=tuple(task.retained_batch(st["gate"], n2) for _ in range(4))))
```
`info` holds loss, s, accepted, rho_hat (one per try), slack, progress_lin (= -g·Δ), step_norm, step_Hnorm, and
nonprogress_zeroed. Counters are in `ms.ops`. Arm config: `probe='sketch'|'exact'`, `k` (0..64), `K` (int, or None for
"build once"), `ridge_rel`, `reject_nonprogress`, `inject`, `lam` (penalty), and `ref_norms`/`c`/`metric='H'|'euclid'`
(stepmatch).

Semantics fixed in code (including the F_REVIEW §4.2 fixes):
- Probe refresh is driven by `ms.calls`, the data index, which is never rolled back. The refresh rule is `calls % K == 0`.
- The Adam counter t advances only on commit (`AdamState.t`), and bias correction uses it.
- Rollback restores (θ, m, v, t) and returns `theta.copy()`. The `AdamState` object is left untouched, so rollback is
  bit-exact. The probe block is not rolled back.
- If g·Δ1 >= 0, the step is a **null step**: no gate tries, nothing committed, θ unchanged. It is counted as
  `nonprogress_zeroed`, not as `rejected_step`.
- Ridge μ = max(1e-6·tr(G0)/k, 1e-12).
- When S is empty or all-zero (k = 0) the correction and the reject rule are both skipped, so the result is bitwise
  AdamW (I2, unconditional).
- `vau_blocksplit*`: D is normalised per LoRA factor block per layer, D_A/mean(D_A) and D_B/mean(D_B). This is used only
  as the projection metric; Δ0 is still the true AdamW step.
- `na_lora` (InfLoRA-style null-A LoRA, a known method):
  - At init, U_l is computed from one feature forward on R1 per LoRA module (SVD, relative tol 1e-10), and
    A_l ← A_l(I − U_lU_lᵀ). A is frozen; AdamW state is kept over the B blocks only.
  - Runs must start from `arm.initial_theta(ms)`. Passing an unprojected A raises.
  - Probe cost: 1 forward, 0 VJPs.
- LoRA-only arms raise `NotApplicable` on the linear family; `arm_status()` returns `'NOT_APPLICABLE'`.
- adamw_stepmatch commits AdamW state from the raw g even when the target norm is 0.

## Test status (`meter.py E_tests_final2`, 222 passed / 0 failed / 2 skipped (NOT_APPLICABLE), 7.8 CPU-s)
- **FD (I1):** central differences in FP64, rel err <= 1e-6. Covered: GEM/CPR/anchor/penalty gradients; JVP, VJP and the
  dense Jacobian for all 3 families; the full-chain erasure gradient; the adjoint identity. A mutation check (×1.0001 in
  the attention VJP) is caught.
- **AdamW (I7):** matches an independent per-element loop at rtol 1e-12 over 40 steps for 3 hyper-parameter settings
  (including both lerp branches). `peek` is pure. Step 1 = -lr·sign(g).
- **Projector:** S Δ1 = 0 (1e-12, μ = 0); the ridge residual identity holds; the minimiser matches scipy SLSQP and an
  independent KKT solve; ||Δ1||_{D⁻¹} <= ||Δ0||_{D⁻¹}; H-progress >= 0 when m̂ = g; sketch = Σ J.
- **Gate:** accepts within budget; rejects and rolls back bit-exactly (I9; 3 families × 4 arms); backtracks to s = 1/2
  exactly on the linear model.
- **Other:** LoRA zero-init gives an exactly zero A-gradient at step 1 (mlp, flow); op counters are exact for 9 arm
  variants × 2 families plus the reject path; invalid inputs (shape, dtype, NaN/Inf, missing gate batches, bad configs,
  non-Generator rng) raise.
- **Integrity checks:**
  - I2: k = 0 + no gate == AdamW bitwise over 20 steps, for vau_nogate/vau/pre_proj/post_euclid × 3 families.
  - I3: constant D gives the Euclidean projection (1e-12).
  - I4: see infidelity 8.
  - I5: lr = 0 leaves θ bitwise unchanged for every arm.
  - I6: τ = ∞ gate == ungated, bitwise, for 6 arms × 2 families.
  - I8: CRN; the sketch is shared across k.
  - I10: hash helper.
  - I11: seed guard.
  - I12: flow endpoints at LoRA = 0 equal the base endpoints bitwise.
  - Also: v = b exactly at init; knobs (anisotropy, κ/φ, rare, shift, contrast down to 1e-6, eps_n = 0, gauge
    first-order neutral but drifting, inject hook, early stop, K = None).
  - Native: static AST checks, and `adamw_peek` transliterated and checked bitwise against numpy.
- **F_REVIEW additions:**
  - A null step skips the gate and commits nothing (vau, vau_nogate, post_euclid_reject).
  - The Adam counter advances only on commit.
  - The μ floor holds.
  - The blocksplit metric is normalised per block.
  - na_lora: probed retained outputs are unchanged to 1e-12 after 40 AdamW steps at lr 0.1, while target outputs move
    (MLP, and the attention model with N_r = 1).
  - The factor-wise constraints dA U = 0, dB(AU) = 0 with B = 0 and rank(AU) = r keep B ≡ 0.
  - Op counters for all new arms on 3 families; NOT_APPLICABLE on linear.

## Smoke (`meter.py E_smoke`; `results/E_smoke_output.txt`, `results/E_smoke.json`)
Measured CPU time of `arm.step` over 5 steps (placeholder hyper-parameters, nothing tuned):
| | linear (P=40) | mlp (P=88) | flow (P=128) |
|---|---|---|---|
| adamw | 1.9 ms | 2.1 ms | 13.7 ms |
| vau | 5.7 ms | 6.4 ms | 27.5 ms |

na_lora: 2.4 ms (mlp) and 15.5 ms (flow). Its ops per 5 steps are forward 6/63, vjp 5/55, 1 probe build.

Per 5 steps for vau with k = 8, K = 2, N_r = 8, n2 = 16 (calls/samples):
- forward 8/79, vjp 29/247, teacher 10/110, gate_forward 5/80, 5 solves, 3 probe builds.
- flow also has 55 rollout forwards.

adamw per 5 steps: forward 5/55, vjp 5/55, teacher 10/110. The whole smoke job takes 2.2 CPU-s. Sprint usage by Agent E
is about 14 metered CPU-s plus a few unmetered seconds.

## Infidelities and deviations (numpy surrogate vs native torch/GEM)
1. **FP64 everywhere.** GEM computes the loss in fp32 (`.float()`) on an fp16 base. The fp16 output noise floor that the
   gate would see is absent here.
2. **No LoRA dropout.** peft applies dropout 0.1 to the LoRA input in the student pass. The surrogate is deterministic.
   The native file states its policy: probes and gate in eval mode.
3. **Model (c) is a toy.** It has one single-head joint attention block. There is no adaLN modulation, MLP, RoPE,
   multi-head attention, VAE latents, guidance embedding or pooled text. The rollout ignores GEM's guidance 3.0 and the
   loss's guidance 1.0. The time shift uses the assumed diffusers exponential formula with μ = 0.630. Prompts are fixed
   random token tables, not text encoders.
4. **GEM details reproduced:** a prefix of 11 states after steps 0..10, the state after step k paired with σ_k
   (EG:214-215 quirk), a student rollout with no grad, teachers with the adapter disabled, sum then γ/n_latents.
   **Not reproduced:** chunking into 6+5 (it gives the same gradient sum), the step-budget arithmetic, and prompt
   augmentation.
5. **Retained data for model (c):** queries are base-trajectory states at σ_k with k uniform over 0..27. In the
   protocol these are generated on the fly by the runner and are not charged to arms. Native GEM has no retained data.
6. **Op units:** a "sample" is one per-input network evaluation. The exact probe is counted as N_r·d single-sample VJPs.
   The gate's base outputs are counted (`gate_base`) but are cacheable, and the protocol does not charge them. FE
   conversion is left to the runner.
7. **AdamW:** `theta + delta0` equals torch's `theta*(1-lr·wd) - step_size·m/denom` only up to the last ulp. The
   foreach/fused kernels can also differ in the last ulp. Torch source line numbers are UNVERIFIED.
8. **I4 versus the frozen ridge.** With μ = 1e-6·tr(G0)/k the exact-J residual ||JΔ1||/||JΔ0|| is about 3e-6, not
   <= 1e-8. The residual identity μ(G0+μI)⁻¹SΔ0 is tested. I4 passes (<= 1e-15) only with `ridge_rel = 0`, which leaves
   only the 1e-12 floor and requires J to have full row rank (that test uses 4 R1 samples). **Coordinator decision
   needed:** either amend I4 to use ridge_rel = 0, or accept about 1e-5.
9. **I2 is now unconditional.** The reject rule is skipped together with the correction when S is empty. This is my
   reading of "skip the correction when S = 0"; it is a spec interpretation and should be stated in the candidate.
10. **Native compute.** At GEM defaults with n1 = 32, the probe charge 3k·n1/K is about 307 FE/step against 66 FE/step
    for native GEM, i.e. above the protocol's 4× compute rule (INTEGRATION_NOTES §10). In this toy n1 = 8.
11. **Sketch draw order differs** between numpy (one row-major Z of k_max × N_r·d) and the torch sketch. The two are
    equal in distribution but not in values.
12. **Protocol naming conflict.** eval_protocol.json defines `post_euclid` WITH the reject rule. The coordinator's
    update asked for a separate `post_euclid_reject`. Here `post_euclid` has no reject rule and `post_euclid_reject`
    is the protocol's "VAU-Euclid" with the rule. The runner must map the names accordingly.
13. **na_lora degenerates when probe features have full rank.** If U_l spans all of d_in then A_l = 0 and there is no
    erasure capacity. In the toys:
    - flow, N_r = 8: rank 8 of dm = 8 for all four modules, so A = 0 and drift and erasure are both exactly 0 in the
      smoke run;
    - mlp fc2, N_r = 32: rank 16 of 16;
    - mlp fc1 is never full rank, because the concept coordinates are absent from retained inputs.
    At Flux scale, 32 items × ~1.5k tokens of width 3072 make a full numerical rank at tol 1e-10 likely (UNVERIFIED).
    na_lora then needs an energy threshold (InfLoRA/GPM-style) rather than 1e-10. That changes the method; it is a
    coordinator/protocol decision and is not implemented.
14. **Not built in this wave:** G3's construction g = cos ψ g_null + sin ψ g_range (needs fixture teacher maps in the
    runner), evaluation observables (E_hinge, E_end, rho_*), FE accounting, penalty_oracle, and the runner itself.

## Wave 3: validation runner (`code/run_validation.py`)
- **Command:** `python code/run_validation.py --phase dev|validation`. It re-executes itself under `code/meter.py`
  (wall limit 6 h) and runs ≤ 4 fork workers, one work unit per (family, cell, seed).
- **Validation guard:** the phase refuses unless `configs/FREEZE.json` exists, its sha256 of eval_protocol.json,
  candidate.json, the code/vau tree and run_validation.py all match, and it carries `calibration` and `timings`.
- **Outputs:** `results/<phase>/raw/<family>/<cell>__s<seed>.jsonl.gz` (one run per line: cfg, checkpoint
  trajectory, per-step log, mechanism log, F3 record, ops); `units.jsonl`; `summary.json` (coverage classes,
  matched tables, Q1–Q4, safety, F1–F5, compute rule, G9 mechanism, GD-B/GD-D, verdict, DECISIONS D1–D23).
- **Covered:** families G1–G9, GD_B, GD_D, penalty_oracle, the G5 gauge subtest, G9 measurement runs, the stepmatch
  paired vau runs, θ0 coverage (C_probe, C_probe_D, eff_rank_99, C_feat) and coverage drift along one vau run per
  unit, the full statistics, the budget hard stop at 6.0 core-h, and crash/timeout recorded as failures (never
  retried).
- **Tests:** `code/tests/test_runner_stats.py` checks crossing/interpolation, config selection, log-ratio rules, the
  exact sign-flip test, bootstrap, Holm, binomial, fallback ladder, Spearman and FE on synthetic inputs.
- **Candidate:** `configs/candidate.json` written. Amendment A2 is recorded in `configs/eval_protocol.json` (v1.2)
  and `docs/EVAL_PROTOCOL.md`.

### Library changes in this wave
- fast linear model (contract θ first), with exact-J and sketch probes via `phi` / `vjp_rows`
- `probe_forward` / `probe_vjp` op categories
- cached gate base outputs
- na_lora energy threshold
- vau_blocksplit on a single block
- `mu_floor` knob
- covered-population and concept-target task modes
- flow erasure batch `(z0, target_id, idx)`
- `TinyFlowDiT.sample_fast`
- G9 seed code

### Dev phase results (DEVELOPMENT ONLY; dev seeds; not evidence)
- `results/dev/integrity.json`: all I1–I15 pass, plus the runner-level I5/I10/I11/I12 checks.
- `results/dev/calibration.json`: seeds 0..7.
- `results/dev/compute_projection.json`: 4.03 core-h before the ladder, 3.69 after L1.
