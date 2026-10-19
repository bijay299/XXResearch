# VAU integration notes: GEM c328fc6 (STATIC, NOT EXECUTED)

Files: `native/vau_torch.py` (torch VAU step wrapped around `torch.optim.AdamW`), `native/gem_integration.patch`
(unified diff against `gem/operators/erasure/erasure_gem.py` @ `c328fc6bec687efe60223498d1b6bed2e5fbcffd`, blob
`96c0311`). The patch was made with `git diff --no-index` from a modified copy in the scratch area. `git apply --check -p1`
passes on the clone, and the clone's tracked files were not changed. **torch is not installed here, so neither file has
been imported or run.** The arithmetic is mirrored in `code/vau` and covered by the numpy tests. `adamw_peek` is also
transliterated and checked bitwise against `vau.adamw.peek` (`tests/test_integrity.py::test_native_arithmetic_mirror_by_transliteration`).
Every torch/diffusers/peft API detail below comes from memory and is UNVERIFIED. Line refs `EG:n` are to the pinned file
(VERIFIED by reading `git show c328fc6:...`).

## 1. Where it hooks (patch hunks)
| hunk | location | change |
|---|---|---|
| 1 | after EG:15 imports | `from gem.operators.erasure.vau_torch import ...` (you must copy `native/vau_torch.py` into `gem/operators/erasure/`) |
| 2 | `GEMConfig`, after EG:45 | `retention_method="none"` plus `retained_prompts_probe`, `retained_prompts_gate`, `vau_*` fields |
| 3 | after EG:86 (`optimizer = AdamW(...)`) | `vau = None` |
| 4 | before the training loop (EG:130) | build the retained pools (teacher rollouts, cached base outputs) and `VAUAdamW` |
| 5 | EG:288 `optimizer.step()` | `optimizer.step()` if `vau is None`, otherwise `vau.step()` plus wandb logging of s / accepted / rho_hat |
With `retention_method="none"` the code path is native GEM: the only addition is the `vau = None` assignment.

## 2. Teacher detachment
- Erasure gradient: this does not change. GEM's own backward at EG:263 still fills `p.grad`. The anchor and target teachers
  run under `no_grad` with `disable_adapter()` (EG:219-231). VAU reads `p.grad` exactly where AdamW would, i.e. after
  the optional clipping at EG:283-284 and the gradient assert at EG:286.
- Probes (`build_probe`): there is one graph-building forward on R1 with the adapter ON. `torch.autograd.grad(outs, lora_params,
  grad_outputs=Sigma_i)` then runs k times with `retain_graph` on all but the last call. No teacher enters the probe
  graph. The retained reference `f(theta0)` is a cached constant (`RetainedItem.base_out`, computed under `no_grad` with
  `disable_adapter()`).
- Gate (`rho_hat`): runs under `torch.no_grad`. Trial parameters are written into the LoRA tensors, a forward runs with the
  adapter ON, and the result is compared with the cached base outputs. theta is restored with a bit-exact `copy_` of the
  saved fp32 flat vector.
- `f(theta0) = f(adapter disabled)` holds exactly only while B = 0 at init (peft default). VAU's reference is the base
  model, which is what the protocol's `rho` measures.

## 3. Batching and reduction
- GEM batches over timesteps (chunks of `max_parallel_timesteps=6`, EG:202-204). Its loss sums over latents and divides
  by n_latents (EG:242, 249). VAU leaves this alone.
- In VAU, the retained output `f_x` is the fp32 flattened velocity of one (prompt, latent, timestep) item.
  rho = mean over items of `min(||f - f0||^2, M)`; this is a squared L2 norm, not a mean-square, so its scale grows with
  seq·C. Set tau² and M in the same units.
- `predict_noise` takes one prompt embedding per call (`pipeline_wrapper_flux.py:48-80`), so `_outputs` loops over items.
  Grouping items that share a prompt into one batched call is an optimisation left to the integrator.

## 4. dtype and epsilon
- LoRA params are fp32 (`factory.py:205-215`); the Flux base is fp16. `adamw_peek` runs in fp32 on the flat vector with
  the torch single-tensor formula (lerp for m, `sqrt(v)/sqrt(bc2)+eps`). The foreach/fused AdamW kernels can differ
  from it in the last ulp.
- The projection's k×k Gram and Cholesky are in fp64. P-sized products are in fp32. Ridge `mu = 1e-6·tr(G0)/k`. In fp32,
  `S diag(D) S^T` can lose the small eigenvalues: with D spanning 1e±8 (eps = 1e-8 floor, zero-gradient A entries at
  step 1 give D = 1e8), fp32 accumulation error can exceed mu. Build G0 in fp64 if P·k allows.
- At step 1, LoRA A has g = 0 exactly (B = 0), so D_A = 1/eps = 1e8 while D_B ≈ 1/|g|. The metric weights therefore span
  many decades at step 1. Prototype test `test_lora_zero_init_A_gradient_exactly_zero` covers this.
- The GEM loss `pairwise_distance` eps = 1e-6 all-ones shift is native and unchanged.
- The fp16 base output gives the gate's rho a rounding floor. If tau² is near that floor, rho_hat at theta0 is not 0.
  Measure rho_hat(theta0) before choosing tau².

## 5. LoRA targets and dropout
- Targets: GEM default `qk_dual` = {to_q, to_k, add_q_proj, add_k_proj} in the 19 dual-stream blocks (76 modules, r = 16,
  alpha = 16, dropout 0.1; A_gem_code_facts §4). VAU projects over the concatenation of all trainable LoRA tensors (one
  global metric projection). `VAUAdamW.params = [p for p in group if p.requires_grad]`.
- Dropout policy (stated, not tested): the erasure gradient keeps GEM's train-mode LoRA dropout 0.1 (EG:142). Probes and
  gate run in `wrapper.eval()` (dropout OFF) and restore `train()` afterwards. The probe Jacobian is therefore that of
  the deterministic network, while g comes from a dropout-perturbed one; the numpy surrogate has no dropout.
  Alternative: keep probes in train mode with a fixed dropout mask. That is not implemented.
- Do not merge adapters during VAU. `normal_adapters_temporarily_merged` would change the numerics of trial evaluations.

## 6. Optimizer state and serialization
- `torch.optim.AdamW` keeps ownership of `exp_avg`, `exp_avg_sq` and `step`. VAU never calls `optimizer.step()`. On
  acceptance it writes theta, m', v' and t' into the params and `optimizer.state[p]`. On rejection it writes nothing,
  so the rollback is exact. The `step` tensor is updated with `fill_`. Torch's `capturable` and `differentiable` flags
  are not supported.
- Additions to checkpoints (`VAUAdamW.state_dict()`): `calls` (data index that drives the probe refresh; never rolled
  back), the current probe block `S` (k×P fp32), the sketch generator state, `theta0`, the gate-stream position and
  the cfg. All of these are needed for exact resume. GEM itself does not checkpoint the optimizer
  (`steps_to_save_checkpoints` saves pipelines). Resume support therefore needs a new save hook (not in the patch).

## 7. Deterministic seeding
- VAU uses its own `torch.Generator(seed=vau_seed)` for the sketch, plus pool generators `vau_seed` / `vau_seed+1`.
  GEM's `train_generator` (EG:125), and so the erasure-noise sequence, is untouched. This keeps adamw-vs-vau
  comparisons paired.
- The sketch draws k_max = 64 rows per refresh and uses the first k, so the stream is identical across k (same rule as
  the numpy package). The draw order differs from numpy, so torch and numpy sketches differ for the same seed.
- GPU nondeterminism (atomics in attention backward) makes the probes non-bitwise-reproducible. Use
  `torch.use_deterministic_algorithms(True)` for integrity runs.

## 8. Failure handling (no silent repair)
- Nonfinite g or theta, a parameter with `grad=None`, inconsistent per-parameter `step`, `amsgrad`/`maximize`, more than
  one param group, a non-PD Gram (Cholesky error), or an exhausted gate pool all raise `ContractViolation`. Gate items are
  never reused, because reuse would void the Hoeffding argument. The pool must hold at least T·R·n2 items: at
  T = 23, R = 4, n2 = 8 that is 736 items, about 67 gate prompts × 11 latents.
- g·Δ1 >= 0 gives a zero step. The gate is still evaluated (literal pseudocode) and its decision decides whether
  (m, v, t) commit.

## 9. Required integration changes (beyond the patch)
1. Copy `vau_torch.py` to `gem/operators/erasure/`.
2. Expose the config fields through the CLI (`gem/train.py`; not inspected).
3. Supply retained prompts. Probe and gate pools must be disjoint from each other, from target/anchor prompts and from
   evaluation prompts.
4. Verify the torch/peft/diffusers API assumptions: optimizer state keys, the `step` tensor type, `disable_adapter`,
   `partially_denoise_until_timestep` returning latents of shape (1, seq, C), and `set_timesteps(seq_len, N)` before
   `predict_noise`.
5. Add an optimizer/VAU checkpoint hook if resume is needed.
6. Memory: the probe keeps the autograd graph of n1 Flux forwards for k backward passes. Chunk R1 (accumulate per chunk
   with separate graphs) or use `is_grads_batched=True`.

## 10. Per-step operation count at GEM defaults (symbolic, not measured)
Defaults: 28 sampler steps, prefix end index 10, so n_lat = 11 latents per iteration; chunks of 6+5; about 23 optimizer
steps (budget 250 // 11, A_gem_code_facts §7). FE = one transformer forward on one latent; backward ≈ 2 FE
(protocol convention).

| item | native GEM per iteration | VAU extra per iteration |
|---|---|---|
| student rollout (adapter ON, no grad, guidance 3.0) | 11 forwards (batch 1) = 11 FE | 0 |
| teacher anchor + target (adapter OFF, no grad) | 2×11 = 22 FE (4 batched calls) | 0 |
| student forward+backward on 11 latents | 11 + 22 = 33 FE (2 fwd + 2 bwd calls) | 0 |
| probe refresh (every K) | – | n1 FE graph forward + k backward passes over n1 items = n1 + 2k·n1 FE per refresh; protocol charges k·3·n1/K per step |
| projection | – | Gram k²P + 2kP flops, k³/3 Cholesky (fp64) |
| gate | – | n2 FE per try, 1..R tries (no grad); base outputs cached once |
| optimizer arithmetic | O(P) | O(P) peek + projection, plus 1 extra P-vector copy |
| memory | θ, m, v (3P) | S (kP), m', v', Δ, D, θ backup (≈5P), probe graph of n1 forwards |
| one-time | – | pool rollouts (11 forwards per retained prompt) + one base output per pool item |

Per iteration: native = 66 FE. VAU = 66 + 3k·n1/K + s_tries·n2. Illustrative instantiation (symbolic arithmetic, not a
measurement): k = 16, K = 5, n2 = 8, R ≤ 4.
- n1 = 32: +307 FE/step (probe) + 8..32 FE (gate), i.e. 5.8–6.1× native per step. **This exceeds the protocol's 4× compute
  rule if n1 = 32 is used on the real model.**
- n1 = 8: +77 + 8..32 FE, i.e. 2.3–2.7×.

Over 23 steps native ≈ 1.5k FE. P for Flux qk_dual ≈ 76 × 16 × (3072 + 3072) ≈ 7.5M params. This is DERIVED from Flux
hidden size 3072 for both streams (UNVERIFIED from memory), which gives S ≈ 0.48 GB at k = 16 in fp32.
