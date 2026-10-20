# ALGORITHM — VAU (Verified Adam-metric Update) for AdamW-trained LoRA erasure

Frozen definition: `configs/candidate.json` (hash in `configs/FREEZE.json`, freeze commit `aec0f38`). Mathematics:
`DERIVATION.tex` (readable: `DERIVATION.html`, `DERIVATION.md`). Selection history: `CANDIDATES.md`. Everything below
is CPU numpy; nothing here touches FLUX, a GPU, or real images.

## 1. Inputs, state, dimensions
| object | shape | meaning |
|---|---|---|
| θ | P | trainable LoRA factors (A, B per adapted layer, flattened); B = 0 at init (peft default) |
| (m, v, c) | P, P, scalar | AdamW moments and step counter; **c advances only on commit** |
| L_e | scalar | native GEM loss `γ·Σ relu(D⁺ − η D⁻)/n_latents` with `F.pairwise_distance` ε-shift, detached teachers (`code/vau/losses.py:gem_loss`) |
| R₁ | n₁ retained inputs | probe batch (n₁ = 32); the student outputs there define f_r ∈ ℝ^{n₁d} |
| S | k × P | sketched retained Jacobian Σ·J_{R₁}(θ), Σ = Z[:k]/√k Gaussian, k = 16 (64 drawn per refresh, common random numbers), k VJPs, refreshed every K = 5 step calls |
| R₂ | n₂ per try | fresh retained verification batch (n₂ = 256; 64 for the flow model), never reused |
| hyper | — | β = (0.9, 0.999), ε = 1e-8, wd = 0.01 (torch defaults; GEM passes only lr), lr tuned per protocol, ridge μ = max(1e-6·tr(SDSᵀ)/k, 1e-12), τ² tuned, M = 2τ², δ = 0.05, R = 4, s ∈ {1, ½, ¼, ⅛} |

## 2. One step (exactly what `code/vau/methods.py: make_arm('vau') = Gated(VAUNoGate)` executes)
1. g = ∇θ L_e(θ) — one student forward+backward; two no-grad teacher forwards (adapter off).
2. Tentative AdamW: m′, v′, c′ = c+1, bias-corrected m̂, v̂; D = 1/(√v̂ + ε); Δ₀ = −lr·D⊙m̂ − lr·wd·θ (`adamw.peek`, no mutation).
3. If the step-call index ≡ 1 mod K: rebuild S (k VJPs on R₁). If S is empty/zero: Δ₁ = Δ₀ and skip 4–5.
4. Δ₁ = Δ₀ − D⊙Sᵀ(SDSᵀ + μI)⁻¹SΔ₀ (Cholesky; `projections.metric_project`).
5. Reject: if g·Δ₁ ≥ 0 → null step: no gate tries, no batch consumed, nothing committed.
6. Gate: for s = 1, ½, ¼, ⅛ draw a fresh R₂; ρ̂ = mean min(‖f_x(θ+sΔ₁) − f_x(θ₀)‖²/Z, M); accept the first s with
   ρ̂ + M√(ln(TR/δ)/(2n₂)) ≤ τ². No acceptance → null step.
7. Commit on acceptance: θ ← θ + sΔ₁; (m, v, c) ← (m′, v′, c′) computed from the **raw** g. On any null step (θ, m, v, c)
   are unchanged bitwise; the data stream and the probe-refresh counter still advance.

Pseudocode with the same semantics: `CANDIDATES.md` §"Exact candidate". Torch transcription (STATIC, NOT EXECUTED):
`native/vau_torch.py`; GEM hook: `native/gem_integration.patch` (replaces `optimizer.step()` at `erasure_gem.py:288` of
GEM `c328fc6`).

## 3. Arms evaluated against it (same probes, same gate batches where applicable)
`adamw` (native GEM optimisation), `adamw_stepmatch`, `penalty` (same R₁), `pre_proj` (GPM-style), `post_euclid`,
`post_euclid_reject` (Adam-NSCL-style post projection with VAU's reject rule), `vau_nogate`, `vau_blocksplit`, `na_lora`
(null-A / InfLoRA-style, known method), each also `gated_<arm>` with the identical gate and `earlystop_<arm>`.
Definitions: `code/vau/methods.py`; protocol: `configs/eval_protocol.json` v1.2, `docs/EVAL_PROTOCOL.md`.

## 4. Reproduction (one command each; CPU only)
```
cd research/flow_method_design_v1
/tmp/claude-0/venv/bin/python -m pytest -q code/tests               # 238 unit/contract/FD tests
python3 code/meter.py dev -- /tmp/claude-0/venv/bin/python code/run_validation.py --phase dev
python3 code/meter.py val -- /tmp/claude-0/venv/bin/python code/run_validation.py --phase validation   # refuses unless FREEZE.json hashes match
```
Environment: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, pytest, sympy (venv with system site packages); torch absent.
Any Python ≥ 3.11 with numpy ≥ 2 and scipy should work; the venv path is only where this sprint installed them.
