# COST — measured CPU costs (toy) and LLaDA-8B projections (labelled)

## 1. Measured on the CPU toy (world 0, 200 steps, 1 thread/process; `results/dev/compare_w0/runs/*.json` counters)

Units: **row-forwards** = one forward pass of one (prompt‖answer) sequence (16 tokens). Averaged over each arm's LR runs.
Training wall time excludes evaluation.

| arm | student fwd/step | student bwd/step | reference fwd/step | rollout fwd/step (run / needed up to τ) | total fwd/step | train ms/step | rollout ms/step |
|---|---|---|---|---|---|---|---|
| SC-TF | 64 | 64 | 32 | 0 | 96 | 34.7 | — |
| SC-TF-cm (T6, B_f=52) | 84 | 84 | 52 | 0 | 136 | 43.7 | — |
| SC (MDU rollout flag, T=8) | 64 | 64 | 32 | 130.4 / 130.4 | 226 | 57.4 (incl. rollout) | (in train) |
| FX-Mix (P1–P3, α=0.25, J=4, R=4) | 64 | 64 | 32 | 40.0 / 35.7 | 136 | 35.2 | 12.1 |
| FX-Mix-greedy (P1 only) | 64 | 64 | 32 | 48.0 / 46.6 | 144 | 34.6 | 9.3 |
| FX-Mix no-trunc (T5b) | 64 | 64 | 32 | 40.0 / 40.0 | 136 | 32.1 | 11.3 |
| FX-Mix ε=0 (T5d) / rnd-TF (T5c) | 64 | 64 | 32 | 40.0 / 36.6 · 36.2 | 136 | 30–31 | 11 |
| MDU τ=1 / NPO | 64 | 64 | 32 | 0 | 96 | 32.2 / 32.4 | — |

- Rollout cost of FX-Mix on the toy: 24 forget prompts × 1 rollout per refresh, configs round-robin (8, 8, 4 steps).
  That is ≈ 160 row-forwards per refresh, every R = 4 steps → 40 per step, i.e. **+42 % forwards over SC-TF**.
  These are no-grad forwards. SC-TF-cm matches FX-Mix at 136 forwards/step (checked against the counters).
- SC follows MDU's code: one random-stop greedy rollout of the whole forget batch at **every** step (32 rows × E[stop] ≈ 4).
  That is 3.3× FX-Mix's rollout forwards per step.
- Truncation at the first F-commit saves little on the toy (needed/run = 0.89), because the batched sampler runs to
  the end unless every row has leaked. The LLaDA port stops a batch once all rows have leaked (`stop_when_all_leaked`).
- Evaluation dominates toy wall time. One checkpoint costs 5.3–5.7 s: 9 configs × (forget 24 prompts × 8 samples +
  retain 84 facts × 2 templates × 4 samples + copy 100), run on a merged-LoRA copy. Base-model cost is the same.
- Run totals (ledger):
  - one 200-step run ≈ 50–57 s CPU, of which ~45 s is evaluation;
  - world-0 comparison: 27 runs ≈ 1427 CPU-s (2 shards: 737 s + 691 s);
  - smoke test 197 s; references 15 s;
  - world 1 (8 arms × 3 LR × 300 steps): see DEV_LOG.

## 2. LLaDA-8B projections (NOT measured — arithmetic from stated assumptions; real-model execution UNTESTED)

**Assumptions:**
- LLaDA-8B: 32 layers, d_model 4096, FFN 12288, |V| = 126,464, ≈ 8.0 B params. Recalled from the LLaDA paper and
  consistent with the dllm copy of `modeling_llada.py`; UNVERIFIED here.
- bf16 weights. Forward ≈ 2·N·tokens FLOPs; LoRA backward ≈ 2× forward (conservative).
- Sequence ≈ 200 tokens (chat-templated TOFU question ≈ 40–80 tokens + gen_length 128).
- FX-Mix defaults from `configs/hyperplane_fxmix.yaml`:
  - B_f = 8 forget states per step (α = 0.25 → 2 TF_ff + 6 pool states);
  - B_r = 8 retain sequences;
  - R = 4;
  - 8 forget prompts rolled out per refresh;
  - pool steps {128, 128, 64, 32} → mean 88 forwards per full rollout.

| quantity per optimisation step | SC-TF | SC (MDU rollout flag, T=16, GT length) | FX-Mix |
|---|---|---|---|
| student fwd + bwd (sequences) | 16 + 16 | 16 + 16 | 16 + 16 |
| reference fwd (adapter-disabled, no grad) | 8 | 8 | 8 |
| rollout fwd (no grad) | 0 | 8 × E[stop] ≈ 8 × 8 = 64 | 8 × 88 / 4 = **176** (upper bound; truncation and all-leaked early stop reduce it) |
| FLOPs (≈ 3.2e12 per sequence-forward) | 1.5e14 + 2.6e13 ≈ **1.8e14** | ≈ 3.8e14 | 1.5e14 + 2.6e13 + 5.6e14 ≈ **7.4e14** (≈ 4.1× SC-TF) |
| projected time/step, 1× H100 at 35 % MFU (bf16 989 TF/s peak) | ≈ 0.5 s | ≈ 1.1 s | ≈ 2.1 s → 300 steps ≈ 11 min (+ eval) |

**Vocab-dependent operations (|V| = 126,464):**
- Student and reference logits:
  - full logits b × L × |V| in bf16 = 8 × 200 × 126,464 × 2 B ≈ **405 MB** per pass;
  - only the masked answer positions are upcast to fp32 (≤ 8 × 128 × |V| × 4 B ≈ **518 MB** per tensor).
- Exclusion target r_ε: three log-sum-exps, a `where` and an `exp` over (n_pos, |V|).
  - Elementwise work ≈ 1e8–1e9 FLOPs, negligible next to the model's ≈ 1e14.
  - Transient peak memory ≈ 2–3 GB (z0, log r, r, log p_θ in fp32).
- Native sampler: as upstream, `add_gumbel_noise` casts the full logits to float64 (b × L × |V| × 8 B ≈ 1.6 GB for
  b = 8, L = 200 at T > 0). Low-confidence remasking also builds a full-vocab softmax.
- Certificate B̂_π: one tempered softmax per step (float64) on logits the sampler already computed. No extra forwards.

**Memory:**
- Model weights, bf16: ≈ 16 GB.
- Reference pass via PEFT `disable_adapter()`: **0 extra**. MDU's released code holds a separate `ref_model` (+16 GB).
- LoRA r = 16 on 7 linears × 32 blocks:
  - per block 4·16·(4096+4096) + 2·16·(4096+12288) + 16·(12288+4096) = 1.31 M; total **41.9 M params**;
  - fp32 params + grads + AdamW m, v at 16 B/param ≈ **0.67 GB**.
- Activations for 16 × 200 tokens with grad: a few GB (gradient checkpointing optional on 80 GB).
- Rollout pool storage (answer part only, int64):
  - 8 prompts × (J + 1 = 9) states × 128 × 8 B ≈ **74 KB** per refresh;
  - generation trace recording ≈ 128 steps × 8 × 128 × 8 B ≈ 1 MB.

**ε fixed point (SPEC v1.3):** at ε = 1e-5 with n̂ ≤ gen_length = 128 decision positions:
- implied fixed-point leak ≤ 1 − (1 − 1e-5)^128 ≈ **1.3e-3**;
- certificate floor ε · n̂_elig: Σ_n|E_n| ≈ 8.3e3 for a single-block lc schedule, so ≈ 0.08 as an upper value; random
  remasking (Σ k_n = 128) gives ≈ 1.3e-3.
- Measure both with `fixed_point_report` on θ0 rollouts before training.

**Evaluation cost on LLaDA:** dominated by native generation over the grid. Per config:
- forget prompts × samples × steps sequence-forwards;
- e.g. 400 TOFU-forget10 prompts × 4 samples × 128 steps ≈ 2e5 sequence-forwards ≈ 6.6e17 FLOPs ≈ 30 min per
  config on one H100 at 35 % MFU.
- Held-out grids must therefore be sized deliberately (subsample prompts for stochastic configs).
