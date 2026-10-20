# COST_MODEL — measured CPU, symbolic scaling, unmeasured GPU projection (kept separate)

## 1. Measured on this CPU host (numpy prototype; `results/ledger.jsonl`, `results/E_smoke.json`, validation `summary.json`)
| item | measured |
|---|---|
| one `arm.step`, 5-step smoke, dev seed 0 (CPU-s per 5 steps) | adamw 2.1 / 2.2 / 15.7 ms (linear / MLP / flow); vau 6.3 / 6.6 / 28.7 ms |
| median CPU per validation run (frozen timings) | 0.12 s (model a), 0.23 s (b), 1.66 s (c) |
| full validation phase (636 units, 12 seeds, 16–17 arms × 4–8 configs) | 13,920 core-s = 3.87 core-h; 3593 s wall on 4 workers; peak RSS 1.53 GiB |
| forward-equivalents per run, median (protocol accounting; FE = one model forward per sample, backward = 2 FE, probe VJP row = 3 FE per sample, gate try = n₂ FE) | G1: adamw 996, penalty 4,433, post_euclid_reject+gate 24,213, **vau 29,245**; G8: adamw 888, penalty 2,444, post_euclid_reject+gate 5,808, **vau 12,032** |
| vau / best control (median FE) | 1.43 (G1), 1.51 (G6), 19.9 (G5), 27.1 (G7), 2.07 (G8) |
| vau / native adamw (median FE) | 20.2 (G5) – 64.1 (G7) |
Where VAU's extra cost goes (from op counters): gate tries (≤ R·n₂ retained forwards per step) dominate; probe VJPs
(k per refresh) second; the k×k Gram + Cholesky per step is negligible at these sizes. Rejected/null steps still pay the
erasure forward/backward and any gate tries.

## 2. Symbolic per-step scaling at GEM defaults (not measured; `native/INTEGRATION_NOTES.md` §10)
Native GEM iteration (verified structure, `agents/A_gem_code_facts.md`): 11-latent student rollout (11 FE) + anchor and
target teachers (22 FE) + student forward/backward (33 FE) = **66 FE**, ≈ 23 iterations.
VAU extra per iteration: probe 3·k·n₁/K FE (k = 16, n₁ retained items, refresh K = 5), gate s_tries·n₂ FE, Gram k²P
+ k³/3 flops, optimizer state +≈ 5P floats (S is k·P), plus a one-time pool of retained rollouts and cached base outputs.
P for FLUX `qk_dual` LoRA r=16 ≈ 76·16·(3072+3072) ≈ 7.5 M (DERIVED; hidden size 3072 from memory, UNVERIFIED) ⇒
S ≈ 0.48 GB fp32 at k = 16. Illustrative: n₁ = 32 ⇒ +307 FE (probe) + 8–32 FE (gate) ≈ 5.8–6.1× native per step;
n₁ = 8 ⇒ ≈ 2.3–2.7×. The frozen protocol's 4× compute rule would fail at n₁ = 32.

## 3. GPU / FLUX (unmeasured projection only)
No GPU was used. Wall-clock or dollar figures for FLUX are **not** estimated here; `docs/REAL_MODEL_PROTOCOL.md` contains
an assumed budget (~88 GPU-h, cap 96) that has not been measured and is not a claim. No speedup or exactness on GPU is
claimed.

## 4. Whole-sprint accounting (reconciled 2026-10-20)
Charged CPU = sum of `results/ledger.jsonl` entries excluding the nested self-meter `E_run_validation` (13,914.8 core-s),
which is contained in the enclosing `VALIDATION_PHASE_v1_2` (13,920.5 core-s): **17,413 core-s ≈ 4.84 core-h** of the
48 core-h ceiling (before the closeout entries appended after this reconciliation; see `docs/AGENT_LEDGER.md`).
Failed attempts included: 1,682 core-s (D_window_toy, F_f4_lora_toy, F_f5_diag, the first E_run_dev crash 1,577 core-s,
and a 0-s validation launch failure). Peak RSS of any metered job 1.53 GiB (cap 16). Gaps: short unmetered shell/Python
calls by agents and the coordinator (file reads, JSON inspection, pip installs) are not in the ledger; the RSS cap was
enforced only via RLIMIT_AS in `meter.py`, and the 4-core cap only by worker count; dollar/credit spend could not be read
or capped from inside the session.
