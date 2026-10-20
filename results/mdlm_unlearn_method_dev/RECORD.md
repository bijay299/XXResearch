# Project record — MDLM unlearning method development

One record, three strictly separated tiers. **Nothing in HISTORICAL may be reinterpreted.**

## Tier 1 — HISTORICAL (closed, preserved, immutable)

| Study | Branch @ commit | Conclusion (as published, not to be revised) |
|---|---|---|
| LLaDA access + feasibility | `research/llada-access-smoke-v1` @ `181189ed0453` | access + native masked-diffusion inference + one real LoRA step verified |
| RWKU baseline qualification | `research/llada-rwku-baseline-v1` @ `aecefcd662c5` | **`INSUFFICIENT_BASE_RECALL`** — LLaDA-8B-Instruct 22/100 EM on RWKU probes, 10 entities |
| EMDU vs MDU on RWKU | `research/llada-emdu-mdu-rwku-v1` @ `e2c87cae87e2` | **`PAPER_ALIGNED_RUN_BLOCKED`** — full-parameter 8B does not fit 2x40GB |
| LLaDA PoC (TD-SSD vs SSD) | `research/llada-poc-v1` @ `c3578a908d6c` | preliminary commit; **TD-SSD did not separate from the SSD control** (best point +0.0625 = 1 probe of 16, worse at 150 steps) |
| ARC (amortized reference caching) | `research/arc-amortized-reference-caching` @ `28154b2f001c` | exactness + 1.257x speedup hold; **efficacy parity fails at 128 facts**; per-fact prediction refuted |
| Relearning robustness for MDLM | `research/relearning-robust-mdlm` @ `a67d610044d0` | MDLM unlearning **reversible** (0.049 -> 0.605 in 10 steps); recovery is **directional not positional**; gradient-norm defense **NOT SUPPORTED** |

**TD-SSD / SSD status:** retained as experimental evidence. Neither is the selected approach.
Either may be improved, replaced or abandoned on evidence.

## Tier 2 — ACTIVE DEVELOPMENT (this project)
Everything under `/data/bijaypandey/mdlm_unlearn`. Development evidence; freely inspected;
may be iterated. See `docs/` for the team's working artifacts and `ATTEMPTS.tsv` for every
attempt including failures.

## Tier 3 — FINAL EVALUATION (sealed until selection is frozen)
Not yet created. Will be opened only after method selection, configs, checkpoint rules,
metrics and interpretation are frozen in `docs/FINAL_EVAL_PREREG.md`.

## Resource ledger
Per-workspace ledgers are cumulative and **never reset**. Prior consumption, carried
forward for accounting (not re-charged):

| workspace | GPU device-h | CPU core-h | attempts |
|---|---|---|---|
| llada_smoke | 0.042 | — | — |
| llada_rwku | 0.090 | — | 8 |
| llada_emdu | 0.523 | — | 6 |
| llada_poc | 1.021 | 19.725 | 27 |
| spex (ARC) | 1.353 | 2.246 | 28 |
| anvil (relearn) | 1.088 | 23.166 | 19 |
| **prior total** | **4.117** | **45.137** | **88** |

This project's own ledger starts at zero in `GPU_SECONDS` / `CPU_SECONDS` and is additive
to the above. Note: CPU core-seconds come from `/usr/bin/time` summed over threads.

## Standing constraints
Max **2 GPUs concurrently**; GPUs 2-3 while another user holds 0-1; never interfere with
other users' jobs. No paid services, no infrastructure permission changes, no additional
model-weight downloads. Caches on `/data` (root fs is full). Base-10 forced in all shell
arithmetic (octal-literal bug, documented twice).
