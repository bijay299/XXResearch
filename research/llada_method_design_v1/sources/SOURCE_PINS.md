# Source pins and access status

| Source | Pin / identity | Access in this sprint | What was inspected |
|---|---|---|---|
| LLaDA official repo | github.com/ML-GSAI/LLaDA @ `9182493720ed723ef8031210d85959364e51cbe0` (2026-07-15) | git clone OK | GUIDELINES.md (pretrain/SFT loss), generate.py (native sampler), get_log_likelihood.py, chat.py |
| LLaDA model code `modeling_llada.py` | HF hub GSAI-ML/LLaDA-8B-Instruct | **BLOCKED** (huggingface.co 403) | not inspected — module names UNVERIFIED |
| MDU paper | arXiv 2605.18253v1 (PDF supplied by user, sha256 `82e1d2adfb95e26824e337dc9a86182d918fb4de40531d3b8ae36a1ead0cf3e0`) | local PDF | full paper incl. App. A (Alg. 1), B (setup, baselines) |
| MDU code | github.com/leegeoru/MDU @ `e5e8e8274ff8f56f16e978803d9e902359eb5272` | git clone OK | `src/unlearn_mdu_llada.py` (sha256 `9f0b9476…0f84337`), configs |
| arxiv.org, huggingface.co, proceedings.mlr.press, openreview.net, semanticscholar | — | **BLOCKED** (proxy 403 / DNS) | see PRIOR_ART.md for per-source status |
