# Environment

- Host: managed cloud container, Linux 6.18.44-fc-v139, 4 CPUs visible, 15 GiB RAM; no GPU used or visible to any job (no CUDA libraries installed).
- Python 3.13.16 in venv /tmp/claude-0/venv (system site packages): numpy 2.5.3, scipy 1.18.1, sympy 1.14.0, pytest 9.1.1.
- torch: NOT installed (download.pytorch.org blocked by proxy 403; PyPI CUDA wheel download aborted at 2.7 GB cache and deleted). All native torch files are STATIC, NOT EXECUTED.
- Network: GitHub clone/ls-remote worked; arXiv/ar5iv/alphaxiv/HF/Semantic Scholar unreachable (Bash 403, WebFetch DNS). Git push was refused (403) until the user reconnected GitHub on 2026-10-19 ~23:20Z.
- Pinned external code (read only, clones in session scratchpad, not vendored): GEM multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd; Adam-NSCL ShipengWang/Adam-NSCL@a2f39b4273aa300739c460b33a5e8d2c674632b2; others in SOURCE_MATRIX.md / configs/source_cache.json.
- Reference snapshot: research/audit-and-matched-effectiveness-protocol @ f269656fe4e20fa7d18d05c7465833790b016c62 (branch base; not modified).
- Cost monitoring: CPU core-seconds, wall time and peak RSS are metered per job (code/meter.py -> results/ledger.jsonl). **Dollar spend / cloud-session credit cannot be read or capped programmatically from inside this session**; no auto-top-up or paid API was used. Token usage per subagent is recorded in docs/AGENT_LEDGER.md from harness notifications.
