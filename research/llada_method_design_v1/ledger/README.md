# Resource ledger

- `resources.jsonl`: one record per command run through `tools/ledger_run.py` (wall seconds, child user+sys CPU
  seconds, child max RSS, thread limit). These are **observed** values for the wrapped child process tree only.
- `manual.jsonl`: coordinator-entered records for work not run through the wrapper (installs, clones), with the
  measurement basis stated per record (`basis: observed|estimated|unmeasured`).
- Enforced limits: per-process BLAS/OpenMP thread caps set by the wrapper (default 1). There is **no** cgroup/CPU-quota
  enforcement and **no** cloud-billing telemetry; no dollar cap is claimed.
- Not captured: LLM-agent reasoning time (not CPU work on this host), editor/file operations, git operations.
- Machine: 4 vCPU, 15 GiB RAM, no GPU. Max possible CPU use over 8 h = 32 core-hours (< 48 core-hour ceiling).
