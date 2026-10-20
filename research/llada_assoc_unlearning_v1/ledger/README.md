# Resource ledger — llada_assoc_unlearning_v1 (fresh budget; nothing inherited from earlier sprints)

Ceilings for this assignment: 8 h elapsed (start 2026-10-20T07:01Z), 24 measured CPU-core-hours, ≤ 4 concurrent CPU
threads, ≤ 8 GiB additional storage, no GPU, no model-weight downloads, no paid APIs.
- `resources.jsonl`: every compute command run through `tools/ledger_run.py` (wall, child user+sys CPU seconds, max RSS,
  thread cap). Observed values for the wrapped process tree only.
- `manual.jsonl`: unwrapped work (clones, installs, aborted attempts) with the measurement basis stated.
- Not captured: agent reasoning, file edits, git. No billing telemetry ⇒ no dollar figure is claimed.
- The PyTorch environment `/tmp/claude-0/env-llada` (1.6 GiB) and pinned source clones in `/tmp/claude-0/src` were created by
  the previous (closed) sprint and are reused read-only; they are not counted as new storage here.
