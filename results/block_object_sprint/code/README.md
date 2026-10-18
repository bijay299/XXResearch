# Code snapshot — read before reusing any of it

This directory is the **executable snapshot** of the sprint, preserved unmodified. The sprint
closed at **`BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS**; see
[`../AMENDMENT_01_CLOSEOUT.md`](../AMENDMENT_01_CLOSEOUT.md), which is authoritative.

Nothing here was changed by the closeout. These warnings are documentation only.

## 1. The measurement these scripts implement is unqualified on the generated domain

`score_presence.py` implements the frozen binary target-presence procedure. It passed its
independent real-photograph controls (20/20 sensitivity and 20/20 specificity per object on
120 annotated COCO images) and **failed to be qualified on generated, heavily stylised
images**: the agent's audit disagreed with the judge on 5 of 40 native-BLOCK Bears images, and
that disagreement was recorded as unresolved.

Those are **agent-versus-judge disagreements**, not errors adjudicated against independent
ground truth, so no verified scorer error rate follows from them. `validate_scorer.py`
validates against real photographs only; **passing it does not establish validity on generated
images**. `analyse.py`'s gate and comparison outputs inherit this limitation.

`score_presence.py` and the release's `scripts/eval/llava.py` call the **same** LLaVA
checkpoint with different prompts. They are **not** independent instruments and do not
cross-validate each other.

## 2. `ledger.py` can lose concurrent updates

`ledger.py` performs an **unlocked read–modify–write** of a single JSON file and stages it
through **one shared temporary filename** (`ledger.json.tmp`). Jobs in this sprint ran
concurrently on two GPUs and appended through this path, so **lost updates were possible**.

No actually-lost entry was established, and none is claimed. The writer was **not** repaired
and **not** tested. The archived `../evidence/ledgers/ledger.json` is preserved as-is;
its known reconciliation gaps — a 4-image difference between the image ledger (563) and the
sum of job-level `images` fields (567), and failure accounting recorded as two *groups*
rather than per attempt — are documented in the amendment §5 and are **unresolved**, not
repaired.

## 3. The native erasure path carries a disclosed ambiguity

`run_arm.py` applies the authors' own `apply_block_erasure`. For Bears that update writes 64
context decoder rows of which only **57 are unique**, so **7 writes are order-ambiguous** and
PyTorch leaves the winner unspecified. Separately, the target-to-empty contribution identity
described in [`../INTEGRATION_AUDIT.md`](../INTEGRATION_AUDIT.md) holds only where the
`clamp(..., min=1e-6)` on the target activation is inactive and only for rows not subject to a
conflicting duplicate assignment. This was **measured and disclosed, not fixed**; do not
assume the update is exact.

## 4. Reproduction notes

`run_arm.py` and the vendored release resolve paths relative to the release root (the harness
`chdir`s there), `ledger.py` writes absolute paths under `/data`, and `env/runenv.sh` pins
host-specific cache directories. None of these are portable as-is.

Running anything in this directory requires GPUs and the pinned environments in
[`../env/`](../env/); **no rerun is authorised by this snapshot.**
