# Corrections — executed chain saw -> parachute diagnostic (reporting closeout)

**Original execution commit corrected: `ffc133f1e94cd04df5a088c720a3370d49037328`**
(protocol frozen at revision 02, `c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80`).

This is a **reporting closeout**, not a new experiment and not authorization for one.
**No model, encoder, detector, checkpoint-tensor computation, edit/replay, generation,
classification, image viewing or reannotation, or GPU allocation/polling was run to
produce it.** Work was limited to reading existing records and executed code, simple CPU
bookkeeping on already-archived numbers, file-level hashing, and documentation updates.

## Unchanged by this pass

**The underlying numerical result does not change anywhere below.** Specifically unchanged:
all per-image classifier predictions and counts; chain saw 100 -> 0 -> 0 with zero raw
relapse candidates and `C=0, U=0` for both methods; parachute 0/100 directly and
sequentially for both methods; every frozen gate value and threshold; the empty candidate
sets; the decision that **neither batch comparator was triggered**; the 4,000-image /
4-of-6-endpoint accounting and the unused 2,000-image allowance; the archived residual
JSONs, logger source, verification logs, analysis tables, panel images and prior commits.

## Issue-to-fix table

| # | Incorrect claim at `ffc133f` | Corrected interpretation | Numerical result changed? |
|---|---|---|---|
| **C1** | UCE's cumulative retention failure described as driven by cassette player, "worst class −24", implying a single-class issue | UCE violates the 15-case per-class cumulative bound on **two classes independently**: cassette player (28 -> 15 -> 4, cumulative **−24**, incremental −11) **and english springer** (89 -> 80 -> 72, cumulative **−17**, incremental −8). Cumulative aggregate is **−6.625** pts (inside the 8-point bound); incremental aggregate −3.25 and worst class −11 pass. Cassette's weak M0 baseline (28/100) is context, **not** grounds to waive the gate, and does not apply to english springer (M0 89/100), which fails independently. UCE must **not** be described as satisfying the full gate conjunction; DP does | No — counts and gate outcomes unchanged (UCE cumulative retention still FAIL) |
| **C2** | Residual values reported as percentages of the parent's response / "weight-level perturbation" — "38.7%", "0.42%", "0.13%", "3.2%"; and "UCE's soft preservation is **~90x weaker** than DP's" | The archived logger computes `q(c) = ‖(W_child − W_parent)c‖₂ / ‖c‖₂` — an **input-normalized isolated-vector response residual** whose denominator is the **input** norm, not `‖W_parent c‖`. So `q=0.3869613` does **not** establish a 38.7% parent-response change or weight perturbation, and `q=0.0042419` does **not** establish 0.42%. Percentage labels are replaced by explicitly defined decimal `q` values plus the already-recorded absolute residuals. The "~90x weaker preservation" ranking is **withdrawn**: a ratio of selected `q` values establishes no general or causal preservation-strength ranking, since token selection, precision pathway and probe inputs differ between methods. A parent-response fraction would need a different denominator and was **not** computed | No — the logged `q` and absolute values are unchanged; only their labels and interpretation |
| **C3** | UCE residuals presented as measurements of its native preservation vectors | The executed logger (`evidence/code/residuals.py`, `encode_concepts`) **hardcodes bf16 text encoders for both methods**, whereas native UCE encodes in **fp32**. Promoting bf16-produced vectors to fp32 does not reconstruct fp32-encoder outputs. UCE's published residuals are therefore **probes on bf16 encodings**, not verified measurements of its native fp32 preservation vectors; the native-input reading is withdrawn. Encoders were **not** rerun and the logged measurements are **not** replaced | No — measurements preserved as archived |
| **C4** | "For UCE, pre- and post-cast are indistinguishable", used to argue the casting asymmetry is DP-specific | The archived values **differ**: UCE T5 `q` 0.3865076 -> 0.3869613; UCE CLIP `q` 0.0316740 -> 0.0319098. Close, but not identical — "indistinguishable" is withdrawn. **Both** methods are evaluated in bf16; DP's asymmetry concerns its **native edit-time/intermediate** casting, not an absence of evaluation rounding for UCE. DP's small captured pre-cast and larger post-cast residuals support a **narrow numerical observation on those probe vectors only** — not exact preservation for contextual prompts or the generator as a whole | No |
| **C5** | English springer decline explained as "a fine-grained label shift, not visible content loss or rendering damage" | **Withdrawn as extrapolation.** Retained: the local observation that the two displayed panel images depict coherent, well-formed dogs. Their recorded top-1 labels are M0 `English springer`/`English springer`; UCE A-parent `English setter`/`English springer`; UCE SEQ `English setter`/`English springer` — so the only disagreement in this pair **already exists at the A-parent**, before the B edit. Two cases explain neither the 8 further cases lost after B nor the 17 cumulative losses; the cause is **not established**. No additional images were inspected to replace the claim | No |
| **C6** | Manifest hardware row read as exclusive use of one physical GPU throughout | The row is the **initial allocation**. GPU 2 ran every edit and all four generations; **GPU 3 was additionally used, concurrently, for residual logging and the `DP_SEQ` classification pass** (verified idle before first use). GPUs 0/1 never used. Annotated as a **post-execution correction** in the manifest rather than rewritten in place | No |
| **C7** | "Host-only … identified by path/bytes/sha256" applied to **all** excluded artifacts, including the 4,000 raw images | Two identification levels now distinguished in `evidence/HOST_ONLY_ARTIFACTS.json`: the four edited checkpoints (and the DP pre-cast capture, C8) are **hash-identified**; the raw generation output is inventoried **only by directory path, image count and total bytes** and is **not hash-indexed per image**. The 140 published panel images are individually hashed in `evidence/panel_hashes.csv`. **No new hashing campaign was run** | No |
| **C8** | DP pre-cast fp32 capture absent from the host-only inventory | Present on host and now inventoried with path, byte count and sha256 (`logs/DP_SEQ.precast_fp32.pt`, 59,770,400 bytes, `5a3995fbd0d1fed27c69e1d7a0b26d91a5159f16f8441014bcd49e40c06da9a8`). Hashed at **file level only** — no tensors were loaded, and the capture is **not** committed | No |
| **C9** | `save_reload_bitwise_identical` described as "save/reload integrity" | Described accurately: the executed code deserializes the **saved file a second time** and compares it against the first deserialization of that same file — a **repeated-deserialization determinism check**, not a documented comparison against the in-memory pre-save tensor. It does not on its own exclude a serialization-time discrepancy. **No further integrity experiment was run**; the field and its value are unchanged | No |
| **C10** | "One recommended next scientific step": re-score existing SEQ images with a second detector | **Replaced by the closeout decision.** No second detector, new pair, reverse order, sweep, ablation or new method is authorized. Detector coverage remains a **stated scope boundary of a closed result**, not a queued follow-up. Any future measurement requires a separately declared protocol and must not overwrite this frozen result | No |

## Preserved historical artifacts (not silently repaired)

The executed logger source (`evidence/code/residuals.py`, including its hardcoded bf16
encoders and its `relative_residual` field) and the original residual JSONs are preserved
**exactly as they ran**. Their field names mean:

- `residual_l2` — absolute `‖(W_child − W_parent)c‖₂` for one probe vector.
- `input_l2` — `‖c‖₂`, the probe vector's own norm.
- `relative_residual` — `q = residual_l2 / input_l2`; **input-normalized**, not a fraction
  of the parent's output (C2).
- `stage` — `pre_cast` / `post_cast`; for UCE, `pre_cast` is the saved fp32 child minus
  fp32 parent (UCE edits fp32 end-to-end), for DP it uses the adapter's fp32 capture.

Executed code was **not** edited to fix these limitations and then presented as the code
that produced the measurements; the limitations are documented here instead.

## Status

**Diagnostic CLOSED.** Scope as published and unchanged: no verified detector-discovered
relapse (`C=0, U=0`), **not** certified semantic absence; one pair, one order, one literal
prompt per target at 100 seeds. **This result does not demonstrate an unmet method gap.**
