# Handoff — audit remediation r5

**Audited snapshot:** `0b435427557e4fb984cc9acff26e6dacc0613cd2`
**Parent commit at start:** `66ff9d3afec54f446d0750b53f1576a1f10e7ab7` (working tree clean)
**Branch:** `canonwork` → `research/mdlm-unlearn-method-dev`
**Work since the audit, preserved:** 49 files / +4,643 lines — the cross-model UTRAP, Meta and UCE
runs and the UCE λ-scale erratum. **Nothing was reverted.** Every register row states whether it
affects those newer results; three do (`FINDING64`'s UTRAP and Meta arms carry the same
implementation defects as their SD-1.4 counterparts).

---

## 1. Files changed

### New code (8 files, 2,143 lines)

| file | lines | what |
|---|---:|---|
| `scripts/attack_core.py` | 190 | **one** relearning-attack update, shared by evaluator and defences. `AttackSpec` auto-computes `divergence_from_evaluation_attacker`; `horizon_role` has no default and `EVALUATION_REFERENCE` is refused for any horizon ≠ 300; `AttackData` makes `provenance` and `split_role` mandatory |
| `scripts/objectives.py` | 111 | both weight-space terms as separate named functions (`anchor_penalty` = ours, `wdr_forget_loss` = faithful) + `meta_terms` (full Eq. 8) and `meta_terms_as_implemented_variantA` |
| `scripts/split_registry.py` | 237 | split registry **v2**: reads prompt templates from script **source text**, fails closed on unregistered defences, 7 overlap tables, punctuation-aware normalisation, FINAL freeze, RNG stream separation |
| `scripts/registry_v1.py` | 346 | result registry keyed by **sha256 checkpoint identity** + target + family + budget + seeds, with admissibility flags; generates all current tables |
| `scripts/erase_wdr_v2.py` | 221 | faithful 'Weight Dist Reg' adaptation (8 axes reconciled) |
| `scripts/meta_unlearn_v2.py` | 304 | full Eq. 8 + scope separation + target-content check + `--order` guard |
| `scripts/erase_utrap_v2.py` | 313 | UTRAP through the shared attack core, evaluator-matching defaults |
| `scripts/test_audit_repairs.py` | 421 | 24 deterministic CPU tests |

### Unchanged on purpose
`scripts/erase_wsr.py`, `scripts/meta_unlearn.py`, `scripts/erase_utrap.py`,
`scripts/attack_relearn.py`, `scripts/gen_eval.py`, `scripts/verify_splits.py` — **not touched**, so
every historical result reproduces bit-for-bit. The corrections live in `*_v2` files with distinct
identities. `attack_relearn.py` in particular was left alone deliberately: refactoring the
evaluation attacker through `attack_core.py` would risk changing the measured adversary and
invalidating comparison with 642 historical records. `attack_core.EVAL_FULL_UNET` instead encodes
its configuration as a frozen constant that v2 defences are checked against.

### New documents (11 files, 2,069 lines)
`CORRECTION_REGISTER_r5.md` (+ `outputs/correction_register_r5.json`),
`ERRATUM_wsr_direction.md`, `ERRATUM_meta_objective.md`, `ERRATUM_utrap_inner_attacker.md`,
`ERRATUM_split_leak.md`, `ERRATUM_interpretations_r5.md`, `T2I_BASELINES_v2.md`,
`PREREG_PRECEDENCE.md`, `PREREG_H90_utrap_v2.md`, `T2I_REPORT_r5.md`,
`TABLES_v1_generated.md`.

### Correction banners added to 17 existing documents
`FINDING43`, `FINDING56`–`FINDING64`, `ERRATUM_r2_claim_corrections`, `AUDIT_CLOSURE_r2`,
`PROTOCOL_FROZEN_v1`, `PREREG_H66`, `PREREG_H80`, `PREREG_H84`, `PREREG_H85`, and a SUPERSEDED
banner on `T2I_REPORT_r4`. **Every original body is unaltered**; the banner states what is withdrawn
and points to the erratum. **No preregistration was rewritten.**

### New machine-readable artifacts
`outputs/registry_v1.json`, `outputs/split_registry_v2.json`,
`outputs/correction_register_r5.json`, `outputs/test_audit_repairs.json`.

---

## 2. Tests actually run, with outcomes

`CUDA_VISIBLE_DEVICES="" venv_t2i/bin/python scripts/test_audit_repairs.py` → **24 passed, 0
failed**. Breakdown: 6 weight-space direction, 4 Meta objective terms, 10 attacker correspondence,
4 split registry.

Two tests failed on the first run and were **test** bugs, not objective bugs:
`t_meta_value_matches_equation` and `t_meta_reduces_to_variantA_only_if_terms_dropped` asserted
float64 tolerances against float32 arithmetic (`0.3 − 0.2 = 0.100000024`). Fixed by computing in
float64. The gradient-sign tests, which are the substantive checks, passed throughout.

### Bounded integration validation (`logs/smoke_v2.log`)

Real pinned SD-1.4, few steps, GPU 0, serial. **No scientific result is claimed from these.**
Every `SMOKE_*` checkpoint was deleted immediately after its artifact was verified.

| script | outcome | what it proved |
|---|---|---|
| `erase_wdr_v2.py` | **exit 0**, artifact written | the faithful objective runs; `forget=-0.009023`, `RMS=0.009024` — sign and value agree exactly; **trust-region stop fired at step 0 and was recorded** in `trust_region_stop` |
| `meta_unlearn_v2.py` | **exit 0**, artifact written | **target content check: 8/8 = 1.0000** classified as church from `--target-source base`; **scope separation works** (`meta 'xattn': 44.0M \| attack 'full': 859.5M`); and the restored `−L_F(θ′)` term is visibly active — `L_F(θ')` **rose 0.00791 → 0.01968 → 0.02809** over 12 steps, which is the behaviour variant A cannot produce |
| `erase_utrap_v2.py` | **exit 1, then exit 0** after a fix | see §3 |

### A real measurement that came out of the smoke run

UTRAP v2's new fixed-noise probe, over K=8 inner steps: **0.14043 → 0.13628, delta −0.0042** —
about **3%** of the probe loss, consistently across outer steps. The 8-step surrogate adapts, but
only marginally, against an evaluation attacker that drives the same loss to near zero over 300
steps. **This is an integration-test observation on 6 outer steps, not a result.** It is also the
first quantitative evidence on whether the surrogate approximates the real attacker at all, and it
is why `PREREG_H90` checks the probe-vs-K relationship **before** interpreting any H outcome.

---

## 3. One bug I introduced and fixed

`erase_utrap_v2.py` crashed with `RuntimeError: Input type (c10::Half) and bias type (float) should
be the same`. Cause: my own provenance repair. Switching the target generator to a **separate fp16
base pipeline** (so the attacker's data provenance matches `attack_relearn.py`) fed fp16 images into
the fp32 VAE. The snapshot's `erase_utrap.py` generated and encoded through one fp32 pipe, so the
mismatch could not arise there. Fixed with `.float()` at `erase_utrap_v2.py:178` and a comment
explaining why the cast is load-bearing. Re-run: exit 0.

Also, earlier in the session, a `pkill -f "H89L_uce_sd15_church_lam10000"` pattern matched **this
shell's own command line** and killed my command mid-execution. Redone by PID after verifying
`/proc/<pid>/cmdline`. No experiment was harmed; one label marker was left stale and is accounted
for in §4.

---

## 4. GPU work completed during the remediation

Not part of the audit, but it finished the task in progress and is accounted for.

**The properly scaled UCE λ sweep on SD-1.5 completed**, bracketing the window from both sides by
measurement (`T2I_REPORT_r5.md` §3.3). λ=1000 gives the first **usable** SD-1.5 UCE arm —
**as-is target 0.2000 / retain8 0.9500** — and 2 of its 3 frozen-grid attacks are gate-valid at
**H = 1.000**; the third (seed 0) had claimed a label without producing output and **is running
now**. λ=10000 and λ=50000 both give target 1.0000 / retain8 1.0000 (`W_new → W_old`, no erasure),
are flagged `vacuous_no_erasure`, and the λ=50000 attacks were independently caught by the validity
gate.

**Resource accounting, continued not reset:** `elapsed 77.8 h | gpu ≥34.71 device-h (FLOOR) |
result JSONs 1196 | storage 174.2 GiB / 300 GiB authorised`. The GPU figure remains a reconstructed
**floor** — only 446 of 1196 records carry a runtime field — and is not presented as a measured
total. Two GPUs at a time throughout; GPUs 2 and 3 were left to other users.

---

## 5. Remaining blockers

| # | blocker | impact | what unblocks it |
|---|---|---|---|
| **1** | **`/data` is at 100% capacity, ~14 GB free.** Shared with another project (`cuig_pilot`, ~386 GB); not a function of our 300 GB authorisation | **No multi-checkpoint campaign can start safely.** E1/E2/E4/E5 need 7–14 GB of checkpoints each | a decision about the other project's footprint, or more disk. **I have deleted nothing**, per instruction |
| **2** | `stabilityai/*` returns **HTTP 401** | **E9** (SD-2.x / SDXL) impossible; the cross-model claim is confined to SD-1.x + one pruned derivative | a Hugging Face token |
| **3** | FID dependencies absent and COCO not on disk | **E8** impossible; no claim about general image quality or global repair | authorisation for `pip install torchvision scipy` and a ~6 GB COCO val2014 download |
| **4** | `gen_eval.py` cannot yet take the FINAL prompt pool | **E6** blocked; every target number stays a **development** measurement | a small flag in `gen_eval.py`; ~20 lines, not yet written because it should land with E6 rather than ahead of it |
| **5** | The three released reference implementations need separate conda environments | **E7** blocked; "we reproduced it" remains unavailable, and "unusable" remains unsubstantiated | confirmation that building separate envs is acceptable and will not disturb `venv_t2i` |
| **6** | No never-trained oracle | reactivation vs relearning unresolvable **by construction** | a design decision: build a tractable never-trained reference, or narrow the paper to the operational relearning threat |

---

## 6. Status, per axis — not a single verdict

Repeating `CORRECTION_REGISTER_r5.md` §C because the audit asked for it explicitly and a one-word
answer would be wrong:

| axis | status |
|---|---|
| **Implementation** | **SUBSTANTIALLY RESOLVED, UNVALIDATED ON MODELS.** Three wrong objectives corrected and unit-tested; attacker definition shared and drift-proofed; split checker fails closed. **Not one corrected implementation has produced a model.** |
| **Evaluation & provenance** | **PARTIALLY RESOLVED.** Leak detected and attributed, two more found, DEV/FINAL separated with FINAL frozen before inspection, RNG streams separated, gated+ungated and all three denominators reported, checkpoints identified by sha256. **No arm evaluated on FINAL**, so every target number remains a development measurement; CLIP classifier revision still unpinned. |
| **Empirical comparison** | **NOT RESOLVED.** No corrected implementation run; the three released references not executed; the unroll-depth question untested with a faithful attacker. **No comparative claim about the published baselines is currently supported.** |
| **Scientific interpretation** | **CORRECTED IN TEXT, STRUCTURALLY LIMITED.** Every saturated-endpoint, mechanism and capacity claim rewritten to what the data supports. The structural limits are **not** fixed by wording — E5, E6 and E10 are what would move them. |

**The mandate remains unmet**: no arm reaches retain8 ≥ 0.85 with max-over-valid H ≤ 0.40. This
remediation does not change that. It changes the **standing** of the negative result — it is now a
statement about **our implementations** of three of the five baselines, not about the published
methods, until E1, E2 and E7 run.

## 7. Reproducing this state

```bash
cd /data/bijaypandey/t2i_unlearn
CUDA_VISIBLE_DEVICES="" venv_t2i/bin/python scripts/test_audit_repairs.py   # 24 passed
CUDA_VISIBLE_DEVICES="" venv_t2i/bin/python scripts/split_registry.py       # LEAK CONFIRMED, FINAL usable
venv_t2i/bin/python scripts/registry_v1.py --tables                         # ~2.5 min (sha256 over checkpoints)
```

The registry and tables are **regenerated**, never hand-edited. If a table disagrees with prose,
the table is right and the prose is stale — that inversion is the point of §7 of the audit.
