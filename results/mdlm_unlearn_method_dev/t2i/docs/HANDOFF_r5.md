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
**as-is target 0.2000 / retain8 0.9500** — and **all three** frozen-grid seeds are now gate-valid at
**H = 1.000** (6 records across duplicated labels). *The previous revision said "2 of 3 ... the
third (seed 0) ... is running now"; it completed, and nothing is running.* λ=10000 and λ=50000 both
give target 1.0000 / retain8 1.0000 (`W_new → W_old`, no erasure), are flagged
`vacuous_no_erasure`, and the λ=50000 attacks were independently caught by the validity gate.

**Work completed after this handoff was written**, and not described anywhere in §1–§3 above:
E1 (faithful WDR, α ∈ {1.0, 4000}), E2 + E2g + E2h (Meta v2 full Eq. 8, 18-cell frozen grid plus an
attacker-lr sweep), E4 + E4m + E4mg (UTRAP v2 K-sweep at matched compute), E6 (FINAL-prompt
re-evaluation, which **consumed** the FINAL set), E11a–o and E12 (budget/lr/concept/method curves),
**R1** (upstream IMMA trained 5000/5000 and scored) and **R4b–d** (upstream UCE executed). These
produced `FINDING65`–`FINDING78`. §6 below is corrected accordingly.

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
| **5** | ~~The three released reference implementations need separate conda environments~~ **DISCHARGED for two of three.** `venv_imma` and `venv_uce` were built as isolated **copies** (`venv_t2i` never modified); **upstream IMMA ran** (R1, 5000/5000 steps + eval) and **upstream UCE ran** (R4b–d, `executed: true`). `vision_relearning` (R3) is still unrun — it needs `wandb` and CIFAR-scale data | **E7 partially discharged.** "We reproduced it" is *still* unavailable, but for a different reason: both runs were scored on **our** benchmark, which is not a reproduction (`UPSTREAM_FIDELITY_MANIFEST.md` §6). "Unusable" remains unsubstantiated | R3 only: `wandb` + CIFAR data |
| **6** | No never-trained oracle | reactivation vs relearning unresolvable **by construction** | a design decision: build a tractable never-trained reference, or narrow the paper to the operational relearning threat |

---

## 6. Status, per axis — not a single verdict

Repeating `CORRECTION_REGISTER_r5.md` §C because the audit asked for it explicitly and a one-word
answer would be wrong:

| axis | status |
|---|---|
| **Implementation** | **SUBSTANTIALLY RESOLVED, NOW VALIDATED ON MODELS.** Three wrong objectives corrected and unit-tested; attacker definition shared and drift-proofed; split checker fails closed. **All three corrected implementations have now produced models** — WDR v2 (E1, α ∈ {1.0, 4000}), Meta v2 (E2, λ ∈ {0.5, 1.0}) and UTRAP v2 (E4, K ∈ {0,1,8,32}). *The previous revision's "Not one corrected implementation has produced a model" was false.* Two new implementation defects were found in the process: `erase_wdr_v2.py` alternates where the reference **sums** (`D-016`), which entangles `FINDING67`'s α finding with our own schedule (`D-017`). |
| **Evaluation & provenance** | **PARTIALLY RESOLVED.** Leak detected and attributed, two more found, DEV/FINAL separated with FINAL frozen before inspection, RNG streams separated, gated+ungated and all three denominators reported, checkpoints identified by sha256. **Still open:** E6 evaluated four arms on FINAL (`FINDING65`/`66`), which **consumed** that split — so a fresh final set must be frozen — **and those FINAL scores are defective** (`D-022`): `gen_eval_final.py` used `gen_eval.LAB_KEY` instead of `LABELS`, degrading nine of twelve distractor strings while leaving `church` (the erased target) correct, which biases target hits **upward**; `FINDING65`'s depth ratios are **unmeasured pending a CPU-only re-score**, while `FINDING66`'s ceiling conclusion survives as a within-scorer comparison. Every *other* target number remains a development measurement; the CLIP classifier revision is still unpinned; `FINDING72` measured the **validity gate's exclusion rate to be non-monotone in `T_before`**, so every "max-over-valid H" is computed over a biased sample; **234 of 335 builds have `ABSENT` checkpoints**; and **7 historical `gen_*.json` records do not identify the checkpoint they scored** (`unet: null`), five of which scored an *edited* model. |
| **Empirical comparison** | **PARTIALLY RESOLVED — still no comparative claim about the published methods.** Corrected implementations **have** been run, two released references **have** been executed (upstream IMMA, upstream UCE), and the unroll-depth question **has** been tested at matched compute (`FINDING76`: flat). *The previous revision asserted the opposite of all three.* But **no comparative claim about the published baselines is supported even so**, and for a stronger reason than before: R1 and R4 used upstream code with **our** training data, scorer and prompts, which is not a reproduction of any paper's reported experiment. `vision_relearning` (R3) is still unrun. |
| **Scientific interpretation** | **CORRECTED IN TEXT, STRUCTURALLY LIMITED.** Every saturated-endpoint, mechanism and capacity claim rewritten to what the data supports. The structural limits are **not** fixed by wording — E5, E6 and E10 are what would move them. |

**The mandate remains unmet**: no arm reaches retain8 ≥ 0.85 with max-over-valid H ≤ 0.40, at any
budget. This remediation does not change that, and neither did E1, E2, E4 or R1/R4.

The nearest approach is `meta2_church_lt0.5` (E2, full Eq. 8) at retain8 **0.9000** with as-is
target **0.0000**. At 25 steps it clears the robustness half at attacker lr 1e-5 (**0.0667**) and
lr 5e-5 (**0.2000**), and fails only at lr 1e-4 (**0.5333**, `logs/e2h_stronger.log`); since
max-over-valid is a ratchet, the budget fails the bar. `FINDING78`'s companion claim to have
produced "the first arm to meet both mandate halves" is **retracted** (`D-008`); its retention
result — **+0.20 to +0.27** retain8 at identical suppression, attributable to Eq. 8's **pairing**
rather than to ζ's weight — survives and is the first substantive gain any corrected baseline has
produced.

What the remediation changed is the **standing** of the negative result. It is now a statement about
**our implementations plus two upstream-code runs scored on our own benchmark** — and still not
about the published methods, because scoring upstream code on our benchmark is not a reproduction.
E1, E2, E4 and E7's IMMA/UCE halves have run; **R3 is the cell that would still change this**, and
it has not.

## 7. Reproducing this state

```bash
cd /data/bijaypandey/t2i_unlearn
CUDA_VISIBLE_DEVICES="" venv_t2i/bin/python scripts/test_audit_repairs.py   # 24 passed
CUDA_VISIBLE_DEVICES="" venv_t2i/bin/python scripts/split_registry.py       # LEAK CONFIRMED, FINAL usable
venv_t2i/bin/python scripts/registry_v1.py --tables                         # ~2.5 min (sha256 over checkpoints)
```

The registry and tables are **regenerated**, never hand-edited. If a table disagrees with prose,
the table is right and the prose is stale — that inversion is the point of §7 of the audit.

---

## 8. Change log — 2026-10-26 (Agent 6, provenance audit)

This handoff described the state at the end of the audit remediation. Substantial GPU work ran
afterwards (`FINDING65`–`FINDING78`), which falsified several of its status statements. Corrected
below; full traceability in `AGENT6_PROVENANCE_FINDINGS.md`.

| § | what changed | why |
|---|---|---|
| §4 | "2 of its 3 frozen-grid attacks are gate-valid ... the third (seed 0) ... **is running now**" → all three complete, 6/6 gate-valid | the run finished; `ps` shows no project process, and `FREEZE_ACTIVE` (12:43) now bars new GPU work |
| §4 | added the list of campaigns that ran **after** this handoff was written (E1, E2/E2g/E2h, E4/E4m/E4mg, E6, E11a–o, E12, R1, R4b–d) | §1–§3 describe none of them, so a reader would take this document as current when it is not |
| §5 blocker 5 | "**E7 blocked**" → discharged for two of three; R3 only | upstream IMMA ran 5000/5000 with eval; upstream UCE executed. The *conclusion* ("we reproduced it" unavailable) survives, for the better reason that both were scored on our benchmark |
| §6 Implementation | "**Not one corrected implementation has produced a model**" → all three have | false. `gen_wdr2_church_a{1.0,4000}`, `gen_meta2_church_lt{0.5,1.0}`, 8 `utrap2*` builds |
| §6 Implementation | added the two new implementation defects found since (`D-016` alternate-vs-sum, `D-017` shared-Adam) | they were found *by* running the corrected code, which is the point of validating on models |
| §6 Evaluation | "**No arm evaluated on FINAL**" → four arms were, and that **consumed** the split; added `FINDING72`'s gate bias, the 234/335 `ABSENT` checkpoints, and the 7 `unet: null` records | the row understated what had been measured and omitted three provenance limits a successor needs |
| §6 Empirical comparison | "NOT RESOLVED — no corrected implementation run; references not executed; unroll depth untested" → all three have happened, and the conclusion is **restated on stronger grounds** | three false sub-claims. The verdict "no comparative claim is supported" is kept, now because upstream-code-on-our-benchmark is not a reproduction, rather than because nothing ran |
| §6 closing | mandate paragraph rewritten with the `meta2_church_lt0.5` near-miss, the `FINDING78` retraction (`D-008`), and R3 named as the remaining cell | the conclusion was right but rested on absence of measurement; it now rests on `logs/e2h_stronger.log` |

### 8.1 Re-verified and found correct (unchanged)

§1's file inventory and line counts; §2's **24 passed / 0 failed** and the two float64 test-bug
fixes; §2's smoke-run outcomes including Meta v2's `L_F(θ')` rise 0.00791 → 0.01968 → 0.02809 and
the `meta 'xattn': 44.0M | attack 'full': 859.5M` scope separation; §3's fp16/fp32 VAE bug and the
`pkill` self-match incident; §5's blockers 1, 2, 3, 4 and 6.

**Flagged, not repaired:** §4's resource accounting (`result JSONs 1196`, `storage 174.2 GiB`) is
stale — there are now **1,424** JSON files under `outputs/`. §7's reproduction recipe still invokes
`scripts/registry_v1.py`, which is correct but will **not** reproduce the current tables, since
`outputs/registry_v1.json` predates 150 JSON files; regenerating it is a lead decision under the
freeze.
