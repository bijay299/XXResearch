# Closure register — 2026-10-28

Issued when MANDATE v3 replaced the audit direction with a constructive-method direction. This
records what is closed, why, and what remains active, so nothing is left half-open.

**Nothing here is deleted.** Every artifact, number and evidence file stays on disk and in the
repository. What changes is status.

---

## 1. CLOSED — the evaluation/audit paper as a deliverable

**Reason:** PI direction, 2026-10-28. An empirical paper showing existing methods break under attack
was rejected; the deliverable must be a constructive method.

**Affected, now superseded in framing only:**

| artifact | status |
|---|---|
| `docs/PI_REPORT_2026-10-28.md` + `.pdf` | **numbers stand, contribution claim retired** |
| `docs/PI_CONTEXT_PACK_2026-10-28.md` + `.pdf` | same |
| `docs/PI_PRESENTATION_SCRIPT_2026-10-28.md` + `.pdf` | same |
| the Claude doc of the same report | same |

**Do not cite these for a proposed contribution.** Their measurements remain valid and are reused as
motivation and baselines under v3.

## 2. CLOSED — incomplete, with reason

**U47 (recovery-direction cosine).** Intended to measure `cos(delta_attack, tau)` to decide whether
erasing orthogonally to tau was well motivated.

- The attack ran; the measurement did **not**. `attack_relearn_v3.py` does not save an attacked
  checkpoint, and the runner looked for a file that never existed.
- The attack cell that did run was **gate-invalid**: `T 0.917 -> 0.542`, a *negative* rise. The
  task-vector arm at target 0.8667 is too weakly erased to attack meaningfully — the same vacuity
  FINDING58 excluded for `gift_church_b0.1`.
- **Not reopened**, because the chosen method (TVI, Option B) uses tau as a cheap attacker estimate
  rather than a direction to be orthogonal to, and therefore does not depend on this result.

## 3. CLOSED — offers not taken up

- Rewriting the deliverables under the audit framing — superseded by §1.
- Pulling the Bau Lab `WhenAreConceptsErased` repo for their released erased weights.
- The representation probe as a *contribution*. Lu et al. (NeurIPS 2025) already publish a
  four-probe suite; it is now **prior work we cite**, not a direction we own.
- Extending D4 build replication beyond the load-bearing arm.
- Attacking `ESD_vangogh_official` — rejected as vacuous at target 0.7333.

## 4. STANDING — closed earlier, unchanged

The trap/meta direction (S1/S1b/S1c), damage-then-distil (U23/U24/U32), selective Fisher damage,
UTRAP, WSR-anchor. All closed under pre-registered rules; conclusions preserved; not to be
retroactively altered.

MDLM remains permanently closed and is not referenced.

---

## 5. ACTIVE — what the work is now

| item | status |
|---|---|
| **MANDATE v3** | governing |
| **TVI** (`PREREG_U48_TVI.md`) | **running** — `lam=0` ablation built, `lam=1` building |
| `scripts/task_vector.py`, `apply_task_vector.py`, `tvi_erase.py` | current method code |
| U46 gate result | tau localises (23/686 tensors hold 50%); direction carries information (0.8667 targeted vs 1.0000 norm-matched random) |

## 6. RETAINED as assets under v3

- **The attack infrastructure**, which exceeds the published standard: LoRA r4/16/32 **and** full
  UNet, 1–1000 steps, lr 1e-5 to 5e-3, n=3, plus a benign downstream rung and image-level audits.
- **ResAlign's verified checkpoints** as the robustness bar: un-attacked 0.0000 reproduced exactly
  against a base control of 0.1817; worst observed recovery at 25 steps 60% of base where every
  other method reaches ~100%.
- **The 164-arm / 1,215-cell measurement base** as motivation and baselines.
- **D1, D2, D4 closed; D3 open** and affecting wording.
