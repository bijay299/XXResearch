# PREREG U32 — genuine-eraser teacher at the utility-preserving damage level

**Declared 2026-10-27, BEFORE execution.** Development data; no confirmatory claim. Frozen on first
result; corrections in a dated amendment. Authorised by the project owner.

## 1. The objective, in the owner's terms

Three bars, in order. Image quality on the target prompt is **not** one of them and is not scored.

1. **Ask for a church, do not get a church.**
2. **Every other concept still works.**
3. **It stays that way after someone fine-tunes the model.**

Bar 3 is the open problem. All 86 concept-matched arms in this project — every published method
included — fully reverse to `H = 1.0000` under 25 steps of fine-tuning. No new scoring axis is
introduced by this study.

## 2. Why this specific cell, and why it is not a re-run

Damage-then-distil (UNDO, arXiv:2506.06278, NeurIPS 2025 — **the mechanism and the name are the
authors'**) has been attempted twice here. Both failed, for **different and now-diagnosed** reasons:

| study | teacher | teacher's real erasure (4 eval seeds, U29-U31 audit) | alpha | outcome |
|---|---|---|---:|---|
| U23 | AdvUnlearn church TE | **leaks** — `h48_adv_a0.3` reads 0.3333 x4 and the student's images show **visible churches** | 0.3 | student inherited the leak; fails bar 1 |
| U24 | UCE | **genuine** — `u24_uce_s6000` reads 0.0000 x4 | 0.5 | erasure transferred, **utility destroyed** (retain8 0.2917); fails bar 2 |
| **U32** | **UCE** | **genuine** — `uce_church` reads 0.0000, 0.0000, 0.1111, 0.0000 | **0.3** | **never built** |

U23 failed on the teacher axis. U24 failed on the damage axis. **The cell where both are set
correctly has never been built** — it was queued as `PREREG_U25_STRONG_TEACHER.md` and closed
unrun. alpha = 0.3 is the level U23 measured as preserving utility (retain8 0.8833 / 0.9250) where
0.5 destroys it (0.4583 / 0.5667).

**First-principles reason it might clear bar 3 where editing methods do not.** Existing methods
re-point the concept rather than remove it — this project's own signature is recovery rising to
1.0000 *while retained capability rises*, which is only possible if nothing was removed. A student
rebuilt from a teacher's outputs can acquire only what the teacher demonstrates; the teacher
demonstrates no churches, so that machinery is not rebuilt. The attacker's problem changes from
*re-point an existing capability* to *learn one*.

**Stated against ourselves:** this mechanism is not ours, our one original addition (Fisher-weighted
selective damage) lost to a random control in U23 and is withdrawn, and `--protect none` is used
here precisely because it is the published baseline mechanism.

## 3. Design

Teacher `adapters/uce_church/unet.pt`. Single arm `u32_uceT_a0.3`.

| held fixed | value |
|---|---|
| `--alpha` | **0.3** |
| `--beta` | 0.1 |
| `--protect` | **none** (plain UNDO) |
| `--steps` | 6000 |
| `--lr` | 1e-5 |
| `--n-images` | 300 |
| `--forget-frac` | 0.2 |
| `--seed` | 0 |
| concept | church |

**Attacker: UNRESTRICTED**, matched to how all baselines in this project were measured —
`attack_relearn_v2.py`, 3 learning rates {1e-5, 5e-5, 1e-4} x 3 seeds {0, 1, 6} x 25 steps,
evidence schema, `--validate-reload`. No restriction, no LoRA.

**Image audit (`audit_teacher.py`, 4 eval seeds) is run on the student before any robustness claim.**
It is a **validity check, not a score**: U30 established that `target_suppression_acc` can be
satisfied by a speckle overlay on a visible church, so bar 1 must be confirmed against the image.
It contributes no number to the success criterion.

## 4. Decision rule, declared before any number exists

Primary quantities: un-attacked `target_suppression_acc`, un-attacked `legacy_retain8`, and
**max-over-admissible** `H_after`, where admissible = gate-valid (`T_rise >= 0.20`) **and** not
destroyed (`retain_after < 0.55` and `retain_before - retain_after >= 0.25`).

**The admissible-cell count is reported beside the baselines' counts.** U30 found our arms admitted
3 of 9 cells against the baselines' 22/24, 16/18 and 21/21, which made an earlier comparison
unmatched. A max taken over few cells is not comparable to one taken over many and will not be
presented as such.

| outcome | reading |
|---|---|
| target <= 0.15 (**image-confirmed**) AND retain8 >= 0.85 AND max-over-admissible H <= 0.70 with an admissible-cell count comparable to the baselines' | **clears all three bars.** Beats UCE on the axis every method fails. Proceed to n=3 builds, then a second concept |
| first two bars clear, H > 0.70 | erasure and utility transfer; **robustness does not.** Report the operating point; the damage-then-distil family does not solve bar 3 at this configuration |
| retain8 < 0.85 | **utility fails again at the lower damage level.** With U24's alpha=0.5 failure this makes the family finished on the utility axis; report and stop |
| target > 0.15, or audit shows visible churches | the teacher's erasure did not transfer. Report; the mechanism does not preserve erasure through distillation |
| admissible cells <= 3 | **no robustness claim is made regardless of H**, because the comparison would not be matched |

**Noise floors:** `H_after` reproduces to within one quantum (1/15 = 0.067); retained capability to
~0.042. Differences smaller than those are not differences.

## 5. What U32 cannot establish

- **n = 1 build.** Five claims in this project have died at n = 3.
- One concept (church), one alpha, one teacher, one attack family, 25-step budget, SD-1.4.
- **Not a novelty claim.** Novelty verification is parked at the owner's direction.
- The UCE teacher's own target images are degenerate texture (U30). Under the owner's stated
  criterion that is acceptable — the church does not appear — but it means the student is distilled
  from degenerate demonstrations on the forget set, and if bar 2 fails that is a candidate cause.
- A single un-audited confound remains: whether our UCE port's edit strength matches the published
  configuration. `uce_church_tuned` exists for comparison and has not been checked.
