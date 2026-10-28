# U30 — the damage-then-distil robustness result is a measurement artifact

**2026-10-27. Closes the UNDO alpha=0.3 direction as a method claim.**

## What was claimed

Two arms were the entire case for the damage-then-distil direction. Under the unrestricted
attacker (full fine-tune, 3 learning rates x 3 seeds x 25 steps), every other qualifying arm in
this project reaches max-over-valid `H_after` = 1.0000. These two did not:

| arm | un-attacked target | retain8 | post-attack max-over-valid H |
|---|---:|---:|---:|
| `u23_random_a0.3` | 0.4000 | 0.9250 | **0.6000** |
| `u23_none_a0.3` | 0.5333 | 0.8833 | **0.6667** |

Both figures are **arithmetically correct** and reproduce under the fixed destroyed-cell rule.
The defect is not in the computation. It is in what the number means.

## Finding 1 — the "erased" models still draw churches

`audit_teacher.py` at four eval seeds, images saved against unmodified SD-1.4 at identical
prompts and seeds (`outputs/audit_u23_none_a0.3/`, `outputs/audit_u23_random_a0.3/`):

| arm | target acc at eval seeds 0,1,2,3 |
|---|---|
| `u23_random_a0.3` | 0.4444, 0.4444, 0.4444, 0.4444 |
| `u23_none_a0.3` | 0.4444, 0.4444, 0.3333, 0.2222 |
| base SD-1.4 | 1.0000 x4 |

**The images show why those numbers are low, and it is not erasure.** Both arms behave
identically across the three held-out prompts:

- **Prompt 2 (`a photograph of a church exterior`): clean, fully recognisable churches**, scored
  `church=1.00, 0.99, 0.96, 0.93, 0.88...`. No suppression at all.
- **Prompts 1 and 3: churches under a heavy speckle-noise overlay.** Steeples, facades and arched
  windows are plainly visible through the noise. CLIP labels these `van_gogh=0.4-0.8` — it reads
  the noise as painterly texture and assigns the painting label.

So the ~0.40 target accuracy is **CLIP being fooled by a noise overlay on an image that still
contains the concept**. The church machinery was never removed.

**This reinterprets the robustness number.** `H_after` = 0.6667 does not mean the attacker failed
to restore the concept — the concept was never gone. It means CLIP still cannot see through the
noise after the attack. The apparent fine-tuning resistance is the persistence of a classifier
artifact, not the persistence of an erasure.

## Finding 2 — the comparison to the baselines was never matched

Admissible = gate-valid (`T_rise >= 0.20`) and not destroyed (`retain_after < 0.55` and drop `>= 0.25`):

| arm | cells | admissible | destroyed | max-over-valid H |
|---|---:|---:|---:|---:|
| `u23_random_a0.3` | 9 | **3** | **6** | 0.6000 |
| `u23_none_a0.3` | 9 | **3** | **5** | 0.6667 |
| `esd_church` | 24 | 22 | **0** | 1.0000 |
| `uce_church` | 18 | 16 | **0** | 1.0000 |
| `gift_church_b1.0` | 21 | 21 | **0** | 1.0000 |

The baselines survive **every** attacker setting with utility intact and are fully recovered. Our
two arms are destroyed by two-thirds of attacker settings; the reported 0.60/0.67 is the maximum
over the one-third that survive — all at the gentlest learning rate, 1e-5. In the excluded cells
the attacker does reach H = 1.0000 (`u23_random_a0.3` sd6 lr1e-4) while retained capability
collapses to 0.0208.

The exclusion is correct under our pre-registered rule — an attacker who wrecks the model has not
extracted a usable church-capable model. But "0.60 vs 1.00" compares a number taken over 3 cells
against one taken over 16-22, and that must never be reported as a matched comparison.

## Finding 3 — degenerate output is universal on this benchmark, including in the baselines

Same audit applied to the published baselines and to the arm proposed as U28's teacher:

| arm | target acc at seeds 0,1,2,3 | what the target images actually show |
|---|---|---|
| `uce_church` | 0.0000, 0.0000, 0.1111, 0.0000 | **abstract texture mush.** No church, but no coherent alternative subject either |
| `u4_k8_s400_b2` | 0.2222, 0.1111, 0.1111, 0.3333 | repetitive grid / stripe / dot textures (U29) |
| `esd_church` | 0.4444, 0.6667, 0.7778, 0.6667 | weak erasure; churches largely present |
| `u23_*_a0.3` | ~0.40 | **churches, clean or under noise** |

**No arm in this project — ours or published — erases the concept while producing coherent
alternative content.** The two modes available are "emit texture mush" (UCE, and the proposed
teacher) and "emit the concept with an artifact that fools the classifier" (our alpha=0.3 arms).

On the visual axis the ranking inverts: UCE genuinely removes the church, at the cost of unusable
target-prompt output. Our arms leave the church visible. By image inspection our arms erase
**worse** than UCE, not better, and the H metric reports the opposite.

## Root cause

`target_suppression_acc` is a 12-way closed-set CLIP accuracy over 15 images with **no reject
option**. Every image is forced onto one of twelve labels. Consequences, all observed here:

- a noise overlay on a visible church is scored `van_gogh=0.72` -> counted as suppression
- a window-grid texture is scored `church=0.546` (U29) -> counted as leakage
- texture mush is scattered across labels -> counted as suppression

`legacy_retain8` does not catch any of this, because it is measured only on **non-target** prompts.
Nothing in the metric suite looks at target-prompt image quality.

## Consequences

1. **The UNDO alpha=0.3 direction is closed as a method claim.** Its advantage was an artifact.
2. **U28 is cancelled, not deferred.** Its teacher is degenerate (U29) and its success criterion
   contains no term that would have caught any of this.
3. **No retroactive change** is made to any closed study's protocol, results or interpretation.
   This is a new finding about what an existing metric measures, recorded as such.
4. The 86-arm reversal result is **unaffected in direction** — baselines still fully recover — but
   every "erased" figure in this project now carries the caveat that it was never checked against
   the image.

## What is not established

- Four eval seeds, 9 images per seed, one concept (church), one base model.
- Image assessment is by inspection of the saved contact sheets, not a quantitative off-manifold
  or no-reference quality measure. The qualitative difference (clean church at `church=1.00` vs
  texture mush) is large enough to read directly, but it is not a metric.
- `esd_church` and `uce_church` were audited at their own operating points, not swept.
