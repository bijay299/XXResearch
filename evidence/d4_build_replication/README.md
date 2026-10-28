# D4 — the load-bearing arm replicates across build seeds

**2026-10-28. 3 builds, no failures.**

D4 is "n=1 is pervasive". Attacks in this project are replicated at n=3, but **every arm is a single
build**. Five claims here have already died on seed replication (F35, F59, F62, F69, F71), so an
unreplicated build is the project's most reliable way of being wrong.

`esd_church` carries the most weight of any arm. U34, U36 and U37 all use it for the headline that
**benign fine-tuning on non-target concepts completely undoes erasure** — verified in images, and
confirmed by the self-source control. That rested on one build at seed 0.

## Result

ESD, church, 500 steps, lr 1e-5, three build seeds:

| build seed | un-attacked target suppression | retain8 |
|---:|---:|---:|
| 0 | 0.4667 | 0.9917 |
| 1 | 0.5333 | 1.0000 |
| 6 | 0.4667 | 1.0000 |
| **mean** | **0.4889** (sd 0.0385) | **0.9972** (sd 0.0048) |

**The full range is 0.0667 — exactly one quantum** (1/15), the metric's own resolution. Utility is
flat at ~1.00 with sd 0.005.

So ESD's erasure level is a property of the method **as we implement it**, not of a lucky build
seed. The benign-recovery headline is now replicated at the **build** level, which no other result
in this project has.

## What this does and does not close

**Closes:** the specific D4 exposure on the arm carrying the headline. The claim "ESD erases to
~0.49 and is fully recovered by benign fine-tuning" survives build replication.

**Does not close:** D4 in general. The other 163 arms remain single builds. What this establishes is
that build-seed variance on a training-based erasure method is **at or below the metric's
resolution**, which makes the single-build design defensible rather than merely convenient — but
that is one method, one concept.

**Not applicable to UCE:** it is a closed-form edit with no stochastic training, so build seeds do
not exist for it.

## Defect status

| | status |
|---|---|
| **D1** inconsistent label text | **closed** — moves metrics ~0.01 against a 0.067 noise floor |
| **D2** validity-gate bias | **closed** — confirmed, runs against our claim, conclusion survives unfiltered |
| **D3** fidelity | **open, widened** — no arm is an official reproduction; wording corrected, fix is to attack author-released checkpoints |
| **D4** pervasive n=1 | **closed on the load-bearing arm**; general exposure remains |

## Limits

- One method (ESD), one concept (church), three seeds, SD-1.4.
- Replicates the **un-attacked** erasure level. The benign-recovery measurement itself was already
  n=3 over fine-tuning seeds (U34/U37); this adds replication of the thing being fine-tuned.
- sd from three points is a weak estimate; the range is the more honest statistic and it is one
  quantum.
