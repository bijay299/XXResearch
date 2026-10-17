# Final report — EraseAnything efficiency pilot

**Status: `INCOMPLETE` — effectiveness measurement BLOCKED by scorer adequacy; a measured
engineering result is delivered; no algorithmic advantage is established, and no advantage
over current methods is verified.**

Read with [`RUN_MANIFEST.md`](RUN_MANIFEST.md) (the contract) and
[`AMENDMENT_01.md`](AMENDMENT_01.md) (the competitor check and the stop decision).

---

## 1. What was asked, and what happened

The question was whether **one** algorithmic change could cut EraseAnything's adaptation
cost while preserving deletion, preservation and robustness. Three things were established,
in this order:

1. **The native reference runs and was profiled** — the measured bottleneck is exactly where
   static reading predicted.
2. **Ordinary engineering already clears the pilot's 25% cost bar** (35.0% measured), so by
   the brief's own rule this is an **engineering result**, not a contribution.
3. **The effectiveness half cannot be measured** — the frozen panel fails its M0 adequacy
   gate, and the failing component is our **scorer**, not the model.

Algorithm development was therefore stopped before any prototype was trained, and the one
authorised contemporary comparator arm was deliberately not taken (`AMENDMENT_01.md` §6).

## 2. Native reference — established and measured

The author's `calculate_upper_loss` / `calculate_lower_loss` were executed **unchanged**;
only the loop, placement, RNG control and timing are ours. Fidelity checks passed at
runtime: 76 LoRA tensors / 1,867,776 trainable params on `add_k_proj`+`add_q_proj`, 1160
backbone tensors frozen, all losses finite, adapter saved and reloadable.

**Profile, development concept, 6 steps (3 upper + 3 lower), 2 devices:**

| phase | measured |
|---|---|
| `step_lower` | **19.91 s/step** |
| `step_upper` | **5.96 s/step** |
| backward | 0.25 s/step |
| text encode | 0.049 s/call |
| model load | 15.0 s |
| frozen LLM call | 4.5 s (once) |
| **transformer forwards** | **435 in 6 steps** |
| **peak memory** | **38.38 GB** |

**435 is exactly the statically predicted 3×32 + 3×113.** The lower step is **77%** of step
time, and **112 of its 113** forwards are the K+1 = 4 full 28-step no-grad trajectories, of
which **one** attention map at a single random index is consumed and **27 of 28 discarded**
(`calc_loss.py:572, 595, 598`). Peak 38.4 GB on a 40 GB card confirms the two-device
placement was necessary, not a convenience.

**Extrapolated native cost to produce one adapter** (300 steps, 2 devices occupied):
**2.166 GPU-device-hours**. This is an extrapolation from a 6-step profile plus measured
preparation, not a completed 300-step timing run — labelled as such.

## 3. Measured engineering result (the main positive finding)

Two **exact** redundancy removals, each verified rather than assumed:

| configuration | wall, 6 steps | forwards | peak GB | lower s/step | vs native | extrapolated GPU-device-h |
|---|---|---|---|---|---|---|
| **A** native (author attention, full InfoNCE) | 82.65 s | 435 | 38.4 | 19.91 | — | 2.166 |
| + explicit softmax only in the consumed block | 72.47 s | 435 | 35.9 | 17.38 | **−12.3%** | 1.893 |
| **B** + early-stop InfoNCE at the consumed index | **53.72 s** | **311** | 35.9 | **11.11** | **−35.0%** | **1.372** |

- **Shared attention.** The author hardcodes `vis_attn = True`, so *every* dual-stream block
  materialises a full `[B, heads, 1280, 1280]` softmax while only the **last** block's matrix
  is returned. Verified in float64 that the consumed matrix is **bit-identical** either way
  (the model sample agrees to 3.3e-16 in fp64 — within tolerance, **not** bit-identical).
- **Early-stop InfoNCE.** The attention map at sampler index `j` depends only on `z_0…z_j`,
  so stopping there returns the same map bit for bit. Expected saving
  `1 − E[j+1]/28 = 48.2%` on those trajectories; measured forwards fell 435 → 311.

**This is ordinary removal of redundant computation and is reported as engineering.** It
already exceeds the pilot's ≥25% screening threshold, which is precisely why a proposed
algorithm would have had to beat 35% to add anything.

A real bug was found and charged, not hidden: the first early-stop measurement showed *no*
saving because the trainer bound `calculate_lower_loss` before the patch installed. Fixed;
affected measurement rerun and the wasted run charged.

## 4. The blocker — the panel does not qualify

| concept | canonical | perturbed | retention | qualified |
|---|---|---|---|---|
| Frida Kahlo | 32.5% | 20.0% | 55.0% | **NO** |
| Katsushika Hokusai | 32.5% | 25.0% | 60.0% | **NO** |
| Vincent van Gogh | 15.0% | 20.0% | 65.0% | **NO** |

The gate needs ≥80% on both target panels. Generic-object retention is **100%**, so the
pipeline and scorer work for object content.

**Correction to an earlier working note of ours.** An agent visual check of the fixed-order
M0 contact sheets shows the panels are unmistakable artist pastiche — the Van Gogh sheet has
Starry-Night skies, sunflowers in a yellow vase, wheat fields, the café terrace, worn boots,
cypresses and irises. The **generator is fine; the scorer is the failing component.** The
earlier note attributing this to FLUX not rendering style at 512 px is **withdrawn**. CLIP
ViT-L/14 zero-shot artist-identity discrimination is not sensitive enough for this task,
which is why the style-erasure literature uses a *trained* style classifier. This is our
methodological error in instrument choice. (Agent judgement, explicitly not human review.)

Three bounded diagnostics were run and **all** reported
(`measured/scorer_adequacy_diagnostic.json`): 1024×1024 regeneration (10%/45%), null-label
removal (35/55/35%), alternative template (47.5/45/22.5%). None reaches 80%. Searching
further would be choosing the instrument by its outcome, so it stopped there.

**Consequence.** With `p_M0` ≈ 0.15–0.33, `S = 1 − p_edited/p_M0` has a tiny noisy
denominator — exactly what the gate exists to prevent. The ≥75%-suppression rule and the
±5 pp noninferiority rule **cannot be evaluated**. The concepts are disclosed as
unqualified; none was replaced.

## 5. Contemporary methods (paper + author code, no GPU)

Full findings in `AMENDMENT_01.md` §3. The two that bear on this pilot:

- **EraseAnything++** (`f73e5c1`): its `calc_loss.py` is **byte-identical** to original
  EraseAnything — it does **not** touch the InfoNCE cost. Its "single backward pass"
  replaces the original's *alternating* schedule with **both** objectives every step plus
  periodic extra loss evaluations, which on our measured numbers (19.91 s lower vs 5.96 s
  upper) **increases** per-step cost. The paper reports **no** wall-clock or GPU-time
  comparison at all, so it is not an established efficiency baseline either.
- **BLOCK** (`db4ab11`): per-target removal on FLUX.1-dev in **1.020 s** after a one-time
  **5428 s** transcoder training, plus a permanent **0.43 s/image** inference overhead; beats
  EraseAnything on FLUX nudity detections (153 vs 199) with slightly worse FID (22.78 vs
  21.75) and low style IRA (36.1%). Not blanket dominance — a **different cost regime**
  (amortised per model, then nearly free per target, taxed at inference) versus
  EraseAnything (per concept, untaxed at inference).

**The unresolved challenge, stated plainly:** BLOCK's ~1 s per target sits against our
measured 2.166 GPU-device-hours per adapter — about three orders of magnitude, in a
regime our 35% engineering saving does not engage. **Any advantage of this result over
current methods is unverified**; no matched measurement against EA++ or BLOCK exists here.

## 6. Budget, images, deadline

| | consumed | cap |
|---|---|---|
| setup/feasibility/debug | 0.49 | 8 |
| development | 0.57 | 24 |
| held-out / eval / recovery | 0.92 | 64 |
| **TOTAL GPU-hours** | **1.97** | **96** |
| newly decoded images | **441** | 3000 |
| auxiliary LLM requests | **3** | 20 |

Failure costs charged, not discounted: **0.695 GPU-h** to an aborted M0 attempt that
CPU-thrashed, **0.026 GPU-h** to two crashed trainer starts, **2** failed LLM requests.
Deadline **2026-10-31T22:19:05Z**, unchanged. An external user's process on GPU 2 was
avoided, never preempted.

## 7. Limitations

- **No effectiveness result exists.** No deletion, preservation, robustness or quality
  comparison was produced for any arm. No equivalence, matched-effectiveness or
  semantic-absence claim is made.
- **No arm C, no comparator arm.** The 35% figure is a *cost* measurement of ordinary
  engineering on the development concept only — not a held-out result, not validated against
  any effectiveness criterion, and not compared to any contemporary method.
- The 2.166 / 1.372 GPU-device-hour figures are **extrapolations** from 6-step profiles plus
  measured preparation, not completed 300-step runs.
- Single development concept, one profile repetition per configuration; no variance estimate
  over repeats. Timings share a host with other users, though contention was checked.
- Image originals (441 PNGs) are **not in git** — too large for sensible git storage and no
  release upload was available. `evidence/IMAGE_MANIFEST.json` carries byte counts and
  SHA256 for all 430 retained originals, which remain on the work volume at
  `/data/bijaypandey/cuig_pilot/eraseanything_efficiency_pilot/`. Fixed-order review grids
  for all 9 M0 concept×panel combinations **are** published.

## 8. One evidence-based recommendation

**Re-specify the effectiveness instrument before any further EraseAnything efficiency work,
and treat the engineering result as already delivered.** Concretely: replace CLIP zero-shot
artist discrimination with a *trained* style classifier validated to ≥80% M0 target presence
on this exact panel, re-score the 360 existing M0 images (no new generation needed), and only
then decide whether an A-vs-B held-out comparison is worth 1440 images. The 35% ordinary
saving should be adopted regardless, since both components are exact.

This is a recommendation only. It is **not** launched, and nothing here authorises further
computation.
