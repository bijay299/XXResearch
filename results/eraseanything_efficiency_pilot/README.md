# EraseAnything efficiency pilot — EXECUTED, INCOMPLETE

Bounded pilot on the training efficiency of **EraseAnything** (ICML 2025,
`arXiv:2412.20413v1`) on **FLUX.1-dev**: reproduce an executable native reference, measure
its bottleneck, and test at most one improvement against an ordinary control.

> **Amended once — read [`AMENDMENT_01.md`](AMENDMENT_01.md) alongside this.** A targeted
> check of **EraseAnything++** and **BLOCK** was completed, after which **algorithm
> development was stopped** and the one authorised comparator arm was deliberately not
> taken. Budgets, start timestamp, deadline, concepts, cases, seeds and screening rules are
> unchanged.

## Status: `INCOMPLETE`

**Effectiveness measurement is BLOCKED by scorer adequacy. A measured engineering result is
delivered. No algorithmic advantage is established, and no advantage over current methods is
verified.**

### The three findings

**1 — The native reference runs, and its bottleneck is where static reading predicted.**
6-step profile: **435 transformer forwards**, exactly the predicted `3×32 + 3×113`. The
lower (InfoNCE) step is **19.91 s** against **5.96 s** for the upper step — **77%** of step
time. Within it, **112 of 113** forwards are four full 28-step no-grad trajectories from
which **one** attention map is consumed and **27 of 28 discarded**. Peak memory **38.4 GB**
on a 40 GB card. Extrapolated native cost per adapter: **2.166 GPU-device-hours**.

**2 — Ordinary engineering already clears the pilot's 25% bar: 35.0% measured.**

| configuration | wall (6 steps) | forwards | peak GB | vs native |
|---|---|---|---|---|
| **A** native (author-faithful) | 82.65 s | 435 | 38.4 | — |
| + explicit softmax only in the consumed block | 72.47 s | 435 | 35.9 | −12.3% |
| **B** + early-stop InfoNCE at the consumed index | **53.72 s** | **311** | 35.9 | **−35.0%** |

Both changes are **exact on the consumed quantity** (bit-identical attention maps, verified
in float64). Per the brief, when ordinary engineering explains the gain this is an
**engineering result, not a contribution** — and a proposed algorithm would have had to beat
35% to add anything.

**3 — The frozen panel fails its M0 adequacy gate, and the fault is ours.**
Canonical target presence 32.5% / 32.5% / 15.0% against a required ≥80%. An agent visual
check of the fixed-order contact sheets shows the images are unmistakable artist pastiche:
**the generator is fine, our CLIP zero-shot scorer is not sensitive enough.** An earlier
working note blaming 512 px rendering is **withdrawn**. Three bounded diagnostics
(1024 px, null-label removal, alternative template) are all reported and none reaches 80%.
Consequently the ≥75%-suppression and ±5 pp rules **cannot be evaluated**, and the concepts
are disclosed as unqualified rather than replaced.

### What this is not

No deletion, preservation, robustness or quality comparison exists for any arm. No arm C was
trained; no comparator arm was run; no held-out outcome was ever inspected. **BLOCK reports
~1 s per target on FLUX.1-dev after a one-time 5428 s transcoder training, against our
measured 2.166 GPU-device-hours per adapter** — a gap our 35% saving does not engage.

## Files

| file | contents |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | measured results, the blocker, contemporary methods, limitations, one recommendation |
| [`AMENDMENT_01.md`](AMENDMENT_01.md) | the EA++/BLOCK/ARC check, the technical distinction, the stop decision, budgets |
| [`RUN_MANIFEST.md`](RUN_MANIFEST.md) | pins, native-method reading, concepts/cases/seeds, scoring, 4 documented patches, append-only deviation log |
| [`code/`](code/) | runnable trainer, arms, generation+scoring, case freezer, patch script, ledger |
| [`frozen/`](frozen/) | 360 frozen cases + seeds, training-image manifests, frozen LLM responses, measured summary |
| [`measured/`](measured/) | per-case M0 scores (3×120), 3 profile `run_meta.json`, budget ledger, scorer diagnostic, attention-equivalence measurement |
| [`evidence/`](evidence/) | 9 fixed-order M0 review grids + SHA256 manifest of all 430 retained originals |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, bytes, sha256 of every published file |

## Budget

| | consumed | cap |
|---|---|---|
| GPU-hours (all phases, failures charged) | **1.97** | 96 |
| newly decoded images | **441** | 3000 |
| auxiliary LLM requests | **3** | 20 |

Start **2026-10-17T22:19:05Z**, deadline **2026-10-31T22:19:05Z**. Charged failures include
0.695 GPU-h to a CPU-thrashing aborted run and 2 failed LLM calls. No utilisation discount.

## Reproducing

Pins: author code `tomguluson92/EraseAnything@67d8c0bd`, model
`black-forest-labs/FLUX.1-dev@3de623fc`, diffusers 0.32.0 + the ported patch in
`code/apply_ea_patch.py`, scorer `openai/clip-vit-large-patch14`. FLUX.1-dev is gated;
credentials are not in this repository. Image originals are not in git (see
`FINAL_REPORT.md` §7); their SHA256 are in `evidence/IMAGE_MANIFEST.json`.

## Scope

Closed work elsewhere in this repository is untouched: the gap map at `edaec97`, the
changing-replacement closure at `5786864`, the FLUX DP/UCE diagnostic at `0e88d26` and its
unused budget, the SD1.x/RECE and human-review closures, and the Minimalist/SSMG design
sprint at `855240c`. None was rerun, re-scored or reinterpreted.
