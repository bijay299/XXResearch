# EraseAnything efficiency pilot — EXECUTED, INCOMPLETE, CLOSED

> ## ⚠ READ [`AMENDMENT_02_CLOSEOUT.md`](AMENDMENT_02_CLOSEOUT.md) FIRST
>
> **Final status: `INCOMPLETE — PILOT CLOSED WITH CORRECTIONS`.** A review found that this
> pilot's "exact / bit-identical / safe to adopt" claims are **not supported**: arm B
> **resamples a different, shorter sampler schedule** rather than stopping the native one,
> so it is a **changed variant, not an exact control**. The closeout amendment **supersedes
> every conflicting claim below**, and supersedes all earlier recommendations — to adopt the
> optimisation, to replace or retrain the scorer, to re-score images, or to continue.
> **No successor work is authorised.**

Bounded pilot on the training efficiency of **EraseAnything** (ICML 2025,
`arXiv:2412.20413v1`) on **FLUX.1-dev**: reproduce an executable native reference, measure
its bottleneck, and test at most one improvement against an ordinary control.

> **Amended once — read [`AMENDMENT_01.md`](AMENDMENT_01.md) alongside this.** A targeted
> check of **EraseAnything++** and **BLOCK** was completed, after which **algorithm
> development was stopped** and the one authorised comparator arm was deliberately not
> taken. Budgets, start timestamp, deadline, concepts, cases, seeds and screening rules are
> unchanged.

## Status: `INCOMPLETE — PILOT CLOSED WITH CORRECTIONS`

**No qualified edited-model deletion, retention, robustness or quality comparison exists.
No algorithmic advantage is established. The timing observation below is a one-profile
observation of a CHANGED variant, not an exact or adoptable optimisation.**

### The three findings

**1 — The native reference runs, and its bottleneck is where static reading predicted.**
6-step profile: **435 transformer forwards**, exactly the predicted `3×32 + 3×113`. The
lower (InfoNCE) step is **19.91 s** against **5.96 s** for the upper step — **77%** of step
time. Within it, **112 of 113** forwards are four full 28-step no-grad trajectories from
which **one** attention map is consumed and **27 of 28 discarded**. Peak memory **38.4 GB**
on a 40 GB card — which shows the tested two-device placement ran, **not** that
single-device or offload alternatives are impossible. The previously quoted 2.166 / 1.372
GPU-device-hour "per adapter" figures are **withdrawn from full-cost claims**: they omit the
measured backward, optimizer and remaining loop overhead, and no replacement estimate is
offered.

**2 — A changed variant measured ≈35% less wall time in one profile (NOT an exact control).**

| configuration | wall (6 steps) | forwards | peak GB | vs native |
|---|---|---|---|---|
| **A** native (author-faithful) | 82.65 s | 435 | 38.4 | — |
| + explicit softmax only in the consumed block | 72.47 s | 435 | 35.9 | −12.3% |
| **B** + early-stop InfoNCE at the consumed index | **53.72 s** | **311** | 35.9 | **−35.0%** |

**Corrected (`AMENDMENT_02_CLOSEOUT.md` §2–§3).** Neither change is exact. Early-stop passes
`j+1` as a schedule *length*, and `set_train_timesteps` draws a **new random schedule** of
that length — so the time coordinates, latent states and consumed maps all change. Selective
explicit attention is not bf16-equivalent either: the raw profiles differ at **step 0**
(ESD 0.0196533203125 vs 0.019775390625; grad-norm 0.1015625 vs 0.1005859375). The figures
above are **one-profile observations of a changed variant** and establish neither equivalent
engineering, nor full-adapter savings, nor a safe optimisation. The claims that a proposed
algorithm "had to beat 35%" and that ordinary engineering "explains" any arm-C gain are
withdrawn: **no arm C result and no arm B effectiveness result exist.**

**3 — The frozen panel fails its M0 adequacy requirement, so no effectiveness result exists.**
Canonical 13/40, 13/40, 6/40; perturbed 8/40, 10/40, 8/40; retention 22/40, 24/40, 26/40 —
all target panels below the frozen ≥80%. Consequently the ≥75%-suppression and ±5 pp rules
**cannot be evaluated**, and the concepts are disclosed as unqualified rather than replaced.
**Corrected:** the categorical claim that "the generator is fine; the scorer is the failing
component" is **withdrawn**. A scorer mismatch is plausible but **not independently
established** — contact-sheet impressions are not labelled ground truth, and several cited
visual cues were already named in the subject prompts. The recommendation to train a
classifier until these same M0 images pass 80% is withdrawn as outcome-selected.

### What this is not

No deletion, preservation, robustness or quality comparison exists for any arm. No arm C was
trained; no comparator arm was run; no held-out outcome was ever inspected. BLOCK's
author-reported quantities (upfront 5428 s, per-target 1.020 s, per-image 0.43 s) are kept
**separate and author-reported**; the earlier "three orders of magnitude" ratio against our
**incomplete extrapolated** GPU-hour figure is **withdrawn** as not like-for-like.

## Files

| file | contents |
|---|---|
| [`AMENDMENT_02_CLOSEOUT.md`](AMENDMENT_02_CLOSEOUT.md) | **read first.** Corrections, withdrawn claims, corrected accounting, final status |
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
| GPU-hours (all phases, failures charged) | **2.016** | 96 |
| newly decoded images | **441** | 3000 |
| auxiliary provider requests | **UNKNOWN** (≥3; see closeout §5) | 20 |

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
