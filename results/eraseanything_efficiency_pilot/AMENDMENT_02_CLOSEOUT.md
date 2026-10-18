# AMENDMENT 02 — closeout with corrections

**Documentation-only. No experiment, no model load, no GPU execution, no re-scoring, no new
API call was performed for this amendment.** It corrects the publication at
[`cb9cf23`](https://github.com/bijay299/XXResearch/tree/cb9cf23dc7cbc44408386ccd51e34a5bf058770d/results/eraseanything_efficiency_pilot),
which is preserved unchanged in Git history.

## Final status: `INCOMPLETE — PILOT CLOSED WITH CORRECTIONS`

This amendment **supersedes conflicting historical claims** in `README.md`,
`FINAL_REPORT.md` and `AMENDMENT_01.md`, and **supersedes every earlier recommendation** —
to adopt the optimisation, to replace or retrain the scorer, to re-score the M0 images, or
to continue the pilot in any form. **No successor assignment is proposed and none is
authorised.** Unused allowance does not authorise continuation.

---

## 1. Chronology

| when | event |
|---|---|
| 2026-10-17T22:19:05Z | pilot start |
| — | setup, pins, frozen cases, 360 M0 images, native profile, 3 profiled configurations |
| `cb9cf23` | pilot published as `INCOMPLETE` with `AMENDMENT_01` (EA++/BLOCK/ARC check; algorithm development stopped) |
| **this amendment** | review identified that the published "exact / bit-identical / safe to adopt" claims are **not supported**; corrected and closed |

## 2. The central correction — arm B is a CHANGED variant, not an exact control

**Withdrawn:** that early-stopping each InfoNCE trajectory at the consumed index `j`
returns the same attention map "bit for bit", that the change is "exact", that the
objective is "unchanged", and that it is an "ordinary exact removal" safe to "adopt
regardless".

**Why the published reasoning fails.** It assumed the sampler schedule is fixed, so that
truncating at `j` keeps the prefix `z_0…z_j`. It is not fixed:

- `code/ea_arms.py:163-173` computes `n_steps = j + 1` and passes that **length** into
  `latent_sample`.
- Upstream at pin `67d8c0bd…`, `utils/esd_utils.py:61` calls
  `scheduler.set_train_timesteps(timesteps, device=…)` with that length.
- `tools/scheduler_process.py:93-109` (`linear=False`, the default, and no caller overrides
  it) then draws a **new** random vector `torch.sigmoid(torch.randn((num_timesteps,)))`,
  maps it to `(1 − t)·1000` and sorts descending.

So arm B does **not** stop a 28-point schedule at `j`; it **resamples a different, shorter
full schedule of length `j+1`**. Time coordinates differ, the latent states differ, the
consumed attention map differs, and the amount of Torch RNG consumed differs, which can
perturb subsequent draws. The published claim of bit-identity was wrong, and the fp64 check
cited in its support tested a different property entirely (see §3).

**What survives, and at what scope.** The timing and work-count observations stand as
observations **of the changed variant**, from **one profile each**, on the development
concept only:

| | forwards | training seconds (6 steps) |
|---|---|---|
| A native | 435 | 82.651 |
| B (shared attn + early-stop) | 311 | 53.719 |

The ≈35% reduction is a **single-profile timing observation of a variant that changes the
computation**. It does **not** establish equivalent engineering, an exact control,
full-adapter savings, or a safe optimisation.

## 3. Selective explicit attention is not operationally equivalent in bf16

**Withdrawn:** that materialising the explicit softmax only in the consumed block is
output-preserving in training, and that the float64 check established this.

- Upstream blocks run fused SDPA, which changes the states entering the final explicit
  block, so the "consumed matrix is bit-identical either way" argument does not carry to
  the training path.
- The published raw profiles already contradict equivalence at **step 0**
  (`measured/profile/*.run_meta.json`): ESD loss **0.0196533203125** (A) vs
  **0.019775390625** (B1); gradient norm **0.1015625** (A) vs **0.1005859375** (B1);
  attention loss 62.0 vs 61.75.
- The fp64 check was a CPU check on a tiny randomly-initialised transformer. It does not
  establish bf16 training equivalence on FLUX.1-dev.

**Correction of a numeric error in the active text.** `measured/attn_equivalence.json`
records **max 165**, mean **1.4943949381510417**, differing-pixel fraction
**0.6395390828450521**. The figure "max 135" that appeared in the narrative came from a
different, unindexed timing script on a different prompt. Note also that this artifact
measures **decoded-image differences between two inference paths** — it is **not** evidence
about attention-map or gradient identity, and it was previously cited as if it were
supporting equivalence.

Historical files retain the original wording as evidence: `RUN_MANIFEST.md` §5 and
`code/gen_eval.py` still contain the "135" figure and are deliberately left unchanged; see
`code/README.md`.

## 4. Arm C — no result, and unsupported claims withdrawn

Arm C source exists (`code/ea_arms.py`) but **was never trained or evaluated**. Withdrawn:

- that its common-latent substitution reduces variance or preserves the contrast semantics
  — it **changes the sampling distribution**, and generic common-random-numbers reasoning
  does not establish either property here;
- any appeal to a "worked check in derivation_note.md" — **no such file was ever published**
  in this result set;
- that arm C "would have had to beat 35%", and that ordinary engineering "explains" arm C's
  gain. **No arm C gain exists and no arm B effectiveness result exists**, so neither
  statement is meaningful.

**The measurement failure (§6) independently justifies stopping.** Stopping does not rest
on the withdrawn cost reasoning.

## 5. Cost and accounting corrections

**2.166 / 1.372 GPU-device-hours are withdrawn from full-cost and end-to-end claims.** They
were built from `step_upper`/`step_lower` plus preparation only, and **omit the measured
backward, optimizer and remaining loop overhead** recorded in the same profiles. They are
incomplete phase extrapolations, not measured costs to produce a usable adapter. Where they
appear historically they are labelled with that omission. **No replacement full-run estimate
is offered**, because producing one would require a run that is not authorised.

**Ledger figures (authoritative, from `measured/ledger.jsonl`, raw entries preserved):**

| phase | GPU-hours |
|---|---|
| setup/feasibility/debug | **0.492141** |
| development | **0.606821** |
| held-out / evaluation | **0.916710** |
| **TOTAL** | **2.015672** → **2.016** |
| newly decoded images | **441** |

The **"held-out" category includes M0 generation**; it does **not** imply any edited-model
held-out exposure. There was none.

**Auxiliary provider requests: actual count UNKNOWN.** The ledger records 3 at
*invocation* granularity, but that number cannot be trusted as a provider-call count:

- our wrapper hardcodes `"requests": 1` per MoE invocation
  (`frozen/irrelevant_Frida_Kahlo.json` records `requests: 1`);
- native `MoE` (`tools/ir_concept.py`) issues **two** GPT calls per invocation under
  `model_list=["gpt"]`, `final_expert=["gpt"]` — a suggestion call and a final-expert
  arbitration call;
- SDK-level retries are not visible to us;
- **no retained log records any provider attempt**, so the count cannot be reconstructed.

Therefore: provider requests are **at least 3 and plausibly more; the true number is
unknown**, and no remaining allowance is asserted. No new API call was made to resolve this.

**Placement.** Peak **38.378 GB** establishes that the tested two-device placement ran. It
does **not** establish that single-device or CPU-offload alternatives are impossible; no such
alternative was tested. The earlier "the two-device placement was necessary, not a
convenience" is withdrawn.

## 6. Scorer and panel — narrowed

**Preserved raw counts (frozen scorer, unchanged):**

| concept | canonical | perturbed | retention |
|---|---|---|---|
| Frida Kahlo | **13/40** | **8/40** | **22/40** |
| Katsushika Hokusai | **13/40** | **10/40** | **24/40** |
| Vincent van Gogh | **6/40** | **8/40** | **26/40** |

All target panels **fail** the frozen ≥80% requirement. **This pilot therefore provides no
qualified edited-model deletion, retention, robustness or quality comparison.** That
conclusion is unchanged and is the pilot's primary limiting finding.

**Withdrawn:** the categorical claim that *"the generator is fine; the scorer is the failing
component."* A scorer mismatch remains **plausible but not independently established**:

- agent contact-sheet impressions are **not labelled style ground truth**;
- several visual cues cited as evidence of style (sunflowers, wheat fields, a café terrace,
  boots, cypresses, irises) were **already specified in the subject prompts**, so their
  presence does not demonstrate artist-style rendering;
- 100% generic-object recognition does **not** validate artist-style discrimination;
- the 1024 px probe does not isolate all resolution effects.

**Withdrawn:** the recommendation to select or train a style classifier "validated to ≥80%
M0 target presence on this exact panel". M0 recognition rate is **not** independent
classifier validation, and tuning an instrument until these same images pass would select
the measurement by its outcome. Any future measurement qualification would require
independent labelled evidence with separate calibration and validation sets. **That
assignment is not designed or launched here, and this frozen pilot is not retroactively
requalified.**

**Withdrawn:** the proposed automatic A-versus-B continuation. For the record, its
arithmetic was also wrong: 2 arms × 2 concepts × 2 seeds × 120 cases = **960**, not 1440.
Recording that correction is **not** a revised execution plan.

## 7. Competitor findings — preserved but narrowed

- **EA++ loss-file byte identity is supported** (`image/utils/calc_loss.py` vs
  `utils/calc_loss.py`, diff = 0 lines). The further observations — both objectives every
  step instead of alternation, plus periodic extra loss evaluations — are **static
  work-count observations of the code**, **not** measured EA++ runtime, convergence or total
  cost. We ran no EA++ training. The statement that this "raises per-step cost" is an
  inference from work counts, not a measurement.
- **BLOCK quantities are AUTHOR-REPORTED and kept separate**: upfront **5428 s**, per-target
  **1.020 s**, per-image **0.43 s**. BLOCK **Appendix E.1 reports 40 GB A100 training
  hardware**; this correction supersedes our earlier "hardware not specified" remark, which
  reflected the sections we had inspected. Exact matched Table 11 timing conditions remain
  **unmeasured here**.
- **Withdrawn:** the "about three orders of magnitude" ratio. It compared an author-reported
  per-target second count against our **incomplete, extrapolated, two-device** GPU-hour
  figure — not a like-for-like comparison.
- **Replaced:** "the mechanism is not covered / no overlap" becomes **"the specific
  substitution was not identified in the sections and code we inspected."** Absence of
  identification is not absence of prior art.

**No novelty claim and no claim that the problem is solved or unsolved is made.**

## 8. Evidence limitations

- **441 newly decoded outputs; 430 retained originals on the host;** original PNGs are
  **not published**. Fixed-order review grids (9) and `evidence/IMAGE_MANIFEST.json` hashes
  are available. **Hashes do not substitute for image access** — a reader cannot
  independently re-inspect or re-score the originals from this repository.
- **"Not published" is not "never performed", and vice versa.** The fp64 attention check and
  the scorer recalibrations were performed and are described, but their raw artifacts were
  not published. The arm-C "derivation note" referenced in historical text was **never
  produced**. No missing artifact has been backfilled for this amendment.
- Single development concept; one profile per configuration; no repeat measurements and no
  variance estimate. Timings shared a host with other users.
- `RUN_MANIFEST.md`, `AMENDMENT_01.md`, `frozen/`, `measured/`, `evidence/` and `code/` are
  **unchanged** as historical evidence and therefore still contain the superseded wording.
  This amendment governs where they conflict.

## 9. What this pilot supports, in full

1. The native reference **executed** at the stated pins, with runtime fidelity checks
   passing (76 LoRA tensors / 1,867,776 params; 1160 backbone tensors frozen; finite losses).
2. Measured per-step structure on the development concept, one profile: 435 forwards in
   6 steps; `step_lower` 19.91 s vs `step_upper` 5.96 s; peak 38.378 GB.
3. A **changed** variant (arm B) measured 311 forwards and 53.719 s against 82.651 s, one
   profile each.
4. The frozen panel **fails** its adequacy requirement on all three concepts, so **no
   edited-model effectiveness comparison exists**.
5. EA++'s loss file is byte-identical to the original's.
6. Budget: **2.016 GPU-hours**, **441 images**, provider-request count **unknown**.

Everything beyond this list that appears in earlier documents is either withdrawn above or
should be read as superseded.

**Pilot closed. No new experiments, comparator runs, re-scoring or follow-up project are
authorised or planned.**
