# AMENDMENT 01 — access conclusion and readiness

**Retained status: `DATA_ACCESS_BLOCKED` — PRACTICE/UI PREVIEW ONLY.**
**Zero study items; not ready for independent study review. Do not collect research
responses from this package.**

This amendment is **authoritative** over conflicting wording in the documents published at
commit `8d9ecee757cd9db53b5273d4bf1ede7cdcc158dd`.

It is **documentation-only**. No GPU was allocated; nothing was generated, scored, trained,
downloaded as model or data, relabelled, rebuilt or repackaged; no code, schema or decision
rule was changed; no test or browser validation was re-run; no reviewer was contacted; no
response was imported; and no outcome was analysed. Code claims below were checked by
**static reading only** of the already-published files.

---

## 1. Corrections: old claim → corrected claim → supporting record

### 1.1 The access conclusion was overstated

| | |
|---|---|
| **Old claim** | "no eligible data source exists"; "Every modern-backbone erasure paper located releases **code only**"; "**zero genuine hits**" presented as establishing that no modern erased checkpoints are published. |
| **Corrected claim** | **No eligible matched non-sensitive object assets were identified during the bounded access check.** |
| **Supporting record** | Zero GitHub releases and zero metadata-filtered search hits are **absence of evidence**, not evidence of absence. They cannot establish nonexistence. |

**Withdrawn**, explicitly: that no modern erased checkpoints exist; that all relevant modern
papers are code-only; and that no eligible data source exists anywhere.

### 1.2 The GEM row was wrong

| | |
|---|---|
| **Old claim** | "GEM … **No** — code only (`multimodal-ai-lab/GEM`)". |
| **Corrected claim** | **GEM publishes FLUX erasure reproductions for sensitive targets.** The project links released checkpoints `MAI-Lab/GEM-Flux-Nudity` and `MAI-Lab/GEM-Flux-Bloody-Gore`, both FLUX.1-dev derivatives in safetensors. They are **ineligible for this frozen non-sensitive object study** because their targets are sensitive content, not concrete non-sensitive objects. |
| **Supporting record** | [GEM README @ `c328fc6`](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/README.md) · [`MAI-Lab/GEM-Flux-Nudity`](https://huggingface.co/MAI-Lab/GEM-Flux-Nudity) · [`MAI-Lab/GEM-Flux-Bloody-Gore`](https://huggingface.co/MAI-Lab/GEM-Flux-Bloody-Gore) |

As the model cards themselves state, these are **reproductions, not the original paper
checkpoints**: *"It is not the original checkpoint from the paper, but a reproduction."* That
distinction is preserved here rather than flattened into "the paper's weights".

### 1.3 Why the original search could not have found them

This is the concrete reason the old existence claim was unsafe, not a hypothetical.

Both GEM repositories **are** correctly tagged `concept-erasure`, `unlearning` and
`base_model:black-forest-labs/FLUX.1-dev`. The access check's most precise instrument was a
`base_model:`-filtered query crossed with the terms `erase` / `eras` / `unlearn`. Re-reading
that query shows the HuggingFace `search=` parameter matches the **model name**, not the tag
set — and neither `GEM-Flux-Nudity` nor `GEM-Flux-Bloody-Gore` contains "eras" or "unlearn"
in its name. The filtered query therefore returned a single unrelated name match and
**silently missed two correctly-tagged FLUX erased checkpoints**.

A keyword-on-name search cannot enumerate a namespace. The bounded check remains a
reasonable bounded check; it was never capable of supporting a nonexistence claim.

### 1.4 What this correction does and does not change

It does **not** resolve the missing item. No matched original/erased **non-sensitive object**
assets on a modern backbone were identified. The frozen study excludes sensitive content, so
the GEM reproductions cannot fill this panel.

**Status is retained: `DATA_ACCESS_BLOCKED`.** No further source hunt, literature review,
download or source-selection exercise was performed or is authorised.

---

## 2. Task A is practice/UI preview only

| | |
|---|---|
| **Old claim** | Reviewer-facing distribution instructions: send the ZIP to three reviewers with aliases and order variants, have them "Export final", and return the JSON; the package was described as a package for independent review. |
| **Corrected claim** | **The published ZIP is a practice / UI preview only.** It contains **zero study images** and **12 non-study practice images**, and its study-item array is **empty**. It must **not** be distributed as a study questionnaire, and **no research responses may be collected from it**. |
| **Supporting record** | Read directly from the published archive: `window.__CFG__.items` is `[]`; `package_id` `msv1-taskA-v1.0.0`; 15 zip entries; 12 `practice/*.png`. |

**Withdrawn:** the instruction to distribute this package as a study questionnaire and
collect Final JSON from reviewers.

**Preserved distinctions:**

- The **planned** designs — 180 images (3 targets × 20 cases × 3 conditions), or a 120-image
  fallback with ranking claims dropped — are **plans only**. The **actual** study panel is
  empty. The two must not be conflated.
- The **normal viewer correctly disables Final submission at zero items**
  (`btnFinal.disabled = !(n > 0 && done === n)`), so it does not invite an empty study
  return. An empty **draft** export is a UI artefact, **not a human study response**.
- No participants, no human labels, no evaluator-error findings and no research contribution
  exist.

**Archive preserved unchanged.** The ZIP, package id, manifest, previews and image bytes are
**not** rebuilt, repackaged or retargeted:

| | |
|---|---|
| bytes | `18,641,021` |
| SHA256 | `604e3227d28bcc3c1548a59da33fd5a01f27fed9f95e35409035a547c3fb2246` |

The generic completion instructions **inside** that historical archive (`README.txt`, and the
viewer's own "send the downloaded JSON to your coordinator" wording) are **superseded by this
amendment**. They remain inside the archive because the archive is preserved byte-for-byte;
they do not authorise collection. No historical tag or release is retargeted.

---

## 3. Privacy and blinding, stated precisely

What the published record supports:

- **No actual Task-B reference associations, arm keys or automated scores were found in the
  published package.** The published `window.__CFG__` contains no `ref_file`.
- **Generic shared viewer code containing an unused reference-file field is not itself a
  mapping leak.** `code/build_viewer.py` is the shared builder for both stages and references
  `it.ref_file` when it emits a stage-B page; no association data accompanies it.
- **`COMMITMENTS.json` explicitly describes the arm-key, case-provenance and prediction
  material as empty JSON objects (`{}`)**, because no study cases exist. The commitments
  attest that nothing was withheld at publication time — **not** that a populated hidden
  mapping exists.
- **Task B is an empty private scaffold**, not a populated matched-reference review package.
- **Private seed/salt storage and backups are host-reported.** The coordinator did **not**
  independently verify their contents or permissions. The earlier report of verified backup
  digests and `0600`/`0700` permissions is a statement of what the host reported, nothing more.
- **Meaningful masking of actual study arms has not yet been exercised**, because there are no
  study arms. The masking audit shows the published bundle is clean of the listed material; it
  does not demonstrate that blinding works on a real panel.

Commitments are **unchanged**. Task B, salts, seeds and other private material remain outside
Git, and nothing private is published here to "prove" privacy.

**Any future actual study requires separately authorised prospective case freezing, new
commitments and a new package version.** This frozen empty panel must not be silently
populated.

---

## 4. Readiness: planned versus implemented

Checked by **static reading** of the published `analysis/` code. **Nothing was repaired**; the
code is published unchanged and these are prospective limitations requiring separate
authorisation before any future collection or analysis.

| # | Planned (PROTOCOL / ANALYSIS_PLAN) | Implemented (static reading) | Gap |
|---|---|---|---|
| 1 | "≥ 2 **complete independent** returns" admitted to the primary analysis | `analyze.py:141` — `primary_analysis_runnable = len(finals) >= 2`, where `finals` is just files with `export_kind == "final"` | Counts **files**, not distinct, complete, independently eligible reviewers. Two files sharing an alias both count. `analyze.py` checks no `package_id` / `protocol_id` / `stage` / `schema_version`, and no completeness. A positive study-size check is absent and would anyway be insufficient. |
| 1b | Validator gates admissibility | `validate_responses.py` errors only if `responses` is **not a list**; an **empty list passes**. Count mismatches, missing rows and a disagreeing `answered` flag are **warnings** | A **zero-row or partial Final validates as OK**. (Distinct from the viewer, which correctly blocks an empty Final.) |
| 2 | "every rate … accompanied by an interval … assigning **all `no_reference`** and all evaluator-`uncertain` cases first in the most favourable and then in the least favourable direction" | `rates()` builds `pos`/`neg` from reference-present/absent only; `nor` is **counted but never swung**. `band()` widens solely by evaluator-`uncertain` **within** `pos`/`neg` | `no_reference` cases are **excluded from the bounds** although the code's own `note` string asserts they are assigned both ways. `coverage` is a bare scalar with **no bounds**. |
| 3 | "cluster bootstrap: **resample targets, then cases within target**", preserving pairing and shared contexts | `idx[target].append(nid)` passes **individual image ids**; `cluster_bootstrap` then draws `rnd.choice(cases)` over those image ids (the local variable is named `cases` but holds images) | Resampling is over **images, not matched case bundles**, so an original can be drawn without its erased counterpart. Original/erased **pairing and shared contexts are not preserved** as claimed. |
| 4 | Secondary estimand: removal effect from evaluator vs from human reference, **and the difference**, with bounds | `paired_transitions()` emits transition rows only; the string "suppression" appears **zero times** in `analyze.py` | The planned **suppression-estimate contrasts and their bounds are not implemented**. |
| 5 | "Primary analysis requires **≥ 2 complete independent returns**" (§1) | "If a reviewer's file is **partially complete**, their answered items still contribute to the reference for those items" (§5) | The plan is **internally inconsistent** about whether partial returns may contribute. |

### The 47/47 browser checks, correctly scoped

The reported **47/47** are **synthetic-fixture UI checks**, supported by the published suite
(`validation/run_validation.js`) and log (`validation/validation_results.json`). They
establish that the interface behaves as intended on fixtures.

They do **not** establish statistical correctness, independent reviewer admission, real-arm
blinding, or study readiness. The three viewer defects they caught were genuine, and that
remains true; it is a statement about the UI, not about the study.

**No scientific result is invalidated, because no study result exists.**

---

## 5. Accounting — retained, and qualified

Recorded preparation totals, retained unchanged:

| | |
|---|---|
| GPU-hours | **0.042485** (≈ **0.0425**) |
| generated practice images | **12** |
| study images | **0** |
| recorded failed GPU jobs | **0** |

The rounded **152.9 s** duration is **consistent** with the rounded GPU-hour report
(`152.9 s × 1 device ÷ 3600`). There is **no accounting discrepancy**, and none is manufactured
here.

These are **recorded totals from this study's own ledger**, not independent proof of complete
host history.

The original **12 GPU-hour, 240-image and 72-hour** figures are **historical ceilings of the
preparation assignment**. They are **not** renewed permission to spend unused allowance.

---

## 6. No-new-experiment declaration

No new experiment, generation, scoring, training, model or data download, image relabelling,
source hunt, literature review, source selection, code repair, schema or decision-rule change,
test or browser re-run, fixture generation, analysis execution, reviewer contact, response
import, Task-B release, unblinding, or reopening of closed work occurred in producing this
amendment, and none is authorised by it.

This amendment corrects documentation and stops.
