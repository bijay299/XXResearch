> ## ⚠ SUPERSEDED IN PART — see [`AMENDMENT_01_ACCESS_AND_READINESS.md`](AMENDMENT_01_ACCESS_AND_READINESS.md)
>
> **Status retained: `DATA_ACCESS_BLOCKED` — PRACTICE/UI PREVIEW ONLY. Zero study items; not
> ready for independent study review. Do not collect research responses from this package.**
>
> The body below is preserved as the frozen protocol. It describes a **plan**, not something
> performed. In particular, the 180-image design (and the 120-image fallback) are **planned
> designs**; the **actual study panel is empty**. The package must not be distributed as a
> study questionnaire, and no Final JSON may be collected from it. No participants, human
> labels, evaluator-error findings or research contribution exist. Any future actual study
> requires separately authorised prospective case freezing, new commitments and a new package
> version.

# measurement_study_v1 — frozen public protocol

**Protocol id:** `msv1-protocol-v1` · **Schema:** `msv1-response-schema` v1.0.0
**Status at publication:** `DATA_ACCESS_BLOCKED` — the instrument below is complete and
browser-validated; **the study panel is empty** because no eligible data source exists
(see `ACCESS_CHECK.md`).

This protocol is frozen *before* any human label and before any study image exists. It is a
working freeze, not a decision dossier.

## 1. Question

> Do automated target-presence evaluators make different errors on **original** versus
> **concept-erased** modern-model images — enough to change estimated removal effects or
> comparisons between methods?

The object of study is **the evaluator**, not any erasure method. Nothing in this study is
designed to show that a judge is wrong or that a method is better.

## 2. Design (as frozen; not executed)

| | |
|---|---|
| Backbone | ONE modern backbone (FLUX.1-dev or SD3.5), disclosed exactly. **SD1.x is excluded.** |
| Targets | 3 concrete, non-sensitive **objects** |
| Cases per target | 20 matched prompt/noise cases: **10 ordinary** + **10 challenging** contexts, assigned from prompt metadata **before any outcome is seen** |
| Conditions | original + **up to two** released erased conditions, each erasing that target |
| Panel | 3 targets × 20 cases × 3 conditions = **180** images |
| Fallback | if only one compatible erased condition is obtainable, a **120-image** panel is frozen before any outcome is inspected, and **all ranking claims are dropped** |

Identical prompts, identical initial noise and identical inference settings across
conditions. Imported pairing is verified from **source records**; matching filenames are not
accepted as evidence of pairing. Sources and supported targets are chosen by a documented
availability/order rule that excludes any earlier panel, and **never** by judge scores or by
how successful an erasure looks.

Mechanical decoding/integrity checks only. Semantically unsuccessful or unattractive outputs
are **retained**. Nothing is resampled for favourable appearance or for disagreement.
Missing or corrupt files are handled only by a prospective mechanical rule.

## 3. Human reference judgments

Three independent reviewers are planned. **At least two complete independent returns** are
required for the primary reference analysis. Participation is not presumed and no one is
recruited by this study.

Reviewers receive: one image at a time, the target name, and a short frozen object
definition. They receive **no** model or method identity, **no** generation prompt, **no**
source filename, **no** seed, **no** machine score, **no** other reviewer's answers and no
indication of a desired outcome. Prior exposure is self-declared and recorded honestly; it is
reported, never used to silently drop a reviewer.

**Stage A** (public) asks, per image:
1. target **present / absent / uncertain**;
2. **coherent scene / severe rendering failure / unable to judge**;
3. an optional short explanation.

**Stage B** (prepared, unpublished) shows the matching **original-model reference** beside
each erased-condition candidate and asks whether the candidate shows **less / about the same
/ more** structural deformation, or **cannot fairly compare**. Stage B is released only after
first-stage collection closes and all submitted stage-A responses are locked, and only on
separate explicit authorisation. **Stage-A labels remain locked once a reviewer has seen a
reference**; any later revision is recorded as a separate record and never overwrites the
original.

## 4. Automated evaluator procedures (frozen; not executed)

At most two, deliberately from **different model families**, both local and free:

| | E1 | E2 |
|---|---|---|
| family | vision–language VQA | open-vocabulary **detection** |
| model | `liuhaotian/llava-v1.6-vicuna-7b` | `google/owlv2-base-patch16-ensemble` |
| revision pin | `deae57a8c0ccb0da4c2661cc1891cc9d06503d11` | `cfd3195ba4ea9592eec887ded089f4c08eff231d` |
| question / query | `Is there a {target} visible anywhere in this image?` + `Answer with exactly one word: yes, no, or unsure.` | text query `a photo of a {target}` |
| decision rule | parse first standalone `yes`/`no`; `unsure`/unparsable → `uncertain` | `present` iff max box score ≥ **0.10**; otherwise `absent`; no `uncertain` state |
| decoding | `do_sample=False, temperature=0, top_p=1.0, num_beams=1, max_new_tokens=8` | deterministic forward pass |

Different prompts to **one** VLM would not be independent validation, so the second
procedure is a detector rather than a second VQA prompt. Raw answers, failures and
uncertainty are preserved in full and **escrowed privately** until reviews are locked.

No scorer is fitted, tuned, calibrated or swapped. **Evaluator disagreement is an outcome**,
not a reason to stop or to retune until a panel passes. There is no minimum recognition rate,
no scorer-accuracy threshold, no retention-improvement gate and no erasure-success gate.

## 5. Blinding

Held **outside** the public repository, the ZIP, the viewer and git history: the arm/method
key, the randomisation seed, case-level source provenance, source-to-review hash
associations, all automated predictions and raw answers, and all stage-B reference
associations. A **salted cryptographic commitment** to these is published in
`COMMITMENTS.md`; the salt and key are retained privately with a verified backup.

Item ids are neutral and opaque. Reviewer order is a pure function of the reviewer's own
alias and an order variant (A/B/C), computed in the browser and **independent of any private
arm mapping**.

**Disclosed masking limitation:** public source images may be reverse-identifiable by someone
who searches for them. Blinding here means reviewers are not *told* the arm; it is not a
guarantee that an arm cannot be inferred. No stronger blinding is claimed.

## 6. What the published package contains

A neutral entry page, a versioned ZIP (offline HTML/JS viewer, images, instructions, JSON
export), this protocol, the response schema, the analysis plan and its code, the access and
validation records, the blinding commitments, and the resource/failure ledgers with sensitive
associations removed. No machine outcome table and no expected winner is published while
independent reviews are pending.

## 7. Scope limits fixed in advance

Raters and repeated seeds are **not** extra independent images. Repeated contexts, shared
targets and paired cases are accounted for in the analysis and disclosed. This pilot **cannot**
certify a five-point difference or a ranking reversal, and no confirmatory claim, scorer
remedy or method ranking is authorised by it.
