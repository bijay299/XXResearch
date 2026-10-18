# BLOCK object-erasure sprint — CLOSED

**Final status: `BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS.**

## Read in this order

1. **[`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md)** — authoritative. Corrects the
   status, discloses runs made after the stopping condition was known, corrects the retention
   and causal claims, narrows the measurement claims, and reconciles the accounting.
2. [`REPORT.md`](REPORT.md) — the sprint report, corrected in place to match the amendment.
3. [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) — every tracked file with SHA256, plus the
   release asset holding the 560 full-resolution scored originals.

## In one paragraph

A runnable source-derived implementation of **BLOCK** (arXiv:2606.25548v1,
`sprintml/block-concept-removal@db4ab11a`) was built for object erasure on **FLUX.1-dev**
(`@3de623fc`). Both transcoders were trained natively because none are released. Independent
real-photograph scorer controls passed (20/20 sensitivity and 20/20 specificity per object on
120 annotated COCO images). Baseline gates 2–6 were recorded as passed at the scope tested.
**Gate 1 was not satisfied**: generated-domain scorer qualification remained unresolved, which
under the frozen manifest required `BASELINE_UNQUALIFIED`, publication and a stop. **No
prototype and no control were developed, and no qualified method advantage or contribution
was established.** Native BLOCK results on three objects are preserved as descriptive judge
outputs from an unqualified measurement — including 160 images produced **after** the stopping
condition was known, disclosed as a protocol deviation.

## Historical records (preserved, superseded framing)

[`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md) ·
[`PROTOTYPE_DECISION.md`](PROTOTYPE_DECISION.md) ·
[`INTEGRATION_AUDIT.md`](INTEGRATION_AUDIT.md)

Their bodies are unchanged; each carries a supersession notice pointing to the amendment.

## Unchanged frozen artifacts

[`RUN_MANIFEST.md`](RUN_MANIFEST.md), [`cases.json`](cases.json), all raw
scores/answers/labels under [`evidence/`](evidence/), the ledgers, the manifests, the corpora,
the grids, the executable code in [`code/`](code/) (see
[`code/README.md`](code/README.md) for its limitations) and the patches in
[`patches/`](patches/).

## Not authorised by anything in this directory

No prototype, control, comparator, rescoring, relabelling, scorer-development, re-audit or
successor task. A stated measurement prerequisite is not an assignment.
