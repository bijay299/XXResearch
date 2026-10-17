# RESEARCH SCOPING — NO EXPERIMENTS

Evidence-first gap map for concept erasure in modern text-to-image models, including
flow-based backbones. Scoping only: a decision about what might be *worth a separately
reviewed next step*, not a result.

## Decision: **NONE — SCOPING PASS CLOSED WITH CORRECTIONS**

**No sufficiently specified and justified additional contribution was selected in this
bounded pass.** No field-wide impossibility or solvedness conclusion follows: the three
assessed directions are **not** shown to be solved, impossible or exhausted.

> **Amended 2026-10-17** — see [`CORRECTIONS.md`](CORRECTIONS.md). NONE is retained;
> supporting claims published at `5b2d184` are corrected (a backwards repair direction for
> DVE's Eq. 9, an unsupported expressiveness separation, an over-general "the frontier is
> inference-time" claim, prior-art reasoning that treated existing comparisons as disproof,
> and blanket field-closure wording). **No new experimental evidence was produced or
> required.**

All three cards are retained and their novelty gate is now **UNRESOLVED — insufficiently
specified**, not FAIL. Relevant comparisons do exist (DVE's FLUX baselines; CoME-Bench's
fused-MACE evaluation with non-target preservation metrics), and **repeating them would not
establish novelty** — but they do not prove any objective solved or exclude a stronger
algorithmic or theoretical result. What blocked selection was under-specification. An
unresolved essential gate prevents selection just as a failure would.

| start here | for |
|---|---|
| [`GAP_MAP.md`](GAP_MAP.md) | the decision, the capability-and-assumptions matrix (deployment contract kept distinct from mechanism), and the gate table with reasons |
| [`CANDIDATES.md`](CANDIDATES.md) | the three retained claim cards, each with its strongest objection and why it was not selectable, plus directions not advanced in this pass |
| [`CORRECTIONS.md`](CORRECTIONS.md) | **read alongside the map.** The replaced claims (maths, prior-art reasoning, scope) and corrected wording, with a pointer to the original publication |
| [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) | every source with version, **inspection depth** and **claim-support** status, secondary-only mentions, and access limitations |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Limits of this decision

- **No experiments.** No models or encoders, no checkpoint-tensor loading, no feature
  extraction, no edit/replay execution, no generation, no classifiers/detectors, no image
  viewing or reannotation, no GPU allocation or polling, no installs or downloads, no
  method implementation or execution adapters, no new datasets or evaluation packages, no
  reviewer/author contact, and no imported research code executed. Paper and source reading
  only, plus ordinary CPU bookkeeping and hashing.
- **NONE does not mean no gap exists in this field**, and does not rule out that any of
  C1–C3 could become selectable if specified sharply. Inspection was bounded: EMMA,
  M-ErasureBench and 2511.07970v2 were read at abstract/listing level only; no code was
  audited except the already-pinned DP repository; MACE, ESD, CA, EAP and Meta-Unlearning
  are known only as baselines named inside other papers. **No exhaustive novelty search is
  claimed.**
- **Method design is an admissible contribution class.** A useful nontrivial theoretical
  characterization, an algorithmic improvement, or a rigorously supported empirical
  limitation all qualified; "build a better editor" was simply too vague to select. The
  earlier statement that method design was an insufficient class is withdrawn.
- **Flow-model erasure is not an inference-time-only landscape.** Of the flow-targeted work
  inspected, Minimalist Concept Erasure (parameter masking, FLUX.1-schnell) and
  EraseAnything (LoRA, FLUX.1-dev) are **weight-space**, while DVE and Mosaic are
  inference-time. The inference-time observation applies to **those two inspected methods**,
  not the field.
- DVE's reported comparisons are **not uniformly first-party**: Table 1 marks some baseline
  numbers as sourced from EraseAnything, and its robustness table covers a smaller baseline
  subset than §5.1 lists.
- **Unknown coverage is not failure**, and a missing evaluation, an untested combination or
  a stated limitation does not by itself establish a gap.
- Backbone versions differ and are not interchangeable: Mosaic, DVE and EraseAnything use
  **FLUX.1-dev**; Minimalist Concept Erasure and our own closed records use
  **FLUX.1-schnell**.

## Relationship to closed work

Closed findings supplied **context only, never fresh confirmation data**, and nothing was
re-scored, regenerated or modified:

- The changing-replacement candidate remains **NO_GO — CLOSED_CORRECTED** (`5786864`) and
  is **not** revived here under renamed policy, guide-refresh or dependency-graph
  terminology. That closure is **specific to that formulation** and does not close the
  sequential/continual field; likewise the previously unselected cached-evaluation
  formulation does not close evaluation or benchmark research generally.
- The FLUX chain-saw→parachute diagnostic remains closed at `0e88d26` (DP passed its stated
  numeric gates; UCE failed cumulative retention; neither showed detector-discovered
  relapse). No re-scoring, expanded prompts, reruns, or spending of the unused image
  allowance.
- SD1.x/human-review closures and pauses, and the previously unselected projection,
  unrestricted-certificate and evaluation-reuse formulations, are preserved as recorded.

**Nothing in this map authorizes implementation or experiments**, no successor assessment
is scheduled, and no follow-up job was created.
