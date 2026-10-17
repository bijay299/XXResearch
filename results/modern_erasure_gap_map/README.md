# RESEARCH SCOPING — NO EXPERIMENTS

Evidence-first gap map for concept erasure in modern text-to-image models, including
flow-based backbones. Scoping only: a decision about what might be *worth a separately
reviewed next step*, not a result.

## Decision: **NONE**

No candidate was selected. Three distinct candidates were formulated and gated; **all three
fail the plausible-novelty gate**, on two separable grounds:

- **Established coverage** — CoME-Bench already evaluates a weight-editing baseline (MACE
  fused into DiT linear weights) on compositional selective removal with a
  Selective-Alignment metric; DVE already benchmarks ESD/UCE/CA/MACE/EAP/Meta-Unlearning/
  EraseAnything on FLUX.1-dev with deletion, utility, four-suite robustness and latency.
- **Insufficient claim class** — what remains is a model port, a benchmark extension, an
  elementary expressiveness observation, or a hyperparameter/attack study, each explicitly
  insufficient on its own.

| start here | for |
|---|---|
| [`GAP_MAP.md`](GAP_MAP.md) | the decision, the capability-and-assumptions matrix (deployment contract kept distinct from mechanism), and the gate table with reasons |
| [`CANDIDATES.md`](CANDIDATES.md) | the three claim cards in full, each with its strongest objection and why it failed, plus the directions deliberately not advanced |
| [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) | every source with version, **inspection depth** and **claim-support** status, secondary-only mentions, and access limitations |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Limits of this decision

- **No experiments.** No models or encoders, no checkpoint-tensor loading, no feature
  extraction, no edit/replay execution, no generation, no classifiers/detectors, no image
  viewing or reannotation, no GPU allocation or polling, no installs or downloads, no
  method implementation or execution adapters, no new datasets or evaluation packages, no
  reviewer/author contact, and no imported research code executed. Paper and source reading
  only, plus ordinary CPU bookkeeping and hashing.
- **NONE does not mean no gap exists in this field.** It means no candidate I could justify
  from the inspected evidence cleared the bar. Inspection was bounded: EMMA,
  M-ErasureBench and 2511.07970v2 were read at abstract/listing level only; no code was
  audited except the already-pinned DP repository; MACE and EraseAnything are known only as
  baselines named inside other papers. **No exhaustive novelty search is claimed.**
- **Unknown coverage is not failure**, and a missing evaluation, an untested combination or
  a stated limitation does not by itself establish a gap.
- Mosaic and DVE both target **FLUX.1-dev**; our own closed records concern
  **FLUX.1-schnell**. Different checkpoints, not treated as interchangeable.

## Relationship to closed work

Closed findings supplied **context only, never fresh confirmation data**, and nothing was
re-scored, regenerated or modified:

- The changing-replacement candidate remains **NO_GO — CLOSED_CORRECTED** (`5786864`) and
  is **not** revived here under renamed policy, guide-refresh or dependency-graph
  terminology.
- The FLUX chain-saw→parachute diagnostic remains closed at `0e88d26` (DP passed its stated
  numeric gates; UCE failed cumulative retention; neither showed detector-discovered
  relapse). No re-scoring, expanded prompts, reruns, or spending of the unused image
  allowance.
- SD1.x/human-review closures and pauses, and the previously unselected projection,
  unrestricted-certificate and evaluation-reuse formulations, are preserved as recorded.

**Nothing in this map authorizes implementation or experiments**, no successor assessment
is scheduled, and no follow-up job was created.
