# ANALYTICAL ASSESSMENT — NO EXPERIMENTS

Candidate question: *when later deletion requests make a previously acceptable replacement
concept unacceptable, do existing continual-erasure methods already revise earlier
replacement behaviour adequately, or is a useful capability missing?*

## Decision: **NO_GO — no distinct contribution has been established under the assessed constraints. The candidate is closed as formulated.**

> **Amended 2026-10-17** — see [`CORRECTIONS.md`](CORRECTIONS.md). The decision is
> **retained and narrowed**; twelve supporting overstatements in the original assessment
> (commit `998aab8`) are corrected. **No new empirical findings were produced.**

**Reason:** existing adaptive-target, neutral-anchor, replay and recomputation
alternatives are **important baselines**. The elementary incompatibility observation and
the illustrative refresh repair establish **no nontrivial new contribution**, and this
assessment **has not established a useful residual failure that survives those
alternatives**.

It is **not** claimed that the expanded old-prompt requirement is universally solved, that
every repair is exact or cheap, or that any contrary case must be artificial. Conversely,
uncertainty about full old-prompt coverage **does not** establish novelty, justify GO, or
authorise experiments. Key corrected points: LACU's target-invalidation passage is **§5.3**
and is labelled *related scenario explicitly addressed; expanded old-prompt semantic
cleanup/guarantee unverified by the cited evidence*; a neutral anchor helps **only while it
stays outside the forbidden set** (a task assumption); and the single-edit timings do
**not** bound cumulative-repair cost.

| start here | for |
|---|---|
| [`ASSESSMENT.md`](ASSESSMENT.md) | decision first, then the precise task, the prior-art and obvious-repair comparisons, the separation of elementary/prior-art/inferred/novel claims, the falsification condition, and an illustrative exact-arithmetic appendix |
| [`CORRECTIONS.md`](CORRECTIONS.md) | the twelve corrected overstatements, each with its precise replacement, and what did not change |
| [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) | every source with exact section/equation/code location, version, date, verification status and access limitation |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Limits of this assessment

- **No experiments.** No model, encoder, detector, checkpoint-tensor loading, feature
  extraction, edit/replay, generation, classification, image viewing or annotation; no GPU
  allocation; no installs or downloads; no imported research code executed.
- **Not an exhaustive novelty search.** CEASE had no released implementation **as of the
  in-repo record dated 2026-10-11, not re-verified here**, so its behaviour is cited at
  paper level only — and that availability status is **not** used as evidence of a missing
  capability; `2511.07970v2` was read at **abstract level only** and its treatment of this
  scenario is recorded as *unverified*, not absent.
- The appendix arithmetic is **synthetic and illustrative** — not empirical evidence, and
  its transfer to a real diffusion backbone is labelled inference.
- **NO_GO is a decision not to advance this candidate as formulated**, because no distinct
  contribution was established against strong existing baselines — **not** a demonstration
  that those baselines fully solve the expanded task. It is not an impossibility theorem,
  and it is not a finding that any method fails.

## Relationship to closed work

The `modern_flux_dp_uce` sequential-erasure diagnostic remains **CLOSED_CORRECTED** at
`0e88d26` and is unchanged by this assessment: nothing was re-scored, regenerated or
modified, and its unused 2,000-image allowance was not spent. Its findings (DP passed the
stated numeric gates, UCE failed cumulative retention, both had zero detector-discovered
relapse) do **not** establish this candidate and were not used as if they did — only its
recorded per-edit wall times and its standing measurement cautions were reused. The
SD1.x/human-review work remains closed or paused; previously unselected projection,
unrestricted-certificate and evaluation-reuse proposals remain unselected.

**No follow-up is authorized by this assessment.**
