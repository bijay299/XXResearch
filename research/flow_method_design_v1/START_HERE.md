# START HERE — flow_method_design_v1 (CPU-only method-development sprint)

**Computed verdict (frozen runner, unchanged): `NO_SUPPORTED_CANDIDATE`.**
Reasons recorded by the runner: F1 triggered; safety failures in G1, G2, G3, G4, G6, G9. Basis, quoted rules and the
disclosed specification ambiguities: [`docs/INTERPRETATION_NOTE.md`](docs/INTERPRETATION_NOTE.md).

Branch `claude/epic-fermat-xrwssq` of `bijay299/XXResearch`, based on reference snapshot `f269656` of
`research/audit-and-matched-effectiveness-protocol` (not modified). Freeze commit
`aec0f3837aba5842c2e30eb4f758bfa9750fb99d` (protocol v1.2, candidate, code and runner hashes in `configs/FREEZE.json`).
Validation results complete in commit `7277a5b`.

## What was accomplished
1. Three mechanism tracks (loss geometry B, optimizer/shared parameters C, trajectories D) each produced one specification;
   all three reported no method-level novelty (`CANDIDATES.md`, `agents/`). C's was selected: **VAU** — correct the actual
   AdamW LoRA step in Adam's metric against a sketched retained-Jacobian probe, reject non-progress steps, and accept only
   after a fresh-batch Hoeffding check, with full optimizer-state rollback (`ALGORITHM.md`, `configs/candidate.json`).
2. Mathematics with proofs and counterexamples (`DERIVATION.tex` / `.html` / `.md`); independent review by Agent F
   narrowed several claims (`review/F_REVIEW.md`). One early pivot (factor-wise exact LoRA projection) was **considered and
   declined**: it provably collapses under zero-init LoRA, and its working form is the known null-A/InfLoRA construction,
   which became the control `na_lora`.
3. A numpy CPU prototype with 13 arms (+ gated/early-stop variants), 238 tests incl. finite-difference checks, a frozen
   evaluation protocol with pre-declared thresholds, and a static torch/GEM integration patch (`code/`, `native/`).
4. One frozen synthetic validation: 636/636 units, 12 fresh seeds, 3.87 core-h (`results/RESULTS.md`).

## Exact contribution claim and its limits
- **Claimed:** an implemented, tested and adversarially reviewed negative result. In covered linear-reference cells, the
  Adam-metric correction gave 1.01–1.35× the first-order erasure progress of the Euclidean post-projection (seed medians
  1.14×, 1.07×), below the pre-declared 1.2× ⇒ C-H1(i) falsified there (F1). At matched erasure, the full VAU procedure tied
  the best controls in every testable family (Q1 GMR 1.011 / 0.946 / 0.984), and gated VAU often failed to reach the target
  erasure (safety failures).
- **Coverage/testability:** G2, G5 and **G8 (the tiny flow model)** had no COVERED cell ⇒ NOT_TESTABLE. READY was therefore
  unavailable regardless of other outcomes, and the narrowed hypothesis was **not tested on the flow model**. Results there
  are not a falsification of the covered-regime claim; they support only "this sprint did not establish a useful candidate".
- **Originality (`NOVELTY_ASSESSMENT.md`):** a potentially distinct composition (two verified differences from Adam-NSCL);
  versus the strongest straightforward combination (`post_euclid_reject+gate`) it differs only in a textbook metric
  component; no arrangement-specific useful behaviour shown.
- **Mathematical correctness:** sound within stated scopes (bounds are on the probe batch; the gate certifies a clipped
  mean; HMT constant UNVERIFIED).
- **Cost (`COST_MODEL.md`):** 20–64× native AdamW forward-equivalents in the prototype, dominated by gate tries.
- **Not established by anything here:** object removal, image quality, FLUX behaviour, or superiority over any published method.

## Single next scientific decision
Do **not** launch the real-model protocol for VAU (`docs/REAL_MODEL_PROTOCOL.md`, retained as written). The only open
question this sprint leaves is whether the covered regime is reachable at all for flow/LoRA erasure. Answering it would
need a separately budgeted, pre-registered study (e.g. measuring retained-response probe coverage on a real model before
any training) — not proposed or run here.

## Map
| need | file |
|---|---|
| sources and pins | `SOURCE_MATRIX.md`, `configs/source_cache.json`, `agents/A_gem_code_facts.md` |
| candidates and selection | `CANDIDATES.md`, `agents/{B,C,D}_candidate_spec.md` |
| math | `DERIVATION.tex` (`DERIVATION.html`, `DERIVATION.md`) |
| algorithm and reproduction | `ALGORITHM.md` (one command per phase) |
| frozen configuration | `configs/eval_protocol.json`, `configs/candidate.json`, `configs/FREEZE.json`, `docs/EVAL_PROTOCOL.md` |
| raw and summarised results | `results/validation/`, `results/RESULTS.md`, `results/dev/` (development only) |
| verdict interpretation | `docs/INTERPRETATION_NOTE.md` |
| reviews | `review/F_REVIEW.md`, `review/F_FULL_ALGORITHM_NOVELTY.md`, `NOVELTY_ASSESSMENT.md` |
| integration and cost | `native/`, `COST_MODEL.md` |
| future protocol | `docs/REAL_MODEL_PROTOCOL.md` |
| agents, resources, environment, evidence | `docs/AGENT_LEDGER.md`, `docs/ENVIRONMENT.md`, `docs/EVIDENCE_INDEX.md`, `results/ledger.jsonl` |
