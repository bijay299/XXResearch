# START HERE — flow_method_design_v1 (CPU-only method-development sprint)

**Computed verdict (frozen runner, unchanged): `NO_SUPPORTED_CANDIDATE`.**

> **Amended 2026-10-20 by [`AMENDMENT_01.md`](AMENDMENT_01.md)** (documentation only; no experiment run). It corrects the
> bilinear explanation (contrary G9 evidence), F1's observable, "tied" wording, the scope of several mathematical claims,
> the coverage/open-question statement, and accounting/record reconciliation. Where this file and the amendment differ,
> the amendment governs.
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
   declined**: under its specific constraints (dA·U = 0, dB·(A·U) = 0) with B = 0 and rank(A·U) = r it provably collapses
   (zero-init LoRA alone is not sufficient), and its working form is the known null-A/InfLoRA construction,
   which became the control `na_lora`.
3. A numpy CPU prototype with 13 arms (+ gated/early-stop variants), 238 tests incl. finite-difference checks, a frozen
   evaluation protocol with pre-declared thresholds, and a static torch/GEM integration patch (`code/`, `native/`).
4. One frozen synthetic validation: 636/636 units, 12 fresh seeds, 3.87 core-h (`results/RESULTS.md`).

## Exact contribution claim and its limits
- **Claimed:** an implemented, tested and adversarially reviewed negative result. In covered linear-reference cells, the
  Adam-metric correction gave 1.01–1.35× the **cumulative held-out hinge progress** (10-step or earlier-cutoff procedure;
  not the first-order −g·Δ) of the Euclidean post-projection (seed medians 1.14×, 1.07×); 9/12 (d = 2) and 11/12 (d = 3)
  seeds fell below the pre-declared 1.2× (some individual seeds exceeded it) ⇒ C-H1(i) falsified there (F1). At matched
  erasure, VAU **did not establish the preregistered qualifying advantage** in any testable family (Q1 GMR 1.011 / 0.946 /
  0.984 with wide CIs, which do not establish equivalence). Safety qualification failed in six families; several of those
  ratios are dominated by gated VAU **failing to reach matched erasure** (scored by convention), not by measured damage.
- **Bilinear explanation withdrawn:** the frozen G9 diagnostic found VAU's bilinear share γ ≈ 0.00416 % (layer-level
  squared-norm ratio) and drift 1.69× Euclidean — contrary to the predicted ≥ 10× with γ > 0.99 (AMENDMENT_01 §1).
- **Coverage/testability:** all families ran, including the tiny flow model G8. G2, G5 and G8 had no COVERED cell ⇒
  NOT_TESTABLE, so READY was unavailable regardless of other outcomes and the narrowed hypothesis was **not tested within the
  qualifying COVERED regime** in those families. No FLUX experiment ran. F1 and the safety failures each support stopping
  independently of the disclosed F4/F5 ambiguities. Results there
  are not a falsification of the covered-regime claim; they support only "this sprint did not establish a useful candidate".
- **Originality (`NOVELTY_ASSESSMENT.md`):** a potentially distinct composition (two verified differences from Adam-NSCL);
  versus the strongest straightforward combination (`post_euclid_reject+gate`) it differs only in a textbook metric
  component; no arrangement-specific useful behaviour shown.
- **Mathematical correctness:** the conditional KKT/projection result, ridge-residual identity and clipped-mean acceptance
  certificate are correct under their stated assumptions; the equivalence and optimality statements are narrowed in
  AMENDMENT_01 §3 (idealized μ = 0 projectors only; collinearity ≠ trajectory equivalence; metric match is sufficient, not
  necessary); bounds are on the probe batch; HMT constant UNVERIFIED.
- **Cost (`COST_MODEL.md`):** 20–64× native AdamW forward-equivalents in the prototype, dominated by gate tries.
- **Not established by anything here:** object removal, image quality, FLUX behaviour, or superiority over any published method.

## Single next scientific decision
Stop: do **not** launch the real-model protocol for VAU (`docs/REAL_MODEL_PROTOCOL.md`, retained as written). The earlier
statement that coverage reachability is "the only open question" is withdrawn (AMENDMENT_01 §4); no follow-up study is
proposed here.

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
| corrections (governing) | `AMENDMENT_01.md` |
| reviews | `review/F_REVIEW.md`, `review/F_FULL_ALGORITHM_NOVELTY.md`, `NOVELTY_ASSESSMENT.md` |
| integration and cost | `native/`, `COST_MODEL.md` |
| future protocol | `docs/REAL_MODEL_PROTOCOL.md` |
| agents, resources, environment, evidence | `docs/AGENT_LEDGER.md`, `docs/ENVIRONMENT.md`, `docs/EVIDENCE_INDEX.md`, `results/ledger.jsonl` |
