# EVIDENCE INDEX — research/flow_method_design_v1

Tags: VERIFIED (read at a pinned location) · DERIVED (proof) · COMPUTED (reproducible script/run) · UNVERIFIED.
Frozen = hashed in `configs/FREEZE.json` (freeze commit `aec0f3837aba5842c2e30eb4f758bfa9750fb99d`).

| claim / artifact | evidence | tag |
|---|---|---|
| GEM code facts (loss, rollout prefix σ≈0.77–1.0, AdamW lr only, no retention, LoRA targets) | `agents/A_gem_code_facts.md` (file:line @ GEM c328fc6) | VERIFIED |
| Competitive matrix (8 priority + 3 threats) | `SOURCE_MATRIX.md`, `agents/A_source_matrix.md`, `configs/source_cache.json` | VERIFIED code / CLAIM-UNVERIFIED papers |
| Adam-NSCL projects the actual Adam update via feature-covariance null-space (Euclidean, normalised) | `SOURCE_MATRIX.md` §0; Adam-NSCL@a2f39b4 `optim/adam_svd.py:78-90,93-114` | VERIFIED |
| Track B: GEM(η=1) identities, offset amplification H-B1 | `agents/B_candidate_spec.md`, `B_scripts/b_checks_output.json` | DERIVED + COMPUTED |
| Track C: pre-Adam projection non-neutral (Prop 1), gate certificate | `agents/C_*`, `DERIVATION.tex` Props 1, Thm 8 | DERIVED + COMPUTED |
| Track D: endpoint bound, IS unbiasedness, prefix-vs-uniform | `agents/D_*`, `D_scripts/PREREG.txt` | DERIVED + COMPUTED (P-D4 inconclusive, P-D5 falsified in validation GD-D) |
| Selection, narrowing, declined pivot | `CANDIDATES.md`; `code/scratch_checks/{sign_check,factor_exact_check,factor_collapse_check}.py` | COMPUTED (development draws) |
| Derivation (Props 1–3, Lemmas 5–6, Thms 7–8, factor-wise exactness/collapse) | `DERIVATION.tex` (+ `.html`, `.md`) | DERIVED; reviewed in `review/F_REVIEW.md` |
| Independent proof/novelty review; full-algorithm assessment | `review/F_REVIEW.md`, `review/F_scripts/`, `review/F_FULL_ALGORITHM_NOVELTY.md`, `NOVELTY_ASSESSMENT.md` | DERIVED + COMPUTED (dev toys) |
| Implementation, 238 tests (FD gradients, AdamW equivalence, projector invariants, rollback) | `code/vau/`, `code/tests/`, `results/dev/integrity.json`, `results/dev/pytest_junit.xml` | COMPUTED |
| Protocol, thresholds, verdict rules | `configs/eval_protocol.json` v1.2, `docs/EVAL_PROTOCOL.md` | Frozen |
| Candidate settings | `configs/candidate.json` | Frozen |
| Development phase (seeds 0..7; not evidence) | `results/dev/` | COMPUTED (development only) |
| Validation raw results (636 units, seeds 7000–7011) | `results/validation/raw/`, `units.jsonl`, `summary.json` | COMPUTED (frozen runner) |
| Computed verdict NO_SUPPORTED_CANDIDATE and its basis | `results/validation/summary.json` → `verdict`; `docs/INTERPRETATION_NOTE.md` | COMPUTED + interpretation |
| Coverage table, F1, family/safety stats | `results/RESULTS.md` | COMPUTED |
| Post-hoc descriptive arm table (not a decision input) | `results/validation/posthoc_arm_table.{py,json}` | COMPUTED post hoc |
| Native integration (STATIC, NOT EXECUTED) | `native/vau_torch.py`, `native/gem_integration.patch`, `native/INTEGRATION_NOTES.md` | static; no torch available |
| Costs | `COST_MODEL.md`, `results/ledger.jsonl` | measured CPU / symbolic / unmeasured GPU kept separate |
| Real-model falsification protocol (proposal; not launched; not recommended under the verdict) | `docs/REAL_MODEL_PROTOCOL.md` | proposal |
| Agents, resources, environment | `docs/AGENT_LEDGER.md`, `docs/ENVIRONMENT.md` | records |
