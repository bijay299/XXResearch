# Full-algorithm novelty assessment — VAU (coordinator synthesis; one bounded assessment)

> **Amended by [`AMENDMENT_01.md`](AMENDMENT_01.md) §1–§3** (documentation only): equivalence and optimality statements
> narrowed, "tied"/first-order wording corrected, bilinear explanation withdrawn. Earlier wording is preserved in git history
> (commit `7fde561`).

Basis: Agent F's bounded assessment `review/F_FULL_ALGORITHM_NOVELTY.md` (no models run; one sympy check), F's proof review
`review/F_REVIEW.md`, code-level evidence in `SOURCE_MATRIX.md` §0 (Adam-NSCL@a2f39b4 re-checked by the coordinator), and
the unchanged frozen results (`results/RESULTS.md`). Applied rule (user clarification, 2026-10-20): a well-motivated
combination is an eligible contribution if its particular arrangement gives useful behaviour that existing methods or
straightforward combinations do not; known ingredients alone are no reason to reject, and a new name or extra gate alone is
no reason to accept.

**Correction of scope, not of evidence.** `configs/candidate.json` (frozen) records `"novelty": "NONE"` and `CANDIDATES.md`
says "No component is new". Those statements concerned components and are preserved. This assessment judges the complete
stateful procedure; its conclusion ("potentially distinct composition, utility not shown") replaces "NONE" only as a
description of the composition, and adds no evidence.

## 1. Exact contribution claim under assessment
VAU = AdamW on LoRA with (i) the actual tentative AdamW step Δ₀ (moments, bias correction, decoupled wd) corrected by the
Adam-metric projector Δ₁ = Δ₀ − D Sᵀ(SDSᵀ+μI)⁻¹SΔ₀ against a Gaussian sketch S = ΣJ_{R₁} of the retained-output Jacobian
refreshed every K steps; (ii) a null step if g·Δ₁ ≥ 0; (iii) a backtracking gate on fresh i.i.d. retained batches with a
Hoeffding/union-bound acceptance rule; (iv) commit of (m′, v′, c+1) from the raw gradient only on acceptance, full rollback
of (θ, m, v, c) otherwise.

## 2. Known components versus potentially original coupling
| component | classification |
|---|---|
| GEM loss; AdamW semantics; Gaussian sketch (HMT); OGD-style output-gradient constraints | existing primitives |
| projecting the **actual post-Adam update** with raw-gradient moments | existing — same state policy as Adam-NSCL (verified at code) |
| D-metric projector | existing primitive (scaled gradient projection / SQP with metric D⁻¹); **verified difference** from Adam-NSCL's Euclidean, feature-covariance projector |
| end-to-end retained-Jacobian sketch instead of layer-input feature covariance | **verified difference** from Adam-NSCL; existing primitive in itself |
| reject rule, backtracking gate, fresh-batch confidence bound | existing primitives, shared by every gated control arm |
| rollback of (θ, m, v, c), counter advanced only on commit | composition detail; known pattern (skipped optimizer steps) |
| metric × gate interaction (metric steps can be larger in Euclidean norm; the gate converts excess drift into backtracking) | the only **potentially distinct coupling**; no useful behaviour unique to it demonstrated |

## 3. Closest prior art, equivalences, distinguishing behaviour, unresolved questions
- **Closest complete prior algorithm (verified at code):** Adam-NSCL — same skeleton (Adam, raw-g state, projected actual
  update); differs in the metric and in using an end-to-end Jacobian sketch rather than feature covariance; adds reject/gate/rollback.
- **Strongest straightforward combination:** `post_euclid_reject+gate` (Euclidean post-projection of the same Adam step, same
  sketch, reject rule, gate, rollback) — present as a frozen arm. VAU differs from it **only in the metric**.
- **Equivalence (narrowed):** for the **idealized, unregularized projectors** (μ = 0, S of full row rank), Π_D = Π iff
  range(Sᵀ) is D-invariant (D = cI special case). This is not automatically an "if and only if" statement about the
  positive-ridge, stateful, gated implementation. When dim null(S) = 1 the projected proposals are **collinear** at that step;
  this does not establish equivalence of complete Adam trajectories under learning-rate tuning. Whitening (y = D^{-1/2}θ) shows the projection component is known; it does
  not by itself make the whole stateful algorithm equivalent.
- **Distinguishing example (exact, F §4):** S = [1,1,1], D = diag(1/100,1,1), g = (1,½,−1): per Adam-metric step norm VAU has
  1.62× the Euclidean variant's first-order progress; per Euclidean norm 0.87×. If the step budget is measured in Adam's metric D⁻¹, VAU's
  direction is optimal (conditional sufficient result); global proportionality of a damage metric to D⁻¹ is not necessary —
  agreement on the feasible subspace, or for the particular gradient, can suffice.
- **Unresolved (evidence or access missing):** Adam-NSCL paper text, InfLoRA, CPO/TRPO, EA++/AEGIS/DoCo algorithms were not
  re-read (arXiv unreachable); no faithful Adam-NSCL-on-LoRA arm, no state-commit-policy ablation, no construction-batch-gate
  arm, no projected-SGD arm, no H-norm-matched Euclidean arm exist in the frozen study (disclosed limitations, not added).

## 4. What the unchanged frozen results do and do not establish
- **Do establish (synthetic, covered linear-reference cells):** the metric component's distinguishing prediction was tested
  directly. F1 measured VAU-noGate / post_euclid_reject **cumulative held-out hinge progress** (10-step or earlier-cutoff; not the
  first-order −g·Δ) of 1.01–1.35× (seed medians 1.14× d=2, 1.07× d=3); 9/12 and 11/12 seeds were below the pre-declared
  1.2× (a few individual seeds above), so C-H1(i) is falsified there. Inside the identical stateful procedure VAU did not
  establish the preregistered qualifying advantage at matched erasure (Q1 GMR 1.011 / 0.946 / 0.984 in G1 / G6 / G7; the CIs
  do not establish equivalence).
- **Do not establish:** the narrowed claim within the COVERED regime for the flow-model family (G8) or G5/G2 — all ran, but
  none had a COVERED cell (NOT_TESTABLE); any validated explanation of VAU's failure (the bilinear explanation was contradicted
  by the G9 diagnostic, AMENDMENT_01 §1); the
  value of rollback (no ablation); superiority or inferiority versus faithful Adam-NSCL; real-model utility; novelty.

## 5. Four separate conclusions
1. **Mathematical correctness:** the conditional KKT/projection characterisation, the ridge-residual identity and the
   clipped-mean acceptance certificate are correct under their stated assumptions; the equivalence and optimality statements
   are narrowed as above (F_REVIEW §1 gaps stand: R₁-only bounds, UNVERIFIED HMT constant).
2. **Originality:** potentially distinct composition relative to Adam-NSCL (two verified differences); relative to the
   strongest straightforward combination it differs in one textbook component. No arrangement-specific useful behaviour shown.
3. **Empirical utility:** not supported on the frozen synthetic evidence where testable; untested within the COVERED regime
   on the flow model (which ran); no FLUX experiment ran.
4. **Computational cost:** vs `post_euclid_reject+gate`, a per-step k²P Gram + k³ solve (small); vs native GEM, 20–64×
   forward-equivalents in this prototype, dominated by gate tries shared with all gated arms (`COST_MODEL.md`).

## 6. Decision under the existing rules
The computed verdict is `NO_SUPPORTED_CANDIDATE` (`docs/INTERPRETATION_NOTE.md`). Under the existing decision rules the
evidence supports **stopping** this candidate: do not launch the real-model protocol for VAU. F1 and the safety failures each
support stopping independently of the disclosed F4/F5 ambiguities. The narrowed claim was not tested within the COVERED
regime on the flow model (which ran); no follow-up study is proposed.
