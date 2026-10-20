# F_FULL_ALGORITHM_NOVELTY — VAU assessed as a complete stateful procedure (Agent F, 2026-10-20)

**Scope.** This applies the coordinator's clarification: a well-motivated combination is an eligible contribution if its particular arrangement gives
useful behaviour that neither existing methods nor straightforward combinations give. It is one bounded assessment from existing material:
`F_REVIEW.md`, the `F_scripts` outputs, `SOURCE_MATRIX.md` §0, `agents/A_source_matrix.md`, the current `CANDIDATES.md` pseudocode, `configs/candidate.json` and
`configs/eval_protocol.json` arms. The only computation was one exact sympy arithmetic check (example in §4); no model was run.

**What changed from my earlier review.** `F_REVIEW` §3 called VAU "relabelled" because its *projection component* is scaled gradient projection
(Euclidean projection in Adam-whitened coordinates). That statement stands. The change here is a change of **scope**, not of evidence: I now judge the
whole stateful procedure against the closest *complete* algorithms. No new evidence was added.

## 1. Component classification (VAU = current `CANDIDATES.md` / `candidate.json`)

| # | Component | Class | Basis |
|---|---|---|---|
| 1 | Native GEM loss, detached teachers | existing primitive | A_gem_code_facts §1 (VERIFIED) |
| 2 | AdamW moments, bias correction, D = 1/(√v̂+ε), decoupled wd | existing primitive | torch semantics (line numbers UNVERIFIED) |
| 3 | Retained-output probes: Gaussian sketch Σ·J_r, k VJPs, refresh every K data steps (counter never rolled back) | existing primitives: OGD-style output-gradient constraints (UNVERIFIED) and the HMT sketch (UNVERIFIED constant) | — |
| 4 | Projection of the **actual post-Adam step**, raw-g moments committed, only the displacement projected | **existing; identical to Adam-NSCL's state policy** | Adam-NSCL@a2f39b4 `adam_svd.py:156` updates `exp_avg` from raw grad and `:78-90` projects the update (VERIFIED) |
| 5 | **D-metric** projector Δ₁ = Δ₀ − D Sᵀ(SDSᵀ+μI)⁻¹SΔ₀ | existing primitive (scaled gradient projection / SQP / CPO with metric D⁻¹). Using it inside an Adam-NSCL-type loop is a **verified difference** from Adam-NSCL | Adam-NSCL code uses a Euclidean, per-layer, feature-covariance projector normalised by 1/‖·‖_F (VERIFIED) |
| 6 | End-to-end retained-Jacobian sketch instead of layer-input feature covariance | **verified difference** from Adam-NSCL; existing primitive in itself | SOURCE_MATRIX §0 |
| 7 | Reject if g·Δ₁ ≥ 0: null step, no state commit, no gate batch consumed | existing primitive (descent / sign test; AEGIS GRP is a sign rule, UNVERIFIED) | — |
| 8 | Backtracking gate s = 1, ½, …; fresh i.i.d. R₂ per try; Hoeffding plus a union bound over a fixed T·R | existing primitives (backtracking line search; held-out confidence-bound safety tests). The per-step sequential use with an adaptive trial count is a **potentially distinct composition** detail, but standard | F_REVIEW Thm 8 check |
| 9 | Full rollback of (θ, m, v, c) on reject; Adam counter advanced only on commit | potentially distinct composition **detail**. Skipping an optimizer step on a failed check is a known pattern (e.g. AMP loss-scale skipping; UNVERIFIED) | candidate.json `commit`, `adam_step_counter` |

## 2. Closest complete algorithm and strongest straightforward combination

- **Closest complete prior algorithm (verified at code):** Adam-NSCL. It is the same stateful skeleton: Adam with raw-g state and a projected actual update.
  VAU differs from it in components 5 and 6 (VERIFIED) and adds components 7–9. CPO/TRPO (constrained step + backtracking) are close in spirit but
  carry no Adam state, so they are **UNVERIFIED** as complete-algorithm matches (sources not re-read). EA++/AEGIS/DoCo use gradient surgery on *gradients* before the
  optimizer (CLAIM/UNVERIFIED, A_source_matrix) and are therefore closer to `pre_proj`.
- **Strongest straightforward combination:** Adam-NSCL-style Euclidean post-projection on the same Jacobian sketch, plus the same reject rule, plus the same gate
  and rollback. This is exactly the frozen arm **`post_euclid_reject+gate`**.
- **Precise relation:** VAU ≡ `post_euclid_reject+gate` **if and only if** Π_D = Π on every step. That holds iff range(Sᵀ) is D-invariant (DERIVED: Π_D = Π
  ⇔ DSᵀ(SDSᵀ)⁻¹S = Sᵀ(SSᵀ)⁻¹S ⇔ range(DSᵀ) = range(Sᵀ)); D = cI is a special case. It is also equivalent **up to step scale** whenever dim null(S) = 1,
  because both steps then lie on the same line and lr tuning removes the difference apart from sign, which the shared reject rule handles.
  Otherwise the two are **different**: identical state policy, gate and probes, but a different feasible direction on each step.
  Versus faithful Adam-NSCL on LoRA (feature space): different (verified structural difference); the empirical comparison is **unresolved**, because no such arm exists.

## 3. Which failure each coupling addresses, and where it backfires

- **Pre-Adam projection is not neutral (Prop 1).** Fixed by *any* post-optimizer projection; `post_euclid` fixes it too. Not specific to VAU.
- **Euclidean post-projection ignores Adam's metric (Prop 2/3).** VAU-specific. It is optimal per **H-norm**. It is worse per Euclidean norm (example in §4).
  The sign case is neutralised by `post_euclid_reject`.
- **Adaptive construction-batch verification and state contamination by rejected trials.** Fixed by the fresh-batch gate with rollback. That gate is **shared by every gated arm**, so it is not VAU-specific.
- **Arrangement-specific interaction.** VAU's metric step can be several times larger in Euclidean norm (F_REVIEW f5: ‖Δ₁‖/‖Δ₀‖ up to 6.9). In that case the gate
  turns a would-be drift excess into backtracking. This metric×gate coupling is the only behaviour that belongs to the arrangement rather than to a single part.
- **Counter-productive case.** Under zero-init LoRA, g_A ≈ 0 makes D_A/D_B up to 3e6. The H-metric makes A "cheap", routes the correction through A, and produces ~50× the
  Euclid per-step bilinear drift (dB·dA). That drift is >99.9 % of the actual change (f5). `vau_blocksplit` tests whether per-block normalisation removes this.

## 4. Distinguishing worked example (exact; sympy)
Take S = [1, 1, 1], D = diag(1/100, 1, 1), m̂ = g = (1, ½, −1), λ_wd = 0. Here null(S) has dimension 2, and range(Sᵀ) is not D-invariant.

| step direction | first-order progress gᵀx | progress / ‖x‖_H | progress / ‖x‖₂ |
|---|---|---|---|
| VAU Π_D D g = (5/402, 299/402, −152/201) | 917/804 | **1.068** | 1.075 |
| Euclid post Π D g = (13/75, 199/300, −251/300) | 161/120 | 0.659 | 1.240 |
| projected SGD Π g | 13/6 | 0.257 | **1.472** |

**Prediction.** At a matched step size, VAU beats `post_euclid_reject` by a factor of **1.62** if the post-projection residual drift scales with the H-norm.
It *loses*, by a factor of **0.87**, if the residual drift scales with the Euclidean norm.
More generally (DERIVED), under a budget ‖Δ‖_N ≤ r the best feasible direction is the N-metric projection of N⁻¹g, so VAU is the optimum iff N ∝ D⁻¹.
Its usefulness is therefore exactly the empirical claim that *retained drift after projection is measured in Adam's metric*. The zero-init LoRA bilinear
term (§3) is a case where that fails.

## 5. Can the frozen arms distinguish the claim?
- **Metric effect within the full procedure:** `vau` vs `post_euclid_reject+gate`, and `vau_nogate` vs `post_euclid_reject`. This is a complete 2×2 of metric × gate
  (reject rule shared): **present**. A/B routing: `vau_blocksplit`. Step-size artefacts: lr tuning, plus `adamw_stepmatch` (matches AdamW's H-norm to VAU's,
  not post_euclid's).
- **Missing (disclosed as limitations, not added):**
  - (a) A **state-commit-policy ablation**: no-rollback on reject, or momentum built from the projected step. The rollback is credited to no arm separately; all gated arms share it.
  - (b) A **faithful Adam-NSCL-on-LoRA** arm: feature covariance, right-multiplication, ‖·‖_F normalisation. `post_euclid` is only a Jacobian-sketch analogue, and `na_lora` is InfLoRA-style.
  - (c) A **construction-batch gate** (R₁ reused for verification), to show empirically that fresh batches are necessary. Only C's counterexample covers this.
  - (d) Projected SGD (Πg), the per-Euclidean-norm optimum in §4.
  - (e) A post_euclid arm matched to VAU's H-norm step for step.

## 6. Four separate conclusions
1. **Mathematical correctness.** The derivations are correct within the stated scopes, with the gaps listed in F_REVIEW §1 (Thm 7/Lemma 6 cover R₁ only; the gate certifies a clipped mean
   under fresh i.i.d. draws; the HMT constant is UNVERIFIED). The equivalence condition in §2 and the example in §4 are exact.
2. **Originality.** VAU is a **potentially distinct composition**. As a complete procedure it differs from the closest verified algorithm (Adam-NSCL) in two verified ways (metric and
   end-to-end sketch) and adds a standard reject rule, gate and rollback. Against the strongest straightforward combination (`post_euclid_reject+gate`) it differs in **one** component, the
   metric. That component is a textbook primitive and is inert when range(Sᵀ) is D-invariant or dim null(S) = 1. No arrangement-specific useful behaviour is
   demonstrated yet. The best candidate for one is the metric×gate coupling.
3. **Empirical utility.** **Unknown until validation.** My development toys are mixed: VAU beat Euclid by 1.38×/1.12×/1.03× only in the probe-covered regime, and lost to
   the penalty or to lr-matched AdamW elsewhere (F_REVIEW f4). The frozen results can establish whether the metric beats the Euclidean projection inside the identical
   stateful procedure on toy families a/b/c. They cannot establish real-model utility, the value of the rollback policy (no ablation), superiority over faithful Adam-NSCL, or novelty.
4. **Computational cost.** Relative to `post_euclid_reject+gate`, VAU adds a per-step recomputation of SDSᵀ, because D changes every step while SSᵀ can be cached between refreshes.
   That is k²P flops plus a k³ solve per step, small next to a transformer forward. Probes (k VJPs every K steps) and the gate (≤ R·n₂ retained forwards per step) are shared with the
   gated controls. Adam-NSCL's per-layer feature covariance and SVD avoid VJPs and are cheaper.
