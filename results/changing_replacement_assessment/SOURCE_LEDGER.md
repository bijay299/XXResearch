# Source ledger — "replacement concept later becomes forbidden" assessment

Every row records what was inspected, where, when, and **how far it was verified**.
Verification status is deliberately conservative: *verified* means I read the stated
location; *inferred* means I reasoned from it; *unverified* means I could not confirm it
from a primary source in this pass. **No code was executed, no model/encoder/detector was
run, and no image was viewed.** Fetched papers were read as rendered HTML/abstract pages.
Assessment date: **2026-10-17**. This is **not** an exhaustive novelty search.

## Primary sources fetched this pass

| # | Source | Version / date inspected | What it was used for | Status |
|---|---|---|---|---|
| S1 | **CEASE**, `arxiv.org/html/2610.01989v1` | v1 HTML, fetched 2026-10-17 | §3.1 sequential formulation ("each step solves for a new update against the running weights… retain set held fixed"); §3.3 AIC `K₂^AIC=[K₂, c∗]`, `Δᵢ K₂^AIC = 0`, and *"the closed-form correction enforces Δᵢc∗=0 up to numerical precision"*; §3.3 HOC orthogonal projection onto complement of historical update subspace (Eq. 12–13); §4.1 *"The per-task anchors are 'person' for celebrities, 'art' for styles, and 'object' for instances."*; App. A.6 *"CEASE completes a request in about 13 seconds"* | **Verified** (paper text). Behaviour is **paper-level only** — see L1 |
| S2 | **LACU v2**, `arxiv.org/html/2512.02657v2` | v2 HTML, fetched 2026-10-17 | §3 / Eq. 3 per-prompt mapping selection from `M=10` LLM+CLIP candidates; *"Selection is recomputed at every continual step using the frozen teacher ϵθ^{i−1}, because the model's denoising landscape changes after each update."*; **the decisive quote**: *"if a concept previously used as a mapping target later becomes a forget concept, a fixed mapping can degenerate into an ineffective self-mapping. LACU instead re-scores candidate trajectories with the current frozen teacher at each step."*; locality-aware replay of top-`N_r=10` nearest **retain** concepts; soft `ℒ_total = λ_unlearn ℒ_unlearn + λ_retain ℒ_retain + λ_reg ℒ_reg`; ~60 min/update on 3×H100 | **Verified** (paper text). Code **not** run or inspected |
| S3 | **AGE**, `arxiv.org/html/2501.18950v3` | v3 HTML, fetched 2026-10-17 | Abstract/§1 adaptive target selection vs. *"fixed generic concept… or just an empty text prompt"*; Eq. 3 minimax `min_θ′ E_{ce} max_{ct}`; Eq. 4 Gumbel-Softmax relaxation; §4 `L₂` preserves the chosen target; §5 training 1000 steps, batch 1; App. B `k=100` nearest concepts | **Verified** (paper text) for selection/cost. **Negative** statements (no sequential handling; scenario not discussed) are *verified-absent at the inspected sections*, recorded as **not applicable to the continual setting**, not as a defect |
| S4 | **Continual-unlearning regularization study**, `arxiv.org/abs/2511.07970v2` | v2 **abstract page only**, fetched 2026-10-17 | Sequential-request framing; *"popular unlearning methods suffer from rapid utility collapse"*; cumulative parameter drift; add-on regularizers + gradient-projection method | **Partially verified — abstract only.** Full text not inspected; its treatment of replacement invalidation is **unverified**, not asserted absent |
| S5 | **DP**, pinned source `czhang024/CVPR_DouleProjection@3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (already cloned on `/data`) | commit verified at inspection | `Flux/Double_proxy.py:174-189` — guide vectors rebuilt from the **entire** `guide_concepts` list for every erase token, then projected onto their joint span (pooled, not per-target); `:295` — `edit_concepts` split on `;`, so cumulative multi-concept erase sets are a native call | **Verified by source inspection.** Not executed in this pass |
| S6 | DP paper, `arxiv.org/abs/2604.10032` | cited by ID | Identified only; not re-fetched this pass | **Not inspected this pass** |

## Existing in-repo records reused (no re-derivation)

| # | Record | Used for | Status |
|---|---|---|---|
| L1 | `docs/research/FEASIBILITY_BRIEF_2026-10-11.md` method table + `docs/research/feasibility_sources_2026-10-11.json` | CEASE has **no released implementation** (*"Code coming soon"*, `contains_method_code: false`, fetched 2026-10-11); CEASE anchor = one fixed shared anchor per task; LACU = per-prompt score-selected mapping from 10 LLM+CLIP candidates; SPEED `--anchor_concepts` required; ConAbl anchor always supplied, no native selection | **Verified** (existing repo record) |
| L2 | Same brief, §"Current vs accumulated anchor protection" | This group **already** classified `AIC-accum` (protect every anchor used so far) as **our configuration variant** of an existing mechanism — not a new method | **Verified**; directly supports the "routine bookkeeping" finding |
| L3 | `results/modern_flux_dp_uce/sequential_diagnostic_execution/` (CLOSED_CORRECTED at `0e88d26`) | Recorded per-edit wall times **1.07 s (DP)**, **2.47 s (UCE)** from `logs/edit_*.log`; and the standing cautions that a detector-defined `C=0, U=0` is **not** a semantic-absence certificate and that input-normalised residuals are not output-presence measures | **Verified** (existing records read, not regenerated). That diagnostic remains **closed and unmodified** |

## Access limitations and explicit non-claims

- **CEASE**: no public implementation as last recorded (L1). Its described behaviour could
  not be code-verified; HOC's trigger/rank-cut remain unspecified per the same record.
- **S4**: abstract only. Absence of a replacement-invalidation discussion there is
  **unverified**, not an established omission.
- **AGE**: assumes a fixed erasure set upfront, so the continual scenario is **outside its
  stated setting**. Its silence is recorded as inapplicability, not as a missing capability.
- No claim of exhaustive coverage of the continual-erasure literature is made.
- The Appendix arithmetic in `ASSESSMENT.md` is **illustrative and synthetic**. It is not
  empirical evidence about any real model, and transfer of its conclusion to a diffusion
  backbone is explicitly labelled **inference**.
- Nothing in the closed `modern_flux_dp_uce` diagnostic was re-scored, regenerated or
  modified; no unused image allowance was spent.
