# RECE golf-ball → English-springer: direct-vs-sequential editing + stage-ablation diagnostic

Two bounded, completed experiments on the author-released RECE `GolfBall_ep0.pt`
checkpoint (`CompVis/stable-diffusion-v1-4`, closed-form concept erasure,
Gong et al., https://github.com/CharlesGong12/RECE, pinned commit
`403a67fd86dce6c124b88798340e001e625b28c9`).

| start here | for |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | does deleting a second concept (English springer) from an already-edited checkpoint behave differently than deleting it from pristine SD1.4? Both pass their numeric suppression/retention gates; the visual review finds two sequential-specific quality costs the numbers don't show. |
| [`refinement_ablation/FINAL_REPORT.md`](refinement_ablation/FINAL_REPORT.md) | **follow-up diagnostic**: is that corruption already present after RECE's preliminary-UCE stage, or does it arise when the adversarial-refinement stage is added? Answer: it arises during refinement, isolated to the sequential branch. |
| [`refinement_ablation/CLOSEOUT.md`](refinement_ablation/CLOSEOUT.md) | the existing-evidence reconciliation that preceded the diagnostic, including the full-resolution re-verification that produced corrections C1/C2 below |
| [`reference_A_intervention/FINAL_REPORT.md`](reference_A_intervention/FINAL_REPORT.md) | **causal diagnostic**: does using parent A instead of pristine M0 as the adversarial-embedding reference prevent the refinement corruption found above? Answer: partial, not an unqualified rescue — golf-ball degeneration drops 87.5%→82.5% (net +2 clean cases, after `CORRECTIONS.md` C7's re-review; originally reported as 77.5%/+4) at the cost of English-springer suppression weakening 100%→95% (0/20→1/20); retained classes unaffected. |
| [`embedding_cross_swap/FINAL_REPORT.md`](embedding_cross_swap/FINAL_REPORT.md) | **causal diagnostic**: does corruption follow the recovered adversarial embedding, the working checkpoint, or their interaction (reference held fixed at M0)? A 2x2 design crossing the direct/sequential working checkpoints with their own recovered embeddings. Answer: mixed — on the erasure target (English springer), corruption follows the embedding cleanly in both directions; on the historical golf-ball class, it is a genuine interaction, severe only when the sequential checkpoint and sequential embedding are combined. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | **read this alongside all four reports.** Two measured values in `FINAL_REPORT.md` were corrected upward after full-resolution re-verification (both make the finding stronger, not weaker); one summary sentence in the ablation report needs a checkpoint qualifier; C4/C5 correct prior verification-check scope and the preliminary-UCE slicing description; C6–C12 (added while publishing the embedding cross-swap diagnostic) fix a listing bug and re-review five case labels in `reference_A_intervention`, correct a prompt-family claim, decompose an embedding-norm figure, correct an own-script assertion-enforcement claim, correct the token-suffix-length claim again, and retract an unestablished "ordinary noise" characterization |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | exact path, byte count and sha256 of every file published here — verify any row without trusting the table |

## What is here

- **`evidence/`** — recognition-count tables (`ANALYSIS_RESULTS.json`),
  per-case visual-review labels and paired transitions
  (`VISUAL_REVIEW.json`), frozen run manifests, per-edit checkpoint hashes
  and runtime verification logs (`edit_logs/`), the upstream frozen
  100-case manifest and the completed screen's own keyed golf-ball labels
  that this work's corrections were checked against
  (`upstream/`), and the exact code used to reconstruct the two
  preliminary-UCE-only diagnostic endpoints plus the vendored native
  RECE functions it calls (`code/`).
- **`images/grids/`** — fixed-order, case-id-labeled contact-sheet review
  grids for golf ball and English springer (the two classes with findings)
  at all four newly-generated endpoints (B, A->B, B-UCE, AB-UCE).
- **`images/corrected_cases/`** — full-resolution crops for the specific
  cases named in `CORRECTIONS.md` (golf-ball cases 22, 34, 39; English-
  springer cases 84, 88, 92, 94), at both the Stage-1-only and final
  checkpoints, so the before/after contrast behind the stage-localization
  finding is directly inspectable; plus `AB_REF_A_case*.png` for the
  reference-A intervention's own recovered/newly-damaged/persisted golf-
  ball cases and its one English-springer exact-hit case.
- **`reference_A_intervention/`** and **`evidence/reference_A_intervention/`**
  — the reference-only diagnostic (parent A vs. pristine M0 as the
  `close_form_emb_regzero` adversarial-embedding reference), including the
  frozen pre-execution manifest, the image-free replay that validated the
  reconstructed AB-UCE intermediate against the existing final checkpoint
  exactly, full-tensor verification logs, diagnostic traces (embedding
  norms, the G-matrix identity check), and the new endpoint's recognition
  table, visual-review labels, and per-case classifier predictions.
  `CORRECTED_LABELS.json` here holds C6–C12's re-derivations.
- **`embedding_cross_swap/`** and **`evidence/embedding_cross_swap/`** —
  the 2x2 embedding/working-checkpoint cross-swap diagnostic: frozen
  pre-execution manifest, full-tensor verification logs (including the
  image-free direct-branch compatibility gate, exact match against the
  existing `B_english_springer.pt`), per-cell diagnostics (embedding
  norms/cosine, edit displacement, recovery-fit and erase-fit residuals,
  retain-projection drift), the two new endpoints' recognition tables,
  visual-review labels, and per-case classifier predictions.
  `EMBEDDING_PROVENANCE.json` records the excluded recovered-embedding
  tensors' shape/dtype/sha256 (the tensors themselves are not published,
  per this repo's no-checkpoint-binaries convention).

## What is NOT here

Model checkpoints (the 3.44 GB `GolfBall_ep0.pt`, the pristine SD1.4
snapshot, and the two newly-edited `.pt` files produced by this work —
identified by sha256 in the edit logs and upstream manifest, not vendored),
the pinned RECE GitHub source tree (public at the commit above; the two
functions actually used are vendored verbatim instead of cloning the whole
repo), the full ~400-image raw generation output for both experiments, and
any host venv/cache. See `PUBLISHED_EVIDENCE_INDEX.md` for the complete,
explicit exclusion list.

**All visual-quality statements in both reports are AI (Claude) visual
screening, not independent human review** — stated explicitly in each
report and in `VISUAL_REVIEW.json`'s own `method` field.
