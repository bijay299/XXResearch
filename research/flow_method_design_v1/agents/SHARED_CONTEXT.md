# Shared context for all specialist agents (read this once; do not reload full histories)

Sprint: XXResearch `flow_method_design_v1`. Goal: develop ONE implementable concept-erasure candidate for
flow-matching text-to-image models (FLUX-class, LoRA, GEM-style training loss), with math, a CPU prototype,
discriminating tests, prior-art comparison, cost model, and a future real-model falsification protocol.
Real-model superiority is NOT claimed or tested. Do not manufacture novelty. A negative result is acceptable.

## Hard constraints
- CPU only. No GPUs, no Hyperplane, no FLUX/base-weight downloads, no private checkpoints/images/keys, no
  real-model training/generation/detector inference, no paid APIs, no human questionnaires.
- Budget: 48 CPU-core-hours total for the whole sprint (all agents), 4 cores, 16 GiB RSS, 5 GiB new files.
  Run any non-trivial computation through `python3 research/flow_method_design_v1/code/meter.py <label> -- <cmd>`
  (appends to `results/ledger.jsonl`). Keep your own usage small (a few core-minutes unless told otherwise).
- Work ONLY under `/home/user/XXResearch/research/flow_method_design_v1/` (the "sprint dir"). Do NOT edit anything
  outside it. Do NOT run git commit/push/checkout/branch (the coordinator does all git). Never touch other studies.
- Python: `/tmp/claude-0/venv/bin/python` (numpy, scipy, pytest, sympy; torch CPU may appear shortly — check
  `import torch`). Run Python reading untrusted downloaded files with `-I`.
- Each agent writes ONLY to its own output path given in its task (e.g. `agents/B_*`), and returns a SHORT
  (<= 400 words) summary to the coordinator. Put detail in files, not in the reply.

## Pinned local material (read-only)
- Reference snapshot `f269656fe4e20fa7d18d05c7465833790b016c62` of branch
  `research/audit-and-matched-effectiveness-protocol` is checked out read-only at
  `/tmp/claude-0/-home-user-XXResearch/c4deba81-1b02-5cf2-bed1-7e39bf025f9e/scratchpad/ref` ("REF"). The same files
  also live in the sprint branch's working tree (repo root `/home/user/XXResearch/results/...`).
- Native GEM official code is cloned at `.../scratchpad/ext/GEM` (github.com/multimodal-ai-lab/GEM). The
  integration reference commit is `c328fc6bec687efe60223498d1b6bed2e5fbcffd`; verify with
  `git -C <clone> cat-file -t c328fc6...` and read via `git show c328fc6:gem/operators/erasure/erasure_gem.py`.
- Existing verified records to REUSE (do not repeat the work):
  - `REF/results/cpr_gem_cpu_prototype/code/cpr_gem/losses.py` (native GEM loss transcription, CPR, centered
    smooth radial control, anchor regression, axial margin, margin-truncated anchor) and
    `.../FINAL_REPORT.md` + `AMENDMENT_01_STATUS_AND_LIMITATIONS.md` (closed algebra: matched first-order
    progress, A/B/C/gamma_cross quantities, the pairwise_distance all-ones eps shift artifact; CPR is
    control-equivalent to a centered smooth radial loss at v=b).
  - `REF/results/minimalist_erasure_design_sprint/{DERIVATION,ALGORITHM,COST_MODEL,CORRECTIONS,SOURCE_LEDGER}.md`
    (Minimalist Concept Erasure, paper arXiv 2507.13386v1, code YaNgZhAnG-V5/minimalist_concept_erasure @
    d178324ec4a48d0536a595b25fdad587b441add8).
  - `REF/results/modern_erasure_gap_map/{GAP_MAP,CANDIDATES,SOURCE_LEDGER,CORRECTIONS}.md` (DVE arXiv 2602.01089v1,
    Mosaic/CoME-Bench 2605.25574v1, EraseAnything 2412.20413v1, EMMA 2512.17320v2, M-ErasureBench 2512.22877v1).
  - `REF/results/changing_replacement_assessment/SOURCE_LEDGER.md` (CEASE 2610.01989v1, LACU 2512.02657v2,
    AGE 2501.18950v3, continual-unlearning regularization 2511.07970v2, DP czhang024/CVPR_DouleProjection
    @3fc3d8afe03a400dfa71aaaa34b647fef6550e0f / arXiv 2604.10032).
  - `REF/docs/research/FEASIBILITY_BRIEF_2026-10-11.md` + `feasibility_sources_2026-10-11.json`.
  - `REF/results/block_object_sprint/` (BLOCK; code pin db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5).
  - `REF/results/eraseanything_efficiency_pilot/` (EraseAnything FLUX LoRA code pilot; EraseAnything++ if present).
  - CPR detector controls (`REF/results/cpr_*`): results are INCONCLUSIVE. Do NOT tune against, re-analyse, or
    cite their held-out cases as fresh evidence. Do not rerun closed experiments or baseline searches.
- Network: GitHub clones work. `arxiv.org` is BLOCKED from Bash (proxy 403). Try the WebFetch / WebSearch tools
  (load via ToolSearch) for arXiv/HTML; otherwise rely on local verified records and mark `unverified`.
  Never fabricate a citation, equation number, or line number; say `unverified` instead.

## GEM native target loss (verified in REF losses.py lines 233-249 of erasure_gem.py @ c328fc6)
v = student velocity/score prediction (flattened per sample), a = detached anchor teacher, b = detached target
teacher; `D+ = ||v-a+eps_n*1||`, `D- = ||v-b+eps_n*1||` (F.pairwise_distance, eps_n=1e-6),
`loss = gamma * sum_i relu(D+_i - eta*D-_i) / n_latents`. CPR replaces D- by the signed axial coordinate
alpha=(b-v).e_tau, e_tau=(b-a)/||b-a||. Closed result: at v=b CPR equals a centered smooth radial control.
Do not re-announce these as novel.

## Rules for claims
Agent agreement is not evidence. Every material claim needs: a source location (paper section/eq, file:line at a
pinned commit), a complete proof, or a reproducible computation. Mark each as VERIFIED / DERIVED / COMPUTED /
UNVERIFIED. Separate authors' claims from verified facts. Do not manufacture novelty; if a candidate is a
relabelled standard method, say so. State falsification conditions in advance.

## Output conventions
Markdown files, concise tables, explicit ID tags for claims (e.g. B-C1, C-C2). Code files runnable with
`/tmp/claude-0/venv/bin/python` from the repo root. Keep every file < 300 KB.
