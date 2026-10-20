# SHARED CONTEXT — llada_assoc_unlearning_v1 (read fully before working)

Coordinator-maintained. Sprint start 2026-10-20T07:01Z; hard stop 15:01Z (8 h incl. publication).
Repo `/home/user/XXResearch`, branch `claude/sleepy-feynman-a8dzok`, sprint dir `research/llada_assoc_unlearning_v1/` (= `$R`).
Do not git commit/push (coordinator does). Write only inside your assigned paths.

## 1. Assignment (condensed)
Develop ONE implementable unlearning method for masked diffusion LMs (LLaDA), from first principles, ready for a
bounded real-model (Hyperplane) test. A novel, non-obvious combination of existing techniques is eligible. Do not
manufacture novelty or promise benchmark superiority.
**Initial concrete problem (association-selective forgetting):** forget "Alice lives in Paris" while preserving
"Bob lives in Paris", the ability to use the word "Paris", and unrelated knowledge about Alice. This is a problem
specification, not a solution hint. **Do not assume** token exclusion, counterfactual answers, projection, LoRA or
rollout training is the solution.
Required: precise problem account (what is forgotten; what must still work; observables separating success from
refusal / answer corruption / vocabulary suppression / detector evasion; what supervision the algorithm may use; what
finite evaluations can and cannot establish) → derive an update procedure/architecture → identify the actual
algorithmic operation that might matter (a generic constrained objective + standard optimizer is not enough) →
central hypothesis of the form **"Under conditions C, operation X improves outcome Y over control Z, because mechanism M
predicts observable behaviour W."** At most two mechanisms initially; select one; one substantive development
revision allowed before the validation freeze; no redesign after.
Final status (one): `IMPLEMENTATION_READY_HYPOTHESIS` | `IMPLEMENTED_WITH_SPECIFIC_BLOCKER` | `NO_QUALIFIED_CANDIDATE`
(and say whether a negative arises from known-method equivalence, unsupported benefit, mathematical failure or
infrastructure).

## 2. Environment / budget (hard)
- CPU only (4 vCPU, 15 GiB). Python with torch 2.9.1 CPU: `/tmp/claude-0/env-llada/bin/python` (numpy, scipy, sympy, pytest).
- Budget for THIS assignment: 24 measured CPU-core-hours, **≤ 4 concurrent CPU threads in total across all agents**
  (use 1 thread per process unless the coordinator says otherwise), ≤ 8 GiB extra storage, no GPU, no model downloads,
  no paid APIs. Run every compute command through the ledger wrapper:
  `/tmp/claude-0/env-llada/bin/python $R/tools/ledger_run.py --agent <ID> --label "<what>" --threads 1 -- <command>`
  Keep single runs ≤ 10 min wall unless agreed.
- Network: github.com via `git clone` and raw.githubusercontent.com work; **arxiv.org, aclanthology.org,
  huggingface.co, openreview, proceedings.mlr.press are blocked** (curl and WebFetch). WebSearch returns model-written
  summaries + links: label knowledge from it `SNIPPET-ONLY`. Local PDFs (supplied by the user):
  `/tmp/claude-0/src/MDU_2605.18253v1.pdf`, `/tmp/claude-0/src/EMDU_2610.25447v1.pdf` (+ `EMDU.txt`).
- Pinned read-only clones in `/tmp/claude-0/src/`: LLaDA @ 9182493 (GUIDELINES.md, generate.py, get_log_likelihood.py),
  MDU @ e5e8e82, dllm @ ca17675 (contains a vendored `modeling_llada.py` copy — module names still unverified against
  the HF checkpoint), t5patches @ 09285b3 (TNT/TNFF), open-dlu @ fb98433 (LRCU), dlm-priming-vulnerability @ b2c9769.

## 3. LLaDA mechanics (verified from the pinned source in the previous sprint)
- Bidirectional Transformer mask predictor; `[MASK]` = 126336, EOS 126081, EoT 126348, vocab 126464.
- Pretraining: t~U(0,1), every token (no prompt/answer distinction) masked iid w.p. (1−1e-3)t+1e-3; loss Σ_masked CE/p_mask.
- SFT: prompt never masked; answer incl. EOS padding masked iid w.p. p_mask; CE/p_mask/answer_len.
- Native `generate`: fixed gen_length of [MASK], semi-AR blocks, steps per block, Gumbel-max at temperature T,
  remasking `low_confidence` (commit highest-probability candidates) or `random`; committed tokens never re-masked;
  optional CFG whose unconditional branch replaces **prompt tokens by [MASK]** (so masked-prompt inputs are a supported
  use of LLaDA-Instruct).

## 4. Closed studies — read-only constraints (do NOT restart, rerun, extend, retune or repair them)
- **FX-Mix / `research/llada_method_design_v1/`** (closed; status NO_DISTINCT_METHOD_AFTER_CONSTRUCTION). Findings you may
  use as facts about the toy: (i) the TNT/TNFF exclusion target r_ε = p0 with fact tokens removed (UNDIAL γ→∞) is known;
  (ii) training on the model's own sampler states (multi-config rollouts, first-commit truncation) gave no gain over
  random teacher-forced states in a 2-layer toy (parameter sharing generalised); (iii) under shared parameters an
  exclusion edit acts largely like a **token ban**: 91–96 % of the forgotten token's logit drop was carried by shared
  parameters, damaging retained facts that share the token (C's objection, `agents/OBJ_C_to_B.md`); (iv) the exclusion
  target's fixed point (decode-time ban) gave only ~58 % well-formed forget answers under greedy decoding vs 100 % for a
  retrain-without-forget model — the renormalised tail of a memorised prediction is partly junk; (v) MDU's
  prompt-masked anchor still predicts the fact when the answer restates the subject; EMDU (anchor-first KL to a
  prompt+answer-masked anchor) forgets mostly by emitting empty/degenerate answers; (vi) THEORY there: exact first-commit
  decomposition of leakage for the native sampler; escape characterisation in the zero-sharing class.
  Its code (`src/mdmu/`: LLaDA-faithful SFT loss, port of `generate`, LoRA, toy fact world, evaluation) may be **copied
  as read-only reference code** into `$R/src/` with attribution; never modify files under `llada_method_design_v1/`.
- **VAU / flow_method_design_v1** (FLUX; Adam-metric projection of LoRA steps against a retained-Jacobian probe +
  acceptance gates): closed; not to be re-proposed under any name. **CPR** studies: closed.

## 5. Reference list given by the user (start here for prior art; targeted, not a broad catalogue)
MDU arXiv 2605.18253 (PDF local) · EMDU arXiv 2610.25447 (PDF local) · TNT arXiv 2406.13660 · UNDIAL arXiv 2402.10052 ·
Selective Span-Level Unlearning (ACL 2026 short, aclanthology 2026.acl-short.35) · D2F arXiv 2610.17654 · CUSE arXiv 2610.14763.

## 6. Roles (agents) — direct challenges are required
P problem/formalization · O objective designer · U update/architecture designer · D prior-art adversary ·
E mathematical critic · F implementation engineer · V independent verifier. Coordinator owns budget, decision log and
final selection. Every designer receives another agent's strongest objection and must answer with a proof, code-level
change, counterexample or explicit concession; changes of mind are recorded.
