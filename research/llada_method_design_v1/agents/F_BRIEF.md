# F brief — implementation engineer (wave 3)

Authoritative spec: `$R/SPEC.md` (v0 now; v1 will follow after critique — keep every knob configurable).
Substrate you inherit and own from now on: `$R/src/mdmu/` (`README_SUBSTRATE.md`), tests `$R/tests/test_substrate.py`.
Dev checkpoints: `$R/results/dev/substrate/w0_i0/{base,retrain}.pt` (world seed 0, init 0).
Python: `/tmp/claude-0/env-llada/bin/python`; run everything through `$R/tools/ledger_run.py --agent F --threads <=2`.

## Deliverables (in order of priority)
1. `src/mdmu/fxmix.py` — reference implementation (readable, commented with SPEC equation numbers):
   - `exclusion_target(ref_logits, fact_ids, eps)` → target probs r_ε (closed form; ε=0 ⇒ −inf on F then softmax);
     handle p0(F) ≤ ε (target = p0), F empty, and p0(F)→1 numerically (log-space; never divide by ~0).
   - `fact_free_tf_states(ids, prompt_len, fact_ids, generator)` (fact positions always masked; t~U for others).
   - `collect_rollout_states(model, prompts, fact_sets, config, J, generator)` using `diffusion.generate(...,
     return_trace=True)` semantics; truncate at the first step whose commit contains an F token (keep that pre-commit
     state); return only fact-free states; also return per-rollout leak flags (free diagnostic).
   - `fx_forget_loss(model, ref_fn, states, fact_sets, eps)` = mean over states of mean over masked answer positions
     of KL(r ‖ p_θ) (implemented as CE to soft target minus target entropy; report both).
   - Control losses on the same interfaces: NPO (β=0.2, MDU App. B form), MDU (τ ∈ {0,1}; KL(p_θ ‖ anchor^τ/Z),
     anchor = reference with prompt positions → MASK), SC (r_0 on MDU's random TF states incl. fact-visible states).
2. `src/mdmu/unlearn.py` — one training loop for all arms: LoRA (rank 8, all 7 linears, both layers), AdamW, clip 1.0,
   forget batch + retain batch (LLaDA SFT loss, λ_r=1), rollout pool refresh every R steps, checkpoint evaluation every
   k steps with `evaluate.fact_eval` under the evaluation grid; **compute counters**: student fwd, student bwd,
   reference fwd, rollout fwd (no-grad), plus wall-clock; JSON output per run.
3. `tests/test_fxmix.py` — focused analytic fixtures and autograd/finite-difference checks: r_ε(F)=ε exactly and r_ε is
   the minimiser of KL(q‖p0) over q(F)≤ε (compare against random feasible q and a numeric solver); ∂/∂z of CE(r, softmax z)
   = softmax(z) − r (autograd vs analytic vs finite differences); at θ=θ0 the forget gradient vanishes at positions with
   p0(F) ≤ ε; TF states never show F tokens and always mask F positions; rollout truncation keeps the decision state and
   nothing after; degenerate cases (F empty ⇒ loss 0; p0(F)=1 within float ⇒ finite loss); LoRA: both A and B receive
   non-zero gradient by step 2 (B=0 ⇒ ∂A=0 at step 1 — assert and document).
4. `experiments/dev_compare.py` — dev comparison of arms (SPEC §4) on world 0 (+ world 1 if time): small LR grid
   (e.g. {5e-4, 1e-3, 2e-3}), step budget ≤ 400, eval every 25 steps under the full evaluation grid; matched-operating-
   point tables (rule will be fixed in SPEC v1; implement the selection as a function so it can be locked later);
   report INFEASIBLE when an arm never reaches the threshold. Keep total dev compute ≤ ~25 CPU-min.
5. `integration/llada_fxmix.py` + `configs/hyperplane_fxmix.yaml` — LLaDA-8B integration against the pinned interfaces:
   HF `AutoModel.from_pretrained(..., trust_remote_code=True)`, `model(input_ids).logits`, mask id 126336, EOS 126081,
   EoT 126348, prompt/answer layout of GUIDELINES.md SFT, a faithful `generate` port with `return_trace` and first-F-commit
   truncation (identical semantics to `/tmp/claude-0/src/LLaDA/generate.py`), PEFT LoRA target module names marked
   **UNVERIFIED** (HF modeling file not inspectable here), fact-set builder from a tokenizer (answer tokens not in the
   prompt, not special/punctuation/stop-list, + leading-space/case variants), the training step, and a dry-run test
   `tests/test_integration_dryrun.py` against a tiny random model exposing `.logits` (real-model execution: UNTESTED).
6. `results/dev/COST.md` — measured CPU costs (per step, per arm) and separately labelled LLaDA-8B projections
   (forward/backward counts, rollout forwards, teacher forwards, vocab-dependent ops on |V|=126,464, memory for LoRA
   optimiser state, reference pass via adapter-disable, rollout pool storage).

## Rules
- Write only under `$R/src/`, `$R/tests/`, `$R/experiments/`, `$R/integration/`, `$R/configs/`, `$R/results/dev/`.
- Keep substrate tests passing. No git commits. Thread cap 2 per process; total of all your processes ≤ 2 threads.
- Record every non-trivial run in the ledger. Report failures and infeasible arms; never tune on final seeds
  (final worlds are seeds ≥ 2 and belong to G).
