# V_REVIEW — independent implementation check of TD-SSD (X) / SSD (Z) (Agent V, 2026-10-20 ~07:50Z)

Scope: implementation correctness only. Reviewed `llada/tdssd.py` (07:40:43, rewritten to the FROZEN spec O_FINAL §2),
`llada/train_tdssd.py` (07:43:46), `llada/eval_tdssd.py` + `compare_tdssd.py` (07:45:57/07:45:30), configs (07:43:25).
**Every result below is a TINY-MODEL check (own 1-layer bidirectional model, own LoRA, own word tokenizer). It is not
LLaDA verification.** No efficacy experiments and no sweeps were run.
Commands (both via `tools/ledger_run.py --agent V --threads 1`):
`python agents/V_scripts/v_check.py` → `SUMMARY 46/47 PASS` (3.1 s CPU; the one FAIL is cosmetic bug B5) ·
`python -m pytest -q llada/tests/` (test_tdssd 07:42:58 + test_eval_tiny 07:45:57) → `18 passed`.

## Per-item verdicts (v_check.py ids in brackets)
1. **Losses/gradients — PASS.** [1a] L_F equals my own fp64 KL(t‖softmax z_A) computed from the recorded forwards,
   for both arms and for "mean" and "sft" weighting (0.658926 vs 0.658926). [1d] L_F value identical in X and Z with
   the same seed. [1b] logit gradients from retained `.grad`: X gives dL/dz_A = e and dL/dz_c = −e (max err 6e-9),
   so the parameter gradient is (J_A − J_c)ᵀe. Z gives dL/dz_A = e and dL/dz_c = λ_T(p_c − p0(·|x_c))/|M|
   (rotating anchor); [1c] the anchor teacher input is exactly the control canvas, with the adapter off.
   [1e] kl_rows = KL(target‖student), not the reverse. [1f] t(v) = T(σ(v)): t(subject tok) = T(teacher tok) and
   vice versa. Weighting: "mean" (frozen) = mean over M, then mean over rows; "sft" = 1/(p_m·|region|), checked
   against a replicated generator. Note: O_FINAL dropped 1/p_m, so the task's "1/(p_m|a|)" holds only for `sft`.
2. **Signal on answer tokens — PASS.** [2a] the gradient is exactly 0 at unmasked positions (prompt included).
   [2b] the same M on x_A, x_T and x_c; the prompt is never masked. [2c] x_c = σ_k(x_A), answer included; the unmasked
   part of x_T equals the clean twin; [5e] the clean x_T = σ_ν(clean x_A). [2d] over 300 draws the prompt was never
   masked, and the EOS batch-alignment padding past L was maskable (GUIDELINES: answer length includes padding).
3. **Reference detached, adapter off, LoRA-only — PASS.** [3a] every teacher forward (x_T; x_c for Z) runs with the
   adapter off and under no_grad; one student forward runs with the adapter on (2B rows). [3b] after `train()` the
   base params are bit-identical and the LoRA params have changed.
4. **Edge cases — PASS, with a cosmetic issue.** [4a] an empty mask gets exactly one answer position. [4c] at
   p_m = 1e-3 the sft weights are finite and the mean weights sum to 1. [4d] −inf target entries give a finite KL and
   a finite gradient. [4e] KL = inf only if a student logit is −inf. [4f] bf16 logits give an fp32, finite loss.
   [4b] **FAIL (cosmetic)**: see B5.
5. **Stranger construction — PASS, with caveats.** [5a] lengths are equal. [5b] σ is an involution and σ(pA) = pN.
   [5c,5d] teacher and control name tokens are disjoint from all data tokens, and control tokens are also disjoint
   from ȳ. [5f] a y_ν that restates the stranger gives a ȳ carrying the subject's tokens (intended). [5g] a y_ν with
   a subject token is rejected. Caveats: B1, B2, B6.
6. **Interface — PARTIAL.** [6a] `generate_port` == upstream `generate` @9182493 (transformers stubbed; greedy,
   low_confidence; 3 step/block configs) on the tiny model. The prompt tokenization (chat template +
   add_special_tokens=False) matches upstream `generate.py` main(). The vendored dllm `modeling_llada.py` has
   LLaDALlamaBlock {q,k,v}_proj, attn_out, ff_proj, up_proj, ff_out (7, matching the regex); the head
   `transformer.ff_out` sits outside `blocks.N` and is therefore excluded. `--dry-run-tiny` crashes (B7).
   Read-through of the updated eval/compare: D_F, Δ_R and ρ follow the O_FINAL formulas (B4).

## Concrete bugs (file:line @ versions above)
- **B1 wrong-science (moderate)** `tdssd.py:322`: the eval strangers E are just the next untaken full names. They
  bypass `accept_name` (no length or forbidden check), and their words are not disjoint from the teacher and
  control names. Simulation with name_seed 1234 (8 + 8×16 taken names): 13/13 E words occur in taken names. ρ
  (O_FINAL's m-immune held-out-stranger observable) would then be measured on tokens that X trained directly
  through z_c and that Z anchored. Fix: reserve disjoint FIRST/LAST sub-lists for E, run E through `accept_name`,
  and require E tokens ∩ (teacher ∪ control tokens) = ∅.
- **B2 wrong-science (low/moderate)** `tdssd.py:151-153`: only the differing tokens are checked, so a name sharing a
  word with the subject is accepted ([5h]: "James Smith" for "James Zed" swaps only Zed↔Smith). That twin is
  half-identical, and the common-mode cancellation then also removes Alice's own first-name feature. Fix: require
  every subject-span token to differ (n_span == len(subject tokens in pA)), or reject names sharing a word with s_A.
- **B3 wrong-science if misused** `train_tdssd.py:100` + `configs/tdssd.json` (lam_T = 0.0): `--config tdssd.json
  --arm ssd` silently trains Z with no rotating anchor. Fix: `assert cfg.arm != "ssd" or cfg.lam_T > 0` in
  `total_loss`/main, or derive lam_T from the arm.
- **B4 (resolved 07:45:57; one cosmetic issue left)** `eval_tdssd.py` now reads `names.json`, computes D_F on the
  fully masked answer from the ν_A canvas, per-phrasing retain log p, and E-canvas log p. `compare_tdssd.py`
  computes Δ_R = mean[log p0 − log p_θ] on R_ho, ρ = mean|Δ stranger| / mean|Δ subject|, and the sorted-D_F Z
  frontier, all as in O_FINAL. Left: `--gen-length` (eval:115) is not read from the training config.json, so a
  mismatch silently changes D_F. Training logs `D_F_state_mean`, a random-state mean, which is not the matching D_F.
- **B5 cosmetic** `tdssd.py:374`: with an empty region (n = 0), `ensure_nonempty` masks index plen−1, a prompt token.
  This is unreachable for gen_length ≥ 1. Fix: `if n == 0: continue`.
- **B6 cosmetic/low** `tdssd.py:299-301`: teacher-name tokens in y_ν outside a restated span are silently mapped to
  subject tokens in ȳ and are only counted (`n_name_tokens_in_ynu`). O_FINAL forbids b_i outside the name spans of ȳ.
  Fix: reject when the occurrences do not form whole contiguous name spans.
- **B7 cosmetic** `train_tdssd.py --dry-run-tiny`: on example_data it fails with `KeyError: "Quint's"`, because the
  tiny test tokenizer lacks the paraphrase words that `prepare` now tokenizes (`tdssd.py:279`).
- **B8 cosmetic** `tdssd.py:63,108`: L_a is fixed at 32, not "max answer length rounded up to ×8", and longer answers
  or y_ν are truncated silently. Fix: assert that len(answer) ≤ L−1, and that y_ν ends in EOS/EoT.

## Unresolved native checks (Hyperplane; none verifiable here)
- N1 LoRA regex on the HF checkpoint: block_group_size=1 (else `block_groups`), weight_tying False, 7/block, head untouched.
- N2 `PeftModel.disable_adapter()` really yields p0: the init gate (on−off=0) is trivial at B=0; add per-ckpt on≠off + ref_invariance≈0.
- N3 PEFT adapter dtype (fp32 autocast on bf16 base) + AdamW on it; bf16→fp32 logits at masked positions.
- N4 `save_pretrained` → `PeftModel.from_pretrained` on the same merged base reproduces `probe_logits.pt` (assert in eval).
- N5 tokenizer revision: chat template/one BOS; did the memorisation SFT put `<|eot_id|>` before EOS pad (`answer_ids` adds none)?
- N6 upstream `generate` on a PeftModel (`model.device`, `model(x, attention_mask=None).logits`) + the port-agreement gate.
- N7 name acceptance on real BPE (leading-space contexts, multi-token names; `forbidden` from joined text): inspect names.json.
- N8 memory: 24 grad rows × (P+32) × V=126464, perms_T[rows] int64 (~0.3–0.6 GB), no grad checkpointing; record peak_mem.
- N9 no attention_mask with right EOS padding (GUIDELINES-consistent). N10 pin HF revision for trust_remote_code; `torch_dtype` kwarg.
