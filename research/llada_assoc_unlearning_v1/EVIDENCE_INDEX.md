# EVIDENCE_INDEX — what each claim rests on (all files tracked on branch `claude/sleepy-feynman-a8dzok`)

| claim | evidence | kind |
|---|---|---|
| Loss values (L_F, L_T) equal an independent float64 loop; L_F value identical across arms | `llada/tests/test_tdssd.py::test_loss_values_match_independent_reference`, `::test_LF_value_identical_across_arms`; `evidence/pytest_focused.txt` | tiny-model check |
| X gradient = (J_A − J_c)ᵀe, Z gradient = J_Aᵀe; the arms differ; missing stop-gradient is detected | `::test_forget_gradient_matches_declared_equation`, `::test_broken_variant_missing_stopgrad_is_detected`; `agents/V_REVIEW.md` item 1 (logit-level dL/dz_A = e, dL/dz_c = −e, error ≈ 6e-9) | tiny-model check (two independent implementations) |
| Teacher is the frozen reference (adapter off, no grad); only LoRA trains; reference invariant after training | `::test_teacher_is_frozen_reference_detached_and_only_lora_trains`; `evidence/dry_run/*/ckpt_4_ref_invariance.json` (0.0) ; V_REVIEW item 3 | tiny-model check |
| Masking: prompt never masked, same mask on all three canvases, EOS padding maskable, empty mask → one position, finite KL with −inf targets and bf16 logits | `::test_mask_semantics_weights_and_empty_mask`, `::test_kl_rows_edge_cases`; V_REVIEW items 2, 4 | tiny-model check |
| Answer tokens receive signal at init (L_F > 0, non-zero LoRA gradient), Z anchor = 0 at init | `::test_at_init_signal_on_answer_tokens_and_ssd_anchor_zero` | tiny-model check |
| Name pools disjoint, swap involution, twin/control canvases differ only on name tokens; rejections (token count, collision, partial overlap) | `::test_name_pools_swaps_and_canvases`, `::test_name_rejected_on_unequal_token_count_and_collision`, `::test_engineering_fixes_from_V_review` | tiny-model check |
| Both arms execute training steps, rotate controls, checkpoint, through the CLI | `::test_both_arms_train_rotate_controls_and_save`; `evidence/dry_run/{tdssd,ssd}/` (CLI `--dry-run-tiny`) | tiny-model interface check |
| Eval helpers (D_F, key log-prob, failure classes) and verdict rules behave as declared | `llada/tests/test_eval_tiny.py` | tiny-model / synthetic-record check |
| Port of `generate` agrees with upstream @9182493 | `agents/V_REVIEW.md` item 6 (tiny model, 3 settings) | tiny-model check; LLaDA agreement = native gate N3 |
| Mechanism (common-mode cancellation; limiting cases; Adam counterexample) | `agents/O_FINAL.md` §1, §3; `agents/O_scripts/o_final_check.py` + `_out.json` | linear-model derivation + 0.6 s numeric check; NOT a guarantee for LoRA LLaDA |
| Prior-art position (Z ≈ WHP+ adaptation; X ≈ frozen-target CLP variant; CUSE unresolved) | `agents/D_INTAKE.md`, `agents/D_NOVELTY.md` | code-read / PDF-read / snippet-only, labelled per source |
| Efficacy of X or Z on any model | **none** | — |
| LLaDA-8B behaviour, memory, time | **none** (projections only, `HYPERPLANE_PLAN.md` §4) | — |

Ignore-rule check: `runs/` (scratch toy probe) is git-ignored and referenced by nothing; all evidence above lives in
`evidence/`, `agents/`, `llada/` (no ignored extensions).
