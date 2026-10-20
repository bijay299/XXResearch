# START HERE — llada_assoc_unlearning_v1

**Final status: `IMPLEMENTATION_READY_HYPOTHESIS`.** One experimental algorithm (TD-SSD) and one closest simple control
(SSD) are implemented for LLaDA-8B + LoRA and pass the focused tiny-model correctness checks.
- **Not established:**
  - its benefit (no efficacy result on any model);
  - its novelty: it is a combination of a WHP+-style stranger teacher with Counterfactual-Logit-Pairing-style
    differencing;
  - real-model behaviour: Cloud did not load LLaDA. Native gates in `HYPERPLANE_PLAN.md` §3 cover this.
- **Protocol deviation.** On the user's 07:25Z priority update, the original 8-hour plan's CPU efficacy validation was
  dropped (DECISION_LOG #6). Its toy substrate (`src/aslu/`) is unvalidated and is not evidence for anything.

**Code commit:** `b5d9a1e483090b8ca77c5773ffcee7bb365e7322`. Branch `claude/sleepy-feynman-a8dzok`; directory
`research/llada_assoc_unlearning_v1/`.

## The algorithm in one paragraph
- **Setting.** To forget "Alice lives in Paris" without hurting "Bob lives in Paris", train a LoRA on the memorised
  LLaDA checkpoint.
- **Training target.** At every masked answer position of Alice's canvas, the target is what the **frozen memorised
  checkpoint p0** predicts on the same canvas with Alice replaced by a fixed stranger ν_A. Stranger-name tokens in that
  prediction are mapped back to Alice's.
  - The target is two-sided: Paris keeps its stranger-level probability rather than being banned.
  - It is fluent, because it is p0's own answer about the stranger, generated once.
  - It is never empty.
- **The operation under test.** The forget error is back-propagated through z_A − z_c + sg(z_c). Here z_c are the
  logits on a **control canvas**: Alice's canvas with her name swapped for a control name, redrawn every step from 16
  names.
  - The loss value is unchanged.
  - The parameter step becomes (J_A − J_c)ᵀe, which cancels the update components that Alice's canvas shares with any
    name-swapped twin: template, relation, answer-word row and [MASK] directions.
  - This is the "token-ban" channel that carried 91–96 % of the collateral damage in the closed study.
- **Control Z (SSD).** Identical target, states, data and optimiser, without the difference (gradient J_Aᵀe), plus a
  rotating anchor on the control canvas.
- **Retain term.** Both arms include a retain KL to p0 on half of the retain facts.

## Central hypothesis and the one prediction
- **Conditions.** Memorised forget and retain facts share answer words; the target is a fixed stranger from the frozen
  memorised checkpoint; LoRA, optimiser, LR, steps and states are identical.
- **Claim.** Twin-differenced back-propagation (X) improves held-out shared-answer retain damage Δ_R, at matched forget
  distance D_F, over plain stranger-swap distillation (Z).
- **Mechanism.** The logit change at a state x is ∝ K(x,A) − K(x,c), which cancels the common mode.
- **Predicted observable.** Stranger drift ρ(X) ≈ 0 while ρ(Z) > 0.
- **Verdict rules.**

  | verdict | condition |
  |---|---|
  | **Supported** | Δ_R(X) ≤ 0.7·Δ_R(Z) and ρ(X) ≤ 0.3·ρ(Z) at every comparable X checkpoint |
  | **Contradicted** | Δ_R(X) ≥ Δ_R(Z) at every comparable X checkpoint |
  | **Inconclusive** | otherwise |
  | **NOT_COMPARABLE** | X does not forget enough |

  Computed by `llada/compare_tdssd.py`. One seed gives a directional result only.

## Strongest objection (open)
- **Shared attribute.** If the memorised model stores "Paris" as an attribute inside Alice's own representation,
  shared with other Paris residents (m > 0), a name-swapped twin cannot cancel it. In the linear analysis the gain then
  shrinks: X 0.33 vs Z 0.44 at m/i = 0.5, and X 0.67 vs Z 0.73 at m/i = 2. This is plausible for real memorised
  facts.
- **Adam.** Adam breaks exact cancellation. With one fixed twin, Bob moves by −0.17 instead of 0; the rotating
  16-name pool reduces this to about 0.06.
- **Novelty.** The differencing is a frozen-target variant of Counterfactual Logit Pairing (Garg 2019), so the novelty
  claim is limited to this combination. CUSE is unresolved.

## For tomorrow's presentation (short)
- **Experimental algorithm.** "Twin-differenced stranger-swap distillation", a combination/adaptation and not a new
  method.
  - Teacher: the frozen memorised LLaDA's own answer about a stranger.
  - Update: the forget error is pushed only through what distinguishes the forgotten subject from name-swapped twins.
- **One controlled test.** On LLaDA-8B with LoRA, 8 forget and 16 retain facts that share answer words, the test asks
  whether this differencing hurts held-out shared-answer facts less than the same distillation without it, at matched
  forgetting, and whether it leaves strangers untouched.
- **Interpretation.**
  - Success is forget answers falling to the stranger base rate while staying well-formed, with shared-answer retain
    facts and word copying intact.
  - Empty, degenerate or refusal answers count as failures, not forgetting.

## Files (all at the code commit above unless noted)
| file | content |
|---|---|
| `METHOD.md` | equations, mechanism, assumptions, analytical result, limiting cases, counterexample, prior art |
| `agents/O_FINAL.md` | frozen spec, answer to the objection (concession + code change), proposition with assumptions |
| `HYPERPLANE_PLAN.md` | inputs, commands for 2 GPUs, native gates, PROJECTED resources, predeclared comparison |
| `llada/tdssd.py` | candidate (`arm=tdssd`) and control (`arm=ssd`) losses, data preparation, training loop |
| `llada/train_tdssd.py` | Hyperplane entry point (PEFT LoRA, frozen reference via `disable_adapter`, checkpoints, native gates) |
| `llada/eval_tdssd.py`, `llada/compare_tdssd.py` | raw-generation evaluation; predeclared verdict |
| `llada/adapt_data.py` | data adapter (requires a `subject` field per forget fact) |
| `llada/configs/tdssd.json`, `llada/configs/ssd.json` | the two arms (identical except `arm`, `lam_T`) |
| `llada/tests/` | 19 focused tiny-model checks (`evidence/pytest_focused.txt`); CLI dry runs in `evidence/dry_run/` |
| `agents/V_REVIEW.md` | independent verifier: 46/47 own checks pass (one cosmetic); bugs found and fixed (DECISION_LOG #9) |
| `agents/D_INTAKE.md`, `agents/D_NOVELTY.md` | prior-art intake and complete-algorithm novelty qualification |
| `agents/P_PROBLEM.md`, `agents/O_PROPOSAL.md`, `agents/U_PROPOSAL.md` | problem account; the two initial mechanisms (U's not selected) |
| `DECISION_LOG.md`, `agents/AGENT_LEDGER.md`, `ledger/SUMMARY.md`, `EVIDENCE_INDEX.md` | decisions and changes of mind, agents, resources, evidence |

## Commands (detail in `HYPERPLANE_PLAN.md` §2)
```
python research/llada_assoc_unlearning_v1/llada/train_tdssd.py --config research/llada_assoc_unlearning_v1/llada/configs/tdssd.json \
   --model <memorised> --forget forget.jsonl --retain retain.jsonl --llada-repo LLaDA --out runs/X      # candidate
python research/llada_assoc_unlearning_v1/llada/train_tdssd.py --config research/llada_assoc_unlearning_v1/llada/configs/ssd.json \
   --model <memorised> --forget forget.jsonl --retain retain.jsonl --llada-repo LLaDA --out runs/Z      # control
```

## Readiness decision
- **Ready** for one bounded Hyperplane comparison under a separate instruction: 2 GPUs, about 1–1.5 GPU-hours
  PROJECTED.
- **Resource use here.** 0.03 measured CPU core-hours, plus about 0.06 estimated unwrapped (`ledger/SUMMARY.md`). No
  GPU, no model weights, no paid APIs.
- **Most likely blocking risks.**
  - LoRA module names (gate N1).
  - Name acceptance on the real tokenizer for fictional subjects (gate N6).

  Both are checked before training starts.
