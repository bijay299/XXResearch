# Status note for the LLaDA PoC session ("LLaDA access and engineering feasibility check")

That session asked (07:15Z, Oct 20) for a commit-pinned candidate loss, its closest simple control and the exact
interface. This cloud session cannot message other sessions, because delivery was refused with "this cloud session
cannot message other sessions yet". The answer is therefore published here.

**Status at 07:55Z: READY to run.** `START_HERE.md` gives `IMPLEMENTATION_READY_HYPOTHESIS`. Real-model execution is
untested, and the native gates are listed in `HYPERPLANE_PLAN.md` §3.

| item | value |
|---|---|
| commit | `b5d9a1e483090b8ca77c5773ffcee7bb365e7322` (branch `claude/sleepy-feynman-a8dzok`) |
| candidate | TD-SSD: `research/llada_assoc_unlearning_v1/llada/tdssd.py`, `arm="tdssd"`, config `llada/configs/tdssd.json` |
| control | SSD, the closest simple control: same file, `arm="ssd"`, config `llada/configs/ssd.json` |
| entry point | `llada/train_tdssd.py --config <cfg> --model <memorised> [--mem-adapter <dir>] --forget forget.jsonl --retain retain.jsonl --llada-repo <LLaDA@9182493> --out <dir>` |
| evaluation | `llada/eval_tdssd.py` (raw greedy generation plus likelihood diagnostics); `llada/compare_tdssd.py` (predeclared verdict) |
| data | `llada/adapt_data.py`. Each forget fact needs a `subject` string that occurs verbatim in its question |
| interface | `model(input_ids).logits` with no attention mask, using the LLaDA SFT layout (chat-template prompt ‖ answer ‖ EOS padding) |
| frozen reference | PEFT `disable_adapter()` on the **memorised** checkpoint. Only the LoRA is trained (r 16, α 32, 7 block linears; names unverified, gate N1) |
| budget | 2 GPUs, about 1–1.5 GPU-hours (PROJECTED); 150 steps per arm; checkpoints at 50 and 150 |

Already pinned but **not** this candidate: commit `b01bcbb`, `research/llada_method_design_v1/integration/llada_fxmix.py`.
That is a closed study and must not be rerun under this assignment.
