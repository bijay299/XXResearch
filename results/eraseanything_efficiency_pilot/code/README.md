# ⚠ Archived pilot code — known limitations, not a validated control or a ready optimisation

These files are an **unchanged historical snapshot** of what was executed. They are
published as evidence, **not** as a recommended implementation. Their code and comments
still contain claims that
[`../AMENDMENT_02_CLOSEOUT.md`](../AMENDMENT_02_CLOSEOUT.md) **withdraws**. The files are
deliberately left as-run; do not read their comments as current findings.

## Known limitations in this snapshot

**1. `ea_arms.py` — the early-stop path does NOT preserve the native schedule.**
`ea_arms.py:163-173` sets `n_steps = j + 1` and passes it into `latent_sample` as a schedule
**length**. Upstream (`utils/esd_utils.py:61` at pin `67d8c0bd…`) that length reaches
`scheduler.set_train_timesteps`, and `tools/scheduler_process.py:93-109` then draws a **new
random vector** of that length, rescales and sorts it. So the path **resamples a different,
shorter full schedule** rather than stopping the original 28-point schedule at `j`. Time
coordinates, latent states, the consumed attention map and subsequent Torch RNG consumption
can all change.

The module docstring in `ea_arms.py` asserts this is "ORDINARY, EXACT, BIT-IDENTICAL" and
that the maps are returned "bit for bit". **That assertion is wrong and is withdrawn.**
Arm B is a **changed variant**, not an exact control.

**2. `ea_arms.py` / `apply_ea_patch.py` — selective explicit attention is not bf16-equivalent.**
The claim that restricting the materialised softmax to the consumed block preserves outputs
was supported only by a float64 check on a tiny randomly-initialised transformer. Upstream
blocks run fused SDPA, which changes the states entering the final explicit block, and the
published raw profiles already differ at **step 0** (ESD `0.0196533203125` vs
`0.019775390625`; grad-norm `0.1015625` vs `0.1005859375`). **The equivalence claim is
withdrawn.**

**3. `gen_eval.py` — the cited attention-difference figure is wrong.**
Its docstring reports "max |diff| 135 of 255". The indexed artifact
`../measured/attn_equivalence.json` records **max 165**, mean **1.4943949381510417**,
differing-pixel fraction **0.6395390828450521**. That artifact also measures **decoded-image
differences between two inference paths** — it is **not** evidence about attention-map or
gradient identity, and it was previously cited as if it were.

**4. `ea_arms.py` — the arm-C (`shared_prefix`) path was never trained or evaluated.**
Its docstring's variance-reduction and common-random-numbers reasoning is **not established**
for this objective: the substitution changes the sampling distribution. The docstring also
refers to a "worked check in derivation_note.md"; **no such file was ever produced**.

**5. `train_ea.py` / `ledger.py` — auxiliary-provider accounting is understated.**
The wrapper records one request per `MoE` invocation, but native `MoE` issues **two** GPT
calls under `model_list=["gpt"], final_expert=["gpt"]` (a suggestion call and a final-expert
arbitration call), and SDK retries are invisible. The true provider-request count is
**unknown**; see the closeout §5.

## What this snapshot does support

The pins, the documented third-party patches, the device placement actually used, the frozen
case construction, the instrumentation, and the ledger mechanics — all as executed. The
measured work counts and timings it produced are preserved in `../measured/profile/` and
remain valid **as one-profile observations of the configurations actually run**.

No file in this directory has been edited for the closeout.
