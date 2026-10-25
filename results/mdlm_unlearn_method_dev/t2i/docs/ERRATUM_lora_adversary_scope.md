# Erratum — the LoRA adversary's scope was mis-described throughout this phase

**Date:** 2026-10-25 · **Found:** while smoke-testing the IMMA port, which reuses
`scripts/attack_lora.py`'s injection helper.
**Affects the DESCRIPTION, not the numbers**, in `FINDING53`, `FINDING54`, `FINDING58` §3a and
`PREREG_H81`/`H82` wherever the LoRA adversary is called "LoRA rank-16 on cross-attention".

---

## What is wrong

`inject_lora(unet, r, alpha, scope="cross")` is described as adapting the cross-attention layers.
It does not. It iterates `named_modules()` and requires `".attn2."` — with a trailing dot — in the
module name. The `Attention` module itself is named `...transformer_blocks.0.attn2`, which contains
no trailing dot, so **it is skipped**, along with its `to_q`, `to_k` and `to_v` children. Only
`...attn2.to_out` matches, because that name does contain `".attn2."`.

**Measured consequence:** the adversary injects **16 linears, all `attn2.to_out.0`** — the
cross-attention *output* projections only.

```
down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0
down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0
...  (16 total)
```

## What is unaffected

**Every reported number stands.** The parameter count published throughout — **0.399M, 0.046% of the
UNet** — is exactly right for 16 `to_out.0` projections at rank 16, and it is the figure all the
LoRA results were computed and reported with. No attack was weaker or stronger than its recorded
budget; the budget was simply described by the wrong name.

## Why the corrected description makes the finding stronger, not weaker

`FINDING58` §3a observed that a 0.046% LoRA adapter reverses GIFT's immunized arm to H = 1.000 while
GIFT edits 44.0M cross-attention parameters. The corrected scope sharpens this: the adapter touches
**only the 16 output projections** of the very layers GIFT immunizes — not their queries, keys or
values — and that is still sufficient to restore the concept completely. The same holds for the LoRA
reversals in `FINDING53` and `FINDING54`.

So the honest phrasing, used from here on, is **"LoRA rank-16 on the 16 cross-attention output
projections (0.399M params, 0.046%)"**, not "on cross-attention".

## What is NOT being done, deliberately

The helper is **not** being changed to widen the scope. Doing so would alter the adversary that every
LoRA result in this phase was measured with, and would make new numbers incomparable to published
ones. A wider-scope LoRA is a **new, strictly stronger adversary** and belongs under a new label if
it is ever run — the `FINDING35` §4 ratchet means a stronger adversary can only raise max-over-valid
H, and every LoRA arm here is already at the 1.000 ceiling, so there is nothing for it to raise.

The one place the narrow scope **was** about to cause a real error is the IMMA port: its `S`
(cross-attention, per the paper) was being discovered by matching `".base."`, which would have
silently restricted `S` to the 16 LoRA-wrapped `to_out` projections — **12.4M params instead of the
paper's 44.0M**. That is fixed in `scripts/erase_imma.py` before any IMMA arm was built, and the fix
is noted in the code.
