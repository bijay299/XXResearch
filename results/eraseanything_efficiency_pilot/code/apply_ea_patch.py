#!/usr/bin/env python3
"""Port EraseAnything's two method-critical diffusers modifications onto stock diffusers 0.32.0.

WHY A PATCH AND NOT THE VENDORED FORK
-------------------------------------
The author repo (tomguluson92/EraseAnything @ 67d8c0bd5b153ff0e394b45e893b6671bc40e30b)
vendors a *partial* copy of diffusers: 229 .py files, with `utils/` and `schedulers/`
entirely absent, so `import diffusers` from that tree fails
(`ModuleNotFoundError: No module named 'diffusers.utils'`). Of the vendored files that do
exist, 134 are byte-identical to released diffusers 0.32.0 and 92 differ -- almost all by
version drift between the author's declared base `0.32.0.dev0`
(`train_flux_lora.py: check_min_version("0.32.0.dev0")`) and the 0.32.0 release.
Overlaying the vendored `attention_processor.py` wholesale onto 0.32.0 fails
(`ImportError: cannot import name 'SanaLinearAttnProcessor2_0'`), because the author's base
predates processors that 0.32.0 requires elsewhere.

So we install released diffusers 0.32.0 and port ONLY the two method-critical changes:

  (1) models/attention_processor.py
      - add `sam_scaled_dot_product_attention`, copied verbatim from the author tree
        (source/ea/diffusers/src/diffusers/models/attention_processor.py:1776-1798):
        an explicit materialised-softmax attention that also returns the attention matrix.
      - in `FluxAttnProcessor2_0.__call__`, use it instead of `F.scaled_dot_product_attention`
        and return the attention matrix as a third tuple element.

  (2) models/transformers/transformer_flux.py
      - thread that third element out of `FluxTransformerBlock.forward`
      - have `FluxTransformer2DModel.forward` return `(output, attn_maps)` when
        `return_dict=False`, matching the author's
        "2024.11.25 output layer 19 attn_maps for loss calculation."

FIDELITY NOTE: in the author tree `vis_attn = True` is hardcoded in BOTH the processor and
the transformer, so EVERY dual-stream block materialises a full [B, heads, L, L] softmax
matrix while only the LAST dual-stream block's matrix is ever returned (the loop overwrites
`attn_maps` each iteration; FLUX.1-dev has 19 dual-stream blocks, so index 18 == "layer 19").
We reproduce that exactly by default. We additionally expose switches so the same code can
run a flash/SDPA path and a maps-from-one-block path; these are used for measurement and as
shared engineering / ordinary controls, and they are NOT bit-identical to the explicit
softmax (verified only within tolerance).

Switches (module-level, in diffusers.models.attention_processor):
    EA_STATE["explicit_attn"] : bool  -- materialise softmax + return maps at all
    EA_STATE["maps_only_last"]: bool  -- if True, only the last dual-stream block
                                        materialises; others use fused SDPA
Set via ea_runtime.configure_attention(...). Default reproduces the author behaviour.

Idempotent: re-running detects the markers and does nothing.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

MARK = "# ---- EA_PILOT_PATCH ----"

AP_HELPER = '''
# ---- EA_PILOT_PATCH ----
# Ported from EraseAnything @67d8c0bd (diffusers/src/diffusers/models/attention_processor.py
# :1776-1798). Explicit materialised-softmax attention that also returns the attention
# matrix. Kept verbatim except for the removal of commented-out dead branches.
EA_STATE = {"explicit_attn": True, "maps_only_last": False,
            "_want_maps": True, "_last_maps": None}


def ea_sdpa_with_maps(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None):
    import math as _math
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / _math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight
# ---- /EA_PILOT_PATCH ----
'''

AP_OLD = """        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states"""

AP_NEW = """        # ---- EA_PILOT_PATCH ----
        _ea_maps = None
        _ea_want = (
            encoder_hidden_states is not None
            and EA_STATE["explicit_attn"]
            and EA_STATE["_want_maps"]
        )
        if _ea_want:
            hidden_states, _ea_maps = ea_sdpa_with_maps(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        # ---- /EA_PILOT_PATCH ----
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if _ea_maps is not None:                      # ---- EA_PILOT_PATCH ----
                # Stash rather than widen the return tuple: a 3rd element would be
                # consumed by stock's IP-adapter path (`hidden_states + ip_attn_output`).
                EA_STATE["_last_maps"] = _ea_maps
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states"""




def patch_attention_processor(p):
    t = p.read_text()
    if MARK in t:
        return "already patched"
    if t.count(AP_OLD) != 1:
        raise SystemExit(f"FAIL: SDPA block not unique in {p} ({t.count(AP_OLD)})")
    anchor = "class FluxAttnProcessor2_0:"
    if t.count(anchor) != 1:
        raise SystemExit(f"FAIL: FluxAttnProcessor2_0 anchor not unique in {p}")
    t = t.replace(anchor, AP_HELPER + "\n\n" + anchor, 1)
    t = t.replace(AP_OLD, AP_NEW, 1)
    p.write_text(t)
    return "patched"


TF_CALL_OLD = """                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )"""

TF_CALL_NEW = """                # ---- EA_PILOT_PATCH ----
                # EraseAnything needs the attention matrix of the LAST dual-stream block
                # ("layer 19"). The author materialises it in EVERY dual-stream block and
                # keeps only the last (the loop overwrites `attn_maps`); we reproduce that
                # by default, and `maps_only_last` restricts materialisation to the block
                # whose matrix is actually consumed.
                from ..attention_processor import EA_STATE as _EA
                _is_last_ds = index_block == (len(self.transformer_blocks) - 1)
                _EA["_want_maps"] = bool(
                    _EA["explicit_attn"] and ((not _EA["maps_only_last"]) or _is_last_ds)
                )
                _EA["_last_maps"] = None
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if _EA["_last_maps"] is not None:
                    _ea_attn_maps = _EA["_last_maps"]
                    _EA["_last_maps"] = None
                # ---- /EA_PILOT_PATCH ----"""

TF_RET_OLD = """        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)"""

TF_RET_NEW = """        if not return_dict:
            # ---- EA_PILOT_PATCH ---- EraseAnything consumes (sample, attn_maps).
            if _ea_attn_maps is not None:
                return (output, _ea_attn_maps)
            return (output,)

        return Transformer2DModelOutput(sample=output)"""

TF_FWD_ANCHOR = """        hidden_states = self.x_embedder(hidden_states)"""


def patch_transformer_flux(p):
    t = p.read_text()
    if MARK in t:
        return "already patched"
    for name, s in (("dual-stream call", TF_CALL_OLD), ("forward return", TF_RET_OLD),
                    ("forward anchor", TF_FWD_ANCHOR)):
        if t.count(s) != 1:
            raise SystemExit(f"FAIL: {name} not unique in {p} ({t.count(s)})")
    t = t.replace(TF_FWD_ANCHOR,
                  "        _ea_attn_maps = None  # ---- EA_PILOT_PATCH ----\n" + TF_FWD_ANCHOR, 1)
    t = t.replace(TF_CALL_OLD, TF_CALL_NEW, 1)
    t = t.replace(TF_RET_OLD, TF_RET_NEW, 1)
    p.write_text(t)
    return "patched"


def main() -> int:
    import diffusers
    root = pathlib.Path(diffusers.__file__).parent
    backup = pathlib.Path(__file__).resolve().parent / "stock_backup"
    for rel, fn in (("models/attention_processor.py", patch_attention_processor),
                    ("models/transformers/transformer_flux.py", patch_transformer_flux)):
        p = root / rel
        b = backup / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists():
            shutil.copy2(p, b)
        print(f"{rel}: {fn(p)}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
