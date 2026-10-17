"""Preservation-residual logging (protocol section 4), computed POST HOC.

Records, per edited module and per preserve-concept vector, using each method's OWN
native vector selection (DP: every content token + CLIP pooled; UCE: last content token
+ CLIP pooled):

  (a) pre-cast residual  ||dW @ c_i||, from the as-computed fp32 update, where available
      (DP SEQ only -- captured by the adapter at the moment of the native bf16 cast).
  (b) post-cast residual ||(W_child - W_parent) @ c_i||, with BOTH weight matrices taken
      as represented at evaluation (exactly what generate_erased_flux.py loads: bf16)
      and promoted to fp32 only for this arithmetic. Casting dW alone is NOT used.
  (c) numerical context: vector norms, matrix norms, and the relative residual.

No exact-zero assertion is imposed. Nothing here alters any edit; this reads saved
checkpoints and re-encodes the frozen preserve strings with the pinned text encoders.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from safetensors import safe_open

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CTX = "context_embedder.weight"          # consumes T5 token vectors   [3072, 4096]
CLP = "time_text_embed.text_embedder.linear_1.weight"  # CLIP pooled   [3072, 768]
SNAP = ("/data/bijaypandey/cuig_pilot/hf_home/hub/models--black-forest-labs--FLUX.1-schnell"
        "/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9")


def pristine_weights() -> dict:
    """Read the two pristine transformer matrices straight from the model shards."""
    idx = json.load(open(os.path.join(SNAP, "transformer",
                                      "diffusion_pytorch_model.safetensors.index.json")))
    wm = idx["weight_map"]
    out = {}
    for key in (CTX, CLP):
        shard = os.path.join(SNAP, "transformer", wm[key])
        with safe_open(shard, framework="pt") as f:
            out[key] = f.get_tensor(key)
    return out


def encode_concepts(concepts, max_sequence_length=256, device="cuda:0"):
    """Reproduce the native embedding extraction for both methods' selections."""
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, vae=None, transformer=None,
        torch_dtype=torch.bfloat16, safety_checker=None).to(device)
    res = {}
    for c in concepts:
        t_emb = pipe.encode_prompt(prompt=c, prompt_2=None, device=device,
                                   num_images_per_prompt=1,
                                   max_sequence_length=max_sequence_length)
        attn = pipe.tokenizer_2(c, padding="max_length", max_length=max_sequence_length,
                                return_overflowing_tokens=False, truncation=True,
                                return_length=False, return_tensors="pt")["attention_mask"]
        valid = int(attn.sum().item())
        # DP: every content token (skip EOS), native loop range(0, valid-1)
        dp_t5 = [t_emb[0][:, i, :].squeeze(0).float().cpu() for i in range(0, valid - 1)]
        # UCE: single last content token, native index attn.sum()-2
        uce_t5 = t_emb[0][:, valid - 2, :].squeeze(0).float().cpu()
        clip = t_emb[1].squeeze(0).float().cpu()
        res[c] = {"dp_t5": dp_t5, "uce_t5": uce_t5, "clip": clip, "n_t5_tokens": len(dp_t5)}
    del pipe
    torch.cuda.empty_cache()
    return res


def residual_rows(W_delta_fp32, vecs, label):
    rows = []
    for name, v in vecs:
        r = (W_delta_fp32 @ v.unsqueeze(-1)).squeeze(-1)
        vn = float(v.norm().item())
        rn = float(r.norm().item())
        rows.append({
            "stage": label, "vector": name,
            "residual_l2": rn,
            "input_l2": vn,
            "relative_residual": (rn / vn) if vn > 0 else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--method", required=True, choices=["DP", "UCE"])
    ap.add_argument("--child", required=True)
    ap.add_argument("--parent", default=None,
                    help="A-parent checkpoint; omit to use pristine M0 weights")
    ap.add_argument("--preserve_concepts", required=True)
    ap.add_argument("--precast_dump", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    preserve = [c.strip() for c in a.preserve_concepts.split(";") if c.strip()]
    child = load_file(a.child)
    if a.parent:
        parent = load_file(a.parent)
        parent_src = a.parent
    else:
        parent = pristine_weights()
        parent_src = f"pristine M0 shards @ {SNAP}"

    enc = encode_concepts(preserve, device=a.device)

    rec = {
        "endpoint": a.endpoint, "method": a.method,
        "child": a.child, "parent_source": parent_src,
        "preserve_concepts_in_order": preserve,
        "evaluation_representation": "both child and parent cast to bfloat16 (exactly what "
                                     "generate_erased_flux.py loads), then promoted to fp32 "
                                     "for the residual arithmetic; dW is never cast alone",
        "rows": [],
        "notes": [],
    }

    for key, kind in ((CLP, "clip"), (CTX, "t5")):
        # ---- post-cast: evaluation-represented child minus evaluation-represented parent
        c_eval = child[key].to(torch.bfloat16).float()
        p_eval = parent[key].to(torch.bfloat16).float()
        d_post = c_eval - p_eval
        vecs = []
        for cname in preserve:
            e = enc[cname]
            if kind == "clip":
                vecs.append((f"{cname}::clip_pooled", e["clip"]))
            else:
                if a.method == "DP":
                    for i, t in enumerate(e["dp_t5"]):
                        vecs.append((f"{cname}::t5_tok{i}", t))
                else:
                    vecs.append((f"{cname}::t5_last", e["uce_t5"]))
        rows = residual_rows(d_post, vecs, "post_cast")
        for r in rows:
            r["module"] = key
            r["delta_fro_norm"] = float(d_post.norm().item())
        rec["rows"] += rows

        # ---- pre-cast: as-computed fp32 update, DP SEQ only (adapter capture)
        if a.precast_dump and os.path.exists(a.precast_dump):
            caps = torch.load(a.precast_dump)
            shape_key = str(tuple(child[key].shape))
            if shape_key in caps:
                w_f32_final = caps[shape_key].float()
                d_pre = w_f32_final - parent[key].float()
                rows = residual_rows(d_pre, vecs, "pre_cast")
                for r in rows:
                    r["module"] = key
                    r["delta_fro_norm"] = float(d_pre.norm().item())
                rec["rows"] += rows
                rec["notes"].append(
                    f"{key}: pre-cast residual from adapter-captured fp32 matrix "
                    f"(observational clone at the native cast site)")
            else:
                rec["notes"].append(f"{key}: no pre-cast capture for shape {shape_key}")
        elif a.method == "UCE":
            # UCE's native edit dtype is fp32 end-to-end: the SAVED child IS the
            # as-computed update, with no cast between computation and storage. The
            # fp32-level residual is therefore exactly child_fp32 - parent_fp32.
            d_pre = child[key].float() - parent[key].float()
            rows = residual_rows(d_pre, vecs, "pre_cast")
            for r in rows:
                r["module"] = key
                r["delta_fro_norm"] = float(d_pre.norm().item())
            rec["rows"] += rows
            rec["notes"].append(
                f"{key}: pre-cast residual taken directly from the saved fp32 child minus "
                f"fp32 parent (UCE edits in fp32 end-to-end; no cast occurs at edit time)")
        else:
            rec["notes"].append(
                f"{key}: pre-cast residual unavailable (endpoint ran through the unmodified "
                f"native CLI, so no in-run fp32 capture was taken)")

    # compact summary
    summ = {}
    for r in rec["rows"]:
        k = (r["stage"], r["module"])
        s = summ.setdefault(f"{r['stage']}|{r['module']}",
                            {"n": 0, "max_residual_l2": 0.0, "max_relative": 0.0})
        s["n"] += 1
        s["max_residual_l2"] = max(s["max_residual_l2"], r["residual_l2"])
        if r["relative_residual"] is not None:
            s["max_relative"] = max(s["max_relative"], r["relative_residual"])
    rec["summary"] = summ

    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(summ, indent=2))
    print("->", a.out)


if __name__ == "__main__":
    main()
