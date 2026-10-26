"""UCE — Unified Concept Editing (Gandikota et al., arXiv 2308.14761). OUR RE-IMPLEMENTATION.

Closed-form edit of the cross-attention K/V projections. For each such matrix W, UCE solves

    W_new = ( sum_{i in E} v_i* c_i^T + sum_{j in P} W_old c_j c_j^T + lam * W_old )
            ( sum_{i in E} c_i  c_i^T + sum_{j in P}        c_j c_j^T + lam * I     )^{-1}

where c are text-embedding directions, E is the erase set, P the preserve set, and
v_i* = W_old c_target is the value the erased concept should instead produce (we map it to the
embedding of a neutral target, " " / "", which is the paper's erasure setting).

Labelled as a re-implementation throughout (PREREG_H38 S4): the original authors' tuning is not
reproduced, so published numbers are not claimed. There are 32 K/V matrices in SD v1.4.

NOTE ON THE CONCEPT DIRECTION. We take the text-encoder hidden states from the concept's final
content token through position 76, matching UCE's official implementation. See `concept_dirs` for
why the single-token version measured earlier does not erase at all. This choice is held IDENTICAL
across every arm and baseline so comparisons are internally consistent.
"""
import argparse, json, os
import torch

# CROSS-MODEL OVERRIDE (PREREG_H85 / FINDING64). Defaults reproduce the pinned SD-1.4 exactly when
# BASE_MODEL/BASE_REV are unset, so no existing result changes.
import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None


PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"

FORGET_PROMPTS = {
    "church": "church", "parachute": "parachute", "garbage_truck": "garbage truck",
    "van_gogh": "Van Gogh",
}
# preserved concepts: the retain neighbours the utility axis is measured on
PRESERVE = ["cassette player", "chain saw", "english springer spaniel", "french horn",
            "gas pump", "golf ball", "tench", "Monet", "a photo", "a painting"]


@torch.no_grad()
def concept_dirs(pipe, texts, device, paired=None):
    """Hidden states from each prompt's final CONTENT token through position 76.

    CORRECTED. A first version used only the single final-content-token hidden state; UCE then
    changed the weights by just 5.7% and did not erase at all (church stayed at CLIP acc 1.000).
    The reason is structural: SD cross-attention attends over ALL 77 positions, and CLIP's text
    encoder is CAUSAL, so the concept is propagated into every trailing position -- the trailing
    "padding" states carry the sequence summary the UNet actually reads. Editing one position
    therefore leaves the concept reachable. This matches UCE's official implementation, which
    takes the token range from the concept token to the end of the sequence.

    Returns (sum_positions, 768). With `paired`, returns the SAME positions of the paired prompts
    so the erase direction and its target value are position-aligned.
    """
    tok, te = pipe.tokenizer, pipe.text_encoder
    out, out_p = [], []
    for k, t in enumerate(texts):
        ti = tok([t], padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt").to(device)
        h = te(ti.input_ids)[0]
        last = max(int(ti.attention_mask[0].sum().item()) - 2, 0)
        idx = torch.arange(last, h.shape[1], device=device)
        out.append(h[0, idx].float())
        if paired is not None:
            tp = tok([paired[k]], padding="max_length", max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").to(device)
            hp = te(tp.input_ids)[0]
            out_p.append(hp[0, idx].float())          # SAME positions
    C = torch.cat(out)
    return (C, torch.cat(out_p)) if paired is not None else C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=sorted(FORGET_PROMPTS))
    ap.add_argument("--target", default="", help="what the erased concept should map to")
    ap.add_argument("--lam", type=float, default=0.1,
                    help="regularisation toward W_old. WARNING (ERRATUM_uce_lambda_scale.md): this "
                         "is added to c^T c, whose max eigenvalue is ~8.6e4 for SD's 768-dim CLIP "
                         "embeddings, so values of order 1 are INERT -- they change nothing. The "
                         "usable range starts around 1e3-1e4. The 0.1 default is kept only so every "
                         "previously published arm reproduces exactly.")
    ap.add_argument("--preserve-scale", type=float, default=0.1)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, revision=(BASE_REV_ENV or pin["revision"]), torch_dtype=torch.float32,
        variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)

    c_e, c_t = concept_dirs(pipe, [FORGET_PROMPTS[a.concept]], dev, paired=[a.target])
    c_p = concept_dirs(pipe, PRESERVE, dev)
    print(f"[uce] erase directions: {tuple(c_e.shape)}  preserve directions: {tuple(c_p.shape)}",
          flush=True)

    mods = [(n, m) for n, m in pipe.unet.named_modules()
            if n.endswith("attn2.to_k") or n.endswith("attn2.to_v")]
    print(f"[uce] editing {len(mods)} cross-attention K/V matrices for '{a.concept}' -> '{a.target}'",
          flush=True)
    d = c_e.shape[1]
    I = torch.eye(d, device=dev)
    tot = 0.0
    for n, m in mods:
        W = m.weight.data.float()                     # (out, 768)
        V_star = (W @ c_t.T).T                        # (n_e, out) position-aligned target values
        # sum_i v_i c_i^T  and  sum_i c_i c_i^T, over every erase position
        num = V_star.T @ c_e + a.preserve_scale * (W @ c_p.T) @ c_p + a.lam * W
        den = c_e.T @ c_e + a.preserve_scale * (c_p.T @ c_p) + a.lam * I
        W_new = num @ torch.linalg.inv(den)
        tot += (W_new - W).norm().item() / max(W.norm().item(), 1e-9)
        m.weight.data = W_new.to(m.weight.dtype)
    print(f"[uce] mean relative weight change: {tot/len(mods):.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in pipe.unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "t2i_uce_reimpl", "label": a.label, "concept": a.concept,
               "hparams": vars(a), "n_matrices_edited": len(mods),
               "mean_rel_weight_change": tot / len(mods),
               "note": "OUR RE-IMPLEMENTATION of arXiv 2308.14761; original tuning not reproduced"},
              open(f"/data/bijaypandey/t2i_unlearn/outputs/erase_{a.label}.json", "w"), indent=1)
    print(f"[uce] saved {a.out}/unet.pt", flush=True)


if __name__ == "__main__":
    main()
