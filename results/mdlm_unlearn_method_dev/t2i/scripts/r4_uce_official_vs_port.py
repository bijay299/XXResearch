"""R4 — compare the OFFICIAL UCE closed form against our port on MATCHED inputs.

Official reference: unified-concept-editing @ 7c724d9, `trainscripts/uce_sd_erase.py`:

    mat1 = lamb * w_old
    mat2 = lamb * I
    for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
        c_i = embeds[erase_concept].T ; v_i_star = guide_outputs[guide_concept][idx].T
        mat1 += erase_scale   * (v_i_star @ c_i.T)
        mat2 += erase_scale   * (c_i @ c_i.T)
    for preserve_concept in preserve_concepts:
        c_i = embeds[preserve_concept].T ; v_i_star = guide_outputs[preserve_concept][idx].T
        mat1 += preserve_scale * (v_i_star @ c_i.T)
        mat2 += preserve_scale * (c_i @ c_i.T)
    W_new = mat1 @ inverse(mat2)

Ours (`scripts/erase_uce.py`):

    num = V_star.T @ c_e + preserve_scale * (W @ c_p.T) @ c_p + lam * W
    den = c_e.T @ c_e    + preserve_scale * (c_p.T @ c_p)     + lam * I
    W_new = num @ inv(den)

THE COMPARISON THIS MAKES. Both are implementations of the same equation, so the question is
whether our transcription is numerically equivalent on identical inputs, and where it is not. Two
candidate discrepancies are visible from source and are tested rather than asserted:

  (a) **erase_scale.** The official form carries `erase_scale` (default **1**) on BOTH the erase
      numerator and denominator terms. Our port has no such coefficient -- equivalent only if the
      intended erase_scale is exactly 1.
  (b) **lamb default.** Official default is **0.5**; ours is **0.1**. A default mismatch is not a
      formula error, but it means "UCE at default settings" means different things in the two
      codebases, and every historical arm of ours used 0.1.

Method: build the SAME conditioning embeddings and the SAME W_old from the pinned SD-1.4, feed them
to both formulas at matched (lamb, erase_scale, preserve_scale), and report per-matrix relative
error. No attack, no generation -- a numerical check, which is what task item 4 asks for before
robustness is interpreted.

Writes outputs/r4_uce_official_vs_port.json. Runs on one GPU for a few minutes.
"""
import argparse, json, os, sys
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--guide", default=" ", help="official --guide_concepts; '' or ' ' erases toward nothing")
    ap.add_argument("--lamb", type=float, default=0.5, help="official default")
    ap.add_argument("--erase-scale", type=float, default=1.0, help="official default")
    ap.add_argument("--preserve-scale", type=float, default=1.0, help="official default")
    a = ap.parse_args()
    dev = "cuda"
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)
    tok, te, unet = pipe.tokenizer, pipe.text_encoder, pipe.unet

    # ---- matched conditioning: the official code embeds the raw concept strings
    import importlib.util as iu
    spec = iu.spec_from_file_location("euce", f"{P}/scripts/erase_uce.py")
    euce = iu.module_from_spec(spec); spec.loader.exec_module(euce)
    erase_prompts = list(euce.FORGET_PROMPTS.get(a.concept, [a.concept]))
    preserve_prompts = list(euce.PRESERVE)

    def embed(prompts):
        with torch.no_grad():
            ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            return te(ids)[0].reshape(-1, 768)          # [n*77, 768]

    c_e = embed(erase_prompts)                 # [Ne, 768]
    c_p = embed(preserve_prompts)              # [Np, 768]
    c_g = embed([a.guide] * len(erase_prompts))

    # ---- the cross-attention K/V projections both codebases edit
    mods = [(n, m) for n, m in unet.named_modules()
            if n.endswith("attn2.to_k") or n.endswith("attn2.to_v")]
    print(f"[r4] comparing {len(mods)} attn2 K/V projections", flush=True)

    rows, worst = [], 0.0
    for name, m in mods:
        W = m.weight.data.float()                               # [out, 768]
        V_star = (c_g @ W.T)                                    # guide outputs  [Ne, out]
        V_pres = (c_p @ W.T)                                    # preserve outputs [Np, out]

        # --- OFFICIAL form, transposed to our row convention
        mat1 = a.lamb * W + a.erase_scale * (V_star.T @ c_e) + a.preserve_scale * (V_pres.T @ c_p)
        mat2 = (a.lamb * torch.eye(768, device=dev)
                + a.erase_scale * (c_e.T @ c_e) + a.preserve_scale * (c_p.T @ c_p))
        W_off = mat1 @ torch.inverse(mat2)

        # --- OUR form, exactly as erase_uce.py writes it (note: no erase_scale)
        num = V_star.T @ c_e + a.preserve_scale * (V_pres.T @ c_p) + a.lamb * W
        den = c_e.T @ c_e + a.preserve_scale * (c_p.T @ c_p) + a.lamb * torch.eye(768, device=dev)
        W_ours = num @ torch.inverse(den)

        rel = float((W_off - W_ours).norm() / max(float(W_off.norm()), 1e-30))
        worst = max(worst, rel)
        rows.append({"module": name, "rel_err": rel,
                     "norm_official": float(W_off.norm()), "norm_ours": float(W_ours.norm()),
                     "norm_W_old": float(W.norm())})

    # --- does the discrepancy vanish at erase_scale == 1 (candidate (a))?
    equal_at_1 = worst < 1e-5
    rec = {
        "study": "r4_uce_official_vs_port",
        "upstream": "unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e "
                    "trainscripts/uce_sd_erase.py",
        "ours": "scripts/erase_uce.py",
        "config": {"concept": a.concept, "guide": a.guide, "lamb": a.lamb,
                   "erase_scale": a.erase_scale, "preserve_scale": a.preserve_scale,
                   "n_erase_prompts": len(erase_prompts), "n_preserve_prompts": len(preserve_prompts)},
        "n_modules": len(mods),
        "max_relative_error": worst,
        "mean_relative_error": sum(r["rel_err"] for r in rows) / max(len(rows), 1),
        "numerically_equivalent_at_this_config": equal_at_1,
        "default_mismatch_note": "official --lamb default is 0.5; erase_uce.py default is 0.1. "
                                 "Every historical arm of ours used 0.1, so 'UCE at defaults' "
                                 "denotes different edits in the two codebases. Separately, "
                                 "ERRATUM_uce_lambda_scale.md measured lambda inert below ~1e3 "
                                 "against c^T c's max eigenvalue of 85,850.",
        "erase_scale_note": "our port carries no erase_scale coefficient; it is equivalent to the "
                            "official form only when erase_scale == 1 (the official default). "
                            "This run tests that directly.",
        "per_module": rows[:8],
    }
    json.dump(rec, open(f"{OUT}/r4_uce_official_vs_port.json", "w"), indent=1)
    print(f"[r4] max rel err = {worst:.3e}  mean = {rec['mean_relative_error']:.3e}  "
          f"equivalent={equal_at_1}", flush=True)


if __name__ == "__main__":
    main()
