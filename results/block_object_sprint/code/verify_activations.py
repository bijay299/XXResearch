#!/usr/bin/env python3
"""Correctness check on the collected activation pairs, before spending GPU time training.

Verifies, for BOTH transcoders:
  * shapes and counts match what paper E.1 prescribes;
  * all values finite;
  * the stored TARGET really is the ORIGINAL FLUX module applied to the stored INPUT
    (recomputed with the live module on a random sample) -- this is the property the
    transcoder is trained to imitate, so getting it wrong would invalidate everything.
"""
import os, sys, json, torch, random

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
REPO = f"{SPRINT}/code/block-concept-removal"
sys.path.insert(0, REPO); os.chdir(REPO)
from diffusers import FluxPipeline

A = f"{SPRINT}/artifacts/activations"
FN = "activations_flux_style.pt"
rep = {}


def load(layer, which):
    return torch.load(f"{A}/{layer}_{which}/{FN}", map_location="cpu", weights_only=False)


dev = "cuda:0"
pipe = FluxPipeline.from_pretrained(os.environ["FLUX_PATH"], torch_dtype=torch.bfloat16).to(dev)
mods = {"context_embedder": pipe.transformer.context_embedder,
        "text_embedder": pipe.transformer.time_text_embed.text_embedder}
expect = {"context_embedder": (4096, 3072), "text_embedder": (768, 3072)}

for layer in ("context_embedder", "text_embedder"):
    x, y = load(layer, "in"), load(layer, "out")
    r = {"x_shape": list(x.shape), "y_shape": list(y.shape),
         "x_dtype": str(x.dtype), "y_dtype": str(y.dtype),
         "n_pairs": int(x.shape[0]),
         "x_finite": bool(torch.isfinite(x.float()).all()),
         "y_finite": bool(torch.isfinite(y.float()).all()),
         "x_std": float(x.float().std()), "y_std": float(y.float().std())}
    assert x.shape[0] == y.shape[0], f"{layer}: pair count mismatch"
    assert (x.shape[1], y.shape[1]) == expect[layer], f"{layer}: dim mismatch"

    # recompute the target with the live original module on a random sample
    g = torch.Generator().manual_seed(7)
    idx = torch.randperm(x.shape[0], generator=g)[:4096]
    with torch.no_grad():
        xs = x[idx].to(dev)
        yr = mods[layer](xs)
    ys = y[idx].to(dev)
    d = (yr.float() - ys.float()).abs()
    denom = ys.float().abs().mean().clamp(min=1e-12)
    r["recompute_check"] = {
        "n_sampled": int(idx.numel()),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "rel_mean_abs_diff": float(d.mean() / denom),
        "bit_identical": bool(torch.equal(yr, ys)),
    }
    rep[layer] = r
    print(f"{layer}: n={r['n_pairs']}  x{r['x_shape'][1]}->y{r['y_shape'][1]}  "
          f"finite={r['x_finite'] and r['y_finite']}  "
          f"recompute bit-identical={r['recompute_check']['bit_identical']} "
          f"maxdiff={r['recompute_check']['max_abs_diff']:.3e} "
          f"relmean={r['recompute_check']['rel_mean_abs_diff']:.3e}")

# E.1 expectations
rep["expectations"] = {
    "context_prompts": 17600, "context_rule": "80 prompts x 20 objects x (10 styles + no-style)",
    "pooled_pairs_expected": 330000, "pooled_rule": "30000 COCO prompts x (10 styles + no-style)",
    "pooled_pairs_actual": rep["text_embedder"]["n_pairs"],
    "pooled_matches_E1": rep["text_embedder"]["n_pairs"] == 330000,
    "context_pairs_actual_token_level": rep["context_embedder"]["n_pairs"],
    "context_note": ("context pairs are per VALID T5 TOKEN (the release masks padding), so the "
                     "count is ~15x the 17,600 prompts, not equal to it"),
}
print(json.dumps(rep["expectations"], indent=1))
json.dump(rep, open(f"{SPRINT}/artifacts/activations/verify_activations.json", "w"), indent=1)
print("wrote verify_activations.json")
