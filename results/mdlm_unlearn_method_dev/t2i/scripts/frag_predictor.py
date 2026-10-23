"""FRAG (Forget-Retain Alignment Gap) — OUR PORT of arXiv 2608.25429 Appendix A.1 to SD v1.4.

THEIR PREDICTOR, NOT OURS. Transcribed from Eq. (7)-(9) because the authors' repository holds only
a README (ERRATUM_r2 C19). Pre-registered in docs/PREREG_H67_frag_predictor_transfer.md.

    F_ij = |W0_ij| * x_f,j / (x_r,j + eps)      (Eq. 8)
    R_ij = |W0_ij| * x_r,j / (x_f,j + eps)
    D_l  = (W_u - W0)^2                          <- SQUARED, per Eq. (8)
    A_f  = sum_l <F_l, D_l> / (sqrt(sum_l ||F_l||^2) * sqrt(sum_l ||D_l||^2))   (Eq. 9, ONE
           global cosine across layers, not per-layer cosines averaged)
    FRAG = A_f - gamma * A_r,   gamma = 1, eps = 1e-6

Both activation passes use W0; W_u is read but never executed. Importance is therefore computed
ONCE on the base model and reused for every arm, which is what makes this cheap.

Runs on CPU by default so it does not consume the two-GPU budget.

Run: python scripts/frag_predictor.py --concept church --device cpu
"""
import argparse, glob, json, os, time
import numpy as np
import torch

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
OUT = "/data/bijaypandey/t2i_unlearn/outputs"
ADAPT = "/data/bijaypandey/t2i_unlearn/adapters"
EPS = 1e-6
GAMMA = 1.0


def scope_linears(unet):
    """PREREG_H67 S3(a): all nn.Linear inside BasicTransformerBlocks -- the 128 attention
    projections plus the 32 feed-forward linears. WIDER than H66's FRP scope (MLP-only), exactly
    as the two scopes differ in the paper."""
    keep = {}
    for n, m in unet.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if ".transformer_blocks." in n and (".attn" in n or ".ff.net." in n):
            keep[n] = m
    return keep


@torch.no_grad()
def channel_norms(pipe, prompts, layers, timesteps, dev, seed):
    """Eq. (7): per-input-channel l2 norms of the input activation, accumulated over the
    calibration set. Computed on W0 only."""
    acc = {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in layers.items()}
    npos = {n: 0 for n in layers}
    hs = []

    def mk(n):
        def hook(_m, inp, _o):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double().cpu()
            acc[n] += (x * x).sum(0)
            npos[n] += x.shape[0]
        return hook

    for n, m in layers.items():
        hs.append(m.register_forward_hook(mk(n)))
    unet, te, tok = pipe.unet, pipe.text_encoder, pipe.tokenizer
    g = torch.Generator(device="cpu").manual_seed(seed)
    try:
        for pi, pr in enumerate(prompts):
            ids = tok([pr], padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            emb = te(ids)[0]
            for t in timesteps:
                z = torch.randn(1, 4, 64, 64, generator=g, dtype=torch.float32).to(dev)
                unet(z, torch.tensor([int(t)], device=dev).long(), encoder_hidden_states=emb)
            print(f"[frag]   calib prompt {pi+1}/{len(prompts)}", flush=True)
    finally:
        for h in hs:
            h.remove()
    return {n: acc[n].sqrt().float() for n in acc}, npos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--n-timesteps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = a.device
    t0 = time.time()

    import importlib.util as iu
    spec = iu.spec_from_file_location("ut", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "undo_t2i.py"))
    ut = iu.module_from_spec(spec); spec.loader.exec_module(ut)
    kw = a.concept.split("_")[0].lower()
    D_f = (list(ut.FORGET_POOL[a.concept]) * 8)[:a.n_prompts]
    D_r = [q for q in ut.RETAIN_POOL if kw not in q.lower()][:a.n_prompts]

    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet.eval()
    layers = scope_linears(unet)
    print(f"[frag] scope: {len(layers)} linears inside transformer blocks "
          f"({sum(m.weight.numel() for m in layers.values())/1e6:.1f}M params)", flush=True)

    ts = np.linspace(0, pipe.scheduler.config.num_train_timesteps - 1,
                     a.n_timesteps).astype(int).tolist()
    print(f"[frag] calibrating FORGET side ({len(D_f)} prompts x {len(ts)} timesteps)", flush=True)
    xf, nf = channel_norms(pipe, D_f, layers, ts, dev, a.seed)
    print(f"[frag] calibrating RETAIN side ({len(D_r)} prompts x {len(ts)} timesteps)", flush=True)
    xr, nr = channel_norms(pipe, D_r, layers, ts, dev, a.seed + 1)
    print(f"[frag] positions per layer: forget={min(nf.values())} retain={min(nr.values())} "
          f"(paper: 128x256 = 32768 per side)", flush=True)

    # F and R on the BASE weights, once
    W0 = {n: m.weight.data.detach().cpu().float().clone() for n, m in layers.items()}
    F, R = {}, {}
    for n in layers:
        aw = W0[n].abs()
        f_j, r_j = xf[n].cpu(), xr[n].cpu()
        F[n] = aw * (f_j / (r_j + EPS)).unsqueeze(0)
        R[n] = aw * (r_j / (f_j + EPS)).unsqueeze(0)
    nF = float(sum((F[n].double() ** 2).sum() for n in F)) ** 0.5
    nR = float(sum((R[n].double() ** 2).sum() for n in R)) ** 0.5
    del pipe, unet
    print(f"[frag] base importance tensors built (||F||={nF:.4e} ||R||={nR:.4e})", flush=True)

    # score every arm with a UNet checkpoint
    rows = []
    for ck in sorted(glob.glob(f"{ADAPT}/*/unet.pt")):
        arm = os.path.basename(os.path.dirname(ck))
        try:
            sd = torch.load(ck, map_location="cpu")
            sd = sd.get("unet", sd)
        except Exception as e:
            rows.append({"arm": arm, "error": f"{type(e).__name__}: {e}"}); continue
        dot_f = dot_r = nD2 = 0.0
        missing = 0
        for n in layers:
            k = n + ".weight"
            if k not in sd:
                missing += 1; continue
            D = (sd[k].float() - W0[n]) ** 2                 # Eq. (8): SQUARED
            Dd = D.double()
            dot_f += float((F[n].double() * Dd).sum())
            dot_r += float((R[n].double() * Dd).sum())
            nD2 += float((Dd ** 2).sum())
        nD = nD2 ** 0.5
        if missing:
            rows.append({"arm": arm, "error": f"{missing} scope layers absent from checkpoint"})
            continue
        if nD <= 0:
            rows.append({"arm": arm, "unet_identical_to_base": True,
                         "note": "D == 0 exactly: this arm does not modify the UNet (e.g. a "
                                 "text-encoder erasure or an exact alpha=0 no-op). FRAG is "
                                 "UNDEFINED here, not low. Excluded per PREREG_H67 S3(c)."})
            continue
        Af, Ar = dot_f / (nF * nD), dot_r / (nR * nD)
        l2 = float(sum(((sd[n + ".weight"].float() - W0[n]).double() ** 2).sum()
                       for n in layers)) ** 0.5
        rows.append({"arm": arm, "A_f": Af, "A_r": Ar, "FRAG": Af - GAMMA * Ar,
                     "global_l2": l2, "D_norm": nD})
        print(f"[frag] {arm:34s} A_f={Af:.4f} A_r={Ar:.4f} FRAG={Af-Ar:+.4f} l2={l2:.3e}",
              flush=True)
        del sd

    rec = {"study": "frag_predictor_port", "predictor": "FRAG (Forget-Retain Alignment Gap)",
           "source": "arXiv 2608.25429 Appendix A.1, Eq. (7)-(9)",
           "provenance": "OUR PORT from the paper. Not the authors' code (repo is a README stub, "
                         "ERRATUM_r2 C19). THEIR predictor; a positive result is about their work.",
           "prereg": "docs/PREREG_H67_frag_predictor_transfer.md",
           "hparams": vars(a), "gamma": GAMMA, "eps": EPS,
           "scope": {"n_layers": len(layers), "note": "all nn.Linear inside BasicTransformerBlocks "
                                                      "(128 attention + 32 feed-forward)"},
           "calibration": {"D_f": D_f[:4], "D_r": D_r[:4], "timesteps": ts,
                           "positions_forget": min(nf.values()),
                           "positions_retain": min(nr.values()), "pool_recipe": ut.RECIPE},
           "base_norms": {"F": nF, "R": nR},
           "arms": rows,
           "n_scored": sum(1 for r in rows if "FRAG" in r),
           "n_unet_identical": sum(1 for r in rows if r.get("unet_identical_to_base")),
           "n_error": sum(1 for r in rows if "error" in r),
           "minutes": (time.time() - t0) / 60}
    json.dump(rec, open(f"{OUT}/frag_scores_{a.concept}.json", "w"), indent=1)
    print(f"\n[frag] scored {rec['n_scored']} arms; {rec['n_unet_identical']} have a "
          f"bit-identical UNet (FRAG undefined); {rec['n_error']} errors")
    print(f"[frag] wrote {OUT}/frag_scores_{a.concept}.json in {rec['minutes']:.1f} min")


if __name__ == "__main__":
    main()
