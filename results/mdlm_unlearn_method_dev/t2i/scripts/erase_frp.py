"""Forget-Retain Pruning (FRP) — OUR PORT of arXiv 2608.25429 Algorithm 1 to Stable Diffusion.

THIS IS A PUBLISHED METHOD, NOT OURS. The authors' repository (github.com/Yi1-Chen/FRAG) holds
only a README -- "Code release is in preparation" -- so this is transcribed from the paper's
Algorithm 1, Eq. (3)-(4), Eq. (6) and Table 4, and is labelled a port exactly as our UCE and ESD
re-implementations are. It is not a run of the authors' code and not a reproduction of their
numbers. Port decisions are pre-declared in docs/PREREG_H66_frp_port.md.

Algorithm 1, verbatim structure:
    for each target module with W in R^{d_o x d_i}:
        x_f, x_r <- input-channel norms on D_f, D_r          (Wanda-style calibration)
        F_ij <- |W_ij| * x_f,j / (x_r,j + eps)
        R_ij <- |W_ij| * x_r,j / (x_f,j + eps)
        for each output row i:
            S_ij <- rank_j(F_ij) - beta*rank_j(R_ij) + lambda*rank_j(|W_ij|)
            P_i  <- TopK(S_i,:, floor(rho * d_i))
            W_{i,P_i} <- 0

Paper settings (Table 4): rho = 3% MLP-only, beta = 0.05, damage mode = zero out, calibration
400 seq x 256 tok, seed 42. LAMBDA IS NEVER ASSIGNED A VALUE IN THE PAPER -- see PREREG_H66 S3(c);
the primary is pre-declared at 0.05 and the full sweep is always reported.

Run: python scripts/erase_frp.py --concept church --rho 0.03 --beta 0.05 --lmbda 0.05 \
         --label frp_church_r3
"""
import argparse, json, os, time
import numpy as np
import torch

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
OUT = "/data/bijaypandey/t2i_unlearn/outputs"
EPS = 1e-8


def target_linears(unet):
    """The paper's scope is MLP only. The SD UNet analogue is the feed-forward linears inside each
    BasicTransformerBlock: ff.net.0.proj (GEGLU input projection) and ff.net.2 (output projection).
    32 layers, 148.7M params = 17.3% of the UNet. Attention projections are EXCLUDED, matching the
    paper's MLP-only setting -- and keeping FRP off UCE's cross-attention K/V edit surface."""
    return {n: m for n, m in unet.named_modules()
            if isinstance(m, torch.nn.Linear) and ".ff.net." in n}


@torch.no_grad()
def input_channel_norms(pipe, prompts, layers, timesteps, dev, gen_seed=42):
    """Wanda-style calibration: accumulate sum of squares of each INPUT CHANNEL's activations over
    the calibration set, then take the root. Returns {layer_name: tensor(d_in)} and the position
    count actually observed, which is reported rather than assumed."""
    acc = {n: torch.zeros(m.in_features, device=dev, dtype=torch.float64) for n, m in layers.items()}
    npos = {n: 0 for n in layers}
    handles = []

    def mk(n):
        def hook(mod, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).double()     # (positions, d_in)
            acc[n] += (x * x).sum(0)
            npos[n] += x.shape[0]
        return hook

    for n, m in layers.items():
        handles.append(m.register_forward_hook(mk(n)))

    unet, te, tok, sch, vae = pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.scheduler, pipe.vae
    g = torch.Generator(device=dev).manual_seed(gen_seed)
    try:
        for pr in prompts:
            ids = tok([pr], padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            emb = te(ids)[0]
            for t in timesteps:
                z = torch.randn(1, 4, 64, 64, device=dev, dtype=unet.dtype, generator=g)
                ts = torch.tensor([int(t)], device=dev).long()
                unet(z, ts, encoder_hidden_states=emb)
    finally:
        for h in handles:
            h.remove()
    return {n: acc[n].sqrt().float() for n in acc}, npos


def rank_within_rows(x):
    """rank_j(.) from Eq. (6): rank entries within each output row, larger value -> larger rank.
    Average ranks are NOT used; ties get distinct ranks by argsort order, as a TopK on the score
    is all that is consumed downstream."""
    idx = x.argsort(dim=1)
    r = torch.empty_like(x)
    ar = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
    r.scatter_(1, idx, ar.expand_as(x))
    return r


def frp_prune(layers, xf, xr, rho, beta, lmbda):
    """Algorithm 1 lines 3-7. Returns a per-layer audit so the edit is inspectable, not asserted."""
    audit = []
    for n, m in layers.items():
        W = m.weight.data
        absW = W.abs().float()
        f_j, r_j = xf[n].to(W.device).float(), xr[n].to(W.device).float()
        F = absW * (f_j / (r_j + EPS)).unsqueeze(0)      # Eq. (3)
        R = absW * (r_j / (f_j + EPS)).unsqueeze(0)      # Eq. (4)
        S = rank_within_rows(F) - beta * rank_within_rows(R) + lmbda * rank_within_rows(absW)
        k = int(rho * W.shape[1])                        # floor(rho * d_i)
        if k < 1:
            audit.append({"layer": n, "d_in": W.shape[1], "k": 0, "skipped": "rho*d_i < 1"})
            continue
        P = S.topk(k, dim=1).indices
        before = float(W.norm())
        mask = torch.zeros_like(W, dtype=torch.bool)
        mask.scatter_(1, P, True)
        pruned_mass = float(W[mask].abs().sum())
        W[mask] = 0.0
        audit.append({"layer": n, "d_in": int(W.shape[1]), "d_out": int(W.shape[0]), "k": k,
                      "frac_pruned": k / W.shape[1],
                      "weight_norm_before": before, "weight_norm_after": float(W.norm()),
                      "pruned_abs_mass": pruned_mass})
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--rho", type=float, default=0.03, help="paper Table 4: 3% MLP-only")
    ap.add_argument("--beta", type=float, default=0.05, help="paper Table 4: 0.05")
    ap.add_argument("--lmbda", type=float, default=0.05,
                    help="NEVER ASSIGNED IN THE PAPER; PREREG_H66 S3(c) pre-declares 0.05")
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--n-timesteps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="paper Table 4: 42")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    # pools come from undo_t2i recipe v2, which verify_splits.py certifies disjoint from the
    # evaluation strings -- so FRP's calibration is out-of-sample w.r.t. every reported metric.
    import importlib.util as iu
    spec = iu.spec_from_file_location("ut", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "undo_t2i.py"))
    ut = iu.module_from_spec(spec); spec.loader.exec_module(ut)
    assert a.concept in ut.FORGET_POOL, f"no forget pool for {a.concept}"
    D_f = list(ut.FORGET_POOL[a.concept])
    kw = a.concept.split("_")[0].lower()
    D_r = [q for q in ut.RETAIN_POOL if kw not in q.lower()][:a.n_retain]
    assert len(D_r) >= 8, "retain calibration pool too small"

    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet.eval()
    layers = target_linears(unet)
    n_par = sum(m.weight.numel() for m in layers.values())
    print(f"[frp] scope: {len(layers)} FF linears, {n_par/1e6:.1f}M params "
          f"({100*n_par/sum(p.numel() for p in unet.parameters()):.1f}% of UNet)", flush=True)
    print(f"[frp] D_f = {len(D_f)} prompts, D_r = {len(D_r)} prompts, "
          f"{a.n_timesteps} timesteps each", flush=True)

    ts_grid = np.linspace(0, sch_max := pipe.scheduler.config.num_train_timesteps - 1,
                          a.n_timesteps).astype(int).tolist()
    xf, nf = input_channel_norms(pipe, D_f, layers, ts_grid, dev, a.seed)
    xr, nr = input_channel_norms(pipe, D_r, layers, ts_grid, dev, a.seed + 1)
    pos_f, pos_r = min(nf.values()), min(nr.values())
    print(f"[frp] calibration positions per layer: forget={pos_f} retain={pos_r} "
          f"(paper's budget was 400x256 = 102400 per side)", flush=True)

    audit = frp_prune(layers, xf, xr, a.rho, a.beta, a.lmbda)
    kept = [x for x in audit if x.get("k", 0) > 0]
    tot_pruned = sum(x["k"] * x["d_out"] for x in kept)
    print(f"[frp] pruned {tot_pruned/1e6:.2f}M of {n_par/1e6:.1f}M FF weights "
          f"({100*tot_pruned/n_par:.2f}%) across {len(kept)} layers", flush=True)

    out = a.out or f"/data/bijaypandey/t2i_unlearn/adapters/{a.label}"
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{out}/unet.pt")

    rec = {"study": "t2i_frp_port", "method": "FRP (Forget-Retain Pruning)",
           "source": "arXiv 2608.25429 Algorithm 1, Eq. (3)-(4), Eq. (6), Table 4",
           "provenance": "OUR PORT from the paper. The authors' repo holds only a README "
                         "('Code release is in preparation'), so this is NOT their code and NOT a "
                         "reproduction of their numbers. See ERRATUM_r2 C19 and PREREG_H66.",
           "label": a.label, "hparams": vars(a),
           "lambda_note": "lambda is NEVER assigned a value anywhere in the paper; 0.05 is "
                          "pre-declared in PREREG_H66 S3(c) and the full sweep is reported.",
           "scope": {"n_layers": len(layers), "n_params": n_par,
                     "frac_of_unet": n_par / sum(p.numel() for p in unet.parameters()),
                     "note": "MLP-only per the paper; SD analogue is ff.net.0.proj and ff.net.2. "
                             "Attention projections excluded."},
           "calibration": {"D_f": D_f, "n_D_r": len(D_r), "D_r_sample": D_r[:4],
                           "timesteps": ts_grid, "positions_forget": pos_f,
                           "positions_retain": pos_r, "pool_recipe": ut.RECIPE},
           "pruning": {"total_weights_zeroed": tot_pruned,
                       "frac_of_scope": tot_pruned / n_par, "per_layer": audit},
           "out": out, "minutes": (time.time() - t0) / 60}
    json.dump(rec, open(f"{OUT}/frp_{a.label}.json", "w"), indent=1)
    print(f"[frp] {a.label}: wrote {out}/unet.pt in {rec['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
