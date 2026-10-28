"""Build a concept task vector for SD-1.4, after Pham et al. (arXiv:2404.03631).

tau = theta_finetuned - theta_base, where the fine-tune is on images OF the concept, conditioned on
the EMPTY prompt. Pham's argument for empty guidance: it makes the erasure unconditional rather than
tied to any particular user prompt, which is why their method survives prompt attacks.

Erasure is then theta_base - alpha * tau.

This script only BUILDS tau and measures how it is distributed. It does not erase; U46 applies it.

WHY THE MEASUREMENT MATTERS. Task arithmetic (Pham), Unmerge and PerTA all assume the forget signal
is CONCENTRATED -- a low-rank subspace, a few dominant directions, identifiable per-parameter
importance. This project has already tested the T2I version of that assumption once, in a different
form: Fisher-weighted selective damage LOST TO A RANDOM CONTROL, i.e. no separable forget-importance
signal was found in this UNet. So the concentration of tau is not something to assume here.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

ROOT = "/data/bijaypandey/t2i_unlearn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--prompt-mode", choices=["empty", "concept"], default="empty",
                    help="text condition during the fine-tune. Pham et al. use the EMPTY prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    import importlib.util as iu
    sp = iu.spec_from_file_location("atk3", os.path.join(ROOT, "scripts/attack_relearn_v3.py"))
    atk = iu.module_from_spec(sp); sp.loader.exec_module(atk)
    SPLIT = atk.SPLIT
    T = SPLIT[a.concept]["T"]          # training prompts for generating concept images

    # 1. concept images from the BASE model
    base = atk.build_pipe(torch.float16, None, dev)
    base.set_progress_bar_config(disable=True)
    imgs = []
    for i in range(a.n_images):
        g = torch.Generator(device=dev).manual_seed(70_000 + i)
        imgs.append(base([T[i % len(T)]], num_inference_steps=25, guidance_scale=7.5,
                         generator=g).images[0])
    print(f"[tv] {len(imgs)} concept images from base over {len(T)} prompts", flush=True)
    del base; torch.cuda.empty_cache()

    # 2. fine-tune the UNet on them under the chosen text condition
    pipe = atk.build_pipe(torch.float32, None, dev)
    unet, vae, te, tok, sch = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    te.requires_grad_(False); vae.requires_grad_(False)
    theta0 = {k: v.detach().clone() for k, v in unet.state_dict().items() if v.dtype.is_floating_point}

    with torch.no_grad():
        lat = []
        for im in imgs:
            x = torch.from_numpy(
                (torch.frombuffer(im.tobytes(), dtype=torch.uint8)
                 .reshape(im.size[1], im.size[0], 3).numpy())).permute(2, 0, 1)[None].float()
            lat.append(vae.encode((x.to(dev) / 127.5 - 1.0)).latent_dist.sample()
                       * vae.config.scaling_factor)
        lat = torch.cat(lat)
        cap = "" if a.prompt_mode == "empty" else f"a photo of a {a.concept.replace('_',' ')}"
        ids = tok([cap], padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        emb1 = te(ids)[0]

    unet.train().requires_grad_(True)
    opt = torch.optim.Adam(unet.parameters(), lr=a.lr)
    t0 = time.time(); losses = []
    for step in range(a.steps):
        i = torch.randint(0, len(lat), (1,), device=dev)
        z0 = lat[i]
        noise = torch.randn_like(z0)
        ts = torch.randint(0, sch.config.num_train_timesteps, (1,), device=dev).long()
        loss = F.mse_loss(unet(sch.add_noise(z0, noise, ts), ts, encoder_hidden_states=emb1).sample,
                          noise)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0:
            print(f"[tv] step {step+1}/{a.steps} loss={sum(losses[-100:])/100:.5f} "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)

    # 3. tau, and how it is distributed
    unet.eval()
    sd = unet.state_dict()
    tau, per_layer = {}, []
    tot_sq = 0.0
    for k, ref in theta0.items():
        if k in sd and sd[k].shape == ref.shape:
            d = (sd[k].detach() - ref)
            tau[k] = d.half()
            n = float(d.pow(2).sum()); tot_sq += n
            per_layer.append({"name": k, "sq_norm": n, "numel": d.numel(),
                              "base_sq_norm": float(ref.pow(2).sum())})
    for p in per_layer:
        p["frac_of_tau"] = p["sq_norm"] / tot_sq if tot_sq else 0.0
        p["rel_to_base"] = (p["sq_norm"] / p["base_sq_norm"]) ** 0.5 if p["base_sq_norm"] else 0.0
    per_layer.sort(key=lambda r: -r["frac_of_tau"])

    cum, k50, k90 = 0.0, None, None
    for i, p in enumerate(per_layer, 1):
        cum += p["frac_of_tau"]
        if k50 is None and cum >= 0.50: k50 = i
        if k90 is None and cum >= 0.90: k90 = i

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(tau, a.out)
    rec = {"study": "task_vector", "label": a.label, "concept": a.concept,
           "hparams": vars(a), "out": a.out,
           "n_tensors": len(tau), "tau_l2": tot_sq ** 0.5,
           "rel_l2_vs_base": (tot_sq / sum(p["base_sq_norm"] for p in per_layer)) ** 0.5,
           "loss_first50": sum(losses[:50]) / 50, "loss_last50": sum(losses[-50:]) / 50,
           "layers_for_50pct_of_tau": k50, "layers_for_90pct_of_tau": k90,
           "n_layers": len(per_layer),
           "top20_layers": per_layer[:20],
           "minutes": (time.time() - t0) / 60}
    with open(os.path.join(ROOT, f"outputs/tv_{a.label}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[tv] tau saved: {len(tau)} tensors, L2={rec['tau_l2']:.4f}, "
          f"rel to base={rec['rel_l2_vs_base']:.6f}", flush=True)
    print(f"[tv] CONCENTRATION: {k50} of {len(per_layer)} tensors hold 50% of tau's energy; "
          f"{k90} hold 90%", flush=True)


if __name__ == "__main__":
    main()
