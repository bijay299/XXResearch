"""Meta-Unlearning (arXiv 2410.12777) — OUR PORT. Pre-registered in docs/PREREG_H80.

    theta' = theta - alpha * grad L_forget(theta)        # one simulated malicious-finetune step
    L      = L_erase(theta) + L_retain(theta) + lam * ( - L_benign(theta') )

The third term is the paper's contribution: benign concepts related to the target must become BAD
after a simulated attack step, so that relearning triggers self-destruction. The first two keep the
model erased and useful WHEN USED AS IS.

First-order approximation (standard in MAML variants): the gradient of L_benign(theta') is taken at
theta' and applied to theta, without the second-order term. This is a STATED deviation and a live
alternative explanation if the method fails here.
"""
import argparse, copy, json, os, time
import torch
import torch.nn.functional as F

# CROSS-MODEL OVERRIDE (PREREG_H85 / FINDING64). Defaults reproduce the pinned SD-1.4 exactly when
# BASE_MODEL/BASE_REV are unset, so no existing result changes.
import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None


OUT = "/data/bijaypandey/t2i_unlearn/outputs"


def _mods():
    import importlib.util as iu
    d = os.path.dirname(os.path.abspath(__file__))
    out = []
    for n in ("attack_relearn", "undo_t2i"):
        spec = iu.spec_from_file_location(f"mu_{n}", os.path.join(d, f"{n}.py"))
        m = iu.module_from_spec(spec); spec.loader.exec_module(m); out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the ERASED unet to meta-unlearn")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--inner-alpha", type=float, default=1e-5,
                    help="simulated malicious-finetuning step; matches the attack's own lr")
    ap.add_argument("--lam-meta", type=float, default=1.0)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    ar, ut = _mods()
    t0 = time.time()

    # prompt pools, recipe v2 (verify_splits CLEAN)
    kw = a.concept.split("_")[0].lower()
    forget_p = list(ut.FORGET_POOL[a.concept])
    benign_p = [q for q in ut.RETAIN_POOL if kw not in q.lower()][:24]
    assert len(benign_p) >= 8

    pipe = ar.build_pipe(torch.float32, a.teacher, dev)
    unet, vae, te, tok, sch = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    te.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(True)
    teacher = copy.deepcopy(unet).eval().requires_grad_(False)

    # data: the erased model's own generations. No oracle, no concept images from outside.
    gen = ar.build_pipe(torch.float16, a.teacher, dev)
    def make(prompts, n, base_seed):
        lat, emb = [], []
        with torch.no_grad():
            for i in range(n):
                pr = prompts[i % len(prompts)]
                g = torch.Generator(device=dev).manual_seed(base_seed + i)
                im = gen([pr], num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
                x = torch.from_numpy((torch.frombuffer(im.tobytes(), dtype=torch.uint8)
                     .reshape(im.size[1], im.size[0], 3).numpy())).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1.0
                lat.append(vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)
                ids = tok([pr], padding="max_length", max_length=77, truncation=True,
                          return_tensors="pt").input_ids.to(dev)
                emb.append(te(ids)[0])
        return torch.cat(lat), torch.cat(emb)
    f_lat, f_emb = make(forget_p, a.n_images, 30_000)      # target-concept images (attack surrogate)
    b_lat, b_emb = make(benign_p, a.n_images, 40_000)      # benign related concepts
    del gen; torch.cuda.empty_cache()
    print(f"[meta] data: {len(f_lat)} forget, {len(b_lat)} benign", flush=True)

    def dmloss(net, lat, emb, idx, g):
        n = torch.randn(lat[idx].shape, device=dev, dtype=lat.dtype, generator=g)
        ts = torch.randint(0, sch.config.num_train_timesteps, (len(idx),), device=dev,
                           generator=g).long()
        return F.mse_loss(net(sch.add_noise(lat[idx], n, ts), ts,
                              encoder_hidden_states=emb[idx]).sample, n)

    def distil(net, lat, emb, idx, g):
        n = torch.randn(lat[idx].shape, device=dev, dtype=lat.dtype, generator=g)
        ts = torch.randint(0, sch.config.num_train_timesteps, (len(idx),), device=dev,
                           generator=g).long()
        zt = sch.add_noise(lat[idx], n, ts)
        with torch.no_grad():
            tgt = teacher(zt, ts, encoder_hidden_states=emb[idx]).sample
        return F.mse_loss(net(zt, ts, encoder_hidden_states=emb[idx]).sample, tgt)

    opt = torch.optim.Adam(unet.parameters(), lr=a.lr)
    g = torch.Generator(device=dev).manual_seed(a.seed + 777)
    params = [p for p in unet.parameters() if p.requires_grad]
    unet.train()
    log = []
    for step in range(a.steps):
        fi = torch.randint(0, len(f_lat), (1,), device=dev, generator=g)
        bi = torch.randint(0, len(b_lat), (1,), device=dev, generator=g)

        # ---- term 1+2 FIRST, and backward BEFORE theta is touched.
        # BUGFIX: the original order computed these at theta, mutated theta in place to theta',
        # then called .backward() -- whose graph still referenced the overwritten tensors. Torch
        # raised "one of the variables needed for gradient computation has been modified by an
        # inplace operation". The gradients must be accumulated into p.grad while theta is intact.
        opt.zero_grad(set_to_none=True)
        l_erase = distil(unet, f_lat, f_emb, fi, g)
        l_retain = distil(unet, b_lat, b_emb, bi, g)
        (l_erase + l_retain).backward()                     # p.grad now holds terms 1+2

        # ---- term 3: SIMULATED ATTACK, then benign must be BAD there (first-order).
        # autograd.grad RETURNS gradients without accumulating, so p.grad is untouched here.
        l_att = dmloss(unet, f_lat, f_emb, fi, g)
        gatt = torch.autograd.grad(l_att, params, create_graph=False, retain_graph=False)
        backup = [q.detach().clone() for q in params]
        with torch.no_grad():
            for q, gg in zip(params, gatt):
                q.sub_(a.inner_alpha * gg)                  # theta -> theta'
        l_benign_prime = dmloss(unet, b_lat, b_emb, bi, g)  # evaluated AT theta'
        gmeta = torch.autograd.grad(l_benign_prime, params)
        with torch.no_grad():
            for q, b in zip(params, backup):
                q.copy_(b)                                  # restore theta
            for q, gm in zip(params, gmeta):
                if q.grad is None:
                    q.grad = (-a.lam_meta) * gm
                else:
                    q.grad.add_(-a.lam_meta * gm)           # MAXIMISE benign loss at theta'
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "l_erase": float(l_erase), "l_retain": float(l_retain),
                   "l_benign_at_theta_prime": float(l_benign_prime)}
            log.append(rec)
            print(f"[meta] step {step:5d} erase={rec['l_erase']:.5f} "
                  f"retain={rec['l_retain']:.5f} benign@theta'={rec['l_benign_at_theta_prime']:.5f} "
                  f"{(time.time()-t0)/max(step+1,1):.2f}s/step", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "meta_unlearning_port", "source": "arXiv 2410.12777",
               "provenance": "OUR PORT from the paper; first-order meta gradient (stated "
                             "deviation). Not the authors' code.",
               "prereg": "docs/PREREG_H80_meta_unlearning.md",
               "label": a.label, "hparams": vars(a),
               "forget_pool": forget_p, "n_benign_prompts": len(benign_p),
               "pool_recipe": ut.RECIPE, "train_log": log,
               "objective": "L = distil_erase(theta) + distil_retain(theta) "
                            "- lam * L_benign(theta - alpha*grad L_forget(theta))",
               "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/meta_{a.label}.json", "w"), indent=1)
    print(f"[meta] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
