"""UTRAP -- unrolled multi-step adversarial immunization, full-UNet scope. OURS, per PREREG_H84.

    min_theta  L_retain(theta)  -  lam * L_target( A_K(theta) )

A_K(theta) = K steps of the ACTUAL full-UNet relearning attack (the one attack_relearn.py runs, on
16 self-generated target images) starting from theta. The outer objective maximises the attacker's
target loss AT THE END of its trajectory, not at theta and not after one step.

DESIGNED FROM THIS PHASE'S FAILURE MODES, not from a paper:
  * FINDING59 S0 -- IMMA's inner loop approximates the attacker with ONE step and a 3x-longer
    attacker walks past it            -> K steps, K=8 primary;
  * FINDING58 S3a / FINDING59 S3 -- GIFT and IMMA constrain 5.11% of the UNet and a 0.046% LoRA
    reverses both                     -> FULL 859.5M-parameter scope, no psi subset;
  * FINDING60 / FINDING62 -- weight-space proximity and erasure depth are both orthogonal to
    resistance                        -> no anchor term, no deeper-erasure term;
  * FINDING56 S5a -- the one positive signal in the phase was a trigger obstructing an adversary's
    OPTIMISATION                      -> that is the mechanism being scaled up here.

THE APPROXIMATION, STATED IN ADVANCE (PREREG_H84 S2): the trajectory is detached. K attacker steps
are run, grad L_target is evaluated at theta_K, and applied at theta. This is the same
first-order class my Meta-Unlearning port used (FINDING56 S4) -- applied over K steps rather than
one, which FINDING59 identified as the operative difference. It is the port-level limitation most
likely to explain a failure and it is disclosed here, not after the result.

Run: python scripts/erase_utrap.py --teacher adapters/frp_church_r3_l05/unet.pt --concept church \
         --lam 1.0 --inner-k 8 --label utrap_church_l1k8 --out adapters/utrap_church_l1k8
"""
import argparse, copy, importlib.util as iu, json, os, time
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--teacher", required=True, help="the erased unet to immunize")
    ap.add_argument("--lam", type=float, default=1.0, help="PREREG_H84 S2 primary 1.0")
    ap.add_argument("--inner-k", type=int, default=8, help="attacker steps unrolled; primary 8")
    ap.add_argument("--inner-lr", type=float, default=1e-5, help="matches the real attack's primary")
    ap.add_argument("--lr", type=float, default=1e-6, help="outer lr")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    for m in (vae, te):
        m.requires_grad_(False)

    tsd = torch.load(a.teacher, map_location="cpu"); tsd = tsd.get("unet", tsd)
    unet.load_state_dict({k: v.float() for k, v in tsd.items()})
    print(f"[utrap] teacher {a.teacher}", flush=True)

    spec = iu.spec_from_file_location("undo_t2i", f"{P}/scripts/undo_t2i.py")
    undo = iu.module_from_spec(spec); spec.loader.exec_module(undo)
    f_prompts = [f"a photo of a {a.concept}"] * a.n_images
    r_prompts = (undo.RETAIN_POOL * ((a.n_retain // max(len(undo.RETAIN_POOL), 1)) + 1))[:a.n_retain]
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def gen(prompts, tag):
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), 4):
                out.append(pipe(prompts[i:i + 4], num_inference_steps=25, guidance_scale=7.5,
                                generator=g, output_type="pt").images.cpu())
                print(f"[utrap] {tag} {min(i+4,len(prompts))}/{len(prompts)}", flush=True)
        return torch.cat(out)

    f_imgs, r_imgs = gen(f_prompts, "D_target"), gen(r_prompts, "D_retain")

    def lat(imgs):
        z = []
        with torch.no_grad():
            for i in range(0, len(imgs), 4):
                b = imgs[i:i + 4].to(dev) * 2 - 1
                z.append(vae.encode(b).latent_dist.sample(generator=g) * vae.config.scaling_factor)
        return torch.cat(z)

    def embed(prompts):
        with torch.no_grad():
            ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            return te(ids)[0]

    f_lat, r_lat = lat(f_imgs), lat(r_imgs)
    f_emb, r_emb = embed(f_prompts), embed(r_prompts)
    del f_imgs, r_imgs
    torch.cuda.empty_cache()

    unet.requires_grad_(True)
    params = list(unet.parameters())
    n_p = sum(p.numel() for p in params)
    print(f"[utrap] FULL scope: {n_p/1e6:.1f}M params (no psi subset)", flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)

    def dloss(latv, embv, idx):
        z0 = latv[idx]
        noise = torch.randn(z0.shape, device=dev, generator=g)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                          generator=g).long()
        return F.mse_loss(unet(sched.add_noise(z0, noise, t), t,
                               encoder_hidden_states=embv[idx]).sample, noise)

    hist = []
    nf, nr = f_lat.shape[0], r_lat.shape[0]
    for step in range(a.steps):
        # ---- 1. retain gradient at theta (held first so the outer step has it)
        opt.zero_grad(set_to_none=True)
        ri = torch.randint(0, nr, (a.batch,), device=dev, generator=g)
        l_retain = dloss(r_lat, r_emb, ri)
        l_retain.backward()
        g_retain = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        # ---- 2. unroll K steps of the REAL attack from theta, on a backup
        backup = [p.detach().clone() for p in params]
        inner_opt = torch.optim.SGD(params, lr=a.inner_lr)
        l_att_first = l_att_last = None
        for k in range(a.inner_k):
            inner_opt.zero_grad(set_to_none=True)
            fi = torch.randint(0, nf, (a.batch,), device=dev, generator=g)
            la = dloss(f_lat, f_emb, fi)         # the attacker MINIMISES this
            la.backward()
            inner_opt.step()
            if k == 0:
                l_att_first = float(la.detach())
            l_att_last = float(la.detach())

        # ---- 3. grad of the target loss AT THE ATTACKER'S ENDPOINT theta_K
        opt.zero_grad(set_to_none=True)
        fi = torch.randint(0, nf, (a.batch,), device=dev, generator=g)
        l_end = dloss(f_lat, f_emb, fi)
        l_end.backward()
        g_end = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        # ---- 4. restore theta and apply  grad L_retain - lam * grad L_target(theta_K)
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)
        opt.zero_grad(set_to_none=True)
        for p, gr, ge in zip(params, g_retain, g_end):
            if gr is None and ge is None:
                continue
            tot = torch.zeros_like(p)
            if gr is not None:
                tot += gr
            if ge is not None:
                tot -= a.lam * ge        # MAXIMISE the attacker's endpoint target loss
            p.grad = tot
        opt.step()
        del backup, g_retain, g_end

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "L_retain": float(l_retain.detach()),
                   "L_attack_step0": l_att_first, "L_attack_stepK": l_att_last,
                   "L_target_at_thetaK": float(l_end.detach())}
            hist.append(rec)
            print(f"[utrap] step {step:4d} L_retain={rec['L_retain']:.5f} "
                  f"attacker {l_att_first:.5f}->{l_att_last:.5f} "
                  f"L_target(thetaK)={rec['L_target_at_thetaK']:.5f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "utrap", "prereg": "PREREG_H84_unrolled_trap.md", "is_ours": True,
               "designed_from": ["FINDING59 S0 (one-step inner)", "FINDING58 S3a (5% scope)",
                                 "FINDING57 S4 (subspace redundancy)",
                                 "FINDING60/62 (proximity & depth orthogonal)",
                                 "FINDING56 S5a (optimisation obstruction)"],
               "approximation": "detached trajectory; grad L_target evaluated at theta_K and "
                                "applied at theta. First-order class, K steps instead of one. "
                                "Disclosed in PREREG_H84 S2 before any result.",
               "label": a.label, "concept": a.concept, "teacher": a.teacher,
               "scope": "full UNet", "n_params": n_p, "hparams": vars(a),
               "history": hist, "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/utrap_{a.label}.json", "w"), indent=1)
    print(f"[utrap] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
