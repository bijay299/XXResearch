"""GIFT -- OUR PORT of arXiv 2507.13598 (Abdalla et al., Georgetown) to Stable Diffusion v1.4.

THIS IS A PUBLISHED METHOD, NOT OURS. No code is released: the paper carries no code link, no
appendix in the v1 PDF, and the derivation it defers to an appendix is absent. Everything here is
transcribed from the paper's Section 3 and Algorithm 1. Port decisions are pre-declared in
docs/PREREG_H81_gift_port.md. It is labelled a port exactly as our UCE, ESD, FRP and
Meta-Unlearning re-implementations are.

Algorithm 1, verbatim structure -- note it ALTERNATES, it does not sum:

    for each training iteration:
        if inner-loop step:                              # lower level: prior preservation
            theta  <- theta  - a_inner * grad_theta L_prior(x_s, c_s; theta)      (Eq. 9)
        else:                                            # upper level: immunization
            L_max   <- -E_{t,eps}[ ||eps_theta(x_m, c_m, t) - eps||^2 ]           (Eq. 5)
            collect intermediate activations z_m^(j)
            eps_m^(j) ~ N( mean(z_m^(j)), var(z_m^(j)) )                          (per-sample)
            L_noise <- sum_j MSE( z_m^(j), eps_m^(j) )                            (Eq. 7)
            L_imm   <- L_max + beta * L_noise                                     (Eq. 8)
            psi    <- psi    - a_outer * grad_psi L_imm

THE SECOND-ORDER TERM IS EMERGENT, NOT COMPUTED. Section 3.2 Taylor-expands the alternation and
shows the net update carries + a_P a_I grad^2 L_I(psi) grad_psi L_P(theta), the directional
curvature the paper calls "crucial". It arises FROM THE ALTERNATION; Algorithm 1 never forms a
Hessian. So this port is faithful on exactly the point where our Meta-Unlearning port
(scripts/meta_unlearn.py) was not -- that one dropped the second-order term via a first-order
approximation, which FINDING56 section 4 records as a live alternative explanation for its failure.

Run: python scripts/erase_gift.py --concept church --beta 1.0 --label gift_church_b1
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def cross_attn_params(unet):
    """psi: SD's text-conditioned cross-attention projections.

    The paper says "the subset of parameters corresponding to cross-attention layers" and cites
    prompt-to-prompt for concepts living there. In SD that is attn2 (attn1 is self-attention and
    carries no text conditioning). Returns the named parameters so the edit surface is auditable
    rather than asserted -- and it is deliberately the same surface UCE edits, which makes the
    comparison against our UCE arm meaningful.
    """
    keys, params = [], []
    for n, p in unet.named_parameters():
        if ".attn2." in n and any(k in n for k in ("to_q", "to_k", "to_v", "to_out.0")):
            keys.append(n)
            params.append(p)
    return keys, params


def transformer_blocks(unet):
    """z^(j): the BasicTransformerBlock outputs. The paper says "the j-th layer of the U-Net"
    without specifying granularity; block outputs are the natural reading and keep hooks tractable.
    Declared as a port decision in PREREG_H81 section 3."""
    from diffusers.models.attention import BasicTransformerBlock
    return [(n, m) for n, m in unet.named_modules() if isinstance(m, BasicTransformerBlock)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="Eq.8 noising weight. NEVER ASSIGNED IN THE PAPER; PREREG_H81 S3 "
                         "pre-declares 1.0 as primary and reports the sweep regardless.")
    ap.add_argument("--a-inner", type=float, default=1e-5, help="alpha_inner, also unassigned")
    ap.add_argument("--a-outer", type=float, default=1e-5, help="alpha_outer, also unassigned")
    ap.add_argument("--cadence", type=int, default=1,
                    help="inner steps per outer step. Algorithm 1 says 'if inner loop step' and "
                         "never specifies the frequency; 1:1 is pre-declared primary.")
    ap.add_argument("--steps", type=int, default=1500, help="matches H80's budget for comparability")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                    help="SGD is the faithful default: Algorithm 1 writes raw gradient steps and "
                         "S3.2's curvature derivation only holds for plain GD. The paper does not "
                         "say which optimiser it used. adam is for sensitivity testing only.")
    ap.add_argument("--n-images", type=int, default=16,
                    help="D_M size. 16 = exactly what the ATTACKER gets, so the defence is not "
                         "advantaged by data the attack lacks.")
    ap.add_argument("--n-safe", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    # MODEL_PIN.txt is a key=value file; an earlier version of this line took split()[0], which is
    # the MODEL NAME, and passed it as the revision -- every build then died on a Hub metadata
    # fetch. Parse it the way every working script in this phase does.
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    for m in (vae, te):
        m.requires_grad_(False)
    unet.requires_grad_(True)

    # ---- data. D_M: self-generated target images (what the attacker also gets).
    #      D_S: the pools_v2 safe pool, certified disjoint from every evaluation string.
    import importlib.util as iu
    spec = iu.spec_from_file_location("undo_t2i", f"{P}/scripts/undo_t2i.py")
    undo = iu.module_from_spec(spec); spec.loader.exec_module(undo)
    m_prompts = [f"a photo of a {a.concept}"] * a.n_images
    s_prompts = (undo.RETAIN_POOL * ((a.n_safe // max(len(undo.RETAIN_POOL), 1)) + 1))[:a.n_safe]

    g = torch.Generator(device=dev).manual_seed(a.seed)

    def generate(prompts, tag, chunk=4):
        """Generate in CHUNKS. Passing all prompts at once put a 64-image batch through the VAE
        decoder's upsampling convs and OOM'd at a 16 GiB single allocation on a 40 GiB card."""
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), chunk):
                out.append(pipe(prompts[i:i + chunk], num_inference_steps=25, guidance_scale=7.5,
                                generator=g, output_type="pt").images.cpu())
                print(f"[gift] {tag}: {min(i+chunk, len(prompts))}/{len(prompts)}", flush=True)
        return torch.cat(out)

    m_imgs = generate(m_prompts, f"D_M '{a.concept}'")
    s_imgs = generate(s_prompts, "D_S safe")

    def latents_of(imgs):
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

    m_lat, s_lat = latents_of(m_imgs), latents_of(s_imgs)
    m_emb, s_emb = embed(m_prompts), embed(s_prompts)
    del m_imgs, s_imgs
    torch.cuda.empty_cache()

    keys, psi = cross_attn_params(unet)
    blocks = transformer_blocks(unet)
    n_psi = sum(p.numel() for p in psi)
    print(f"[gift] psi = {len(keys)} cross-attn tensors, {n_psi/1e6:.1f}M params "
          f"({100*n_psi/sum(p.numel() for p in unet.parameters()):.2f}% of UNet)", flush=True)
    print(f"[gift] z^(j) hooks on {len(blocks)} BasicTransformerBlocks", flush=True)

    # PLAIN SGD, deliberately, not Adam. Algorithm 1 writes raw gradient steps
    # (theta <- theta - alpha * grad), and Section 3.2's Taylor expansion -- the whole argument for
    # the "crucial" second-order curvature term -- is only valid for plain gradient descent. Two
    # Adam optimisers would also keep SEPARATE moment estimates for psi, which appears in both
    # parameter groups, silently changing the update the analysis describes. The paper does not say
    # which optimiser it used; SGD is what its equations say, so SGD is the faithful choice and
    # --optimizer adam is offered only to test sensitivity.
    mk_opt = (torch.optim.Adam if a.optimizer == "adam" else torch.optim.SGD)
    opt_in = mk_opt(unet.parameters(), lr=a.a_inner)             # lower level: FULL theta (Alg 1 l.4)
    opt_out = mk_opt(psi, lr=a.a_outer)                          # upper level: psi only (Alg 1 l.12)

    # ---- representation-noising hooks
    acts = {}
    capture = {"on": False}

    def mk(name):
        def hook(_mod, _inp, out):
            if capture["on"]:
                acts[name] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in blocks]

    def diffusion_loss(lat, emb, idx):
        z0 = lat[idx]
        noise = torch.randn(z0.shape, device=dev, generator=g)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                          generator=g).long()
        zt = sched.add_noise(z0, noise, t)
        pred = unet(zt, t, encoder_hidden_states=emb[idx]).sample
        return F.mse_loss(pred, noise)

    def noising_loss():
        """Eq. 7: push each activation toward a random draw with ITS OWN first two moments.
        The target is detached -- it is a sample, not a function of theta -- so the gradient flows
        only through z. This destroys structure while keeping scale, which is why the paper says it
        prevents parameters from diverging from pre-training."""
        tot = 0.0
        for n in acts:
            z = acts[n]
            zf = z.detach().reshape(z.shape[0], -1)
            mu = zf.mean(1).view(-1, *([1] * (z.dim() - 1)))
            sd = zf.std(1).view(-1, *([1] * (z.dim() - 1)))
            eps = (mu + sd * torch.randn(z.shape, device=z.device, generator=g)).detach()
            tot = tot + F.mse_loss(z, eps)
        return tot

    hist = []
    nm, ns = m_lat.shape[0], s_lat.shape[0]
    for step in range(a.steps):
        # ---------- Algorithm 1 lines 2-4: inner / lower-level, prior preservation on FULL theta
        for _ in range(a.cadence):
            idx = torch.randint(0, ns, (2,), device=dev, generator=g)
            opt_in.zero_grad(set_to_none=True)
            l_prior = diffusion_loss(s_lat, s_emb, idx)
            l_prior.backward()
            opt_in.step()

        # ---------- Algorithm 1 lines 5-12: outer / upper-level, immunization on psi ONLY
        idx = torch.randint(0, nm, (2,), device=dev, generator=g)
        opt_out.zero_grad(set_to_none=True)
        acts.clear(); capture["on"] = True
        l_diff = diffusion_loss(m_lat, m_emb, idx)
        capture["on"] = False
        l_max = -l_diff                                      # Eq. 5
        l_noise = noising_loss()                             # Eq. 7
        l_imm = l_max + a.beta * l_noise                     # Eq. 8
        l_imm.backward()
        # only psi carries an update here: opt_out holds psi alone, and the full-theta grads this
        # backward also wrote are cleared by opt_in.zero_grad on the next inner step.
        opt_out.step()
        acts.clear()

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "L_prior": float(l_prior.detach()),
                   "L_diff_malicious": float(l_diff.detach()),
                   "L_max": float(l_max.detach()), "L_noise": float(l_noise.detach()),
                   "L_immunize": float(l_imm.detach())}
            hist.append(rec)
            print(f"[gift] step {step:5d} L_prior={rec['L_prior']:.5f} "
                  f"L_diff(mal)={rec['L_diff_malicious']:.5f} L_noise={rec['L_noise']:.5f} "
                  f"L_imm={rec['L_immunize']:.5f}", flush=True)

    for h in handles:
        h.remove()

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    rec = {"study": "gift_port", "prereg": "PREREG_H81_gift_port.md",
           "paper": "arXiv 2507.13598", "is_our_port": True,
           "code_released": False,
           "label": a.label, "concept": a.concept,
           "hparams": vars(a),
           "unassigned_in_paper": ["beta", "a_inner", "a_outer", "cadence"],
           "psi_keys": keys, "n_psi_params": n_psi, "n_z_hooks": len(blocks),
           "history": hist, "minutes": (time.time() - t0) / 60}
    json.dump(rec, open(f"{OUT}/gift_{a.label}.json", "w"), indent=1)
    print(f"[gift] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
