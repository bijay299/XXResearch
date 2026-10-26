"""WSR-anchor -- OUR OWN METHOD: erasure regularised TOWARD the original weights.

!! DIRECTION ERRATUM, RE-CONFIRMED (ERRATUM_wsr_direction.md; independently re-verified by the
Agent-3 fidelity audit 2026-10-26 against the authors' released code). This file was described as
"OUR implementation of the METHOD CLASS proposed by 'From Dormant to Deleted' (arXiv 2505.22310)".
IT IMPLEMENTS THE OPPOSITE DIRECTION. The reference reads:

    upstream/vision_relearning/vision_unlearner.py:366-373
        param_diff       = flatten_param_dict(pretrained.parameters()).detach()
                           - flatten_param_dict(model.parameters())
        weight_dist_loss = - torch.sqrt(torch.mean(torch.square(param_diff)) + eps)
        loss             = unlearning_alpha * weight_dist_loss

i.e. a NEGATED root-mean-square distance, minimised => distance MAXIMISED. The term below is
`+lam_reg * ||theta - theta_base||^2 / ||theta_base||^2`, minimised => distance MINIMISED. Those
are different algorithms, not one algorithm with a sign typo (four further differences: RMS vs sum
of squares; no division by ||theta_base||; the reference's distance term REPLACES the forget
objective rather than being a third term; the reference's retain term is unscaled).

CLASSIFICATION: OUR OWN DESIGN. This file is a legitimate method -- anchored erasure -- and keeps
its FINDING61/62 results under the name `WSR-anchor (ours)`. What is withdrawn is any claim that it
tests, reproduces or discharges arXiv 2505.22310. NO NUMERICAL CHANGE is made here; the faithful
direction lives in `scripts/erase_wdr_v2.py` and `objectives.wdr_forget_loss`.

Historical description retained below for the record:
  "OUR implementation of the METHOD CLASS proposed by 'From Dormant to Deleted', adapted to T2I
   concept erasure" -- WITHDRAWN, see above.

DECLARED UP FRONT, because it matters for how this is read: that paper's experiments are
EXAMPLE-LEVEL unlearning in VISION CLASSIFIERS. It is not portable as-is (FINDING60 S1). What IS
portable is its proposed mechanism -- unlearn while REGULARISING TOWARD THE ORIGINAL WEIGHTS, on the
theory that staying inside the original basin is what confers tamper-resistance. This is therefore
an ADAPTATION OF THEIR IDEA, not a reproduction of their method, and any disagreement with their
results is attributable to that.

It is also the one thing FINDING60 left untested: that finding measured their PREDICTOR (L2 and LMC,
neither of which had explanatory power here) but explicitly recorded "their METHOD is not tested,
only their predictor. Weight-space-regularised erasure in SD is a buildable arm and is NOT run."
This runs it.

Objective:
    L = L_erase(theta)  +  L_retain(theta)  +  lam_reg * ||theta - theta_base||^2 / ||theta_base||^2

  L_erase  = -E[ ||eps_theta(x_f, c_f, t) - eps||^2 ]     (push the target concept away)
  L_retain =  E[ ||eps_theta(x_r, c_r, t) - eps||^2 ]     (hold the retain set)
  the third term is OURS: an explicit anchor TOWARD the original weights. (This line previously
  read "the paper's contribution"; the paper's contribution is the OPPOSITE sign -- see the
  direction erratum at the top of this file.)

A SANITY NOTE ON WHAT THIS CAN SHOW. FINDING60 already found that GIFT sits at L2 = 0.18 -- all but
identical to base -- and is still reversed on 22 valid attacks. So weight-space proximity alone is
known NOT to confer resistance here, and the honest prior is that this arm will also be reversed. It
is run because "their predictor fails" and "their method fails" are different claims, and only one
of them has been measured.
"""
import argparse, importlib.util as iu, json, os, time
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--lam-reg", type=float, default=1.0, help="weight-space anchor strength")
    ap.add_argument("--teacher", default=None,
                    help="FINDING61 S4: an ALREADY-ERASED unet to distil from. With it, L_erase "
                         "becomes a BOUNDED distillation to the teacher instead of an unbounded "
                         "negated diffusion loss -- which is what made the from-scratch sweep fail. "
                         "Omit to reproduce the original (unbounded) from-scratch behaviour.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
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

    # the anchor: a frozen float32 copy of the ORIGINAL weights, on CPU to keep GPU headroom
    anchor = {k: v.detach().clone() for k, v in unet.state_dict().items()
              if v.dtype.is_floating_point}
    anchor_sq = sum(float((v.double() ** 2).sum()) for v in anchor.values())
    print(f"[wsr] anchor ||theta_base||^2 = {anchor_sq:.3e}", flush=True)

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
                print(f"[wsr] {tag} {min(i+4,len(prompts))}/{len(prompts)}", flush=True)
        return torch.cat(out)

    f_imgs, r_imgs = gen(f_prompts, "D_f"), gen(r_prompts, "D_r")

    def lat(imgs):
        z = []
        with torch.no_grad():
            for i in range(0, len(imgs), 4):
                b = imgs[i:i + 4].to(dev) * 2 - 1
                z.append(vae.encode(b).latent_dist.sample(generator=g) * vae.config.scaling_factor)
        return torch.cat(z)

    def emb(prompts):
        with torch.no_grad():
            ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            return te(ids)[0]

    f_lat, r_lat, f_emb, r_emb = lat(f_imgs), lat(r_imgs), emb(f_prompts), emb(r_prompts)
    del f_imgs, r_imgs
    torch.cuda.empty_cache()

    # FINDING61 S4's correction: distil from an already-erased teacher so every term is bounded.
    teacher = None
    if a.teacher:
        from diffusers import UNet2DConditionModel
        tsd = torch.load(a.teacher, map_location="cpu"); tsd = tsd.get("unet", tsd)
        teacher = UNet2DConditionModel.from_config(unet.config)
        teacher.load_state_dict({k: v.float() for k, v in tsd.items()})
        teacher = teacher.to(dev).eval().requires_grad_(False)
        unet.load_state_dict({k: v.float() for k, v in tsd.items()})   # start AT the teacher
        print(f"[wsr] bounded mode: distilling from {a.teacher}, anchoring to base", flush=True)

    unet.requires_grad_(True)
    params = [(n, p) for n, p in unet.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in params], lr=a.lr)

    def dloss(latv, embv, idx):
        z0 = latv[idx]
        noise = torch.randn(z0.shape, device=dev, generator=g)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                          generator=g).long()
        return F.mse_loss(unet(sched.add_noise(z0, noise, t), t,
                               encoder_hidden_states=embv[idx]).sample, noise)

    def distil(latv, embv, idx):
        """Bounded term: match the teacher's noise prediction. Finite optimum by construction."""
        z0 = latv[idx]
        noise = torch.randn(z0.shape, device=dev, generator=g)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                          generator=g).long()
        zt = sched.add_noise(z0, noise, t)
        with torch.no_grad():
            tgt = teacher(zt, t, encoder_hidden_states=embv[idx]).sample
        return F.mse_loss(unet(zt, t, encoder_hidden_states=embv[idx]).sample, tgt)

    hist = []
    nf, nr = f_lat.shape[0], r_lat.shape[0]
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        fi = torch.randint(0, nf, (a.batch,), device=dev, generator=g)
        ri = torch.randint(0, nr, (a.batch,), device=dev, generator=g)
        if teacher is None:
            l_erase = -dloss(f_lat, f_emb, fi)            # UNBOUNDED (the FINDING61 failure mode)
            l_retain = dloss(r_lat, r_emb, ri)
        else:
            l_erase = distil(f_lat, f_emb, fi)            # bounded: hold the teacher's erasure
            l_retain = distil(r_lat, r_emb, ri)           # bounded: hold the teacher's retention
        # the weight-space anchor, normalised so lam_reg is scale-free
        reg = 0.0
        for n, p in params:
            if n in anchor:
                reg = reg + ((p - anchor[n].to(p.device)) ** 2).sum()
        reg = reg / anchor_sq
        (l_erase + l_retain + a.lam_reg * reg).backward()
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "L_erase": float(l_erase.detach()),
                   "L_retain": float(l_retain.detach()), "reg_rel": float(reg.detach())}
            hist.append(rec)
            print(f"[wsr] step {step:5d} L_erase={rec['L_erase']:.5f} "
                  f"L_retain={rec['L_retain']:.5f} ||dtheta||^2/||theta||^2={rec['reg_rel']:.3e}",
                  flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "wsr_erase",
               # CORRECTED 2026-10-26 (Agent-3 audit). These fields claimed this arm adapts and
               # tests arXiv 2505.22310. It implements the OPPOSITE DIRECTION; see
               # docs/ERRATUM_wsr_direction.md and the module docstring.
               "classification": "OUR OWN DESIGN (WSR-anchor): erasure anchored TOWARD the base "
                                 "weights. NOT an implementation of arXiv 2505.22310, whose term "
                                 "MAXIMISES distance from base "
                                 "(vision_unlearner.py:366-373, verified).",
               "direction": "minimises ||theta - theta_base||^2 (pulls TOWARD base)",
               "reference_direction": "maximises RMS(theta_pre - theta) (pushes AWAY from base)",
               "adapts_idea_of": None,
               "is_adaptation_not_reproduction": True,
               "tests_what_FINDING60_left_untested": False,
               "withdrawn_claims": ["adapts_idea_of arXiv 2505.22310",
                                    "tests their METHOD rather than their predictor"],
               "faithful_direction_lives_in": "scripts/erase_wdr_v2.py",
               "domain_caveat": "arXiv 2505.22310 is example-level unlearning in vision "
                                "classifiers; in any case this file does not implement it",
               "label": a.label, "concept": a.concept, "hparams": vars(a),
               "history": hist, "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/wsr_{a.label}.json", "w"), indent=1)
    print(f"[wsr] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
