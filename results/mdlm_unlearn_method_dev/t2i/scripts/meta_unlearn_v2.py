"""Meta-unlearning with the PAPER'S FULL three-term meta objective. Config version `meta_v2`.

WHY A NEW FILE. `scripts/meta_unlearn.py` implements one of the three components of Eq. 8 of
arXiv:2410.12777 (Meta-Unlearning, ICCV 2025). That is documented in `docs/ERRATUM_meta_objective.md`
and it is the reason FINDING56 saw the self-destruct trigger fire on normal users: our port
penalised the LEVEL of benign loss after the simulated attack with nothing rewarding benign loss
being low before it, so destroying benign capability unconditionally was the cheapest way to satisfy
the objective. The old port is left byte-identical; its results stand under their own description.

THE OBJECTIVE, as the paper writes it (Eq. 8 meta term):

    -L_F(theta')  -  zeta * [ L_R(theta') - L_R(theta) ]

accumulated on top of the erasure objective's own gradient, exactly as the released implementation
does (`src/esd_meta.py` builds a signed `sum_grads` and adds it to `param.grad`). With an optimizer
that DESCENDS `param.grad`, the three components mean:

    + zeta * grad L_R(theta)      descend benign loss at theta      -> PRESERVE normal utility
    - zeta * grad L_R(theta')     ascend  benign loss at theta'     -> THE TRAP: utility collapses
                                                                       only if the model is
                                                                       fine-tuned on the target
    - gammaF * grad L_F(theta')   ascend  target loss at theta'     -> ROBUSTNESS: the target stays
                                                                       erased after the attack

The first and second form a DIFFERENCE, not a level. That is the whole point: the objective asks for
benign loss to RISE ACROSS the simulated attack step, not to be high in absolute terms. Our previous
port had the second component only, which is why it collapsed utility outright.

WHAT IS STILL A DEVIATION, stated rather than buried:
  * First-order. theta' is treated as a constant when differentiating; we do not backpropagate
    through the simulated attack. The reference does the same (it adds gradients computed on a
    deepcopy), so this matches the released code, but it is not a full meta-gradient.
  * Our pools, our prompts, our generated data. The paper's R is the union of its `irt` and `tgt`
    pools and its F is the `hrm` pool; we use one benign pool and one target pool, with one `zeta`
    where the reference carries a separate gamma per pool. This is a domain adaptation of the
    objective, not a reproduction of the paper's experiment.
  * Erasure side: we keep teacher distillation on both pools as the base objective, which is what
    our harness already does, rather than the reference's ESD/SDD negative-guidance distillation.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

ROOT = "/data/bijaypandey/t2i_unlearn"
OUT = os.path.join(ROOT, "outputs")
CONFIG_VERSION = "meta_v2"


def _mods():
    """attack_relearn (pipeline builder) and undo_t2i (prompt pools), as the v1 port loads them."""
    import importlib.util as iu
    out = []
    for name in ("attack_relearn", "undo_t2i"):
        q = os.path.join(ROOT, "scripts", "%s.py" % name)
        sp = iu.spec_from_file_location(name, q)
        m = iu.module_from_spec(sp)
        sp.loader.exec_module(m)
        out.append(m)
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the ERASED unet to meta-unlearn")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-alpha", type=float, default=1e-5,
                    help="step size of the simulated malicious fine-tune, theta -> theta'")
    ap.add_argument("--zeta", type=float, default=1.0,
                    help="weight on the RETAIN DIFFERENCE [L_R(theta') - L_R(theta)]")
    ap.add_argument("--gamma-f", type=float, default=1.0,
                    help="weight on the post-attack TARGET term -L_F(theta')")
    ap.add_argument("--n-images", type=int, default=64,
                    help="images per pool; v1 port default, kept for comparability")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    import copy
    ar, ut = _mods()
    torch.manual_seed(a.seed)

    # POOLS AND DATA: replicated line for line from scripts/meta_unlearn.py so the full-objective
    # arm and the one-term arm differ ONLY in the objective. Same prompt pools, same generation
    # seeds (30_000 / 40_000), same count, same sampler settings.
    kw = a.concept.split("_")[0].lower()
    forget_p = list(ut.FORGET_POOL[a.concept])
    benign_p = [q for q in ut.RETAIN_POOL if kw not in q.lower()][:24]
    assert len(benign_p) >= 8

    pipe = ar.build_pipe(torch.float32, a.teacher, dev)
    unet, vae, te, tok, sch = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    te.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(True)
    teacher = copy.deepcopy(unet).eval().requires_grad_(False)

    gen = ar.build_pipe(torch.float16, a.teacher, dev)

    def make(prompts, n, base_seed):
        lat, emb = [], []
        with torch.no_grad():
            for i in range(n):
                pr = prompts[i % len(prompts)]
                gg = torch.Generator(device=dev).manual_seed(base_seed + i)
                im = gen([pr], num_inference_steps=25, guidance_scale=7.5, generator=gg).images[0]
                x = torch.from_numpy((torch.frombuffer(im.tobytes(), dtype=torch.uint8)
                     .reshape(im.size[1], im.size[0], 3).numpy())).permute(2, 0, 1)[None].float()
                x = x.to(dev) / 127.5 - 1.0
                lat.append(vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)
                ids = tok([pr], padding="max_length", max_length=77, truncation=True,
                          return_tensors="pt").input_ids.to(dev)
                emb.append(te(ids)[0])
        return torch.cat(lat), torch.cat(emb)

    f_lat, f_emb = make(forget_p, a.n_images, 30_000)
    b_lat, b_emb = make(benign_p, a.n_images, 40_000)
    del gen
    torch.cuda.empty_cache()
    print("[metaV2] data: %d forget, %d benign" % (len(f_lat), len(b_lat)), flush=True)

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

        # ---- base erasure objective at theta, accumulated into p.grad first (theta still intact)
        opt.zero_grad(set_to_none=True)
        l_erase = distil(unet, f_lat, f_emb, fi, g)
        l_retain_distil = distil(unet, b_lat, b_emb, bi, g)
        (l_erase + l_retain_distil).backward()

        # ---- META COMPONENT 1: + zeta * grad L_R(theta).  Preserve benign capability NORMALLY.
        # This is the term the previous port lacked, and its absence is why the trap fired on
        # ordinary use: nothing opposed the collapse of benign capability.
        l_R_theta = dmloss(unet, b_lat, b_emb, bi, g)
        gR0 = torch.autograd.grad(l_R_theta, params, retain_graph=False)

        # ---- simulate the malicious fine-tune: theta -> theta' = theta - alpha * grad L_F(theta)
        l_att = dmloss(unet, f_lat, f_emb, fi, g)
        gatt = torch.autograd.grad(l_att, params, retain_graph=False)
        backup = [q.detach().clone() for q in params]
        with torch.no_grad():
            for q, gg in zip(params, gatt):
                q.sub_(a.inner_alpha * gg)

        # ---- META COMPONENTS 2 and 3, both evaluated AT theta'
        l_F_prime = dmloss(unet, f_lat, f_emb, fi, g)        # target loss after the attack
        gF1 = torch.autograd.grad(l_F_prime, params, retain_graph=False)
        l_R_prime = dmloss(unet, b_lat, b_emb, bi, g)        # benign loss after the attack
        gR1 = torch.autograd.grad(l_R_prime, params, retain_graph=False)

        with torch.no_grad():
            for q, b in zip(params, backup):
                q.copy_(b)                                   # restore theta before stepping
            for q, g0, g1, gf in zip(params, gR0, gR1, gF1):
                contrib = a.zeta * g0 - a.zeta * g1 - a.gamma_f * gf
                q.grad = contrib if q.grad is None else q.grad.add_(contrib)

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "l_erase": float(l_erase),
                   "l_retain_distil": float(l_retain_distil),
                   "L_R_theta": float(l_R_theta), "L_R_theta_prime": float(l_R_prime),
                   "L_F_theta_prime": float(l_F_prime),
                   "retain_gap": float(l_R_prime) - float(l_R_theta)}
            log.append(rec)
            print("[metaV2] step %5d erase=%.5f | L_R(th)=%.5f L_R(th')=%.5f gap=%+.5f | "
                  "L_F(th')=%.5f | %.2fs/step"
                  % (step, rec["l_erase"], rec["L_R_theta"], rec["L_R_theta_prime"],
                     rec["retain_gap"], rec["L_F_theta_prime"],
                     (time.time() - t0) / max(step + 1, 1)), flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, "%s/unet.pt" % a.out)
    json.dump({
        "study": "meta_unlearning_full_objective", "config_version": CONFIG_VERSION,
        "source": "arXiv 2410.12777 (Meta-Unlearning, ICCV 2025), Eq. 8",
        "provenance": ("OUR implementation of the paper's FULL three-term meta objective, on our "
                       "pools and prompts. Not the authors' code and not a reproduction of their "
                       "reported experiment. Supersedes scripts/meta_unlearn.py, which implemented "
                       "one of the three components (docs/ERRATUM_meta_objective.md)."),
        "objective": ("base: distil_erase(theta) + distil_retain(theta);  meta: "
                      "+zeta*grad L_R(theta) - zeta*grad L_R(theta') - gamma_f*grad L_F(theta'), "
                      "theta' = theta - alpha*grad L_F(theta), first order"),
        "deviations": ["first-order: theta' treated as constant when differentiating",
                       "one benign pool and one zeta, where the reference carries separate "
                       "irt/tgt pools each with its own gamma",
                       "erasure side is teacher distillation, not the reference's ESD/SDD "
                       "negative-guidance distillation"],
        "label": a.label, "hparams": vars(a), "forget_pool": forget_p,
        "n_benign_prompts": len(benign_p), "pool_recipe": ut.RECIPE, "train_log": log,
        "minutes": (time.time() - t0) / 60,
    }, open(os.path.join(OUT, "metaV2_%s.json" % a.label), "w"), indent=1)
    print("[metaV2] wrote %s/unet.pt" % a.out, flush=True)


if __name__ == "__main__":
    main()
