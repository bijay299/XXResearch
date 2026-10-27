"""Meta-unlearning, three-term objective, with a VARIANCE-CORRECT difference estimator.
Config version `meta_v3`.

WHY A THIRD FILE. `meta_unlearn_v2.py` restored the paper's difference term
`-zeta * [L_R(theta') - L_R(theta)]`, but estimated it wrongly, and
`docs/PREREG_S1_AMENDMENT_1_2026-10-27.md` section 6 records the measurement that exposed it: the
two members of the difference were produced by two separate `dmloss` calls, and each call drew FRESH
noise and a FRESH timestep. The difference therefore conflated

  * the parameter change, theta' - theta = -alpha * grad L_F(theta) with alpha = 1e-5, and
  * the change of random draw.

The diffusion loss varies hugely with timestep -- v2's own log shows L_R(theta) ranging 0.00618 to
0.47347 on a model that had barely moved -- so the draw difference dominated the quantity being
estimated by orders of magnitude. v2's logged `gap` swung -0.402, +0.027, +0.179, +0.346, -0.195
across steps with no trend: that is draw noise. After gradient clipping at 1.0 and lr 1e-6 the
result was a slow random walk, and the model's distillation distance from its teacher reached only
1.5e-4 after 1000 steps where v1's one-sided term had moved retain8 from 0.9417 to 0.4750.

v2's builds and results are left in place under their own description; v2 is NOT edited.

TWO CORRECTIONS HERE, both taken from the released upstream rather than invented:

  1. COMMON RANDOM NUMBERS. Noise and timestep are drawn ONCE per pair and reused for both members
     of every difference. Any difference estimator of the form f(theta') - f(theta) must share its
     randomness or it measures the randomness instead.

  2. FIXED TIMESTEPS, as the authors do. `upstream/Meta-Unlearning/src/meta.py:595` reads

         if args.fix_timesteps and not train_set:
             timesteps = torch.ones(bsz, dtype=torch.long, ...) * fixed_time_step

     and the shipped `scripts/meta.sh` passes `--fix_timesteps True --fixed_time_steps 1 2 5 10`.
     Upstream evaluates every meta term at those four fixed low timesteps and SUMS the gradients
     (lines 965-1006); only the inner attack step (`train_set=True`) uses random timesteps. v2 drew
     a random timestep over the full [0, 1000) range for every term, which was an unlisted fidelity
     gap as well as the cause of (1). This file follows upstream: meta terms at fixed timesteps,
     summed; inner attack step at a random timestep.

DEVIATIONS THAT REMAIN, unchanged from v2 and stated rather than buried:
  * First-order: theta' is treated as constant when differentiating. The reference does the same.
  * No `hrm` pool. Upstream carries three pools (hrm / irt / tgt) with five weights; we carry two
    pools and two weights, and our inner attack step trains on the FORGET pool where upstream's
    trains on `hrm_train`. So our simulated attacker is not the paper's attacker. This file cannot
    speak to the published method's performance either way.
  * Erasure side is teacher distillation, and since the model is initialised FROM that teacher the
    base terms start at exactly zero gradient -- this port inherits all suppression from the teacher
    and contains no erasure objective of its own (amendment section 1).
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

ROOT = "/data/bijaypandey/t2i_unlearn"
OUT = os.path.join(ROOT, "outputs")
CONFIG_VERSION = "meta_v3"


def _mods():
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
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-alpha", type=float, default=1e-5)
    ap.add_argument("--zeta", type=float, default=1.0)
    ap.add_argument("--gamma-f", type=float, default=1.0)
    ap.add_argument("--fixed-timesteps", default="1,2,5,10",
                    help="upstream's meta-term timesteps, summed. Empty string = random timestep "
                         "per pair (still common-random-numbers), which isolates correction 1 "
                         "from correction 2.")
    ap.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam's numerical floor. The corrected meta gradient is a "
                         "Hessian-vector product of norm O(alpha), so its per-component magnitude "
                         "can fall near or below the 1e-8 default and be throttled by eps rather "
                         "than by the objective. Remedy declared in PREREG_S1b_META_V3.md s2.2 and "
                         "triggered by the pilot: measured norm ~8.1e-4 over 859,520,964 params "
                         "gives ~2.8e-8 per component, a ratio of ~2.8 to the default.")
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    import copy
    ar, ut = _mods()
    torch.manual_seed(a.seed)

    # Pools and data: replicated from v1/v2 so arms differ ONLY in the objective and its estimator.
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
    print("[metaV3] data: %d forget, %d benign" % (len(f_lat), len(b_lat)), flush=True)

    fixed_ts = [int(x) for x in a.fixed_timesteps.split(",") if x.strip() != ""]

    def draw(lat, idx, g, t=None):
        """One (noise, timestep) draw, to be SHARED by both members of a difference."""
        n = torch.randn(lat[idx].shape, device=dev, dtype=lat.dtype, generator=g)
        if t is None:
            ts = torch.randint(0, sch.config.num_train_timesteps, (len(idx),),
                               device=dev, generator=g).long()
        else:
            ts = torch.full((len(idx),), int(t), device=dev, dtype=torch.long)
        return n, ts

    def dmloss_at(net, lat, emb, idx, n, ts):
        """Diffusion MSE on a GIVEN draw -- no randomness of its own."""
        return F.mse_loss(net(sch.add_noise(lat[idx], n, ts), ts,
                              encoder_hidden_states=emb[idx]).sample, n)

    def distil_at(net, lat, emb, idx, n, ts):
        zt = sch.add_noise(lat[idx], n, ts)
        with torch.no_grad():
            tgt = teacher(zt, ts, encoder_hidden_states=emb[idx]).sample
        return F.mse_loss(net(zt, ts, encoder_hidden_states=emb[idx]).sample, tgt)

    opt = torch.optim.Adam(unet.parameters(), lr=a.lr, eps=a.adam_eps)
    g = torch.Generator(device=dev).manual_seed(a.seed + 777)
    params = [p for p in unet.parameters() if p.requires_grad]
    unet.train()
    log = []

    for step in range(a.steps):
        fi = torch.randint(0, len(f_lat), (1,), device=dev, generator=g)
        bi = torch.randint(0, len(b_lat), (1,), device=dev, generator=g)

        # ---- base erasure objective at theta (random draw; not a difference, so noise here is
        #      ordinary SGD noise rather than a bias in an estimator)
        opt.zero_grad(set_to_none=True)
        nf, tf = draw(f_lat, fi, g)
        nb, tb = draw(b_lat, bi, g)
        l_erase = distil_at(unet, f_lat, f_emb, fi, nf, tf)
        l_retain_distil = distil_at(unet, b_lat, b_emb, bi, nb, tb)
        (l_erase + l_retain_distil).backward()

        # ---- the simulated malicious fine-tune, theta -> theta'. Upstream uses a RANDOM timestep
        #      for this step (train_set=True at src/meta.py:595), so we do too.
        na, ta = draw(f_lat, fi, g)
        l_att = dmloss_at(unet, f_lat, f_emb, fi, na, ta)
        gatt = torch.autograd.grad(l_att, params, retain_graph=False)
        backup = [q.detach().clone() for q in params]

        # ---- META TERMS. For each fixed timestep, ONE shared draw per pool, used at BOTH theta
        #      and theta'. Gradients summed over timesteps, as upstream does.
        ts_list = fixed_ts if fixed_ts else [None]
        pair_draws = []
        for t in ts_list:
            pair_draws.append((draw(b_lat, bi, g, t), draw(f_lat, fi, g, t)))

        acc = [torch.zeros_like(q) for q in params]
        logged = {"L_R_theta": 0.0, "L_R_theta_prime": 0.0, "L_F_theta_prime": 0.0}

        # theta side first, while theta is still intact. Each gradient tuple is folded into `acc`
        # and released immediately: holding all |fixed_timesteps| of them would cost
        # len(ts_list) x 3.4 GiB on this model and will not fit beside Adam's two state copies.
        for (bd, fd) in pair_draws:
            l_R_theta = dmloss_at(unet, b_lat, b_emb, bi, bd[0], bd[1])
            gR0 = torch.autograd.grad(l_R_theta, params, retain_graph=False)
            logged["L_R_theta"] += float(l_R_theta)
            for s, q in zip(acc, gR0):
                s.add_(q, alpha=a.zeta)                       # + zeta * grad L_R(theta)
            del gR0

        with torch.no_grad():
            for q, gg in zip(params, gatt):
                q.sub_(a.inner_alpha * gg)                    # now at theta'

        for (bd, fd) in pair_draws:
            l_R_prime = dmloss_at(unet, b_lat, b_emb, bi, bd[0], bd[1])   # SAME draw as above
            gR1 = torch.autograd.grad(l_R_prime, params, retain_graph=False)
            logged["L_R_theta_prime"] += float(l_R_prime)
            for s, q in zip(acc, gR1):
                s.sub_(q, alpha=a.zeta)                       # - zeta * grad L_R(theta')
            del gR1
            if a.gamma_f != 0.0:
                l_F_prime = dmloss_at(unet, f_lat, f_emb, fi, fd[0], fd[1])
                gF1 = torch.autograd.grad(l_F_prime, params, retain_graph=False)
                logged["L_F_theta_prime"] += float(l_F_prime)
                for s, q in zip(acc, gF1):
                    s.sub_(q, alpha=a.gamma_f)                # - gamma_f * grad L_F(theta')
                del gF1

        with torch.no_grad():
            for q, b in zip(params, backup):
                q.copy_(b)                                    # restore theta before stepping
            for q, s in zip(params, acc):
                q.grad = s if q.grad is None else q.grad.add_(s)

        meta_norm = float(torch.sqrt(sum((s * s).sum() for s in acc)))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            k = max(len(ts_list), 1)
            rec = {"step": step, "l_erase": float(l_erase),
                   "l_retain_distil": float(l_retain_distil),
                   "L_R_theta": logged["L_R_theta"] / k,
                   "L_R_theta_prime": logged["L_R_theta_prime"] / k,
                   "L_F_theta_prime": logged["L_F_theta_prime"] / k,
                   "retain_gap": (logged["L_R_theta_prime"] - logged["L_R_theta"]) / k,
                   "meta_grad_norm": meta_norm,
                   "n_fixed_timesteps": len(ts_list)}
            log.append(rec)
            print("[metaV3] step %5d erase=%.5f | L_R(th)=%.5f L_R(th')=%.5f gap=%+.6f | "
                  "L_F(th')=%.5f | |meta|=%.3e | %.2fs/step"
                  % (step, rec["l_erase"], rec["L_R_theta"], rec["L_R_theta_prime"],
                     rec["retain_gap"], rec["L_F_theta_prime"], rec["meta_grad_norm"],
                     (time.time() - t0) / max(step + 1, 1)), flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, "%s/unet.pt" % a.out)
    json.dump({
        "study": "meta_unlearning_full_objective", "config_version": CONFIG_VERSION,
        "source": "arXiv 2410.12777 (Meta-Unlearning, ICCV 2025), Eq. 8",
        "supersedes": "scripts/meta_unlearn_v2.py for the trap question; see "
                      "docs/PREREG_S1_AMENDMENT_1_2026-10-27.md section 6",
        "provenance": ("OUR implementation of the paper's three-term meta objective with a "
                       "variance-correct difference estimator: common random numbers across each "
                       "difference, and upstream's fixed-timestep meta terms summed over "
                       "--fixed-timesteps. Not the authors' code; not a reproduction."),
        "corrections_vs_v2": [
            "common random numbers: noise and timestep drawn once per pair and shared by both "
            "members of every difference. v2 drew fresh randomness per term, so the difference "
            "measured the draw change rather than the parameter change.",
            "fixed meta-term timesteps summed over the set, following upstream src/meta.py:595 "
            "and scripts/meta.sh (--fix_timesteps True --fixed_time_steps 1 2 5 10). v2 drew a "
            "random timestep over the full [0,1000) range for every term.",
        ],
        "deviations": ["first-order: theta' treated as constant when differentiating",
                       "no hrm pool; inner attack step trains on the FORGET pool where upstream's "
                       "trains on hrm_train, so our simulated attacker is not the paper's",
                       "two pools and two weights where upstream carries three pools and five",
                       "erasure side is teacher distillation from the model's own init, so the "
                       "base terms contribute zero gradient at step 0 and this port inherits all "
                       "suppression from the teacher"],
        "label": a.label, "hparams": vars(a), "forget_pool": forget_p,
        "n_benign_prompts": len(benign_p), "pool_recipe": ut.RECIPE, "train_log": log,
        "minutes": (time.time() - t0) / 60,
    }, open(os.path.join(OUT, "metaV3_%s.json" % a.label), "w"), indent=1)
    print("[metaV3] wrote %s/unet.pt" % a.out, flush=True)


if __name__ == "__main__":
    main()
