"""Meta-unlearning, three-term objective, variance-correct estimator, MICRO-BATCHED meta gradient.
Config version `meta_v4`. This is the S1c instrument of docs/PREREG_S1b_META_V3.md section 3.5.2.

WHY A FOURTH FILE. v3 corrected the difference estimator (common random numbers) and adopted
upstream's fixed meta-term timesteps. Measured outcome: the gradient magnitude was stable at
5.9e-4 to 8.3e-4 across 1500 steps, yet the model's mean parameter drift was 2.046e-5 -- 0.53x a
pure random walk and indistinguishable from v2's broken-estimator 2.27e-5. The objective pushed
continuously but its DIRECTION was redrawn every step, so the updates cancelled.

The cause is the third divergence from upstream found in this stage: `scripts/meta.sh` ships
`--train_batch_size 10` and `--num_images_per_prompt 10`, while v1/v2/v3 draw a SINGLE image index
per step. At batch 1 the meta gradient is

    -alpha * H(theta; xi) . grad L_F(theta; xi')

for one benign image and one noise draw, so its direction is resampled each step. Averaging over
B images cuts that variance by ~sqrt(B) and leaves the expectation unchanged.

v3 is NOT edited; its records and the S1b build stand under their own description.

WHAT THIS FILE CHANGES, and only this:
  * `--meta-batch B` accumulates every term over B independent (forget, benign) samples per step
    before one optimiser step. B = 1 reproduces v3's objective but NOT its RNG stream, so the two
    are not bit-comparable -- which is why this is a separate config_version rather than a flag on
    v3.
  * Micro-batched, not a true batch: each sample is a separate forward/backward at batch 1 and its
    gradient is folded into an accumulator. v3 already peaks at 32.4 GiB of 40 GiB, so a true batch
    of 8-10 would very likely OOM. Micro-batching is mathematically identical to a true batch for
    the gradient, at B x the time and batch-1 memory.
  * The inner attack step uses the B-averaged `grad L_F`, matching upstream, which advances its
    task model with a batch-10 step before evaluating the meta terms.

PRESERVED FROM v3: common random numbers within every difference pair; upstream's fixed meta-term
timesteps summed over --fixed-timesteps; a random timestep for the inner attack step; first-order
treatment of theta'; teacher distillation as the base objective; `--adam-eps`.

DEVIATIONS THAT REMAIN: no `hrm` pool (the inner step trains on the FORGET pool where upstream's
trains on `hrm_train`, so our simulated attacker is not the paper's); two pools and two weights
where upstream carries three and five. This file cannot speak to the published method's performance.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

ROOT = "/data/bijaypandey/t2i_unlearn"
OUT = os.path.join(ROOT, "outputs")
CONFIG_VERSION = "meta_v4"


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
    ap.add_argument("--steps", type=int, default=190)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-alpha", type=float, default=1e-5)
    ap.add_argument("--zeta", type=float, default=1.0)
    ap.add_argument("--gamma-f", type=float, default=1.0)
    ap.add_argument("--fixed-timesteps", default="1,2,5,10")
    ap.add_argument("--meta-batch", type=int, default=8,
                    help="images averaged per step. Upstream ships 10; 8 keeps the step count and "
                         "sample budget aligned with the 1500-sample v1/v2/v3 runs.")
    ap.add_argument("--adam-eps", type=float, default=1e-12)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    import copy
    ar, ut = _mods()
    torch.manual_seed(a.seed)

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
    print("[metaV4] data: %d forget, %d benign | meta-batch=%d steps=%d (samples=%d)"
          % (len(f_lat), len(b_lat), a.meta_batch, a.steps, a.meta_batch * a.steps), flush=True)

    fixed_ts = [int(x) for x in a.fixed_timesteps.split(",") if x.strip() != ""]
    ts_list = fixed_ts if fixed_ts else [None]

    def draw(lat, idx, g, t=None):
        n = torch.randn(lat[idx].shape, device=dev, dtype=lat.dtype, generator=g)
        if t is None:
            ts = torch.randint(0, sch.config.num_train_timesteps, (len(idx),),
                               device=dev, generator=g).long()
        else:
            ts = torch.full((len(idx),), int(t), device=dev, dtype=torch.long)
        return n, ts

    def dmloss_at(net, lat, emb, idx, n, ts):
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
    B = a.meta_batch
    invB = 1.0 / B

    for step in range(a.steps):
        # ---- draw all B micro-samples up front, so each difference pair shares its own draw
        samples = []
        for _ in range(B):
            fi = torch.randint(0, len(f_lat), (1,), device=dev, generator=g)
            bi = torch.randint(0, len(b_lat), (1,), device=dev, generator=g)
            nf, tf = draw(f_lat, fi, g)                       # base distil, forget
            nb, tb = draw(b_lat, bi, g)                       # base distil, benign
            na, ta = draw(f_lat, fi, g)                       # inner attack, random timestep
            pairs = [(draw(b_lat, bi, g, t), draw(f_lat, fi, g, t)) for t in ts_list]
            samples.append((fi, bi, nf, tf, nb, tb, na, ta, pairs))

        # ---- base objective at theta, averaged over the micro-batch
        opt.zero_grad(set_to_none=True)
        l_e_sum = l_r_sum = 0.0
        for (fi, bi, nf, tf, nb, tb, _na, _ta, _p) in samples:
            l_e = distil_at(unet, f_lat, f_emb, fi, nf, tf)
            l_r = distil_at(unet, b_lat, b_emb, bi, nb, tb)
            ((l_e + l_r) * invB).backward()
            l_e_sum += float(l_e); l_r_sum += float(l_r)

        # ---- META, theta side (theta still intact): + zeta * grad L_R(theta)
        acc = [torch.zeros_like(q) for q in params]
        lr_theta = lr_prime = lf_prime = 0.0
        nterm = 0
        for (_fi, bi, _nf, _tf, _nb, _tb, _na, _ta, pairs) in samples:
            for (bd, _fd) in pairs:
                l = dmloss_at(unet, b_lat, b_emb, bi, bd[0], bd[1])
                gR0 = torch.autograd.grad(l, params, retain_graph=False)
                for s, q in zip(acc, gR0):
                    s.add_(q, alpha=a.zeta * invB)
                del gR0
                lr_theta += float(l); nterm += 1

        # ---- the simulated attack, using the B-averaged grad L_F as upstream does
        gatt = [torch.zeros_like(q) for q in params]
        for (fi, _bi, _nf, _tf, _nb, _tb, na, ta, _p) in samples:
            l_att = dmloss_at(unet, f_lat, f_emb, fi, na, ta)
            ga = torch.autograd.grad(l_att, params, retain_graph=False)
            for s, q in zip(gatt, ga):
                s.add_(q, alpha=invB)
            del ga
        backup = [q.detach().clone() for q in params]
        with torch.no_grad():
            for q, gg in zip(params, gatt):
                q.sub_(a.inner_alpha * gg)
        del gatt

        # ---- META, theta' side: - zeta * grad L_R(theta')  - gamma_f * grad L_F(theta')
        for (fi, bi, _nf, _tf, _nb, _tb, _na, _ta, pairs) in samples:
            for (bd, fd) in pairs:
                l = dmloss_at(unet, b_lat, b_emb, bi, bd[0], bd[1])   # SAME draw as theta side
                gR1 = torch.autograd.grad(l, params, retain_graph=False)
                for s, q in zip(acc, gR1):
                    s.sub_(q, alpha=a.zeta * invB)
                del gR1
                lr_prime += float(l)
                if a.gamma_f != 0.0:
                    l2 = dmloss_at(unet, f_lat, f_emb, fi, fd[0], fd[1])
                    gF1 = torch.autograd.grad(l2, params, retain_graph=False)
                    for s, q in zip(acc, gF1):
                        s.sub_(q, alpha=a.gamma_f * invB)
                    del gF1
                    lf_prime += float(l2)

        with torch.no_grad():
            for q, b in zip(params, backup):
                q.copy_(b)
            for q, s in zip(params, acc):
                q.grad = s if q.grad is None else q.grad.add_(s)
        del backup

        meta_norm = float(torch.sqrt(sum((s * s).sum() for s in acc)))
        del acc
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            k = max(nterm, 1)
            rec = {"step": step, "l_erase": l_e_sum * invB, "l_retain_distil": l_r_sum * invB,
                   "L_R_theta": lr_theta / k, "L_R_theta_prime": lr_prime / k,
                   "L_F_theta_prime": (lf_prime / k) if a.gamma_f != 0.0 else 0.0,
                   "retain_gap": (lr_prime - lr_theta) / k,
                   "meta_grad_norm": meta_norm, "n_meta_terms": k, "meta_batch": B}
            log.append(rec)
            print("[metaV4] step %5d erase=%.5f | L_R(th)=%.5f L_R(th')=%.5f gap=%+.6f | "
                  "L_F(th')=%.5f | |meta|=%.3e | %.2fs/step"
                  % (step, rec["l_erase"], rec["L_R_theta"], rec["L_R_theta_prime"],
                     rec["retain_gap"], rec["L_F_theta_prime"], rec["meta_grad_norm"],
                     (time.time() - t0) / max(step + 1, 1)), flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, "%s/unet.pt" % a.out)
    json.dump({
        "study": "meta_unlearning_full_objective", "config_version": CONFIG_VERSION,
        "source": "arXiv 2410.12777 (Meta-Unlearning, ICCV 2025), Eq. 8",
        "supersedes": "scripts/meta_unlearn_v3.py for the trap question; see "
                      "docs/PREREG_S1b_META_V3.md section 3.5",
        "provenance": ("OUR implementation of the paper's three-term meta objective with a "
                       "variance-correct difference estimator AND a micro-batched meta gradient. "
                       "Not the authors' code; not a reproduction."),
        "corrections_vs_v3": [
            "meta gradient accumulated over --meta-batch independent (forget, benign) samples per "
            "step, micro-batched. v3 used a single image index per step, so the meta gradient's "
            "direction was resampled each step and 1500 Adam steps cancelled to a random walk "
            "(measured drift 2.046e-5 vs 3.87e-5 for a pure random walk).",
            "the inner attack step uses the B-averaged grad L_F, matching upstream's batch-10 "
            "task step.",
        ],
        "deviations": ["first-order: theta' treated as constant when differentiating",
                       "no hrm pool; the inner step trains on the FORGET pool where upstream's "
                       "trains on hrm_train, so our simulated attacker is not the paper's",
                       "two pools and two weights where upstream carries three pools and five",
                       "micro-batched rather than a true batch (identical gradient, B x the time)",
                       "erasure side is teacher distillation from the model's own init, so the "
                       "base terms contribute zero gradient at step 0"],
        "label": a.label, "hparams": vars(a), "forget_pool": forget_p,
        "n_benign_prompts": len(benign_p), "pool_recipe": ut.RECIPE, "train_log": log,
        "minutes": (time.time() - t0) / 60,
    }, open(os.path.join(OUT, "metaV4_%s.json" % a.label), "w"), indent=1)
    print("[metaV4] wrote %s/unet.pt" % a.out, flush=True)


if __name__ == "__main__":
    main()
