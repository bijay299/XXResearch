"""UTRAP-v2 (OURS) -- the same outer objective, with the inner attacker now RECONCILED against the
evaluation attacker through `scripts/attack_core.py`. Audit section 4.

    min_theta  L_retain(theta)  -  lam * L_target( A_K(theta) )

WHAT WAS WRONG (independently confirmed; see ERRATUM_utrap_inner_attacker.md). `erase_utrap.py`'s
docstring and `PREREG_H84` both call the inner process "the ACTUAL full-UNet relearning attack (the
one attack_relearn.py runs)". It differed on seven axes simultaneously:

| axis | snapshot inner | evaluation attack | v2 |
|---|---|---|---|
| optimizer | SGD, rebuilt each outer step | Adam, persistent | `--inner-optimizer`, default **adam** |
| grad clipping | none | global norm 1.0 | `--inner-clip`, default **1.0** |
| batch | 2 | 1 | `--inner-batch`, default **1** |
| horizon | K = 4/8/16 | 300 | `--inner-k`, LABELLED (see below) |
| target images | the ERASED teacher's generations | the UNMODIFIED base model's | `--target-source`, default **base** |
| target prompts | `["a photo of a church"] * 16` (an EVAL prompt) | 8 `T` prompts | `--target-prompts`, default **def_train** |
| optimizer state | discarded every outer step | persistent over 300 steps | `--inner-persistent`, default **on** |

Three of these are now defaults that MATCH the evaluator, and the rest are explicit flags whose
values are written into the artifact together with `divergence_from_evaluation_attacker`, computed
by `attack_core.AttackSpec`. Drift of this kind cannot recur silently: the record names it.

THE HORIZON IS LABELLED, NOT RENAMED. K < 300 is declared `SHORT_HORIZON_SURROGATE` and
`attack_core` REFUSES to let a K < 300 unroll be labelled `EVALUATION_REFERENCE`. A shortened
horizon is a legitimate design choice -- it is simply not the 300-step attack at its real budget,
and the artifact now says which it is.

DATA SEPARATION. The audit requires defence-training, attack-training, development and final data
to stay separate. v2's target pool defaults to `--target-prompts def_train`, a DEF_TRAIN pool drawn
from the split registry and DISJOINT from both the attacker's `T` and the evaluation prompts --
unlike the snapshot, which generated defence data from the first evaluation prompt. Selecting
`atk_train` is permitted for a deliberate matched-data ablation and is flagged as such.

WHAT v2 DOES NOT CHANGE. The trajectory is still DETACHED (first-order, K steps). A fully
differentiated unroll remains untested, as before, and is not claimed here.

NO RESULT IS CLAIMED FROM THIS FILE until a build and a frozen-grid evaluation have run. The
existing UTRAP reversal measurements stand; what is withdrawn is the claim that they ruled out
unroll depth, since the surrogate they varied was not the evaluation attacker. See
`PREREG_H90_utrap_v2.md` for the bounded K=0/K=1/K=8/longer comparison with matched compute.
"""
import argparse, importlib.util as iu, json, os, time
import torch

import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
IMPL_VERSION = "utrap/v2-shared-attack-core"
EVAL_STEPS = 300


def _mod(name):
    spec = iu.spec_from_file_location(f"_u_{name}", f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--inner-k", type=int, default=8,
                    help="unrolled attacker steps. 0 selects the K=0 control.")
    ap.add_argument("--inner-lr", type=float, default=1e-5)
    ap.add_argument("--inner-optimizer", default="adam", choices=("adam", "sgd"),
                    help="default adam, MATCHING the evaluation attacker (snapshot used sgd)")
    ap.add_argument("--inner-batch", type=int, default=1,
                    help="default 1, MATCHING the evaluation attacker (snapshot used 2)")
    ap.add_argument("--inner-clip", type=float, default=1.0,
                    help="default 1.0, MATCHING the evaluation attacker (snapshot had none)")
    ap.add_argument("--inner-persistent", type=int, default=1,
                    help="1 keeps Adam moments across outer steps, as the 300-step evaluation "
                         "attacker does; 0 reproduces the snapshot's per-step rebuild")
    ap.add_argument("--target-source", default="base", choices=("base", "teacher"),
                    help="default base, MATCHING the evaluation attacker's data (snapshot used "
                         "the already-erased teacher)")
    ap.add_argument("--target-prompts", default="def_train",
                    choices=("def_train", "atk_train", "eval_leak_reproduce"),
                    help="def_train: a DEF_TRAIN pool disjoint from attacker and evaluation "
                         "prompts (default). atk_train: the attacker's own T pool, for a matched-"
                         "data ablation. eval_leak_reproduce: the snapshot's leaked evaluation "
                         "prompt, available ONLY to reproduce the historical build.")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--probe-every", type=int, default=100,
                    help="cadence of the fixed-noise inner-attacker effectiveness probe")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    core, sr, ar = _mod("attack_core"), _mod("split_registry"), _mod("attack_relearn")

    # ---- the inner attacker's SPEC, validated and labelled before anything runs
    role = ("ABLATION_K0" if a.inner_k == 0 else
            "SHORT_HORIZON_SURROGATE" if a.inner_k < EVAL_STEPS else "MATCHED_HORIZON_SURROGATE")
    spec = core.AttackSpec(
        optimizer=a.inner_optimizer, lr=a.inner_lr, steps=a.inner_k, batch=a.inner_batch,
        clip_grad_norm=(a.inner_clip if a.inner_clip > 0 else None),
        persistent_state=bool(a.inner_persistent), horizon_role=role,
        notes="UTRAP v2 inner surrogate").validate(EVAL_STEPS)
    print(f"[utrap2] inner attacker role={role}", flush=True)
    print(f"[utrap2] divergence from the evaluation attacker: "
          f"{json.dumps(spec.divergence_from_eval())}", flush=True)

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, revision=(BASE_REV_ENV or pin["revision"]), torch_dtype=torch.float32,
        variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    for m in (vae, te):
        m.requires_grad_(False)

    # ---- target prompts, by declared split role
    DEF_TRAIN_TARGET = {
        "church": ["a small roadside church", "a church bell tower in daylight",
                   "a stone church with arched windows", "a whitewashed church by a field"],
        "parachute": ["a parachute pack deploying", "a wide parachute over open ground",
                      "a striped parachute canopy", "a parachute harness in descent"],
        "garbage_truck": ["a garbage truck with its arm extended",
                          "a municipal garbage truck at dawn", "a loaded garbage truck turning",
                          "a garbage truck beside wheelie bins"],
        "van_gogh": ["a Van Gogh orchard in spring", "a Van Gogh harbour scene",
                     "a Van Gogh still life of flowers", "a Van Gogh village lane"],
    }
    if a.target_prompts == "def_train":
        t_prompts, t_role = DEF_TRAIN_TARGET[a.concept], "DEF_TRAIN"
    elif a.target_prompts == "atk_train":
        t_prompts, t_role = list(ar.SPLIT[a.concept]["T"]), "ATK_TRAIN"
    else:
        t_prompts, t_role = [f"a photo of a {a.concept}"], "DEV"
        print("[utrap2] WARNING: reproducing the snapshot's evaluation-prompt leak on purpose",
              flush=True)

    # ---- the target generator. base => the evaluation attacker's own data provenance.
    gen_pipe = pipe
    if a.target_source == "base":
        gen_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, revision=(BASE_REV_ENV or pin["revision"]), torch_dtype=torch.float16,
            variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)
    tsd = torch.load(a.teacher, map_location="cpu"); tsd = tsd.get("unet", tsd)

    undo = _mod("undo_t2i")
    r_prompts = (undo.RETAIN_POOL * ((a.n_retain // max(len(undo.RETAIN_POOL), 1)) + 1))[:a.n_retain]
    f_prompts = [t_prompts[i % len(t_prompts)] for i in range(a.n_images)]
    gdata = torch.Generator(device=dev).manual_seed(sr.seed_for("data_generation", a.seed))

    def gen(p, src, tag):
        out = []
        with torch.no_grad():
            for i in range(0, len(p), 4):
                out.append(src(p[i:i + 4], num_inference_steps=25, guidance_scale=7.5,
                               generator=gdata, output_type="pt").images.cpu())
                print(f"[utrap2] {tag} {min(i+4,len(p))}/{len(p)}", flush=True)
        return torch.cat(out)

    f_imgs = gen(f_prompts, gen_pipe, "D_target")
    if a.target_source == "base":
        del gen_pipe
        torch.cuda.empty_cache()
    unet.load_state_dict({k: v.float() for k, v in tsd.items()})   # teacher AFTER generation
    r_imgs = gen(r_prompts, pipe, "D_retain")

    def lat(imgs):
        # .float() is load-bearing: with --target-source base the generator is a SEPARATE fp16
        # pipeline (so the attacker's data provenance matches attack_relearn.py), while the VAE
        # being encoded into is fp32. The snapshot's erase_utrap.py generated and encoded through
        # one fp32 pipe, so this mismatch did not exist there.
        with torch.no_grad():
            return torch.cat([vae.encode(imgs[i:i + 4].to(dev).float() * 2 - 1).latent_dist
                              .sample(generator=gdata) * vae.config.scaling_factor
                              for i in range(0, len(imgs), 4)])

    def embed(p):
        with torch.no_grad():
            ids = tok(p, padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            return te(ids)[0]

    f_lat, r_lat = lat(f_imgs), lat(r_imgs)
    f_emb, r_emb = embed(f_prompts), embed(r_prompts)
    del f_imgs, r_imgs
    torch.cuda.empty_cache()

    adata = core.AttackData(
        latents=f_lat, embeddings=f_emb, prompts=sorted(set(f_prompts)),
        provenance=f"{'BASE (unmodified)' if a.target_source == 'base' else a.teacher} generations, "
                   f"{a.n_images} images, 25 steps, cfg 7.5",
        split_role=t_role).validate()

    unet.requires_grad_(True)
    params = list(unet.parameters())
    n_p = sum(p.numel() for p in params)
    print(f"[utrap2] FULL scope: {n_p/1e6:.1f}M params", flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)
    gt = torch.Generator(device=dev).manual_seed(sr.seed_for("defence_build", a.seed))

    # ---- fixed-noise probe on the TARGET data: does the inner attacker actually adapt?
    pg = torch.Generator(device=dev).manual_seed(31337)
    pn_i = min(8, len(f_lat))
    p_z0, p_c = f_lat[:pn_i], f_emb[:pn_i]
    p_noise = torch.randn(p_z0.shape, device=dev, generator=pg, dtype=p_z0.dtype)
    p_ts = torch.randint(0, sched.config.num_train_timesteps, (pn_i,), device=dev,
                         generator=pg).long()
    p_zt = sched.add_noise(p_z0, p_noise, p_ts)

    def probe():
        unet.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(
                unet(p_zt, p_ts, encoder_hidden_states=p_c).sample, p_noise))
        unet.train()
        return v

    hist, probes = [], []
    nr = r_lat.shape[0]
    for step in range(a.steps):
        # 1. retain gradient at theta
        opt.zero_grad(set_to_none=True)
        ri = torch.randint(0, nr, (a.batch,), device=dev, generator=gt)
        l_retain = core.diffusion_mse(unet, sched, r_lat, r_emb, ri, gt)
        l_retain.backward()
        g_retain = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        # 2. K steps of the attacker -- THROUGH THE SHARED CORE
        backup = [p.detach().clone() for p in params]
        pr_before = probe() if (step % a.probe_every == 0) else None
        tel = core.run_attack(unet, sched, adata, spec, params=params, generator=gt,
                              eval_steps=EVAL_STEPS)
        if pr_before is not None:
            pr_after = probe()
            probes.append({"step": step, "probe_before": pr_before, "probe_after": pr_after,
                           "delta": pr_after - pr_before,
                           "inner_loss_first": tel["loss_first"],
                           "inner_loss_last": tel["loss_last"],
                           "grad_norm_first": tel["grad_norm_first"],
                           "update_norms": tel["update_norms_sampled"]})
            print(f"[utrap2] inner-attacker probe @{step}: {pr_before:.5f} -> {pr_after:.5f} "
                  f"(delta {pr_after-pr_before:+.5f})", flush=True)

        # 3. target gradient at the attacker's endpoint
        opt.zero_grad(set_to_none=True)
        fi = torch.randint(0, len(f_lat), (a.batch,), device=dev, generator=gt)
        l_end = core.diffusion_mse(unet, sched, f_lat, f_emb, fi, gt)
        l_end.backward()
        g_end = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        # 4. restore theta, apply grad L_retain - lam * grad L_target(theta_K)
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
                tot -= a.lam * ge
            p.grad = tot
        opt.step()
        del backup, g_retain, g_end

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "L_retain": float(l_retain.detach()),
                   "L_attack_step0": tel["loss_first"], "L_attack_stepK": tel["loss_last"],
                   "L_target_at_thetaK": float(l_end.detach())}
            hist.append(rec)
            print(f"[utrap2] step {step:4d} L_retain={rec['L_retain']:.5f} "
                  f"attacker {rec['L_attack_step0']}->{rec['L_attack_stepK']} "
                  f"L_target(thetaK)={rec['L_target_at_thetaK']:.5f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "utrap_v2", "implementation_version": IMPL_VERSION, "is_ours": True,
               "prereg": "docs/PREREG_H90_utrap_v2.md",
               "supersedes_and_does_not_replace": "scripts/erase_utrap.py, whose reversal "
                                                  "measurements stand as historical results",
               "inner_attacker": spec.as_record(EVAL_STEPS),
               "inner_attacker_data": adata.as_record(),
               "inner_attacker_effectiveness_probe": probes,
               "probe_note": "fixed noise, fixed timesteps, fixed batch of the TARGET pool, "
                             "evaluated immediately before and after the K-step unroll. A negative "
                             "delta means the inner attacker is genuinely adapting toward the "
                             "target; a delta near zero means the unroll is doing nothing and no "
                             "claim about unroll depth can be made from the run.",
               "approximation": "detached trajectory; first-order, K steps. A fully differentiated "
                                "unroll remains untested and is not claimed.",
               "scope": "full UNet", "n_params": n_p,
               "split_version": sr.SPLIT_VERSION,
               "rng_streams": {k: sr.seed_for(k, a.seed)
                               for k in ("defence_build", "data_generation")},
               "label": a.label, "concept": a.concept, "teacher": a.teacher, "hparams": vars(a),
               "history": hist, "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/utrap2_{a.label}.json", "w"), indent=1)
    print(f"[utrap2] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
