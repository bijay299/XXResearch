"""WDR-v2 -- a FAITHFUL adaptation of 'Weight Dist Reg' from 'From Dormant to Deleted'
(arXiv 2505.22310 section 6.1), replacing the REVERSED-direction arm in `erase_wsr.py`.

THE ERROR THIS REPAIRS (audit section 2, confirmed independently -- see ERRATUM_wsr_direction.md).
`erase_wsr.py` minimises  +lam * ||theta - theta_base||^2 / ||theta_base||^2, which pulls the model
TOWARD the pretrained weights, on a stated theory that "staying inside the original basin is what
confers tamper-resistance". The cited paper says the opposite, in both the text and the code:

  paper section 6:    "methods with higher L2 distance exhibit higher robustness against
                       relearning attacks"
  paper section 6.1:  "We directly aim to maximize the distance between the pretrained and
                       unlearned models"
  released code (shoaibahmed/vision_relearning @7b9a1cc, vision_unlearner.py L366-373):
        param_diff       = pretrained.detach() - model
        weight_dist_loss = - sqrt(mean(square(param_diff)) + 1e-8)
        loss             = unlearning_alpha * weight_dist_loss

So `erase_wsr.py` implements the opposite mechanism. FINDING61/62 therefore measured an
ANCHORING-TO-BASE method of our own, which remains a valid arm and keeps its results; it did not
test this published competitor, and the claim that it discharged it is withdrawn.

WHAT IS FAITHFUL HERE, term by term (and why a sign flip alone would NOT have been the repair):

  1. DIRECTION      -- the distance term is NEGATED, so minimising it maximises distance.
  2. NORMALISATION  -- ROOT-MEAN-SQUARE over the flattened parameter vector, with eps=1e-8 inside
                       the sqrt. Not a sum of squares, and NOT divided by ||theta_base||.
  3. ROLE           -- it REPLACES the forget objective. The reference sets
                       `forget_loss_holder = weight_dist_loss`; there is no ascent-on-forget-data
                       term at all. `erase_wsr.py` added its anchor as a third term alongside an
                       erase term and a retain term.
  4. RETAIN         -- unscaled (coefficient exactly 1.0) while the distance term carries alpha.
  5. SCHEDULE       -- !! THIS POINT WAS WRONG AND IS WITHDRAWN (Agent-3 fidelity audit,
                       2026-10-26). It claimed "ALTERNATING forget and retain iterations, as the
                       reference's training loop does, not a single summed step." The reference
                       does the OPPOSITE. Alternation there is gated on `current_cycle`, which is
                       non-None only for method names carrying the `alternating_` prefix
                       (`vision_unlearner.py:411`); `weight_dist_reg` has none, so BOTH branches
                       run every iteration and both `.backward()` calls accumulate into the SAME
                       grad buffer before ONE `optimizer.step()` (`vision_unlearner.py:637-682`).
                       The reference SUMS. This file takes TWO separate Adam steps per iteration
                       (see the loop below). That is a real divergence, it has NOT been repaired
                       (repairing it changes numerics), and it has a direct consequence for
                       FINDING67 -- see the ALPHA SCALING note below.
  6. SCOPE          -- all parameters, matching `flatten_param_dict(model.parameters())`.
  7. OPTIMISER      -- Adam (the reference's default `--optimizer-name adam`), alpha default 1.0
                       (`--unlearning-alpha` default 1), no gradient clipping
                       (`--clip-grad-norm` default None).
  8. INITIAL SAFEGUARD -- the reference allows `--initial-safeguard-args`, i.e. starting from an
                       already-unlearned model. `--teacher` reproduces that, and is the T2I analogue
                       of its CIFAR setup. Omit it to start from the base model.

DOMAIN CAVEAT, UNCHANGED AND STILL LOAD-BEARING. The paper's experiments are EXAMPLE-LEVEL
unlearning in VISION CLASSIFIERS. Its retain objective is a cross-entropy; ours must be the
diffusion loss. That substitution is a domain adaptation, declared here, and it is the one
difference that remains after the eight points above. This is a FAITHFUL ADAPTATION, not a
reproduction, and it is not evidence about the authors' CIFAR results.

KNOWN RISK, STATED BEFORE RUNNING. Maximising an unbounded distance is exactly the unbounded-
objective shape that destroyed two previous arms (FINDING61 section 4, and the IMMA alpha=1e-5
build). The reference bounds it in practice only through the retain loss and a step budget. So
`--max-rms` is provided as an explicit trust-region stop: training halts when RMS distance exceeds
it, and the artifact records whether the stop fired. That is a disclosed deviation from the
reference, added because the alternative is a predictable null, and it is recorded in the JSON as
`trust_region_stop`.

ALPHA SCALING -- FINDING67's DERIVATION IS CORRECT, ITS MECHANISM ATTRIBUTION IS NOT
(Agent-3 fidelity audit, 2026-10-26).

The closed form is confirmed analytically. With d = theta_pre - theta, S = mean(d^2),
L = -alpha*sqrt(S + eps):
    dL/dp_j = alpha * d_j / (N * sqrt(S + eps))
    ||grad L|| = alpha*||d|| / (N*sqrt(S)) = alpha*sqrt(N*S)/(N*sqrt(S)) = alpha / sqrt(N)
exactly, independent of distance, for S >> eps. At N = 859,520,964 that is 3.411e-5 at alpha = 1,
matching the measured value to 4 s.f. FINDING67 section 3 is VALID as algebra.

What does NOT follow is the reading that alpha=1 is inert "against a retain-term gradient of
~0.137". The two gradients are never summed in this file -- they go into two separate Adam steps.
Under Adam a step's size is set by g/(sqrt(v)+eps_adam), which is INVARIANT to scaling the loss, so
a gradient-norm ratio between two separately-stepped terms is not the quantity that decides
anything. The reason alpha nevertheless matters here is narrower and is an ARTIFACT of this file's
construction: both steps share ONE Adam instance, so the second-moment buffer `v` is an EMA mixing
both terms and is dominated by the larger (retain) one. The distance step's update is then
~lr*|g_dist|/sqrt(v_retain), which IS linear in alpha until alpha is large enough for the distance
term to dominate `v` itself -- past which alpha saturates and is genuinely inert again.

Three consequences, stated rather than glossed:
  * alpha is inert at BOTH ends -- below ~10 because `v` is retain-dominated, above ~4e3 because
    Adam is scale-invariant once the term dominates `v`. FINDING67's "useful window alpha in
    (1, 4000)" is right; the upper end is a saturation point, not a frontier.
  * the "4000x" factor is a property of the shared-Adam-state two-step schedule, NOT of the
    reference method (which sums into one step) and NOT of the 1/sqrt(N) parity it was attributed
    to. Under the reference's summed schedule the gradient-norm ratio WOULD be the right frame.
  * gradient-norm parity is a DIRECTIONAL HEURISTIC, not an optimum. Nothing establishes that equal
    gradient norms are the best trade-off between two objectives, and under Adam the overall scale
    is normalised away in any case. `--alpha 4000` should be read as "the point at which the term
    starts moving the weights", not as a tuned or optimal value.

SMALL LATENT ISSUE, no repair needed: with `--teacher` omitted, theta == theta_pre at step 0, so
d = 0 and the forget gradient is exactly 0 (the eps guard makes it finite, and zero). The first
forget iteration is a no-op; the term becomes live only after the first retain step moves theta.

Run (no result is claimed from this file until a build and a frozen-grid evaluation have run):
  python scripts/erase_wdr_v2.py --concept church --alpha 1.0 --max-rms 0.02 \
      --teacher adapters/frp_church_r3_l05/unet.pt \
      --label wdr2_church_a1.0 --out adapters/wdr2_church_a1.0
"""
import argparse, importlib.util as iu, json, os, time
import torch
import torch.nn.functional as F

import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
IMPL_VERSION = "wdr/v2-faithful-direction"


def _mod(name):
    spec = iu.spec_from_file_location(f"_w_{name}", f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="reference --unlearning-alpha, default 1 (vision_unlearner.py L1848). "
                         "WARNING (FINDING67): the default is INERT here. This term's gradient norm "
                         "is exactly 1/sqrt(N) = 3.41e-5 at N=859.5M -- constant in distance and "
                         "SHRINKING with model size -- against a retain-term gradient of ~0.137. "
                         "Measured parity is alpha ~= 4.0e3, i.e. the reference's default is ~4000x "
                         "too small on this UNet. Run scripts/diag_wdr_scale.py before choosing a "
                         "value; do not inherit a coefficient across model scales.")
    ap.add_argument("--teacher", default=None,
                    help="reference --initial-safeguard-args analogue: start from an already-erased "
                         "unet. Omit to start from the base model.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1500, help="alternating forget+retain iterations")
    ap.add_argument("--max-rms", type=float, default=0.02,
                    help="DISCLOSED DEVIATION: trust-region stop on RMS distance. The reference "
                         "has no such bound; without one an unbounded ascent predictably destroys "
                         "the model (FINDING61 section 4).")
    ap.add_argument("--n-retain", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    obj, sr = _mod("objectives"), _mod("split_registry")

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, revision=(BASE_REV_ENV or pin["revision"]), torch_dtype=torch.float32,
        variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    for m in (vae, te):
        m.requires_grad_(False)

    # theta_pre: the PRETRAINED weights. The reference's distance is always measured to the
    # pretrained model, never to the safeguarded starting point -- so this snapshot is taken
    # BEFORE any teacher is loaded.
    ref_cpu = [p.detach().clone() for p in unet.parameters()]

    if a.teacher:
        tsd = torch.load(a.teacher, map_location="cpu"); tsd = tsd.get("unet", tsd)
        unet.load_state_dict({k: v.float() for k, v in tsd.items()})
        print(f"[wdr2] initial safeguard: {a.teacher}", flush=True)

    # retain data only: the faithful objective has NO forget-data term (difference 3 above).
    undo = _mod("undo_t2i")
    r_prompts = (undo.RETAIN_POOL * ((a.n_retain // max(len(undo.RETAIN_POOL), 1)) + 1))[:a.n_retain]
    g = torch.Generator(device=dev).manual_seed(sr.seed_for("data_generation", a.seed))

    with torch.no_grad():
        imgs = []
        for i in range(0, len(r_prompts), 4):
            imgs.append(pipe(r_prompts[i:i + 4], num_inference_steps=25, guidance_scale=7.5,
                             generator=g, output_type="pt").images.cpu())
            print(f"[wdr2] D_retain {min(i+4,len(r_prompts))}/{len(r_prompts)}", flush=True)
        imgs = torch.cat(imgs)
        r_lat = torch.cat([vae.encode(imgs[i:i + 4].to(dev) * 2 - 1).latent_dist.sample(generator=g)
                           * vae.config.scaling_factor for i in range(0, len(imgs), 4)])
        ids = tok(r_prompts, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        r_emb = te(ids)[0]
    del imgs
    torch.cuda.empty_cache()

    unet.requires_grad_(True)
    params = list(unet.parameters())
    ref = [r.to(dev) for r in ref_cpu]
    opt = torch.optim.Adam(params, lr=a.lr)   # reference default optimiser; no clipping
    gt = torch.Generator(device=dev).manual_seed(sr.seed_for("defence_build", a.seed))

    hist, stop = [], None
    nr = r_lat.shape[0]
    for step in range(a.steps):
        # ---- FORGET iteration: alpha * ( -RMS(theta_pre - theta) ). Maximises distance.
        opt.zero_grad(set_to_none=True)
        l_forget = obj.wdr_forget_loss(params, ref, alpha=a.alpha)
        l_forget.backward()
        opt.step()

        # ---- RETAIN iteration: unscaled diffusion loss, SEPARATE Adam step.
        # CORRECTED COMMENT (Agent-3 audit): this said "alternating, as reference". The reference
        # SUMS both terms into one step (vision_unlearner.py:411,637-682); only method names with
        # the `alternating_` prefix alternate there, and `weight_dist_reg` is not one. Two separate
        # steps sharing ONE Adam instance is also what makes --alpha appear to matter; see the
        # ALPHA SCALING note in the module docstring. Unchanged: fixing it changes numerics.
        opt.zero_grad(set_to_none=True)
        ri = torch.randint(0, nr, (a.batch,), device=dev, generator=gt)
        z0 = r_lat[ri]
        noise = torch.randn(z0.shape, device=dev, generator=gt)
        ts = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                           generator=gt).long()
        l_retain = F.mse_loss(unet(sched.add_noise(z0, noise, ts), ts,
                                   encoder_hidden_states=r_emb[ri]).sample, noise)
        l_retain.backward()
        opt.step()

        with torch.no_grad():
            rms = float(obj.wdr_rms_distance(params, ref))
        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "l_forget": float(l_forget.detach()),
                   "l_retain": float(l_retain.detach()), "rms_distance": rms}
            hist.append(rec)
            print(f"[wdr2] step {step:5d} forget={rec['l_forget']:.6f} "
                  f"retain={rec['l_retain']:.5f} RMS={rms:.6f}", flush=True)
        if rms > a.max_rms:
            stop = {"fired": True, "step": step, "rms": rms, "max_rms": a.max_rms}
            print(f"[wdr2] TRUST-REGION STOP at step {step}: RMS {rms:.6f} > {a.max_rms}",
                  flush=True)
            break

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "wdr_erase_v2", "implementation_version": IMPL_VERSION,
               "reproduces": "arXiv 2505.22310 section 6.1 'Weight Dist Reg'",
               "reference_code": "shoaibahmed/vision_relearning@7b9a1ccada35a4059604c2688d2822567"
                                 "b3cf40d vision_unlearner.py L366-373",
               "supersedes_and_does_not_replace": "scripts/erase_wsr.py, which implements the "
                                                  "OPPOSITE direction and keeps its own results "
                                                  "as an anchoring-to-base arm",
               "objective": "alternating: (a) alpha * ( -sqrt(mean((theta_pre-theta)^2)+1e-8) ) ; "
                            "(b) diffusion MSE on retain data, coefficient 1.0",
               "faithfulness": {
                   "direction": "MAXIMISES distance (reference)",
                   "normalisation": "RMS over flattened params, eps=1e-8 (reference)",
                   "forget_data_term": "NONE -- the distance term IS the forget loss (reference)",
                   "retain_coefficient": 1.0,
                   # CORRECTED 2026-10-26 (Agent-3 audit): this read "alternating forget/retain
                   # iterations (reference)". It is a DIVERGENCE, not a match.
                   "schedule": "DIVERGENT: two separate Adam steps per iteration. The reference "
                               "SUMS both terms into a single optimizer.step() "
                               "(vision_unlearner.py:411, 637-682); only `alternating_`-prefixed "
                               "methods alternate there and weight_dist_reg is not one.",
                   "schedule_consequence": "the two terms share one Adam instance, so --alpha acts "
                                           "through the shared second-moment buffer rather than "
                                           "through gradient-norm parity; see FINDING67 "
                                           "reinterpretation in docs/AGENT3_FIDELITY_FINDINGS.md",
                   "scope": "all UNet parameters (matches flatten_param_dict(model.parameters()); "
                            "reference excludes buffers, as do we)",
                   "optimiser": "Adam, no grad clipping (reference defaults VERIFIED: "
                                "--optimizer-name default 'adam' vision_unlearner.py:1834, "
                                "--clip-grad-norm default None :1832)",
                   "domain_adaptation": "retain objective is the diffusion MSE, not cross-entropy; "
                                        "the paper is example-level unlearning in vision "
                                        "classifiers",
                   "disclosed_deviation": "--max-rms trust-region stop; the reference has none"},
               "trust_region_stop": stop or {"fired": False},
               "split_version": sr.SPLIT_VERSION,
               "rng_streams": {"data_generation": sr.seed_for("data_generation", a.seed),
                               "defence_build": sr.seed_for("defence_build", a.seed)},
               "label": a.label, "concept": a.concept, "hparams": vars(a),
               "n_params": sum(p.numel() for p in params),
               "history": hist, "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/wdr2_{a.label}.json", "w"), indent=1)
    print(f"[wdr2] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
