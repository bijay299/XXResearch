"""IMMA -- OUR PORT of arXiv 2311.18815 (Zheng & Yeh, ECCV 2024) to Stable Diffusion v1.4.

THIS IS A PUBLISHED METHOD, NOT OURS. Transcribed from Section 4.1, Eq. (3)-(5), Algorithm 1 and
the Implementation-details paragraph. Port decisions are pre-declared in docs/PREREG_H82_imma_port.md.

!! PROVENANCE CORRECTION (Agent-3 fidelity audit, 2026-10-26). This file and its artifact recorded
`code_released: False`. THAT IS WRONG: the authors' code IS public and is now cloned at
`upstream/IMMA` (github.com/amberyzheng/IMMA @2f55041141705031d503134dec22f674f09e21bf). This port
was written from the PAPER and has NOT been reconciled against that code. Four divergences found by
reading it, none of them repaired here (repairing them would change numerics):

  1. SCOPE, 2.13x. The reference filters on `['to_q','to_v','to_k','to_out']` with NO attn2
     restriction (`upstream/IMMA/train/defend_text_to_image_lora.py:561-567`, same code in
     `defend_dreambooth.py:882-889`), so it immunizes SELF-attention (attn1) as well as
     cross-attention (attn2) -- despite the authors naming the variable `xattn_layer_names`.
     Reference scope = attn1 + attn2 to_q/k/v/out = 80 + 80 = 160 tensors, 93.5M params
     (attn1 49.6M + attn2 44.0M). OURS = attn2 only, 80 tensors, 44.0M. Ratio 2.13x.
     Our S follows the paper's TEXT ("cross-attention layers"); the released CODE is broader. Which
     of the two is "IMMA" is genuinely ambiguous, and no result here may be read as a test of the
     released code's scope.
  2. NO OUTER GRADIENT CLIPPING here. The reference clips the upper-level update to
     `--max_grad_norm` 1.0 (`defend_dreambooth.py:1297-1299`). This port clips neither level.
  3. OPTIMIZER. The reference uses AdamW (weight_decay 1e-2); this port uses Adam (wd 0).
  4. HYPERPARAMETERS. `alpha`/`inner_lr`/`I`/`rank` are recorded below as "unassigned_in_paper".
     That is true of the paper but NOT of the code, which supplies them: outer lr 1e-6 (README 1e-5)
     and LoRA lr 1e-4 (`defend_text_to_image_lora.py:213-224`), inner:outer = 1:1
     (`:515-520`), batch 16 (README 1). Our inner_lr 1e-4 coincides; our alpha 1e-5 and batch 2
     do not. Warm-starting phi matches (`--reinitialize_lora` default False, `:366`).

The OBJECTIVE and its SIGNS do match the reference exactly: one unweighted diffusion MSE, descended
on phi at the inner step and ASCENDED on S at the outer step (`accelerator.backward(loss)` vs
`accelerator.backward(-loss)`, `defend_text_to_image_lora.py:944-960`), with two separate
optimizers and no loss coefficients at all. The reference is also strictly FIRST-ORDER -- no
`create_graph`, no `higher`, no `functional_call` anywhere in its tree -- so this port's
alternating-step structure is faithful on the bi-level mechanics.

Eq. (3), the bi-level program:

    max_{theta in S} L_A(x'_I, c'; theta, phi*)   s.t.  phi* = argmin_phi L_A(x'_A, c'; theta, phi)

Algorithm 1, verbatim structure:

    theta_0 = theta_p ; initialise phi_0 per A
    for i = 1..I:
        sample batches x'_A and x'_I INDEPENDENTLY from D
        phi <- phi_{i-1}                                   # WARM START, not reinitialised (line 5)
        phi_i <- argmin_phi L_A(x'_A, c'; theta_{i-1}, phi)   # ONE gradient step in practice
        theta_{i-1 in U} <- phi_i                          # assign overlaps, variant (b) (line 7)
        theta_{i in S} <- theta_{i-1 in S} + alpha * grad_theta L_A(x'_I, c'; theta_{i-1}, phi_i)

WHAT MAKES IMMA DIFFERENT from the two bi-level defences already ported here:
  * the inner loop runs the REAL adversary (LoRA), warm-started across outer iterations -- not a
    simulated single step from scratch (Meta-Unlearning) and not an alternation against a
    prior-preservation loss (GIFT);
  * the upper level ASCENDS the adaptation loss directly (+alpha * grad);
  * it is ADVERSARY-SPECIFIC by construction: an arm immunized against LoRA is not thereby
    immunized against anything else. PREREG_H82 S4 therefore makes LoRA the PRIMARY measurement
    and treats our usual full-UNet grid as a separate transfer question.

Settings the paper DOES specify, and that this port follows rather than substituting our own:
  * S = cross-attention layers only;
  * ADAM for the upper-level task (note: our GIFT port deliberately used plain SGD, because THAT
    paper's curvature derivation required it -- IMMA states Adam, so Adam it is);
  * one gradient step for the lower-level task;
  * overlap variant (b).
Pre-declared by us because unspecified: alpha, inner lr, I, LoRA rank.

Run: python scripts/erase_imma.py --concept church \
         --teacher adapters/frp_church_r3_l05/unet.pt --label imma_church_frp --out adapters/imma_church_frp
"""
import argparse, importlib.util as iu, json, os, time
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def _load(mod, path):
    spec = iu.spec_from_file_location(mod, path)
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--teacher", required=True, help="the ERASED unet to immunize (IMMA S5.1 "
                                                     "immunizes an already-erased model)")
    ap.add_argument("--alpha", type=float, default=1e-5, help="upper-level lr; UNSPECIFIED in paper")
    ap.add_argument("--inner-lr", type=float, default=1e-4, help="lower-level lr; UNSPECIFIED")
    ap.add_argument("--steps", type=int, default=1500, help="I; matches H80/H81 budgets")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank, matching FINDING53's adversary")
    ap.add_argument("--n-images", type=int, default=20,
                    help="|D|. IMMA S5.1 uses 20 images generated by SD BEFORE erasing.")
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

    # D is generated by the PRE-ERASURE model (IMMA S5.1: "20 images generated by Stable Diffusion
    # (SD V1-4), prior to erasing"). So generate BEFORE loading the erased weights.
    g = torch.Generator(device=dev).manual_seed(a.seed)
    prompts = [f"a photo of a {a.concept}"] * a.n_images
    imgs = []
    with torch.no_grad():
        for i in range(0, a.n_images, 4):
            imgs.append(pipe(prompts[i:i + 4], num_inference_steps=25, guidance_scale=7.5,
                             generator=g, output_type="pt").images.cpu())
            print(f"[imma] D (pre-erasure): {min(i+4, a.n_images)}/{a.n_images}", flush=True)
    imgs = torch.cat(imgs)

    sd = torch.load(a.teacher, map_location="cpu")
    unet.load_state_dict({k: v.float() for k, v in sd.get("unet", sd).items()})
    print(f"[imma] loaded erased teacher {a.teacher}", flush=True)

    lat = []
    with torch.no_grad():
        for i in range(0, len(imgs), 4):
            b = imgs[i:i + 4].to(dev) * 2 - 1
            lat.append(vae.encode(b).latent_dist.sample(generator=g) * vae.config.scaling_factor)
    lat = torch.cat(lat)
    with torch.no_grad():
        ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        emb = te(ids)[0]
    del imgs
    torch.cuda.empty_cache()

    # phi: the REAL adversary. Reuses FINDING53's LoRA so the thing IMMA trains against inside its
    # loop is exactly the thing we attack it with afterwards.
    al = _load("attack_lora", f"{P}/scripts/attack_lora.py")
    injected = al.inject_lora(unet, a.rank, a.rank, "cross")
    unet.to(dev)
    phi = [p for n, p in unet.named_parameters() if ".A." in n or ".B." in n
           or n.endswith(".A.weight") or n.endswith(".B.weight")]
    if not phi:
        phi = [p for n, p in unet.named_parameters() if ("lora" in n.lower() or ".A.weight" in n
               or ".B.weight" in n)]
    assert phi, "no LoRA parameters found after injection"

    # S: cross-attention only, per the paper's TEXT -- ALL attn2 projections
    # (to_q/to_k/to_v/to_out), not just the ones LoRA happens to wrap.
    # NB (Agent-3 audit): the authors' RELEASED code omits the ".attn2." test and therefore also
    # immunizes attn1 (93.5M, 2.13x this scope). See the module docstring item 1. Left unchanged so
    # the historical IMMA arms stay reproducible; a matched-scope arm would be a new build.
    # Matching on ".base." alone would have silently restricted
    # S to the 16 to_out projections LoRA injects (12.4M params) instead of the paper's full
    # cross-attention surface (44.0M), because inject_lora's scope is narrower than its name
    # suggests -- see the LoRA-scope correction in docs/.
    S_names, S = [], []
    for n, p in unet.named_parameters():
        if ".attn2." not in n:
            continue
        if n.endswith(".A.weight") or n.endswith(".B.weight"):
            continue                      # phi, not S
        if any(k in n for k in ("to_q", "to_k", "to_v", "to_out")):
            S_names.append(n); S.append(p)
    assert S, "empty S"

    for p in unet.parameters():
        p.requires_grad_(False)
    for p in phi + S:
        p.requires_grad_(True)

    n_S = sum(p.numel() for p in S); n_phi = sum(p.numel() for p in phi)
    print(f"[imma] S = {len(S_names)} cross-attn tensors, {n_S/1e6:.1f}M params", flush=True)
    print(f"[imma] phi = LoRA rank {a.rank} on {len(injected)} linears, {n_phi/1e6:.3f}M params",
          flush=True)

    opt_inner = torch.optim.Adam(phi, lr=a.inner_lr)     # lower level
    opt_outer = torch.optim.Adam(S, lr=a.alpha)          # upper level: Adam, per the paper

    def L_A(idx):
        """The adaptation loss: the diffusion objective with LoRA active."""
        z0 = lat[idx]
        noise = torch.randn(z0.shape, device=dev, generator=g)
        t = torch.randint(0, sched.config.num_train_timesteps, (z0.shape[0],), device=dev,
                          generator=g).long()
        zt = sched.add_noise(z0, noise, t)
        return F.mse_loss(unet(zt, t, encoder_hidden_states=emb[idx]).sample, noise)

    hist = []
    n = lat.shape[0]
    for i in range(a.steps):
        # x'_A and x'_I are sampled INDEPENDENTLY (Alg 1 line 4)
        idx_A = torch.randint(0, n, (a.batch,), device=dev, generator=g)
        idx_I = torch.randint(0, n, (a.batch,), device=dev, generator=g)

        # ---- lower level (Alg 1 lines 5-6): ONE gradient step on phi, warm-started (phi is never
        # reinitialised -- it carries over from the previous outer iteration by construction).
        opt_inner.zero_grad(set_to_none=True)
        l_inner = L_A(idx_A)
        l_inner.backward()
        opt_inner.step()

        # ---- upper level (Alg 1 line 8): ASCEND L_A w.r.t. S, at the adapted phi_i.
        # Line 7 (theta_U <- phi_i) is a NO-OP here: LoRA introduces new parameters, so U is empty.
        # It becomes live only for DreamBooth, where phi is a subset of theta.
        opt_outer.zero_grad(set_to_none=True)
        l_upper = L_A(idx_I)
        (-l_upper).backward()        # maximise => descend the negation
        opt_outer.step()

        if i % a.log_every == 0 or i == a.steps - 1:
            rec = {"step": i, "L_inner_adapt": float(l_inner.detach()),
                   "L_upper_adapt": float(l_upper.detach())}
            hist.append(rec)
            print(f"[imma] step {i:5d} L_A(inner,phi)={rec['L_inner_adapt']:.5f} "
                  f"L_A(upper,S)={rec['L_upper_adapt']:.5f}", flush=True)

    # Strip phi before saving: the immunized artifact is theta_I, NOT theta_I plus a trained
    # adversary. Saving the LoRA would hand the attacker a head start and make the arm
    # incomparable to every other arm in this phase.
    state = {k.replace(".base.", "."): v for k, v in unet.state_dict().items()
             if ".A.weight" not in k and ".B.weight" not in k}
    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in state.items()}, f"{a.out}/unet.pt")

    rec = {"study": "imma_port", "prereg": "PREREG_H82_imma_port.md",
           "paper": "arXiv 2311.18815", "is_our_port": True,
           # CORRECTED 2026-10-26 (Agent-3 audit). This field read `False`, which was wrong.
           "code_released": True,
           "official_code": "github.com/amberyzheng/IMMA@2f55041141705031d503134dec22f674f09e21bf"
                            " (cloned at upstream/IMMA)",
           "reconciled_against_official_code": False,
           "divergences_from_official_code": {
               "scope": "reference filters to_q/to_v/to_k/to_out with NO attn2 restriction, so it "
                        "covers attn1+attn2 = 93.5M (160 tensors); ours is attn2-only 44.0M "
                        "(80 tensors). Ratio 2.13x. Ours follows the paper's text, not the code. "
                        "Source: upstream/IMMA/train/defend_text_to_image_lora.py:561-567",
               "grad_clipping": "reference clips the OUTER update to max_grad_norm=1.0 "
                                "(defend_dreambooth.py:1297-1299); this port clips neither level",
               "optimizer": "reference AdamW (wd 1e-2); this port Adam (wd 0)",
               "hyperparameters": "the CODE supplies outer lr 1e-6 / README 1e-5 and LoRA lr 1e-4; "
                                  "our inner_lr 1e-4 coincides, our alpha 1e-5 and batch 2 do not"},
           "objective_matches_official_code": True,
           "objective_match_note": "one unweighted diffusion MSE, descended on phi and ASCENDED on "
                                   "S, two separate optimizers, no loss coefficients; reference is "
                                   "strictly first-order (no create_graph/higher/functional_call)",
           "label": a.label, "concept": a.concept, "teacher": a.teacher,
           "hparams": vars(a),
           "specified_by_paper": ["S=cross_attention_only", "adam_upper_level",
                                  "one_inner_gradient_step", "overlap_variant_b"],
           "unassigned_in_paper": ["alpha", "inner_lr", "steps", "rank"],
           "unassigned_in_paper_note": "true of the PAPER only; the released code assigns all four",
           "U_is_empty_for_lora": True,
           "n_S_params": n_S, "n_phi_params": n_phi, "S_names": S_names,
           "history": hist, "minutes": (time.time() - t0) / 60}
    json.dump(rec, open(f"{OUT}/imma_{a.label}.json", "w"), indent=1)
    print(f"[imma] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
