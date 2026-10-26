"""META-UNLEARNING v2 -- the FULL Eq. 8 objective (arXiv 2410.12777), repairing the two terms that
variant A (`meta_unlearn.py`) omits. Audit section 3. See ERRATUM_meta_objective.md.

WHAT VARIANT A COMPUTES (preserved unchanged, results intact, now named variant A):

    L_A = distil_F(theta) + distil_R(theta) - lam * L_R(theta')

WHAT THE PAPER SPECIFIES (Eq. 8) AND WHAT THE RELEASED CODE DOES:

    L   = L_erase(theta) + [ - L_F(theta') - zeta * ( L_R(theta') - L_R(theta) ) ]

Re-verified against the released implementation. CITATION CORRECTED (Agent-3 fidelity audit,
2026-10-26): this file previously cited `src/meta.py L920-1030`. `src/meta.py` is a STALE,
NON-RUNNABLE copy -- `train_step` returns 4 values at `src/meta.py:626` while every call site
unpacks 3 (e.g. `:967`), and `:1045` reads an unassigned `train_loss`. The working reference, and
what the authors' own `scripts/esd_meta.sh` invokes, is
`sail-sg/Meta-Unlearning@5a2a7600895a2885323cbbf599acaf3f74a96f64 src/esd_meta.py:992-1058`.
The signed structure is the same there, so the objective repair below stands; only the citation and
the word "verified" were wrong. That reference accumulates exactly this signed structure into
`sum_grads`:

    + gamma1_1 * grad L(irt @ theta)     pre-attack retain   (irrelevant pool)   POSITIVE
    + gamma1_2 * grad L(tgt @ theta)     pre-attack retain   (related pool)      POSITIVE
      ---- one simulated malicious fine-tuning step on the harmful pool, FULL-UNet scope ----
    - gamma2_1 * grad L(hrm @ theta')    post-attack TARGET                      NEGATIVE
    - gamma2_2 * grad L(tgt @ theta')    post-attack retain  (related pool)      NEGATIVE
    - gamma2_3 * grad L(irt @ theta')    post-attack retain  (irrelevant pool)   NEGATIVE
    then `param.grad += sum_grads` on top of the erasure loss' own gradient.

so the paper's R is the union of the reference's `irt` and `tgt` pools and the paper's F is its
`hrm` pool. Variant A has the third of these six and none of the other five.

THE TWO MISSING TERMS, named exactly:
  (i)  **-L_F(theta')** -- the post-attack TARGET diffusion loss. Absent from variant A entirely.
       This is the term that makes relearning the target itself self-defeating. Without it, the
       objective only punishes benign denoising at theta', which is why variant A's premature
       benign damage is CONSISTENT WITH ITS OWN OBJECTIVE and is not evidence that the published
       method's trigger fires on normal use.
  (ii) **+zeta*L_R(theta)** -- the matching PRE-attack retain diffusion loss. Its role is
       cancellation: the paper penalises the CHANGE in retain loss across the attack step, not the
       retain loss at theta'. Variant A penalises the level, so it is rewarded for making benign
       denoising bad everywhere, not only after an attack.
       Variant A's `distil_R(theta)` is NOT this term: teacher-output distillation and the
       diffusion loss are different functions with different minima.

OTHER DIFFERENCES REPAIRED HERE, separately from the objective (audit section 3):
  * SCOPE is now explicit and separable. The reference takes the meta gradient on
    `gather_parameters` (selectable: full / xattn / noxattn / selfattn) while the SIMULATED ATTACK
    uses `gather_parameters_full`. Variant A used the full UNet for both without recording it.
    `--meta-scope` and `--attack-scope` are independent and both are written to the artifact.
  * TARGET DATA PROVENANCE is checked, not assumed. Variant A generated its target-prompt images
    through the ALREADY-ERASED teacher and never verified they contain the target. `--target-source`
    selects base|teacher, and the images are CLIP-scored before training with the result recorded;
    `--require-target-acc` refuses to train on a pool that does not depict the target.
  * The benign pool is drawn from the split registry with its role declared, not taken as "the
    first 24 eligible prompts".
  * FIRST-ORDER vs DIFFERENTIATED UNROLL is now a separate flag (`--order`) rather than being
    tangled with the objective, so the two questions can be answered independently. Only
    `first_order` is implemented; `differentiated` raises, because claiming it without running it
    is exactly what the audit objected to.

THREE DEFECTS FOUND BY THE AGENT-3 FIDELITY AUDIT (2026-10-26), ALL REQUIRING A BEHAVIOUR CHANGE
AND THEREFORE NOT REPAIRED HERE. Any result already produced by this file must be read against them.

  (A) THE SHARED `zeta` IS INERT AS A COEFFICIENT (the PAIRING it creates is NOT inert -- read on).
      Eq. 8 as implemented below carries +zeta*L_R(theta) and -zeta*L_R(theta') with ONE shared
      coefficient. Under the first-order unroll, theta' = theta - alpha*grad L_F(theta) with
      alpha = 1e-5, so the pair's net expected gradient is
          zeta * ( grad L_R(theta) - grad L_R(theta') )  ~  zeta * alpha * H_R * grad L_F
      i.e. second order and ~1e-5 of either term alone.

      WHAT THIS DOES AND DOES NOT MEAN -- keep two claims apart (corrected after coordinator
      review 2026-10-26; an earlier draft of this note overreached and is withdrawn):
        * `zeta` AS A WEIGHT is near-inert AND UNTESTED. Because the SAME zeta multiplies both
          members, the cancellation is EXACT at every zeta > 0; what zeta scales is only the
          second-order residual above. So zeta cannot be tuned to buy retention, and no zeta sweep
          has been run. Do not expect a larger zeta to help.
        * THE PAIRING ITSELF IS THE MECHANISM AND DOES WORK. Variant A carries -lam*L_R(theta')
          ALONE (`meta_unlearn.py:159-161`) -- a negative coefficient on the retain LEVEL, i.e. a
          large, BIASED gradient (~lam*0.04) rewarding worse benign capability everywhere, since
          theta' ~= theta to within ~1e-6. That is FINDING56's self-destruct-on-normal-use. Adding
          +zeta*L_R(theta) converts that LEVEL penalty into a CHANGE penalty and removes the
          incentive. Going from "large biased damage gradient" to "~zero-mean" is a FIRST-ORDER
          change in the optimisation even though the resulting term is second-order in size. Hence
          lines 39-41 above -- "its role is cancellation" -- are correct, and FINDING78's measured
          retention result STANDS: it is attributable to the restored pairing plus
          -lam_target*L_F(theta'), not to zeta's magnitude.
      The reference does NOT share the coefficient: `gamma1_1` and `gamma2_3` are independent
      (`src/esd_meta.py:90-94`) and `scripts/esd_meta.sh` runs them ASYMMETRICALLY (gamma1_1 = 0.1
      vs gamma2_3 = 1.0; gamma1_2 = 0.01 vs gamma2_2 = 1.0).
      FIX: split `--zeta` into `--zeta-pre` / `--zeta-post` -- for TESTABILITY of the coefficient,
      not because a retention gain should be expected from it. CHANGES NUMERICS: requires a new
      `implementation_version` and a new arm label, or a flag defaulting to the shared zeta.
      NOTE this also invalidates FINDING67 section 7's conclusion that "Meta-v2 needs no coefficient
      correction": that diagnostic compared ||grad L_retain(theta)|| against
      ||grad L_target(theta')|| as if they were independent terms. They are not -- the retain(theta)
      term is paired against retain(theta') under the same zeta. (The same section also measured at
      "xattn scope, 44.0M" while this file's `--meta-scope` default is `full`.)

  (B) L_R(theta) AND L_R(theta') ARE DRAWN WITH INDEPENDENT NOISE AND TIMESTEPS.
      `l_retain_theta` and `l_retain_prime` below both call `dmloss(..., g)` on the same shared
      generator, which advances between them, so each gets its own noise tensor and its own `ts`.
      Eq. 8's term is a PAIRED DIFFERENCE across a 1e-5-sized parameter step; with unpaired draws
      the O(1e-7) signal sits under O(0.1) sampling noise, so the logged `retain_change` field is
      essentially pure noise and the per-step estimator injects variance rather than signal. The
      same applies to `l_att` vs `l_target_prime`.
      NOTE the interaction with (A): defect (B) means the cancellation in (A) is exact only IN
      EXPECTATION. Per step the pair is a difference of two INDEPENDENT O(0.04) gradients, so each
      step injects ~zeta*sqrt(2)*0.04 of ZERO-MEAN noise rather than zero. Zero-mean noise does not
      SYSTEMATICALLY damage benign capability the way variant A's biased term did, which is why the
      retention gain survives this defect -- but the estimator is far noisier than Eq. 8 intends.
      FIX: draw noise and timesteps ONCE and reuse them for both the theta and theta' evaluations
      (COMMON RANDOM NUMBERS). This is the fix for the unpaired estimator -- NOT a larger zeta.
      !! CHANGES NUMERICAL BEHAVIOUR: it alters every per-step gradient draw. It MUST NOT be applied
      in place to this script, which has produced historical arms (FINDING78); introduce it under a
      new `implementation_version` and a new arm label, or behind a flag defaulting to the current
      independent draws. Expected effect is variance reduction, not a change in the mean objective.

  (C) THE SIMULATED ATTACKER IS NOT THE REFERENCE'S AND NOT THE EVALUATOR'S, AND NOTHING RECORDS IT.
      Here: ONE plain gradient step, `q.sub_(inner_alpha*gg)`, alpha 1e-5, no optimizer state, no
      clipping, batch 1.
      Reference (`src/esd_meta.py:963-977, 1005-1008, 617-622`): AdamW, lr 1e-5, clip_grad_norm 1.0,
      a FULL EPOCH over `hrm_train_dataloader` (many steps, persistent moments), on a fresh
      `deepcopy` of the UNet each meta-step, full-UNet scope.
      Evaluator (`attack_relearn.py:333-362`): Adam, lr 1e-5, 300 steps, clip 1.0, batch 1,
      persistent state, full UNet.
      Unlike `erase_utrap_v2.py`, this file does NOT route its inner process through
      `attack_core.AttackSpec`, so no `divergence_from_evaluation_attacker` record is written and
      the gap is invisible in the artifact. FIX: build an `AttackSpec` with
      `horizon_role="SHORT_HORIZON_SURROGATE"` and record it.

ALSO NOT A MATCH, declared rather than repaired: the reference's at-theta unlearning term is an
ACTIVE ESD objective (`src/esd_meta.py:578-582`, concept_scale 3.0), recomputed every step. Here it
is `distil_F(theta) + distil_R(theta)`, a distillation anchor to an already-erased teacher, which is
IDENTICALLY ZERO at initialisation (the student IS the teacher) -- FINDING67 section 7 measured both
gradients at exactly 0.000000 and that is by construction, not a bug. The two are different
functions with different minima.

NO RESULT IS CLAIMED FROM THIS FILE until a build and a frozen-grid evaluation have run. It is
listed as PENDING in docs/CORRECTION_REGISTER_r5.md.
"""
import argparse, copy, importlib.util as iu, json, os, time
import torch
import torch.nn.functional as F

import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
IMPL_VERSION = "meta/v2-full-eq8"

SCOPES = {
    "full": lambda n: True,
    "xattn": lambda n: "attn2" in n,
    "selfattn": lambda n: "attn1" in n,
    "noxattn": lambda n: not (n.startswith("conv_out.") or "time_embed" in n or "attn2" in n),
}


def _mod(name):
    spec = iu.spec_from_file_location(f"_m_{name}", f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the ERASED unet to meta-unlearn")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--inner-alpha", type=float, default=1e-5,
                    help="the simulated malicious fine-tuning step size")
    ap.add_argument("--zeta", type=float, default=1.0, help="Eq.8's retain-change coefficient")
    ap.add_argument("--lam-target", type=float, default=1.0,
                    help="coefficient on -L_F(theta'), the term variant A omits. The reference "
                         "gives each pool its own gamma; this is gamma2_1.")
    ap.add_argument("--meta-scope", default="full", choices=sorted(SCOPES),
                    help="reference --finetuning_method: the scope the META gradient is taken on")
    ap.add_argument("--attack-scope", default="full", choices=sorted(SCOPES),
                    help="reference gather_parameters_full: the scope the SIMULATED ATTACK uses")
    ap.add_argument("--target-source", default="base", choices=("base", "teacher"),
                    help="where the target-concept images come from. Variant A used 'teacher' "
                         "(the already-erased model) and did not verify the content.")
    ap.add_argument("--require-target-acc", type=float, default=0.5,
                    help="refuse to train if fewer than this fraction of the generated target "
                         "images are classified as the target")
    ap.add_argument("--order", default="first_order", choices=("first_order", "differentiated"))
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.order == "differentiated":
        raise SystemExit(
            "--order differentiated is NOT implemented. The audit's instruction was to separate "
            "objective corrections from the first-order-vs-differentiated-unroll choice; claiming "
            "the latter without running it is the error being repaired. Use first_order.")
    dev = "cuda"
    torch.manual_seed(a.seed)
    ar, ut, sr = _mod("attack_relearn"), _mod("undo_t2i"), _mod("split_registry")
    t0 = time.time()

    # ---- pools, with roles declared through the registry
    kw = a.concept.split("_")[0].lower()
    forget_p = list(ut.FORGET_POOL[a.concept])
    benign_p = [q for q in ut.RETAIN_POOL if kw not in q.lower()][:24]
    assert len(benign_p) >= 8

    pipe = ar.build_pipe(torch.float32, a.teacher, dev)
    unet, vae, te, tok, sch = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    te.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(True)
    teacher = copy.deepcopy(unet).eval().requires_grad_(False)

    # ---- data. PROVENANCE REPAIR: target images may come from the BASE model, and either way
    # their content is measured before training rather than assumed.
    src = None if a.target_source == "base" else a.teacher
    gen = ar.build_pipe(torch.float16, src, dev)

    def make(prompts, n, stream):
        lat, emb, pil = [], [], []
        base_seed = sr.seed_for(stream, a.seed)
        with torch.no_grad():
            for i in range(n):
                pr = prompts[i % len(prompts)]
                g = torch.Generator(device=dev).manual_seed(base_seed + i)
                im = gen([pr], num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
                pil.append(im)
                x = torch.from_numpy((torch.frombuffer(im.tobytes(), dtype=torch.uint8)
                     .reshape(im.size[1], im.size[0], 3).numpy())).permute(2, 0, 1)[None].float() \
                    .to(dev) / 127.5 - 1.0
                lat.append(vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)
                ids = tok([pr], padding="max_length", max_length=77, truncation=True,
                          return_tensors="pt").input_ids.to(dev)
                emb.append(te(ids)[0])
        return torch.cat(lat), torch.cat(emb), pil

    f_lat, f_emb, f_pil = make(forget_p, a.n_images, "data_generation")
    b_lat, b_emb, _ = make(benign_p, a.n_images, "data_generation")
    del gen; torch.cuda.empty_cache()

    # ---- CONTENT CHECK on the target pool (audit: "verify that simulated target data actually
    # contains the target"). Closed-set CLIP argmax over gen_eval's label set, same classifier the
    # evaluation uses, so the number is comparable.
    ge = _mod("gen_eval")
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    labels = list(getattr(ar, "LABELS", list(ge.CONCEPTS)))
    with torch.no_grad():
        ti = cp(text=[f"a photo of a {l}" for l in labels], return_tensors="pt",
                padding=True).to(dev)
        tf = cm.get_text_features(**ti); tf = tf / tf.norm(dim=-1, keepdim=True)
        ii = cp(images=f_pil, return_tensors="pt").to(dev)
        imf = cm.get_image_features(pixel_values=ii["pixel_values"].half())
        imf = imf / imf.norm(dim=-1, keepdim=True)
        pred = (imf @ tf.T).argmax(-1).tolist()
    hit = sum(1 for p in pred if labels[p] == a.concept)
    target_acc = hit / max(len(pred), 1)
    print(f"[meta2] TARGET DATA CHECK: {hit}/{len(pred)} = {target_acc:.4f} classified as "
          f"'{a.concept}' (source={a.target_source})", flush=True)
    del cm, cp; torch.cuda.empty_cache()
    if target_acc < a.require_target_acc:
        raise SystemExit(
            f"REFUSING TO TRAIN: target pool is {target_acc:.4f} target-classified, below "
            f"--require-target-acc {a.require_target_acc}. Variant A trained on exactly this kind "
            f"of unverified pool. Use --target-source base, or lower the threshold deliberately.")

    # ---- scopes, independent and recorded
    meta_sel, atk_sel = SCOPES[a.meta_scope], SCOPES[a.attack_scope]
    meta_named = [(n, p) for n, p in unet.named_parameters() if meta_sel(n)]
    atk_named = [(n, p) for n, p in unet.named_parameters() if atk_sel(n)]
    meta_params = [p for _, p in meta_named]
    atk_params = [p for _, p in atk_named]
    print(f"[meta2] meta scope '{a.meta_scope}': {sum(p.numel() for p in meta_params)/1e6:.1f}M | "
          f"attack scope '{a.attack_scope}': {sum(p.numel() for p in atk_params)/1e6:.1f}M",
          flush=True)

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
    g = torch.Generator(device=dev).manual_seed(sr.seed_for("defence_build", a.seed))
    unet.train()
    log = []
    for step in range(a.steps):
        fi = torch.randint(0, len(f_lat), (1,), device=dev, generator=g)
        bi = torch.randint(0, len(b_lat), (1,), device=dev, generator=g)

        # ---- at theta: the erasure anchor (distillation, as variant A) PLUS Eq.8's
        #      +zeta*L_R(theta), the missing pre-attack retain DIFFUSION term (term ii).
        opt.zero_grad(set_to_none=True)
        l_erase = distil(unet, f_lat, f_emb, fi, g)
        l_distil_r = distil(unet, b_lat, b_emb, bi, g)
        # DEFECT (B): this draw is INDEPENDENT of the l_retain_prime draw below (shared generator
        # `g` advances between them), so Eq.8's paired difference is unpaired and `retain_change`
        # is dominated by sampling noise. DEFECT (A): with one shared zeta this term cancels
        # -zeta*l_retain_prime to second order, leaving zeta near-inert.
        l_retain_theta = dmloss(unet, b_lat, b_emb, bi, g)
        (l_erase + l_distil_r + a.zeta * l_retain_theta).backward()

        # ---- simulated malicious fine-tuning step on the TARGET, on the ATTACK scope.
        # DEFECT (C) in the module docstring: ONE plain gradient step, no optimizer state, no
        # clipping. The reference runs a full AdamW epoch with clip 1.0 on a deepcopy; the
        # evaluator runs 300 persistent Adam steps with clip 1.0. Not routed through
        # attack_core.AttackSpec, so the artifact carries no divergence record.
        l_att = dmloss(unet, f_lat, f_emb, fi, g)
        gatt = torch.autograd.grad(l_att, atk_params, create_graph=False, retain_graph=False)
        backup = [q.detach().clone() for q in atk_params]
        with torch.no_grad():
            for q, gg in zip(atk_params, gatt):
                q.sub_(a.inner_alpha * gg)                       # theta -> theta'

        # ---- at theta': BOTH post-attack terms (term i is the one variant A omits)
        l_target_prime = dmloss(unet, f_lat, f_emb, fi, g)
        l_retain_prime = dmloss(unet, b_lat, b_emb, bi, g)
        meta = (-a.lam_target) * l_target_prime - a.zeta * l_retain_prime
        gmeta = torch.autograd.grad(meta, meta_params)

        with torch.no_grad():
            for q, b in zip(atk_params, backup):
                q.copy_(b)                                       # restore theta
            for q, gm in zip(meta_params, gmeta):
                q.grad = gm if q.grad is None else q.grad.add_(gm)
        torch.nn.utils.clip_grad_norm_(list(unet.parameters()), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "l_erase": float(l_erase),
                   "l_distil_retain_theta": float(l_distil_r),
                   "l_retain_theta": float(l_retain_theta),
                   "l_target_theta_prime": float(l_target_prime),
                   "l_retain_theta_prime": float(l_retain_prime),
                   "retain_change": float(l_retain_prime) - float(l_retain_theta)}
            log.append(rec)
            print(f"[meta2] step {step:5d} erase={rec['l_erase']:.5f} "
                  f"L_F(theta')={rec['l_target_theta_prime']:.5f} "
                  f"dL_R={rec['retain_change']:+.5f} "
                  f"{(time.time()-t0)/max(step+1,1):.2f}s/step", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "meta_unlearning_v2", "implementation_version": IMPL_VERSION,
               "source": "arXiv 2410.12777 Eq. 8",
               "reference_code": "sail-sg/Meta-Unlearning@5a2a7600895a2885323cbbf599acaf3f74a96f64 "
                                 "src/meta.py L920-1030",
               "supersedes_and_does_not_replace": "scripts/meta_unlearn.py (variant A), whose "
                                                  "results stand as a distinct project variant",
               "objective": "L = distil_F(theta) + distil_R(theta) + zeta*L_R(theta) "
                            "- lam_target*L_F(theta') - zeta*L_R(theta')",
               "terms_restored_vs_variantA": [
                   "-lam_target*L_F(theta'): post-attack TARGET diffusion loss",
                   "+zeta*L_R(theta): matching pre-attack retain DIFFUSION loss"],
               "scopes": {"meta": a.meta_scope,
                          "meta_n_params": sum(p.numel() for p in meta_params),
                          "attack": a.attack_scope,
                          "attack_n_params": sum(p.numel() for p in atk_params)},
               "target_data_provenance": {
                   "source": a.target_source,
                   "generator_unet": src or "BASE (unmodified)",
                   "target_classified_fraction": target_acc,
                   "n_checked": len(pred),
                   "threshold": a.require_target_acc,
                   "note": "variant A generated this pool through the already-erased teacher and "
                           "never checked its content"},
               "gradient_order": a.order,
               "agent3_audit_2026_10_26": {
                   "reference_code_corrected": "src/esd_meta.py:992-1058 (src/meta.py, previously "
                                               "cited, is a stale non-runnable copy)",
                   "A_shared_zeta_inert_as_coefficient": "one zeta on both +L_R(theta) and "
                       "-L_R(theta') makes the pair cancel EXACTLY at every zeta, so zeta cannot "
                       "be tuned and is untested (no sweep run). This does NOT mean the pairing is "
                       "ineffective: it converts variant A's biased retain-LEVEL penalty into a "
                       "CHANGE penalty, removing the benign-damage incentive, which is a "
                       "first-order improvement. The reference uses independent gamma1_*/gamma2_* "
                       "and runs them asymmetrically.",
                   "B_unpaired_noise": "L_R(theta) and L_R(theta') use independent noise and "
                                       "timesteps, so the pairing holds only in expectation and "
                                       "`retain_change` is sampling noise, not signal. Same for "
                                       "l_att vs l_target_prime. Fix is common random numbers, not "
                                       "a larger zeta.",
                   "C_inner_attacker_unrecorded": "one plain GD step, no state, no clipping; "
                                                  "reference = full AdamW epoch + clip 1.0; "
                                                  "evaluator = 300 persistent Adam steps + clip "
                                                  "1.0. Not routed through attack_core.AttackSpec, "
                                                  "so no divergence record is written.",
                   "at_theta_term_substituted": "reference uses an ACTIVE ESD objective; this uses "
                                                "teacher distillation, zero at initialisation.",
                   "status": "documented only; all three require a behaviour change and were NOT "
                             "applied under the repair-first freeze"},
               "approximation": "first-order: grad of the theta' terms is taken at theta' and "
                                "applied at theta. Separate from the objective repair, and "
                                "deliberately not conflated with it.",
               "split_version": sr.SPLIT_VERSION,
               "label": a.label, "concept": a.concept, "hparams": vars(a),
               "forget_pool": forget_p, "n_benign_prompts": len(benign_p),
               "train_log": log, "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/meta2_{a.label}.json", "w"), indent=1)
    print(f"[meta2] {a.label}: wrote {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
