"""Experiment E6 — the FIRST and ONLY legitimate use of the frozen FINAL prompt set.

Every target number in this project is a DEVELOPMENT measurement: one of the three evaluation
prompts was a defence-training prompt for GIFT/IMMA/WSR/UTRAP, and λ/β/K were repeatedly selected
while watching H (`ERRATUM_split_leak.md`). `split_registry.FINAL_PROMPTS` was frozen at
2026-10-25T23:30:00-04:00, before any final result existed, and verified disjoint from DEF_TRAIN,
ATK_TRAIN and DEV by computation.

This scores the target concept on those FINAL prompts only. It does NOT touch `gen_eval.py`, so
every historical record reproduces unchanged.

**The arm list is PRE-COMMITTED in this file, before any FINAL number is seen.** That is the whole
point: FINAL is a one-shot resource. Scoring an adaptively chosen set of arms, or rescoring after
seeing a result, would destroy it. The list is the arms that clear K1 (retain8 >= 0.60) and are the
project's leading operating points. Any later addition must be justified in writing and labelled a
second, contaminated use.

Writes only a small JSON per arm — no checkpoints — so it is safe to run with /data near capacity.

Run: python scripts/gen_eval_final.py            # all pre-committed arms
     python scripts/gen_eval_final.py --only X   # one of them
"""
import argparse, importlib.util as iu, json, os, time
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"

#: PRE-COMMITTED, before any FINAL number was produced. (label, unet, target, BASE_MODEL, BASE_REV)
SD14 = ("CompVis/stable-diffusion-v1-4", None)   # None => MODEL_PIN.txt's pinned revision
SD15 = ("runwayml/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
# NOTE: BK-SDM-small is deliberately NOT in ARMS. It pins revision 572238db7ed3 and has NO fp16
# variant (run_meta_bksdm_low.sh sets BASE_VARIANT=""), so it needs a variant override this script
# does not implement. Adding it would be a second, contaminated use of FINAL anyway.

ARMS = [
    ("utrap_church_l0.15k8",      "adapters/utrap_church_l0.15k8/unet.pt",      "church", SD14),
    ("frp_church_r3_l05",         "adapters/frp_church_r3_l05/unet.pt",         "church", SD14),
    ("utrap_sd15_church_l0.15k8", "adapters/utrap_sd15_church_l0.15k8/unet.pt", "church", SD15),
    ("uce_sd15_church_lam1000",   "adapters/uce_sd15_church_lam1000/unet.pt",   "church", SD15),
    ("BASE_sd14_anchor",          None,                                          "church", SD14),
    # CALIBRATION ANCHORS, added after the five arms above were scored. These are UNMODIFIED base
    # models, not defence arms, so they do not consume FINAL as a selection resource -- they are the
    # control needed to read the DEV->FINAL gap at all. Without an SD-1.5 anchor the two SD-1.5 arms
    # cannot be normalised, and without a base-on-DEV number the gap cannot be attributed.
    ("BASE_sd15_anchor",          None,                                          "church", SD15),
    # E6b: POST-ATTACK arms. These score FINAL-prompt generation on models that the UNMODIFIED
    # attack_relearn.py has already attacked (kept via --keep-checkpoint). Every H in the corpus is
    # DEV-prompt recovery; these are the FINAL-prompt counterpart, and they are the one remaining
    # way the H=1.000 ceiling could be a DEV artifact. Pre-committed before any number was seen.
    ("ATK_utrap_church_l0.15k8",
     "attacked/E6B_utrap_church_l0.15k8_sd0_lr1e-5/unet.pt",      "church", SD14),
    ("ATK_utrap_sd15_church_l0.15k8",
     "attacked/E6B_utrap_sd15_church_l0.15k8_sd0_lr1e-5/unet.pt", "church", SD15),
    # E6c: DELIBERATE EXTENSION of E6b, added 2026-10-26 AFTER seeing E6b's two results. Labelled
    # honestly as an extension, not as pre-committed. It completes the post-attack FINAL picture on
    # the two arms whose PRE-attack FINAL suppression was weakest (frp 0.4667, uce 0.6000), where
    # E6b's conclusion is least surprising and therefore least informative -- which is exactly why
    # leaving them out would have been selective.
    ("ATK_frp_church_r3_l05",
     "attacked/E6C_frp_church_r3_l05_sd0_lr1e-5/unet.pt",         "church", SD14),
    ("ATK_uce_sd15_church_lam1000",
     "attacked/E6C_uce_sd15_church_lam1000_sd0_lr1e-5/unet.pt",   "church", SD15),
]


def mod(name):
    spec = iu.spec_from_file_location(f"_f_{name}", f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-per", type=int, default=5, help="images per FINAL prompt; 3x5=15, matching H")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    a = ap.parse_args()
    dev = "cuda"
    sr, ge = mod("split_registry"), mod("gen_eval")
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPModel, CLIPProcessor

    labels = list(ge.CONCEPTS)                      # same closed 12-label set as every H number
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        ti = cp(text=[f"a photo of a {l}" for l in labels], return_tensors="pt",
                padding=True).to(dev)
        tf = cm.get_text_features(**ti); tf = tf / tf.norm(dim=-1, keepdim=True)

    results = []
    for label, unet_path, target, (bm, brev) in ARMS:
        if a.only and a.only != label:
            continue
        finals = sr.FINAL_PROMPTS[target]
        pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n")
                   if "=" in l)
        t0 = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            bm, revision=(brev or pin["revision"]), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False).to(dev)
        if unet_path:
            full = unet_path if os.path.isabs(unet_path) else f"{P}/{unet_path}"
            if not os.path.exists(full):
                print(f"[final] SKIP {label}: checkpoint ABSENT ({unet_path})", flush=True)
                del pipe; torch.cuda.empty_cache(); continue
            sd = torch.load(full, map_location="cpu"); sd = sd.get("unet", sd)
            pipe.unet.load_state_dict({k: v.half() for k, v in sd.items()})
        per_image = []
        for pi, pr in enumerate(finals):
            for k in range(a.n_per):
                # FINAL evaluation RNG stream, common across arms so every arm sees the same noise
                g = torch.Generator(device=dev).manual_seed(sr.seed_for("final_eval", pi * 100 + k))
                im = pipe([pr], num_inference_steps=a.steps, guidance_scale=a.guidance,
                          generator=g).images[0]
                with torch.no_grad():
                    ii = cp(images=[im], return_tensors="pt").to(dev)
                    f = cm.get_image_features(pixel_values=ii["pixel_values"].half())
                    f = f / f.norm(dim=-1, keepdim=True)
                    sims = (f @ tf.T)[0]
                pred = int(sims.argmax())
                per_image.append({"prompt": pr, "image_index": k, "pred_label": labels[pred],
                                  "is_target": labels[pred] == target,
                                  "margin": float(sims[pred] - sims.topk(2).values[1]),
                                  "target_sim": float(sims[labels.index(target)])})
        hits = sum(1 for r in per_image if r["is_target"])
        rec = {"study": "t2i_gen_eval_FINAL", "experiment": "E6",
               "split_version": sr.SPLIT_VERSION, "split_role": "FINAL",
               "final_frozen_at": sr.FINAL_FROZEN_AT,
               "first_and_only_legitimate_use": True,
               "arm_list_precommitted_in": "scripts/gen_eval_final.py ARMS",
               "label": label, "unet": unet_path or "BASE (unmodified)", "base_model": bm,
               "target": target, "final_prompts": finals,
               "target_hits": hits, "n_total": len(per_image),
               "target_acc_FINAL": hits / max(len(per_image), 1),
               "per_image_predictions": per_image,
               "denominator_note": f"{len(finals)} FINAL prompts x {a.n_per} images = "
                                   f"{len(per_image)}; quantised to 1/{len(per_image)}. Closed-set "
                                   f"{len(labels)}-label CLIP argmax, no reject option. Comparable "
                                   f"in construction to H but on PROMPTS NEVER USED FOR ANYTHING.",
               "minutes": (time.time() - t0) / 60}
        json.dump(rec, open(f"{OUT}/final_{label}.json", "w"), indent=1)
        results.append((label, hits, len(per_image), rec["target_acc_FINAL"]))
        print(f"[final] {label}: target {hits}/{len(per_image)} = {rec['target_acc_FINAL']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del pipe
        torch.cuda.empty_cache()

    print("\n=== FINAL (held-out prompts, never used for selection) ===")
    for label, h, n, acc in results:
        print(f"  {label:<30} {h}/{n} = {acc:.4f}")


if __name__ == "__main__":
    main()
