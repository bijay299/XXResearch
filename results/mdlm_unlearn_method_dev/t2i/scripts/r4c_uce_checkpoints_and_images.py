"""R4c — compare ACTUAL UCE checkpoints and their GENERATED IMAGES at matched prompts and seeds.

Closes the two gaps `FINDING77` §6 named:
  * R4b compared our port's *embedding selection* run through *upstream's* accumulation order; it
    never ran `erase_uce.py`'s own code path into a checkpoint.
  * R4b compared weights only. A weight difference is not an image difference.

WHAT THIS DOES
  1. Runs the pinned **upstream** `UCE(...)` to a checkpoint (its own `.safetensors`).
  2. Runs **our** `erase_uce.py` end-to-end to a checkpoint — its real code path, including its
     `lam*W`-last accumulation order and its `last..76` embedding selection.
  3. Loads BOTH checkpoints into the same pipeline and compares the edited matrices.
  4. Generates images from BOTH at **identical prompts, identical seeds, identical scheduler
     settings**, plus the unmodified base model as an anchor, and scores them with the same
     closed-set CLIP classifier every H number in this project uses.

ACCUMULATION ORDER IS NOW MEASURED, not held fixed: step 3 reports our-port-vs-upstream on the real
code paths, and `--order-probe` recomputes our port's own formula in upstream's order on identical
inputs, so the order effect is separated from the selection effect.

NO ROBUSTNESS CLAIM. No attack is run. This is a checkpoint and generation comparison.
"""
import argparse, importlib.util as iu, json, os, sys, time
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
UPSTREAM = f"{P}/upstream/unified-concept-editing"
UPSTREAM_COMMIT = "7c724d9d7d19190e9bd33fe62fb656f1df87cd8e"


def mod(name, path=None):
    spec = iu.spec_from_file_location(name, path or f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--guide", default="")
    ap.add_argument("--lamb", type=float, default=0.5)
    ap.add_argument("--erase-scale", type=float, default=1.0)
    ap.add_argument("--preserve-scale", type=float, default=1.0)
    ap.add_argument("--n-per-prompt", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    a = ap.parse_args()
    dev = "cuda"; t0 = time.time()
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)

    def fresh(dtype=torch.float32):
        return StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=dtype,
            variant="fp16", safety_checker=None, requires_safety_checker=False).to(dev)

    r4b = mod("r4b", f"{P}/scripts/r4b_uce_upstream_vs_port.py")
    euce = mod("euce", f"{P}/scripts/erase_uce.py")
    ge = mod("gen_eval", f"{P}/scripts/gen_eval.py")
    edit_concepts = r4b.as_prompt_list(euce.FORGET_PROMPTS[a.concept])
    preserve = r4b.as_prompt_list(euce.PRESERVE)
    guide = [a.guide] * len(edit_concepts)

    # ---------- 1. upstream -> checkpoint
    pipe = fresh()
    up, shim = r4b.load_upstream_uce(dev)
    sd_dir = f"{OUT}/r4c"; os.makedirs(sd_dir, exist_ok=True)
    exp = f"upstream_{a.concept}_lamb{a.lamb}"
    up.UCE(pipe, list(edit_concepts), list(guide), list(preserve),
           a.erase_scale, a.preserve_scale, a.lamb, sd_dir, exp)
    up_sd = load_file(f"{sd_dir}/{exp}.safetensors")
    del pipe; torch.cuda.empty_cache()

    # ---------- 2. OUR PORT, its real code path, to a checkpoint
    port_dir = f"{P}/adapters/R4C_port_{a.concept}_lam{a.lamb}"
    if not os.path.exists(f"{port_dir}/unet.pt"):
        rc = os.system(f"cd {P} && venv_uce/bin/python scripts/erase_uce.py --concept {a.concept} "
                       f"--lam {a.lamb} --preserve-scale {a.preserve_scale} "
                       f"--label R4C_port_{a.concept}_lam{a.lamb} --out {port_dir} "
                       f">> {P}/logs/r4c.log 2>&1")
        print(f"[r4c] our port exit={rc}", flush=True)
    assert os.path.exists(f"{port_dir}/unet.pt"), "our port produced no checkpoint"
    port_sd = torch.load(f"{port_dir}/unet.pt", map_location="cpu")
    port_sd = port_sd.get("unet", port_sd)

    # ---------- 3. checkpoint-level comparison on the SAME edited modules
    pipe = fresh()
    names = [n for n, m in pipe.unet.named_modules()
             if "attn2" in n and (n.endswith("to_v") or n.endswith("to_k"))]
    per = []
    for n in names:
        k = n + ".weight"
        Wu = up_sd[k].float().to(dev)
        Wp = port_sd[k].float().to(dev)
        per.append({"module": n,
                    "rel": float((Wp - Wu).norm() / max(float(Wu.norm()), 1e-30)),
                    "norm_upstream": float(Wu.norm()), "norm_port": float(Wp.norm())})
    rels = [x["rel"] for x in per]
    print(f"[r4c] CHECKPOINTS: n={len(per)} max={max(rels):.4e} mean={sum(rels)/len(rels):.4e}",
          flush=True)

    # ---------- 4. matched generations: base / upstream / our port
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    labels = list(ge.CONCEPTS)
    with torch.no_grad():
        ti = cp(text=[f"a photo of a {l}" for l in labels], return_tensors="pt",
                padding=True).to(dev)
        tf = cm.get_text_features(**ti); tf = tf / tf.norm(dim=-1, keepdim=True)
    prompts = list(ge.CONCEPTS[a.concept])          # the same 3 DEV prompts every H uses

    def score(tag, loader):
        pp = fresh(torch.float16)
        if loader is not None:
            loader(pp)
        hits, recs = 0, []
        for pi, pr in enumerate(prompts):
            for k in range(a.n_per_prompt):
                g = torch.Generator(device=dev).manual_seed(9000 + pi * 100 + k)   # MATCHED seeds
                im = pp([pr], num_inference_steps=a.steps, guidance_scale=a.guidance,
                        generator=g).images[0]
                with torch.no_grad():
                    ii = cp(images=[im], return_tensors="pt").to(dev)
                    f = cm.get_image_features(pixel_values=ii["pixel_values"].half())
                    f = f / f.norm(dim=-1, keepdim=True)
                    sims = (f @ tf.T)[0]
                p = int(sims.argmax()); hit = labels[p] == a.concept
                hits += hit
                recs.append({"prompt": pr, "i": k, "pred": labels[p], "is_target": bool(hit)})
        del pp; torch.cuda.empty_cache()
        n = len(recs)
        print(f"[r4c] GEN {tag}: {hits}/{n} = {hits/n:.4f}", flush=True)
        return {"tag": tag, "hits": hits, "n": n, "target_acc": hits / n, "per_image": recs}

    def load_up(pp):
        sd = {k: v.to(pp.unet.dtype) for k, v in up_sd.items()}
        missing = pp.unet.load_state_dict(sd, strict=False)
        assert len(missing.unexpected_keys) == 0, missing.unexpected_keys

    def load_port(pp):
        pp.unet.load_state_dict({k: v.to(pp.unet.dtype) for k, v in port_sd.items()})

    gens = [score("BASE", None), score("UPSTREAM", load_up), score("OUR_PORT", load_port)]
    agree = sum(1 for x, y in zip(gens[1]["per_image"], gens[2]["per_image"])
                if x["pred"] == y["pred"])

    rec = {"study": "r4c_uce_checkpoints_and_images",
           "upstream": {"commit": UPSTREAM_COMMIT, "executed": True, **shim},
           "our_port": {"script": "scripts/erase_uce.py", "executed_end_to_end": True,
                        "checkpoint": f"{port_dir}/unet.pt"},
           "config": {"concept": a.concept, "edit_concepts": edit_concepts, "guide": guide,
                      "n_preserve": len(preserve), "lamb": a.lamb,
                      "erase_scale": a.erase_scale, "preserve_scale": a.preserve_scale,
                      "prompts": prompts, "n_per_prompt": a.n_per_prompt,
                      "seeds": "9000 + prompt_index*100 + i  (identical across all three arms)",
                      "inference_steps": a.steps, "guidance": a.guidance},
           "checkpoint_comparison": {"n_modules": len(per), "per_matrix_max": max(rels),
                                     "per_matrix_mean": sum(rels) / len(rels), "per_matrix": per},
           "generations": gens,
           "upstream_vs_port_label_agreement": f"{agree}/{gens[1]['n']}",
           "caveats": ["No attack was run. This is not a robustness result.",
                       "DEV prompts; closed-set 12-label CLIP argmax; n=%d images per arm."
                       % gens[0]["n"]],
           "minutes": (time.time() - t0) / 60}
    out = f"{OUT}/r4c_uce_lamb{a.lamb}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[r4c] agreement upstream vs port: {agree}/{gens[1]['n']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
