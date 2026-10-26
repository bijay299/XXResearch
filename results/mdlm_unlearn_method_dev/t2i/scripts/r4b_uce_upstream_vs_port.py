"""R4b — UCE: **actually invoke the pinned upstream implementation** and compare it against our
historical port. Supersedes `r4_uce_official_vs_port.py`, which is retained but annotated as void.

WHY R4 (the first attempt) DOES NOT SUPPORT ITS CONCLUSION — all three verified in the artifacts:

  1. **Input bug.** `list(euce.FORGET_PROMPTS.get(concept, [concept]))` — the dict VALUES ARE
     STRINGS, so `list('church')` produced `['c','h','u','r','c','h']`. R4 embedded **six single
     letters** as six concepts. `outputs/r4_uce_official_vs_port.json` records
     `n_erase_prompts: 6`, which is that bug visible in the artifact.
  2. **Neither pipeline was reproduced.** Three different embedding selections exist:
        upstream  `uce_sd_erase.py:34-42`  ONE embedding at attention_mask.sum()-2
        our port  `erase_uce.py:63-65`     positions [attention_mask.sum()-2 .. 76]
        R4        `r4_...py:75`            ALL 77 positions, flattened
     R4 used the third, which is neither.
  3. **No upstream code ran.** R4 compared two LOCAL transcriptions. It could not have detected a
     divergence between our port and the authors' implementation.

WHAT R4b DOES INSTEAD

  * imports and calls the pinned upstream `UCE(...)` **unmodified** (compat shim only, §SHIM);
  * runs our historical `erase_uce.py` path on matched inputs;
  * compares **every** edited matrix, not the first eight;
  * separates per-matrix max, per-matrix mean, and aggregate error;
  * solves the same systems in **float64** to measure actual numerical error rather than inferring
    it from a condition number;
  * isolates embedding-selection, accumulation-order and precision **one factor at a time**;
  * reloads saved checkpoints so serialization precision is included.

It makes NO robustness claim. Weight differences are not image differences and are not attack
resistance.

Run:  venv_uce/bin/python scripts/r4b_uce_upstream_vs_port.py --concept church --lamb 0.5
"""
import argparse, importlib.util as iu, json, os, sys, time
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
UPSTREAM = f"{P}/upstream/unified-concept-editing"
UPSTREAM_COMMIT = "7c724d9d7d19190e9bd33fe62fb656f1df87cd8e"


# ----------------------------------------------------------------- prompt normalisation + test
def as_prompt_list(v):
    """The R4 bug fix: a str is ONE concept, not a sequence of characters.

    `erase_uce.FORGET_PROMPTS` maps concept -> str ('church', 'garbage truck'). A genuine list of
    prompts must be preserved as-is.
    """
    if isinstance(v, str):
        return [v]
    return list(v)


def _regression_as_prompt_list():
    """Required regression check: church is ONE concept; 'garbage truck' stays one whole phrase."""
    assert as_prompt_list("church") == ["church"], as_prompt_list("church")
    assert as_prompt_list("garbage truck") == ["garbage truck"], as_prompt_list("garbage truck")
    assert as_prompt_list(["a", "b"]) == ["a", "b"]
    assert as_prompt_list(("x",)) == ["x"]
    # and the exact failure R4 shipped:
    assert list("church") == ["c", "h", "u", "r", "c", "h"]
    assert as_prompt_list("church") != list("church")
    return True


# ----------------------------------------------------------------- upstream loader (SHIM)
def load_upstream_uce(device):
    """Import the pinned upstream module and supply the globals it expects.

    SHIM, recorded: `uce_sd_erase.py` references a module-level `device` that is only bound inside
    its `__main__` block, and calls `torch.set_grad_enabled(False)` at import. We set `device` on
    the module object and nothing else. The UCE() function body is untouched.
    """
    path = f"{UPSTREAM}/trainscripts/uce_sd_erase.py"
    spec = iu.spec_from_file_location("upstream_uce_sd_erase", path)
    mod = iu.module_from_spec(spec)
    sys.modules["upstream_uce_sd_erase"] = mod
    spec.loader.exec_module(mod)
    mod.device = device                      # <-- the only shim
    return mod, {"shim": "set module-level `device`; UCE() body unmodified",
                 "file": path, "commit": UPSTREAM_COMMIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--guide", default="")
    ap.add_argument("--lamb", type=float, default=0.5)
    ap.add_argument("--erase-scale", type=float, default=1.0)
    ap.add_argument("--preserve-scale", type=float, default=1.0)
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert _regression_as_prompt_list()
    dev = "cuda"
    t0 = time.time()

    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, revision=pin["revision"], torch_dtype=torch.float32, variant="fp16",
        safety_checker=None, requires_safety_checker=False).to(dev)

    # ---- our port's own pools, correctly normalised
    spec = iu.spec_from_file_location("euce", f"{P}/scripts/erase_uce.py")
    euce = iu.module_from_spec(spec); spec.loader.exec_module(euce)
    edit_concepts = as_prompt_list(euce.FORGET_PROMPTS[a.concept])
    preserve_concepts = as_prompt_list(euce.PRESERVE)
    guide_concepts = [a.guide] * len(edit_concepts)

    print(f"[r4b] edit={edit_concepts}  guide={guide_concepts}  "
          f"preserve={len(preserve_concepts)} concepts", flush=True)

    # ================= upstream, ACTUALLY RUN =================
    up, shim = load_upstream_uce(dev)
    save_dir = f"{P}/outputs/r4b_upstream"
    os.makedirs(save_dir, exist_ok=True)
    exp = f"upstream_{a.concept}_lamb{a.lamb}{a.tag}"
    up.UCE(pipe, list(edit_concepts), list(guide_concepts), list(preserve_concepts),
           a.erase_scale, a.preserve_scale, a.lamb, save_dir, exp)
    from safetensors.torch import load_file
    up_sd = load_file(f"{save_dir}/{exp}.safetensors")       # reloaded: serialization included
    print(f"[r4b] upstream produced {len(up_sd)} matrices", flush=True)

    # ================= our port's embedding selection, on the SAME pipe =================
    tok, te = pipe.tokenizer, pipe.text_encoder
    unet = pipe.unet
    mods = [(n, m) for n, m in unet.named_modules()
            if "attn2" in n and (n.endswith("to_v") or n.endswith("to_k"))]
    assert len(mods) == len(up_sd), f"module count mismatch {len(mods)} vs {len(up_sd)}"

    def embed_upstream_style(prompts):
        """ONE embedding per prompt at attention_mask.sum()-2 (upstream)."""
        out = []
        for p in prompts:
            with torch.no_grad():
                emb = pipe.encode_prompt(prompt=p, device=dev, num_images_per_prompt=1,
                                         do_classifier_free_guidance=False)[0]
                n = tok(p, padding="max_length", max_length=tok.model_max_length,
                        truncation=True, return_tensors="pt")["attention_mask"].sum() - 2
                out.append(emb[0, n, :].float())
        return torch.stack(out)

    def embed_port_style(prompts):
        """positions [attention_mask.sum()-2 .. 76] (our erase_uce.py)."""
        out = []
        for p in prompts:
            with torch.no_grad():
                ti = tok([p], padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").to(dev)
                h = te(ti.input_ids)[0]
                last = max(int(ti.attention_mask[0].sum().item()) - 2, 0)
                out.append(h[0, torch.arange(last, h.shape[1], device=dev)].float())
        return torch.cat(out)

    def solve(c_e, c_g_out, c_p, c_p_out, W, lamb, es, ps, dtype):
        """UCE closed form, upstream accumulation order, in the requested dtype."""
        W = W.to(dtype); I = torch.eye(W.shape[1], device=W.device, dtype=dtype)
        mat1 = lamb * W; mat2 = lamb * I
        mat1 = mat1 + es * (c_g_out.to(dtype).T @ c_e.to(dtype))
        mat2 = mat2 + es * (c_e.to(dtype).T @ c_e.to(dtype))
        mat1 = mat1 + ps * (c_p_out.to(dtype).T @ c_p.to(dtype))
        mat2 = mat2 + ps * (c_p.to(dtype).T @ c_p.to(dtype))
        return mat1 @ torch.linalg.inv(mat2)

    results = {}
    for style, embfn in (("upstream_token", embed_upstream_style), ("port_last_to_76", embed_port_style)):
        c_e = embfn(edit_concepts); c_g = embfn(guide_concepts); c_p = embfn(preserve_concepts)
        per = []
        for (name, m) in mods:
            W = m.weight.data.float()
            Wn32 = solve(c_e, c_g @ W.T, c_p, c_p @ W.T, W, a.lamb, a.erase_scale, a.preserve_scale,
                         torch.float32)
            Wn64 = solve(c_e.double(), (c_g @ W.T).double(), c_p.double(), (c_p @ W.T).double(),
                         W.double(), a.lamb, a.erase_scale, a.preserve_scale, torch.float64)
            fp_err = float((Wn32.double() - Wn64).norm() / max(float(Wn64.norm()), 1e-30))
            rec = {"module": name, "float32_vs_float64_rel": fp_err}
            if style == "upstream_token":
                Wup = up_sd[name + ".weight"].float().to(dev)
                rec["vs_upstream_rel"] = float((Wn32 - Wup).norm() / max(float(Wup.norm()), 1e-30))
            per.append(rec)
        results[style] = per

    # upstream (reloaded) vs our-port-selection, every matrix
    cross = []
    c_e_p = embed_port_style(edit_concepts); c_g_p = embed_port_style(guide_concepts)
    c_p_p = embed_port_style(preserve_concepts)
    for (name, m) in mods:
        W = m.weight.data.float()
        Wport = solve(c_e_p, c_g_p @ W.T, c_p_p, c_p_p @ W.T, W, a.lamb, a.erase_scale,
                      a.preserve_scale, torch.float32)
        Wup = up_sd[name + ".weight"].float().to(dev)
        cross.append({"module": name,
                      "rel": float((Wport - Wup).norm() / max(float(Wup.norm()), 1e-30)),
                      "norm_upstream": float(Wup.norm()), "norm_port": float(Wport.norm())})

    def agg(vals):
        return {"per_matrix_max": max(vals), "per_matrix_mean": sum(vals) / len(vals), "n": len(vals)}

    rec = {
        "study": "r4b_uce_upstream_vs_port",
        "supersedes": "outputs/r4_uce_official_vs_port.json (VOID: char-split inputs, neither "
                      "pipeline reproduced, no upstream code executed)",
        "upstream": {"repo": "rohitgandikota/unified-concept-editing", "commit": UPSTREAM_COMMIT,
                     "file": "trainscripts/uce_sd_erase.py", "executed": True, **shim},
        "config": {"model": a.model, "revision": pin["revision"], "concept": a.concept,
                   "edit_concepts": edit_concepts, "guide_concepts": guide_concepts,
                   "preserve_concepts": preserve_concepts, "lamb": a.lamb,
                   "erase_scale": a.erase_scale, "preserve_scale": a.preserve_scale,
                   "dtype": "float32 compute, float64 reference",
                   "n_edited_modules": len(mods),
                   "edited_module_names": [n for n, _ in mods]},
        "A_embedding_selection_upstream_vs_port":
            agg([c["rel"] for c in cross]) | {"note": "FULL pipelines: upstream executed vs our "
                "port's embedding selection. Confounds embedding selection with everything else."},
        "B_same_selection_vs_upstream":
            agg([r["vs_upstream_rel"] for r in results["upstream_token"]]) | {"note":
                "our solver using UPSTREAM's embedding selection, vs upstream's own output. "
                "Isolates solver/accumulation from embedding selection."},
        "C_float32_vs_float64_upstream_selection":
            agg([r["float32_vs_float64_rel"] for r in results["upstream_token"]]) | {"note":
                "MEASURED numerical error of the float32 solve against a float64 solve on "
                "IDENTICAL matrices. Not inferred from a condition number."},
        "C_float32_vs_float64_port_selection":
            agg([r["float32_vs_float64_rel"] for r in results["port_last_to_76"]]),
        "per_matrix_all": cross,
        "caveats": ["Weight differences are NOT image differences and NOT attack resistance.",
                    "This does not establish which implementation is more accurate; the float64 "
                    "columns bound each one's own numerical error, nothing more."],
        "minutes": (time.time() - t0) / 60,
    }
    out = f"{OUT}/r4b_uce_lamb{a.lamb}{a.tag}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[r4b] A upstream-vs-port  max={rec['A_embedding_selection_upstream_vs_port']['per_matrix_max']:.4e} "
          f"mean={rec['A_embedding_selection_upstream_vs_port']['per_matrix_mean']:.4e}", flush=True)
    print(f"[r4b] B same-selection    max={rec['B_same_selection_vs_upstream']['per_matrix_max']:.4e} "
          f"mean={rec['B_same_selection_vs_upstream']['per_matrix_mean']:.4e}", flush=True)
    print(f"[r4b] C fp32-vs-fp64(up)  max={rec['C_float32_vs_float64_upstream_selection']['per_matrix_max']:.4e}", flush=True)
    print(f"[r4b] C fp32-vs-fp64(port) max={rec['C_float32_vs_float64_port_selection']['per_matrix_max']:.4e}", flush=True)
    print(f"[r4b] n modules = {len(mods)}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
