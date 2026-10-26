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
  * reloads UPSTREAM's saved `.safetensors`, so upstream's serialization precision is included.

CORRECTIONS TO THIS DOCSTRING (2026-10-26, AGENT1 P5). Three claims it used to make are false:

  (i)   "isolates embedding-selection, accumulation-order and precision ONE FACTOR AT A TIME."
        IT DOES NOT ISOLATE ACCUMULATION ORDER. `solve()` below uses UPSTREAM's order
        (lam*W first) for EVERY cell, so that factor is held fixed and never varied. Column B
        establishes something narrower but still useful: given identical embeddings AND identical
        accumulation order, our transcription of the closed form reproduces upstream, so there is
        no third hidden discrepancy. Accumulation order is bounded elsewhere -- see
        AGENT1_UCE_FINDINGS D-106/D-160 -- at <1.2e-04 on this GPU path but ~5.6e-02 on CPU, i.e.
        it is BLAS-dependent and this script does not measure it.
  (ii)  "reloads saved checkpoints so serialization precision is included." Only UPSTREAM's is
        reloaded. The port column is computed in memory in float32 and NEVER touches
        `adapters/.../unet.pt`, which `erase_uce.py` writes as FLOAT16. The port's serialization
        floor is 2.07e-04 (AGENT1 D-140) -- ABOVE column B at lam>=0.5 and 1560x above it at
        lam=1000, so column B says nothing about any checkpoint we ever shipped. Use r4c for
        checkpoint-level comparison.
  (iii) The "port" column is our port's EMBEDDING SELECTION run through upstream's accumulation
        order at preserve_scale=1.0. It is NOT `erase_uce.py`'s code path, and preserve_scale=1.0
        matches NO historical arm -- `erase_uce.py` defaults to 0.1 and every SD-1.5 lambda-sweep
        arm, including `uce_sd15_church_lam1000`, used 0.1.

Column C is a MEASURED REALISED float32 error on one realisation, NOT a bound: it is a lower
estimate (the float64 solve carries its own error) and bounds nothing above (one input, one guide,
one concept, one BLAS path). The word "bound" has been removed throughout.

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
    mod.device = device                      # <-- shim 1
    mod.torch_dtype = torch.float32          # <-- shim 2: upstream binds this only in __main__
    return mod, {"shim": "set module-level `device` and `torch_dtype=torch.float32` (both are "
                         "bound only inside upstream's __main__ block, at uce_sd_erase.py:116-117); "
                         "UCE() body unmodified",
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

    def embed_port_style(prompts, pair_with=None):
        """positions [attention_mask.sum()-2 .. 76] (our erase_uce.py).

        BUGFIX: the position count VARIES with token length, so an erase prompt and its guide
        yield different row counts and the closed form will not multiply. erase_uce.py handles this
        with its `paired` path, taking the SAME positions of the paired prompt. `pair_with` supplies
        the prompt whose indices govern; without it a guide/erase pair mismatches (77 vs 76).
        """
        out = []
        for i, p in enumerate(prompts):
            gov = p if pair_with is None else pair_with[i]
            with torch.no_grad():
                tg = tok([gov], padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").to(dev)
                last = max(int(tg.attention_mask[0].sum().item()) - 2, 0)
                ti = tok([p], padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").to(dev)
                h = te(ti.input_ids)[0]
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
        if style == 'port_last_to_76':
            c_e = embfn(edit_concepts); c_g = embfn(guide_concepts, pair_with=edit_concepts)
            c_p = embfn(preserve_concepts)
        else:
            c_e = embfn(edit_concepts); c_g = embfn(guide_concepts); c_p = embfn(preserve_concepts)
        per = []
        for (name, m) in mods:
            W = m.weight.data.float()
            Wn32 = solve(c_e, c_g @ W.T, c_p, c_p @ W.T, W, a.lamb, a.erase_scale, a.preserve_scale,
                         torch.float32)
            # D-130 REPAIR. The previous form passed `(c_g @ W.T).double()` -- a FLOAT32 matmul
            # cast up afterwards -- so the "float64 reference" inherited the float32 rounding of
            # the very numerator products it was meant to measure, making the two errors
            # correlated. Cast FIRST, multiply in float64.
            Wd = W.double()
            Wn64 = solve(c_e.double(), c_g.double() @ Wd.T, c_p.double(), c_p.double() @ Wd.T,
                         Wd, a.lamb, a.erase_scale, a.preserve_scale, torch.float64)
            # Retained so the historical column stays auditable: the OLD mixed-precision
            # reference. Measured on CPU (AGENT1 S9b) at 3.48e-08 from a true float64 solve, i.e.
            # four decades below the float32 errors below, so no previously published column C
            # value needs revising -- but the code was wrong and is fixed.
            Wn64_mixed = solve(c_e.double(), (c_g @ W.T).double(), c_p.double(),
                               (c_p @ W.T).double(), Wd, a.lamb, a.erase_scale, a.preserve_scale,
                               torch.float64)
            fp_err = float((Wn32.double() - Wn64).norm() / max(float(Wn64.norm()), 1e-30))
            rec = {"module": name, "float32_vs_float64_rel": fp_err,
                   "float64_ref_mixed_precision_rel":
                       float((Wn64_mixed - Wn64).norm() / max(float(Wn64.norm()), 1e-30))}
            if style == "upstream_token":
                Wup = up_sd[name + ".weight"].float().to(dev)
                rec["vs_upstream_rel"] = float((Wn32 - Wup).norm() / max(float(Wup.norm()), 1e-30))
            per.append(rec)
        results[style] = per

    # upstream (reloaded) vs our-port-selection, every matrix
    cross = []
    c_e_p = embed_port_style(edit_concepts)
    c_g_p = embed_port_style(guide_concepts, pair_with=edit_concepts)
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
            agg([c["rel"] for c in cross]) | {"note":
                "||solve(OUR selection) - upstream_reloaded|| / ||upstream_reloaded||, all 32 "
                "matrices. NOT erase_uce.py's code path: upstream's accumulation order is used "
                "here, and preserve_scale is this run's value (NOT the port's historical 0.1)."},
        "B_same_selection_vs_upstream":
            agg([r["vs_upstream_rel"] for r in results["upstream_token"]]) | {"note":
                "||solve(UPSTREAM's selection) - upstream_reloaded|| / ||upstream_reloaded||. "
                "Same selection AND same accumulation order as upstream, so it does NOT vary "
                "accumulation order and establishes nothing about it. What it does establish: "
                "given identical embeddings and identical order, our transcription of the closed "
                "form reproduces upstream -- no third hidden discrepancy. NOTE: below the port's "
                "own float16 serialization floor of 2.07e-04, so it says nothing about any "
                "checkpoint we shipped."},
        # A and B share the SAME reference (upstream_reloaded) and the SAME denominator, so the
        # selection-induced difference is bounded two-sidedly by the triangle inequality. A-B is
        # NOT a tensor difference and must not be quoted as one.
        "selection_effect_triangle_bounds": {
            "lower_A_minus_B": (agg([c["rel"] for c in cross])["per_matrix_mean"]
                                - agg([r["vs_upstream_rel"]
                                       for r in results["upstream_token"]])["per_matrix_mean"]),
            "upper_A_plus_B": (agg([c["rel"] for c in cross])["per_matrix_mean"]
                               + agg([r["vs_upstream_rel"]
                                      for r in results["upstream_token"]])["per_matrix_mean"]),
            "note": "||solve(our sel) - solve(upstream sel)|| / ||upstream|| lies in "
                    "[A-B, A+B]. Valid because A and B are distances to a COMMON reference with a "
                    "common denominator. The bound is tight only while B << A."},
        "C_float32_vs_float64_upstream_selection":
            agg([r["float32_vs_float64_rel"] for r in results["upstream_token"]]) | {"note":
                "MEASURED REALISED numerical error of the float32 solve against a TRUE float64 "
                "solve (all products in float64) on IDENTICAL inputs. Not inferred from a "
                "condition number, and NOT a bound -- it is one realisation on one BLAS path."},
        "C_float32_vs_float64_port_selection":
            agg([r["float32_vs_float64_rel"] for r in results["port_last_to_76"]]),
        "float64_reference_mixed_precision_check":
            agg([r["float64_ref_mixed_precision_rel"] for r in results["upstream_token"]]) | {
                "note": "How far the OLD (pre-D-130) mixed-precision reference sits from the "
                        "corrected true-float64 one. Expect ~3e-08, i.e. four decades below "
                        "column C, confirming no published column C value needs revising."},
        "per_matrix_all": cross,
        "caveats": ["Weight differences are NOT image differences and NOT attack resistance.",
                    "This does not establish which implementation is more accurate. Column C is a "
                    "realised float32 error on one realisation -- a lower estimate, not a bound.",
                    "Accumulation order is held at UPSTREAM's in every cell and is NOT measured "
                    "here (AGENT1 D-106/D-160).",
                    "Only upstream's serialization is included; the port column never touches "
                    "erase_uce.py's float16 unet.pt (AGENT1 D-140).",
                    "preserve_scale here is %r; erase_uce.py's historical default is 0.1, so this "
                    "cell may match NO arm in our published tables." % a.preserve_scale,
                    "Upstream at erase_scale=1 with one unexpanded object noun leaves church at "
                    "15/15 CLIP accuracy (r4c/r4d). Weight-space agreement with it is therefore "
                    "not evidence of a working erasure on either side (AGENT1 D-100..D-105)."],
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
