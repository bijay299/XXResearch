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
  5. Measures **upstream vs BASE**, per matrix — without which a non-zero port-vs-upstream norm
     does not prove upstream edited anything at all.

NO ROBUSTNESS CLAIM. No attack is run. This is a checkpoint and generation comparison.

===================== REPAIRS, 2026-10-26 (AGENT1 P6). READ BEFORE CITING. =====================
The previous version of this file advertised behaviour it did not have. Every item below was
confirmed against the artifacts it produced (`outputs/r4c_uce_lamb{0.1,0.5}.json`,
`outputs/r4d_uce_lamb0.5.json`), which were written by the DEFECTIVE version and must be read with
these caveats. Details and numbers: `docs/AGENT1_UCE_FINDINGS.md` D-110..D-115, D-120.

 D-110  `--order-probe` WAS ADVERTISED IN THIS DOCSTRING AND DID NOT EXIST. No order probe was
        computed, so no artifact separates accumulation order from embedding selection. NOW
        IMPLEMENTED, default off (`--order-probe`).
 D-111  SCORED WITH THE WRONG LABEL SET. `labels = list(gen_eval.CONCEPTS)` took the dict KEYS, so
        the CLIP text prompts were "a photo of a garbage_truck", "a photo of a van_gogh",
        "a photo of a english_springer", "a photo of a monet" -- 9 of 12 labels malformed. The
        project's shared closed set is `gen_eval.LABELS` (identical in `attack_relearn.py`), paired
        with `gen_eval.LAB_KEY`. 'church' is index 0 and spelled the same in both, so the TARGET
        accuracies in the existing artifacts are the least-affected quantity, but the competing
        logits came from a corrupted set and the `pred` fields are not interpretable. NOW uses
        LABELS/LAB_KEY.
 D-112  `os.system` RETURN CODE DISCARDED, and the cache guard was `os.path.exists` on a path
        keyed only by (concept, lamb) while the run also depends on preserve_scale, guide and
        --no-preserve. NOW: `subprocess.run` with an explicit env and a checked return code, and
        the cache key covers every input. `--refresh-port` forces a rebuild.
 D-113  **R4d SILENTLY REUSED R4c's PORT CHECKPOINT AND CLOBBERED R4c's UPSTREAM CHECKPOINT.**
        Both runs used `adapters/R4C_port_{concept}_lam{lamb}` and
        `outputs/r4c/upstream_{concept}_lamb{lamb}.safetensors`. Filesystem evidence: the port
        checkpoint's mtime is 12:22 (R4c) while `r4d_uce_lamb0.5.json` is 12:42, and the surviving
        safetensors measures 0.0536 against base -- R4d's 0-preserve value, not R4c's 0.0568.
        So `r4d_uce_lamb0.5.json` records `"n_preserve": 0` while its port arm was built WITH 10
        preserve concepts, and R4c's upstream checkpoint no longer exists (R4c is not reproducible
        from disk). NOW both paths encode the full config, so no run can shadow another.
 D-114  NO IMAGES WERE SAVED. 45 images were generated per run and all discarded. NOW saved under
        `outputs/img_r4{c,d}_<tag>/`, one grid row per arm at matched seeds.
 D-115  (1) `--guide` was never passed to the port (`erase_uce.py`'s equivalent is `--target`), so
        any non-default guide silently compared upstream erasing toward *g* against our port
        erasing toward "". NOW propagated. (2) `--erase-scale` has no port counterpart at all;
        values != 1.0 compared two different equations. NOW refused. (3) `--no-preserve` was
        one-sided: it emptied upstream's pool but `erase_uce.py` always uses its hard-coded
        10-concept PRESERVE. The asymmetry is UNFIXABLE without changing `erase_uce.py`'s
        behaviour, so it is now printed and recorded in the artifact. (4) No retain/utility axis,
        which PREREG_H38 §6 requires; now available opt-in via `--retain-concepts`.
 D-120  `BASE_MODEL`/`BASE_REV`/`BASE_VARIANT` OVERRODE THE PORT SIDE ONLY. This script hard-coded
        SD-1.4 while `erase_uce.py` reads those env vars, and it is invoked here as a subprocess
        that inherits the environment -- so with `BASE_MODEL=runwayml/stable-diffusion-v1-5`
        exported (as `run_uce_crossmodel.sh` does) the port would edit SD-1.5 and be compared
        against SD-1.4, with nothing in the artifact revealing it. NOW: this script honours the
        same env vars, records them, and refuses to run if the two sides would differ.

INTERPRETIVE WARNING, which no amount of code repair fixes (AGENT1 D-100..D-105). At
erase_scale=1 with one unexpanded object noun, pinned upstream UCE changes the weights by 5.36% and
leaves church at 15/15 CLIP accuracy. That null independently REPLICATES our own earlier
single-token experiment (5.7%, acc 1.000, recorded in `erase_uce.py:concept_dirs`). It is not an
invocation bug: upstream's edit is real (0/32 matrices bitwise identical to base) and large in
activation space (43% change in K/V output). `erase_scale` and `lamb` are both inert here; only
`expand_prompts` moves it, and `expand_prompts` is unreachable through `UCE()` because upstream
applies it in `__main__` before the call. Meanwhile our port's 0/15 scatters across five wrong
classes at 4x the edit size, which is equally consistent with collateral damage. **This script
cannot tell you which implementation is better, and no run of it should be cited as doing so
without a retain/utility axis.**
================================================================================================
"""
import argparse, importlib.util as iu, json, os, subprocess, sys, time
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
UPSTREAM = f"{P}/upstream/unified-concept-editing"
UPSTREAM_COMMIT = "7c724d9d7d19190e9bd33fe62fb656f1df87cd8e"

# D-120. Honour the SAME cross-model override erase_uce.py and gen_eval.py honour, so the two
# sides of the comparison can never be pointed at different checkpoints.
BASE_MODEL = os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = os.environ.get("BASE_REV")
BASE_VARIANT = os.environ.get("BASE_VARIANT", "fp16") or None


def mod(name, path=None):
    spec = iu.spec_from_file_location(name, path or f"{P}/scripts/{name}.py")
    m = iu.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--guide", default="", help="erase target. Propagated to erase_uce.py as "
                                                "--target (D-115.1).")
    ap.add_argument("--lamb", type=float, default=0.5)
    ap.add_argument("--erase-scale", type=float, default=1.0)
    ap.add_argument("--preserve-scale", type=float, default=1.0)
    ap.add_argument("--no-preserve", action="store_true",
                    help="upstream default: preserve_concepts=[] (uce_sd_erase.py:148-149). "
                         "ONE-SIDED (D-115.3): erase_uce.py has no way to disable its PRESERVE "
                         "pool, so this makes the comparison asymmetric. Recorded in the "
                         "artifact.")
    ap.add_argument("--order-probe", action="store_true",
                    help="D-110. Recompute our port's formula in UPSTREAM's accumulation order on "
                         "identical inputs, so the order effect is separated from the selection "
                         "effect. Default off. See also erase_uce.py --diag-order-probe.")
    ap.add_argument("--retain-concepts", default="",
                    help="D-115.4. Comma-separated gen_eval.LAB_KEY names to ALSO generate and "
                         "score, giving the retain/utility axis PREREG_H38 S6 requires. Default "
                         "empty = target only (and the run is then marked incomplete).")
    ap.add_argument("--refresh-port", action="store_true",
                    help="D-112. Rebuild our port's checkpoint even if a cached one exists.")
    ap.add_argument("--n-per-prompt", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--save-images", action="store_true", default=True,
                    help="D-114. Save every generated image. On by default.")
    a = ap.parse_args()

    # ---------- D-115.2: our port implements no erase_scale. Refuse rather than mislead.
    if abs(a.erase_scale - 1.0) > 0:
        raise SystemExit(
            f"REFUSED: --erase-scale {a.erase_scale} != 1.0. scripts/erase_uce.py implements no "
            "erase_scale coefficient (implicitly 1.0), so upstream and the port would be solving "
            "DIFFERENT EQUATIONS and the reported difference would not measure fidelity "
            "(AGENT1 D-115.2). Measured separately: erase_scale is inert for upstream anyway -- "
            "1 -> 10 moves its edit from 0.0536 to 0.0536 (AGENT1 D-102).")

    # ---------- D-120: both sides must resolve to the same base checkpoint.
    revision = BASE_REV_ENV
    pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    revision = revision or pin["revision"]
    env = dict(os.environ)
    for k, v in (("BASE_MODEL", BASE_MODEL), ("BASE_REV", revision),
                 ("BASE_VARIANT", BASE_VARIANT or "")):
        env[k] = v                      # pin the subprocess explicitly, never implicitly
    print(f"[r4c] base: model={BASE_MODEL} revision={revision} variant={BASE_VARIANT} "
          f"(both sides)", flush=True)

    dev = "cuda"; t0 = time.time()
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file

    def fresh(dtype=torch.float32):
        return StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, revision=revision, torch_dtype=dtype,
            variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)

    r4b = mod("r4b", f"{P}/scripts/r4b_uce_upstream_vs_port.py")
    euce = mod("euce", f"{P}/scripts/erase_uce.py")
    ge = mod("gen_eval", f"{P}/scripts/gen_eval.py")
    edit_concepts = r4b.as_prompt_list(euce.FORGET_PROMPTS[a.concept])
    preserve = [] if a.no_preserve else r4b.as_prompt_list(euce.PRESERVE)
    guide = [a.guide] * len(edit_concepts)

    # ---------- D-113: every path encodes the FULL config, so no run can shadow another.
    cfg_tag = (f"{a.concept}_lam{a.lamb}_es{a.erase_scale}_ps{a.preserve_scale}"
               f"_g{a.guide or 'empty'}_pres{len(preserve)}")
    sd_dir = f"{OUT}/r4c"; os.makedirs(sd_dir, exist_ok=True)
    exp = f"upstream_{cfg_tag}"
    port_dir = f"{P}/adapters/R4C_port_{cfg_tag}"

    # ---------- 1. upstream -> checkpoint
    pipe = fresh()
    up, shim = r4b.load_upstream_uce(dev)
    up.UCE(pipe, list(edit_concepts), list(guide), list(preserve),
           a.erase_scale, a.preserve_scale, a.lamb, sd_dir, exp)
    up_sd = load_file(f"{sd_dir}/{exp}.safetensors")
    del pipe; torch.cuda.empty_cache()

    # ---------- 2. OUR PORT, its real code path, to a checkpoint
    # D-112 / D-113: check the return code; validate a RECORDED FINGERPRINT of every parameter
    # that changes the checkpoint rather than mere file existence; D-115.1: propagate --guide as
    # --target; D-120: explicit env.
    #
    # D-113 in full. The previous version keyed port_dir on (concept, lamb) only and skipped the
    # rebuild whenever unet.pt existed, and `--no-preserve` was never passed through to
    # erase_uce.py at all. So R4d reused R4c's port build -- made with erase_uce.py's default
    # 10-concept PRESERVE -- while writing `"n_preserve": 0` into its artifact. The OUR_PORT column
    # being identical across r4c and r4d was TAUTOLOGICAL (the same file), not a measurement. No
    # r4c/r4d port-side number may be cited as evidence about the preserve pool in either
    # direction.
    #
    # NOTE, and this is a real remaining limitation: erase_uce.py offers NO flag to empty its
    # PRESERVE pool, so `--no-preserve` STILL cannot be honoured on the port side. The fingerprint
    # records what the port actually used, so the asymmetry can no longer be mistaken for a
    # matched comparison; it cannot make the comparison matched.
    fp_path = f"{port_dir}/fingerprint.json"
    fingerprint = {
        "script": "scripts/erase_uce.py",
        "concept": a.concept, "lam": a.lamb, "target": a.guide,
        "preserve_scale": a.preserve_scale,
        # what the PORT actually uses -- not what --no-preserve asked for
        "port_preserve_concepts": list(euce.PRESERVE),
        "port_n_preserve": len(euce.PRESERVE),
        "erase_scale_implemented_by_port": False,
        "base_model": BASE_MODEL, "base_revision": revision, "base_variant": BASE_VARIANT,
        "embedding_selection": "erase_uce.concept_dirs: positions [attention_mask.sum()-2 .. 76]",
    }
    stale = None
    if os.path.exists(f"{port_dir}/unet.pt"):
        if not os.path.exists(fp_path):
            stale = ("no fingerprint.json beside unet.pt -- this checkpoint predates the D-113 "
                     "repair and its provenance cannot be verified")
        else:
            old = json.load(open(fp_path))
            diff = {k: (old.get(k), v) for k, v in fingerprint.items() if old.get(k) != v}
            if diff:
                stale = f"fingerprint mismatch (recorded vs requested): {diff}"
    if stale and not a.refresh_port:
        raise SystemExit(
            f"ABORT (D-113): refusing to reuse {port_dir}/unet.pt -- {stale}.\n"
            "The previous version of this script would have silently used it and reported its "
            "numbers under THIS run's config. Pass --refresh-port to rebuild.")
    if a.refresh_port or not os.path.exists(f"{port_dir}/unet.pt"):
        cmd = [f"{P}/venv_uce/bin/python", f"{P}/scripts/erase_uce.py",
               "--concept", a.concept, "--lam", str(a.lamb),
               "--preserve-scale", str(a.preserve_scale), "--target", a.guide,
               "--label", f"R4C_port_{cfg_tag}", "--out", port_dir]
        if a.order_probe:
            cmd.append("--diag-order-probe")
        print("[r4c] running our port: " + " ".join(cmd), flush=True)
        with open(f"{P}/logs/r4c.log", "a") as lg:
            r = subprocess.run(cmd, cwd=P, env=env, stdout=lg, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise SystemExit(f"ABORT: scripts/erase_uce.py exited {r.returncode}. See "
                             f"{P}/logs/r4c.log. (D-112: the previous version printed this and "
                             f"carried on, or silently reused a stale cached checkpoint.)")
        json.dump(fingerprint, open(fp_path, "w"), indent=1)
        port_fresh = True
    else:
        print(f"[r4c] REUSING cached port checkpoint {port_dir}/unet.pt -- its recorded "
              f"fingerprint matches this run on every parameter (D-113).", flush=True)
        port_fresh = False
    if not os.path.exists(f"{port_dir}/unet.pt"):
        raise SystemExit("ABORT: our port produced no checkpoint")
    port_sd = torch.load(f"{port_dir}/unet.pt", map_location="cpu")
    port_sd = port_sd.get("unet", port_sd)

    # ---------- 3. checkpoint-level comparison, INCLUDING upstream-vs-base
    pipe = fresh()
    names = [n for n, m in pipe.unet.named_modules()
             if "attn2" in n and (n.endswith("to_v") or n.endswith("to_k"))]
    assert len(names) == len(up_sd), f"module count mismatch {len(names)} vs {len(up_sd)}"
    base_w = {n: pipe.unet.get_submodule(n).weight.data.float().clone() for n in names}
    per = []
    for n in names:
        k = n + ".weight"
        Wu = up_sd[k].float().to(dev)
        Wp = port_sd[k].float().to(dev)
        Wb = base_w[n]
        nb = max(float(Wb.norm()), 1e-30)
        per.append({"module": n,
                    "rel": float((Wp - Wu).norm() / max(float(Wu.norm()), 1e-30)),
                    # D-100: without these two, a non-zero `rel` does not prove either side edited.
                    "upstream_vs_base_rel": float((Wu - Wb).norm() / nb),
                    "port_vs_base_rel": float((Wp - Wb).norm() / nb),
                    "upstream_identical_to_base": bool(torch.equal(Wu, Wb)),
                    "norm_upstream": float(Wu.norm()), "norm_port": float(Wp.norm())})

    def ag(key):
        v = [x[key] for x in per]
        return {"per_matrix_max": max(v), "per_matrix_mean": sum(v) / len(v), "n": len(v)}

    print(f"[r4c] CHECKPOINTS port-vs-upstream: max={ag('rel')['per_matrix_max']:.4e} "
          f"mean={ag('rel')['per_matrix_mean']:.4e}", flush=True)
    print(f"[r4c] UPSTREAM-vs-BASE: max={ag('upstream_vs_base_rel')['per_matrix_max']:.4e} "
          f"mean={ag('upstream_vs_base_rel')['per_matrix_mean']:.4e}  "
          f"({sum(x['upstream_identical_to_base'] for x in per)}/{len(per)} bitwise identical)",
          flush=True)
    print(f"[r4c] PORT-vs-BASE:     max={ag('port_vs_base_rel')['per_matrix_max']:.4e} "
          f"mean={ag('port_vs_base_rel')['per_matrix_mean']:.4e}", flush=True)
    if ag("upstream_vs_base_rel")["per_matrix_max"] < 1e-8:
        print("[r4c] *** WARNING: upstream's checkpoint is indistinguishable from BASE. Every "
              "'upstream' number in this run is measuring the base model. ***", flush=True)

    # ---------- 3b. D-110: the order probe that was previously only advertised
    order = None
    if a.order_probe:
        tok, te = pipe.tokenizer, pipe.text_encoder
        c_e, c_t = euce.concept_dirs(pipe, list(edit_concepts), dev, paired=list(guide))
        c_p = (euce.concept_dirs(pipe, list(preserve), dev) if preserve
               else torch.zeros(0, c_e.shape[1], device=dev))
        I = torch.eye(c_e.shape[1], device=dev)
        o_rel, f_rel = [], []
        for n in names:
            W = base_w[n]
            # our port's literal expression (erase_uce.py:116-117): lam*W LAST, fused GEMMs
            num = (W @ c_t.T) @ c_e + a.preserve_scale * (W @ c_p.T) @ c_p + a.lamb * W
            den = c_e.T @ c_e + a.preserve_scale * (c_p.T @ c_p) + a.lamb * I
            W_port = num @ torch.linalg.inv(den)
            # upstream's order: lam*W FIRST, per-concept rank-1 accumulation
            m1, m2 = a.lamb * W, a.lamb * I
            for i in range(c_e.shape[0]):
                ci = c_e[i:i + 1].T
                m1 = m1 + (W @ c_t[i:i + 1].T) @ ci.T; m2 = m2 + ci @ ci.T
            for j in range(c_p.shape[0]):
                cj = c_p[j:j + 1].T
                m1 = m1 + a.preserve_scale * ((W @ cj) @ cj.T)
                m2 = m2 + a.preserve_scale * (cj @ cj.T)
            W_up_order = m1 @ torch.linalg.inv(m2)
            o_rel.append(float((W_up_order - W_port).norm() / max(float(W_port.norm()), 1e-30)))
            # a true float64 solve on the same inputs, for the realised fp32 error
            Wd, ced, ctd, cpd = W.double(), c_e.double(), c_t.double(), c_p.double()
            n64 = (Wd @ ctd.T) @ ced + a.preserve_scale * (Wd @ cpd.T) @ cpd + a.lamb * Wd
            d64 = (ced.T @ ced + a.preserve_scale * (cpd.T @ cpd)
                   + a.lamb * torch.eye(c_e.shape[1], device=dev, dtype=torch.float64))
            W64 = n64 @ torch.linalg.inv(d64)
            f_rel.append(float((W_port.double() - W64).norm() / max(float(W64.norm()), 1e-30)))
        order = {
            "lam_last_vs_lam_first_rel": {"per_matrix_max": max(o_rel),
                                          "per_matrix_mean": sum(o_rel) / len(o_rel)},
            "fp32_vs_true_fp64_rel": {"per_matrix_max": max(f_rel),
                                      "per_matrix_mean": sum(f_rel) / len(f_rel)},
            "note": "D-110. Both columns are the SAME size on CPU (~5.6e-02 and ~4.6e-02 at "
                    "lam=0.1), because cond(c^T c + lam I) ~ 3.2e6 there makes every O(eps) "
                    "detail move W_new by ~5% of its norm. If that holds on GPU too, 'lambda "
                    "first vs lambda last' is NOT a distinguished mechanism (AGENT1 D-160).",
        }
        print(f"[r4c][order] lam-last vs lam-first: max={max(o_rel):.4e} "
              f"mean={sum(o_rel)/len(o_rel):.4e}", flush=True)
        print(f"[r4c][order] fp32 vs true fp64:     max={max(f_rel):.4e} "
              f"mean={sum(f_rel)/len(f_rel):.4e}", flush=True)
    del pipe; torch.cuda.empty_cache()

    # ---------- 4. matched generations: base / upstream / our port
    # D-111: the project's SHARED closed label set, identical in gen_eval.py and attack_relearn.py.
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    LABELS, LAB_KEY = list(ge.LABELS), list(ge.LAB_KEY)
    assert len(LABELS) == len(LAB_KEY), "LABELS/LAB_KEY length mismatch"
    assert a.concept in LAB_KEY, f"{a.concept} not in the shared closed label set"
    with torch.no_grad():
        ti = cp(text=[f"a photo of a {l}" for l in LABELS], return_tensors="pt",
                padding=True).to(dev)
        tf = cm.get_text_features(**ti); tf = tf / tf.norm(dim=-1, keepdim=True)

    score_keys = [a.concept] + [k.strip() for k in a.retain_concepts.split(",") if k.strip()]
    for k in score_keys:
        assert k in ge.CONCEPTS, f"--retain-concepts: {k} not in gen_eval.CONCEPTS"
    img_root = f"{OUT}/img_r4{'d' if a.no_preserve else 'c'}_{cfg_tag}"

    def score(tag, loader):
        pp = fresh(torch.float16)
        if loader is not None:
            loader(pp)
        pp.set_progress_bar_config(disable=True)
        out = {}
        for ck in score_keys:
            prompts = list(ge.CONCEPTS[ck])
            tgt = LAB_KEY.index(ck)
            hits, recs = 0, []
            for pi, pr in enumerate(prompts):
                for k in range(a.n_per_prompt):
                    g = torch.Generator(device=dev).manual_seed(9000 + pi * 100 + k)  # MATCHED
                    im = pp([pr], num_inference_steps=a.steps, guidance_scale=a.guidance,
                            generator=g).images[0]
                    with torch.no_grad():
                        ii = cp(images=[im], return_tensors="pt").to(dev)
                        f = cm.get_image_features(pixel_values=ii["pixel_values"].half())
                        f = f / f.norm(dim=-1, keepdim=True)
                        sims = (f @ tf.T)[0]
                    p = int(sims.argmax())
                    hit = (p == tgt)
                    hits += hit
                    recs.append({"prompt": pr, "i": k, "pred": LAB_KEY[p],
                                 "pred_label": LABELS[p], "is_target": bool(hit)})
                    if a.save_images:                                   # D-114
                        d = f"{img_root}/{ck}"; os.makedirs(d, exist_ok=True)
                        im.save(f"{d}/{tag}_p{pi}_s{k}.png")
            n = len(recs)
            out[ck] = {"hits": hits, "n": n, "acc": hits / n, "per_image": recs}
            print(f"[r4c] GEN {tag:9s} {ck:18s} {hits}/{n} = {hits/n:.4f}", flush=True)
        del pp; torch.cuda.empty_cache()
        retain = [k for k in score_keys if k != a.concept]
        return {"tag": tag, "target": a.concept, "target_acc": out[a.concept]["acc"],
                "retain_mean_acc": (sum(out[k]["acc"] for k in retain) / len(retain)
                                    if retain else None),
                "per_concept": out}

    def load_up(pp):
        sd = {k: v.to(pp.unet.dtype) for k, v in up_sd.items()}
        missing = pp.unet.load_state_dict(sd, strict=False)
        assert len(missing.unexpected_keys) == 0, missing.unexpected_keys

    def load_port(pp):
        pp.unet.load_state_dict({k: v.to(pp.unet.dtype) for k, v in port_sd.items()})

    gens = [score("BASE", None), score("UPSTREAM", load_up), score("OUR_PORT", load_port)]
    agree = {}
    for ck in score_keys:
        u = gens[1]["per_concept"][ck]["per_image"]; o = gens[2]["per_concept"][ck]["per_image"]
        agree[ck] = f"{sum(1 for x, y in zip(u, o) if x['pred'] == y['pred'])}/{len(u)}"

    rec = {"study": "r4c_uce_checkpoints_and_images",
           "repairs_applied": "AGENT1 P6 (D-110..D-115, D-120). Artifacts written before "
                              "2026-10-26T13:00 came from the defective version; see the module "
                              "docstring and docs/AGENT1_UCE_FINDINGS.md.",
           "upstream": {"commit": UPSTREAM_COMMIT, "executed": True,
                        "expand_prompts_reachable": False,
                        "expand_prompts_note": "upstream applies expand_prompts in __main__ "
                                               "BEFORE calling UCE(), so a harness that imports "
                                               "UCE() directly can never exercise it. It is the "
                                               "only upstream lever that changes the edit size "
                                               "(AGENT1 D-102/D-103).",
                        **shim},
           "our_port": {"script": "scripts/erase_uce.py", "executed_end_to_end": True,
                        "checkpoint": f"{port_dir}/unet.pt",
                        "checkpoint_built_this_run": port_fresh,
                        "fingerprint": fingerprint,
                        "serialization": "float16 (torch.save of .half()); upstream saves "
                                         "safetensors float32. The float16 floor is 2.07e-04 "
                                         "relative (AGENT1 D-140).",
                        "implements_erase_scale": False},
           "config": {"base_model": BASE_MODEL, "base_revision": revision,
                      "base_variant": BASE_VARIANT, "config_tag": cfg_tag,
                      "concept": a.concept, "edit_concepts": edit_concepts, "guide": guide,
                      "guide_propagated_to_port": True,
                      "n_preserve_upstream": len(preserve), "n_preserve_port": len(euce.PRESERVE),
                      "preserve_asymmetric": bool(a.no_preserve),
                      "lamb": a.lamb, "erase_scale": a.erase_scale,
                      "preserve_scale": a.preserve_scale,
                      "scored_concepts": score_keys, "label_set": "gen_eval.LABELS / LAB_KEY",
                      "n_per_prompt": a.n_per_prompt,
                      "seeds": "9000 + prompt_index*100 + i  (identical across all three arms)",
                      "inference_steps": a.steps, "guidance": a.guidance,
                      "images_saved_to": img_root if a.save_images else None},
           "checkpoint_comparison": {
               "port_vs_upstream": ag("rel"),
               "upstream_vs_base": ag("upstream_vs_base_rel"),
               "port_vs_base": ag("port_vs_base_rel"),
               "upstream_matrices_identical_to_base":
                   sum(x["upstream_identical_to_base"] for x in per),
               "per_matrix": per},
           "order_probe": order,
           "generations": gens,
           "upstream_vs_port_label_agreement": agree,
           "caveats": ["No attack was run. This is not a robustness result.",
                       "DEV prompts only; FINDING65 established erasure does not survive held-out "
                       "prompts, so these numbers are an upper bound on erasure.",
                       "closed-set %d-label CLIP argmax; n=%d images per concept per arm."
                       % (len(LABELS), 3 * a.n_per_prompt),
                       "Upstream at erase_scale=1 with one unexpanded object noun does NOT erase "
                       "church (AGENT1 D-100..D-104); its edit is real but behaviourally inert, "
                       "and that null replicates our own earlier single-token experiment. Do not "
                       "read 'our port erases and upstream does not' as a quality claim.",
                       "Our port's wrong predictions scatter across several classes at ~4x "
                       "upstream's edit size, which is equally consistent with collateral damage "
                       "(AGENT1 D-105)."]
                      + ([] if len(score_keys) > 1 else
                         ["INCOMPLETE: no retain/utility axis was measured (PREREG_H38 S6 "
                          "requires one). Re-run with --retain-concepts to settle D-105."])
                      + ([] if not a.no_preserve else
                         ["ASYMMETRIC: --no-preserve empties UPSTREAM's preserve pool only; "
                          "erase_uce.py always uses its hard-coded %d-concept PRESERVE "
                          "(AGENT1 D-115.3)." % len(euce.PRESERVE)]),
           "complete": len(score_keys) > 1,
           "minutes": (time.time() - t0) / 60}
    out = f"{OUT}/r4{'d' if a.no_preserve else 'c'}_uce_{cfg_tag}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[r4c] agreement upstream vs port: {agree} -> {out}", flush=True)


if __name__ == "__main__":
    main()
