#!/usr/bin/env python3
"""THE evaluation harness. One arm per invocation, one GPU per invocation.

Arms
  M0        pristine FLUX.1-dev, nothing replaced
  T0        both native transcoders installed, NO erasure modification
  N         both native transcoders installed + native BLOCK erasure for --erase-target
  (prototype / control arms reuse this file via --variant; see code/variants.py)

Every arm draws its initial noise from torch.Generator(device='cpu').manual_seed(case.init_seed),
with init_seed fixed in cases.json, so the SAME case has the SAME starting latent in every arm.
That is what makes the per-case transitions paired. The generator is rebuilt per image, so
generation order cannot perturb any case's noise.

The erasure itself is the authors' own `apply_block_erasure` from
scripts/unlearn_canvas/diffusion.py (moved verbatim into a function; see INTEGRATION_AUDIT.md),
and the feature identification is the authors' `compute_latent_percentages`. Nothing about the
native update is reimplemented here.
"""
import os, sys, json, time, argparse, hashlib, copy
import torch

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
REPO = f"{SPRINT}/code/block-concept-removal"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{SPRINT}/code")
# The release resolves prompts/ and constants/ relative to its own root, so we must chdir
# there. Capture the caller's cwd FIRST so that a relative --out-dir still means what the
# caller meant, instead of silently landing inside the vendored repo.
ORIG_CWD = os.getcwd()
os.chdir(REPO)

from diffusers import FluxPipeline
from scripts.transcoder.model_inference import Transcoder
from scripts.unlearn_canvas.diffusion import (
    compute_latent_percentages, apply_block_erasure, _main_device)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pipe(dev):
    pipe = FluxPipeline.from_pretrained(os.environ["FLUX_PATH"],
                                        torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install_transcoders(pipe, tdir, dev, rep):
    """Native install path: fp32 transcoder modules replacing the two FLUX gates.

    Kept in fp32 because that is the dtype they were TRAINED in and the dtype the release's
    model_inference.Transcoder constructs; casting them to bf16 would be a silent
    quantisation of the trained module. Input/output are cast at the boundary (PATCH A15).
    """
    sdc = torch.load(f"{tdir}/context_embedder.pt", map_location="cpu")
    sdt = torch.load(f"{tdir}/text_embedder.pt", map_location="cpu")
    tc = Transcoder.from_state_dict(sdc).to(device=dev)
    tt = Transcoder.from_state_dict(sdt).to(device=dev)
    rep["transcoder_files"] = {
        "context_embedder.pt": {"sha256": sha(f"{tdir}/context_embedder.pt")},
        "text_embedder.pt": {"sha256": sha(f"{tdir}/text_embedder.pt")}}
    rep["transcoder_shapes"] = {
        "context": {"n_inputs": tc.n_inputs, "n_outputs": tc.n_outputs,
                    "n_latents": tc.n_latents, "k": tc.k,
                    "param_dtype": str(tc.encoder.weight.dtype)},
        "pooled": {"n_inputs": tt.n_inputs, "n_outputs": tt.n_outputs,
                   "n_latents": tt.n_latents, "k": tt.k,
                   "param_dtype": str(tt.encoder.weight.dtype)}}
    # sanity: the modules we are about to replace must have the shapes the transcoders model
    ce, te = pipe.transformer.context_embedder, pipe.transformer.time_text_embed.text_embedder
    assert (ce.in_features, ce.out_features) == (tc.n_inputs, tc.n_outputs), "context shape mismatch"
    assert (te.linear_1.in_features, te.linear_2.out_features) == (tt.n_inputs, tt.n_outputs), \
        "pooled shape mismatch"
    rep["replaced_modules"] = {
        "transformer.context_embedder": f"Linear({ce.in_features},{ce.out_features}) -> Transcoder",
        "transformer.time_text_embed.text_embedder":
            f"PixArtAlphaTextProjection({te.linear_1.in_features},{te.linear_2.out_features}) -> Transcoder"}
    setattr(pipe.transformer, "context_embedder", tc)
    setattr(pipe.transformer.time_text_embed, "text_embedder", tt)
    return tc, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--object", required=True, choices=["Bears", "Cats", "Horses", "SHARED"])
    ap.add_argument("--strata", nargs="+", default=["target", "related", "unrelated"])
    ap.add_argument("--transcoder-dir", default=None)
    ap.add_argument("--erase-target", default=None, help="concept string, e.g. Bears")
    ap.add_argument("--variant", default=None, help="prototype/control variant name (code/variants.py)")
    ap.add_argument("--variant-arg", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N cases")
    ap.add_argument("--dry-run", action="store_true", help="set up + verify, generate nothing")
    a = ap.parse_args()

    dev = "cuda:0"          # one visible GPU per process (CUDA_VISIBLE_DEVICES pins which)
    for _f in ("out_dir", "transcoder_dir"):
        _v = getattr(a, _f)
        if _v and not os.path.isabs(_v):
            setattr(a, _f, os.path.join(ORIG_CWD, _v))
    os.makedirs(a.out_dir, exist_ok=True)
    rep = {"arm": a.arm, "object": a.object, "strata": a.strata,
           "variant": a.variant, "variant_arg": a.variant_arg,
           "erase_target": a.erase_target, "transcoder_dir": a.transcoder_dir,
           "gpu_name": torch.cuda.get_device_name(0),
           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    spec = json.load(open(f"{SPRINT}/cases.json"))
    ga = spec["generation_arguments_pinned"]
    rep["generation_arguments"] = ga

    # ---- cases for this arm
    if a.object == "SHARED":
        cases = [c for s in a.strata for c in spec["shared_retained"][s]]
    else:
        cases = [c for s in a.strata for c in spec["cases"][a.object][s]]
    if a.limit:
        cases = cases[:a.limit]
    rep["n_cases"] = len(cases)

    t_load0 = time.time()
    pipe = load_pipe(dev)
    rep["t_load_s"] = round(time.time() - t_load0, 2)

    # ---- install transcoders (T0 / N / prototype / control)
    tc = tt = None
    if a.arm != "M0":
        assert a.transcoder_dir, "non-M0 arms need --transcoder-dir"
        t0 = time.time()
        tc, tt = install_transcoders(pipe, a.transcoder_dir, dev, rep)
        rep["t_install_transcoders_s"] = round(time.time() - t0, 2)

    # ---- erasure (N and prototype/control arms)
    if a.erase_target:
        assert tc is not None, "erasure needs transcoders installed"
        t0 = time.time()
        latent_info = compute_latent_percentages(
            pipe, tc, tt, "flux", a.erase_target.replace(" ", "_"), "flux")
        rep["t_feature_identification_s"] = round(time.time() - t0, 2)

        # snapshot for the "intended tensors edited / unrelated unchanged" check
        pre = {"context": tc.W_dec.data.clone(), "pooled": tt.W_dec.data.clone()}
        pre_other = {
            "context_enc": tc.encoder.weight.data.clone(),
            "context_encb": tc.enc_bias.data.clone(),
            "context_decb": tc.dec_bias.data.clone(),
            "pooled_enc": tt.encoder.weight.data.clone(),
            "pooled_encb": tt.enc_bias.data.clone(),
            "pooled_decb": tt.dec_bias.data.clone()}

        t0 = time.time()
        if a.variant:
            from variants import apply_variant
            stats = apply_variant(a.variant, a.variant_arg, tc, tt, latent_info, dev)
        else:
            stats = apply_block_erasure(tc, tt, latent_info, dev, stats={})
        rep["t_erasure_s"] = round(time.time() - t0, 2)
        rep["erasure_stats"] = {k: v for k, v in (stats or {}).items()}

        # ---- correctness: intended rows changed, everything else bit-identical
        chk = {}
        for name, mod, key in (("context", tc, "context"), ("pooled", tt, "pooled")):
            post = mod.W_dec.data
            changed = (post != pre[key]).any(dim=1)
            chk[name] = {
                "rows_changed": int(changed.sum()),
                "rows_total": int(post.shape[0]),
                "all_finite_after": bool(torch.isfinite(post).all()),
                "max_abs_delta": float((post - pre[key]).abs().max()),
            }
        chk["unrelated_tensors_bit_identical"] = {
            k: bool(torch.equal(v, {"context_enc": tc.encoder.weight.data,
                                    "context_encb": tc.enc_bias.data,
                                    "context_decb": tc.dec_bias.data,
                                    "pooled_enc": tt.encoder.weight.data,
                                    "pooled_encb": tt.enc_bias.data,
                                    "pooled_decb": tt.dec_bias.data}[k]))
            for k, v in pre_other.items()}
        rep["edit_verification"] = chk
        del pre, pre_other

    # ---- paired-noise check: the generator for a case is a pure function of its init_seed
    c0 = cases[0]
    g1 = torch.Generator(device="cpu").manual_seed(c0["init_seed"])
    g2 = torch.Generator(device="cpu").manual_seed(c0["init_seed"])
    n1 = torch.randn(4096, generator=g1)
    n2 = torch.randn(4096, generator=g2)
    rep["paired_noise_check"] = {
        "case": c0["case_id"], "init_seed": c0["init_seed"],
        "same_seed_bit_identical": bool(torch.equal(n1, n2)),
        "noise_sha256_first4096": hashlib.sha256(n1.numpy().tobytes()).hexdigest()[:32],
        "note": ("CPU generator rebuilt per image from case.init_seed, so the initial latent "
                 "for a case is identical across arms and independent of device/order")}
    assert rep["paired_noise_check"]["same_seed_bit_identical"]

    if a.dry_run:
        rep["dry_run"] = True
        json.dump(rep, open(f"{a.out_dir}/arm_report.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in rep if k != "generation_arguments"}, indent=1)[:4000])
        return

    # ---- generation, fixed documented order
    torch.cuda.reset_peak_memory_stats()
    recs, t_gen0 = [], time.time()
    for i, c in enumerate(cases):
        g = torch.Generator(device="cpu").manual_seed(c["init_seed"])
        t0 = time.time()
        with torch.no_grad():
            img = pipe(c["prompt"], height=ga["height"], width=ga["width"],
                       num_inference_steps=ga["num_inference_steps"],
                       guidance_scale=ga["guidance_scale"],
                       max_sequence_length=ga["max_sequence_length"],
                       generator=g).images[0]
        fp = os.path.join(a.out_dir, c["filename"])
        img.save(fp)
        recs.append({**c, "order": i, "file": c["filename"],
                     "gen_seconds": round(time.time() - t0, 2),
                     "sha256": sha(fp)})
        if i % 10 == 0:
            el = time.time() - t_gen0
            print(f"[{a.arm}/{a.object}] {i+1}/{len(cases)}  {el:.0f}s  "
                  f"eta {el/(i+1)*(len(cases)-i-1):.0f}s", flush=True)

    rep["t_generation_s"] = round(time.time() - t_gen0, 2)
    rep["t_generation_per_image_s"] = round(rep["t_generation_s"] / max(1, len(cases)), 3)
    rep["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rep["images_written"] = len(recs)
    json.dump(recs, open(f"{a.out_dir}/cases_generated.json", "w"), indent=1)
    json.dump(rep, open(f"{a.out_dir}/arm_report.json", "w"), indent=1)
    print(f"[{a.arm}/{a.object}] wrote {len(recs)} images to {a.out_dir} "
          f"in {rep['t_generation_s']:.0f}s ({rep['t_generation_per_image_s']}s/img)")


if __name__ == "__main__":
    main()
