#!/usr/bin/env python3
"""Generate the 20 primary outputs for one condition (M0 or edited).

Uses GEM's own pipeline construction and wrapper call, at the pinned
implementation's inference settings (28 steps, guidance 3.5, 512x512,
bfloat16). The per-case initial noise is loaded from disk and hash-verified,
and is byte-identical between the two conditions.

Every case is preserved. A case that raises or produces a corrupt output is
recorded explicitly as a missing/failed row -- prompts and seeds are never
replaced and completeness is never manufactured.

Usage:
  run_gen.py --condition m0     --out-dir DIR
  run_gen.py --condition edited --ckpt DIR --out-dir DIR
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402

NUM_CHANNELS_LATENTS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["m0", "edited"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None, help="comma list of case_ids (recovery)")
    a = ap.parse_args()

    if a.condition == "edited" and not a.ckpt:
        raise SystemExit("--ckpt is required for the edited condition")

    os.makedirs(a.out_dir, exist_ok=True)
    prod = C.load_prod_config()
    cases = C.load_cases()["cases"]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        cases = [c for c in cases if c["case_id"] in keep]

    with open(os.path.join(C.ROOT, "manifests", "initial_noise_index.json")) as fh:
        noise_index = {r["case_id"]: r for r in json.load(fh)["cases"]}

    prev = C.pin_base_model_path()
    print(f"[P2] base pinned: {prev!r} -> {C.SNAPSHOT!r}")

    from gem.configs.pipeline_config import PipelineConfig
    from gem.wrappers.factory import PipelineFactory

    pipe_config = PipelineConfig.default_for_model_type(model_type="flux")
    pipe_config = replace(pipe_config, model_dtype="bfloat16")
    if a.condition == "edited":
        pipe_config = replace(pipe_config, model_name_or_path=a.ckpt)

    assert pipe_config.num_inference_steps == 28
    assert pipe_config.inference_guidance_scale == 3.5
    assert pipe_config.image_size == 512

    t0 = time.time()
    pipe, modified = PipelineFactory.create_pipeline(
        pipe_config, return_local_component_names=True)
    wrapper = PipelineFactory.wrap_pipeline(pipe).to("cuda")
    wrapper.eval()
    load_s = time.time() - t0
    print(f"[gen] loaded in {load_s:.1f}s; overwritten components: {modified}")

    if a.condition == "m0":
        assert not modified, f"M0 load must not overwrite anything, got {modified}"
    else:
        assert "transformer" in modified, (
            f"edited load must overwrite the transformer, got {modified}")

    # ---- prove the stored noise equals the native prepare_latents output ----
    probe = cases[0]
    g = torch.Generator(device="cpu").manual_seed(probe["eval_seed"])
    native_lat, _ = pipe.prepare_latents(
        1, NUM_CHANNELS_LATENTS, 512, 512, torch.bfloat16,
        torch.device("cpu"), g, None)
    native_hash = C.tensor_sha256(native_lat)
    stored_hash = noise_index[probe["case_id"]]["tensor_sha256"]
    native_matches = native_hash == stored_hash
    print(f"[gen] native prepare_latents vs stored noise for {probe['case_id']}: "
          f"{'MATCH' if native_matches else 'MISMATCH'}")
    assert native_matches, (
        f"stored noise does not reproduce the native path: "
        f"{native_hash} != {stored_hash}")

    rows = []
    decoded = 0
    for c in cases:
        cid = c["case_id"]
        ni = noise_index[cid]
        npath = os.path.join(C.ROOT, ni["file"])
        lat = torch.load(npath, map_location="cpu")
        h = C.tensor_sha256(lat)
        assert h == ni["tensor_sha256"], f"{cid}: noise hash drift"

        fname = f"{cid}_{a.condition}_seed{c['eval_seed']}.png"
        fpath = os.path.join(a.out_dir, fname)
        row = {
            "case_id": cid, "condition": a.condition, "prompt": c["prompt"],
            "eval_seed": c["eval_seed"], "class": c["class"],
            "category": c["category"],
            "noise_file": ni["file"], "noise_sha256": h,
            "filename": fname,
        }

        try:
            t1 = time.time()
            gen = torch.Generator(device="cpu").manual_seed(c["eval_seed"])
            result = wrapper(
                prompt=c["prompt"],
                height=512, width=512,
                num_inference_steps=28,
                guidance_scale=3.5,
                latents=lat.to("cuda", dtype=torch.bfloat16),
                generator=gen,
            )
            img = result.images[0]
            decoded += 1
            img.save(fpath)
            row.update({
                "status": "ok",
                "seconds": round(time.time() - t1, 2),
                "size": list(img.size),
                "mode": img.mode,
                "file_sha256": C.file_sha256(fpath),
                "bytes": os.path.getsize(fpath),
            })
            print(f"[gen] {cid} ok ({row['seconds']}s) -> {fname}")
        except Exception as e:
            # the decode may or may not have happened; count conservatively
            decoded += 1
            row.update({"status": "failed", "error": f"{type(e).__name__}: {e}",
                        "file_present": os.path.exists(fpath)})
            print(f"[gen] {cid} FAILED: {type(e).__name__}: {e}")

        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    out = {
        "condition": a.condition,
        "checkpoint": a.ckpt if a.condition == "edited" else C.SNAPSHOT,
        "base_revision": C.BASE_REVISION,
        "gem_commit": C.GEM_COMMIT,
        "overwritten_components": modified,
        "load_seconds": round(load_s, 1),
        "inference": {"num_inference_steps": 28, "guidance_scale": 3.5,
                      "height": 512, "width": 512, "dtype": "bfloat16",
                      "batch_size": 1,
                      "latents": "loaded from disk, hash-verified, shared with the other condition",
                      "native_prepare_latents_match": native_matches},
        "deviations_from_gem_generate_and_save_images": [
            "enable_attention_slicing() not called",
            "enable_model_cpu_offload() not called (the 23 GiB transformer fits on one 40 GiB device)",
            "one prompt per call (batch_size=1) with explicit per-case `latents=` instead of a generator list",
        ],
        "n_cases": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "decoded_images": decoded,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "rows": rows,
    }
    C.dump_json(out, os.path.join(a.out_dir, f"index_{a.condition}.json"))

    # tell the launcher how many images were decoded, for the ledger
    label = os.environ.get("MOA_LABEL")
    if label:
        with open(os.path.join(C.ROOT, "logs", f"{label}.images"), "w") as fh:
            fh.write(str(decoded))

    print(f"\n[gen] condition={a.condition} ok={len(ok)}/{len(rows)} "
          f"decoded={decoded} peak={out['peak_gpu_gib']} GiB")
    sys.exit(0 if len(ok) == len(rows) else 5)


if __name__ == "__main__":
    main()
