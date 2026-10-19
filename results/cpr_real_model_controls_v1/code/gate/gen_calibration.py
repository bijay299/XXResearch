#!/usr/bin/env python3
"""Generate the 64 frozen calibration images from pristine M0 (no adapter).

These are SEPARATE development calibration cases and are never part of the
scientific matrix. They exist only to qualify the measurement instrument.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ["CPR_ROOT"])
sys.path.insert(0, os.path.join(os.environ["CPR_ROOT"], "pydeps"))
import torch

from cprlib import ledger as LG
from cprlib import flux_harness as FH

ROOT = os.environ["CPR_ROOT"]


def main():
    LG.init("gen_calibration", devices=(os.environ.get("CUDA_VISIBLE_DEVICES", "?"),))
    wd = LG.Watchdog(max_wall_s=2 * 3600).start()
    cases = json.load(open(os.path.join(ROOT, "configs", "cases.json")))["calibration_cases"]
    outdir = os.path.join(ROOT, "images", "calibration")
    os.makedirs(outdir, exist_ok=True)

    H = FH.FluxHarness(device="cuda:0", dtype=torch.bfloat16)
    H.set_eval_mode()                      # pristine M0: no adapter attached at all
    recs = []
    t0 = time.time()
    for i, c in enumerate(cases):
        p = os.path.join(outdir, f"{c['case_id']}.png")
        if os.path.exists(p):
            recs.append({**c, "path": p, "reused": True})
            continue
        emb = H.encode_prompt(c["prompt"])
        nh = H.initial_noise_hash(c["eval_seed"])
        img = H.generate(emb, seed=c["eval_seed"])
        img.save(p)
        LG.record_images(1, "calibration", c["case_id"])
        recs.append({**c, "path": p, "initial_noise_sha256": nh, "reused": False})
        if (i + 1) % 16 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(cases)}  {el:.0f}s  ({el/(i+1):.2f}s/img)")
    with open(os.path.join(ROOT, "artifacts", "calibration_images.json"), "w") as f:
        json.dump(recs, f, indent=2, sort_keys=True)
    wd.stop()
    LG.close(bucket="entry")
    print(f"generated/reused {len(recs)} calibration images in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
