#!/usr/bin/env python3
"""Generates the 12 permitted NON-STUDY practice items for the review viewer.

These are instructional aids only. They are not study cases, not empirical labels, and
not a reviewer-accuracy gate. They use a practice object ("bicycle") chosen to sit
OUTSIDE any future study target set, so they cannot coach real cases.

Two deliberately degraded items are produced by running the sampler for very few steps,
which is the honest way to obtain a genuine "severe rendering failure" example from a
strong base model. How each item was produced is recorded and published.

Base model only. No erasure, no editor, no training, no tuning.
"""
import os, sys, json, time, hashlib, argparse
import torch

ROOT = "/data/bijaypandey/cuig_pilot/measurement_study_v1"
sys.path.insert(0, f"{ROOT}/code")
import ledger

FLUX = ("/data/bijaypandey/cuig_pilot/hf_home/hub/models--black-forest-labs--FLUX.1-dev"
        "/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
OUT = f"{ROOT}/assets/practice"

# (id, prompt, steps, seed, what it is meant to illustrate)
ITEMS = [
    ("P01", "a red bicycle leaning against a weathered brick wall, afternoon light", 28, 1101,
     "object plainly present, coherent scene"),
    ("P02", "a weathered brick wall in afternoon light, empty pavement", 28, 1102,
     "object plainly absent, coherent scene"),
    ("P03", "a wide empty park path at dawn with a tiny bicycle far away near the horizon", 28, 1103,
     "object present but very small / hard to resolve"),
    ("P04", "a close-up of a chrome bicycle wheel and spokes filling the frame", 28, 1104,
     "only a part of the object is visible"),
    ("P05", "a vintage motorcycle parked on a cobblestone street", 28, 1105,
     "a different but related object; the practice target is absent"),
    ("P06", "a busy bicycle parking rack crowded with many bicycles", 28, 1106,
     "many instances of the object present"),
    ("P07", "an oil painting in heavy impasto of a bicycle in a sunlit courtyard", 28, 1107,
     "heavily stylised painting that is still a coherent scene"),
    ("P08", "a cluttered garage interior with tools, boxes and a bicycle in shadow", 28, 1108,
     "object present but partly occluded in clutter"),
    ("P09", "a quiet living room with a sofa, lamp and bookshelf", 28, 1109,
     "ordinary coherent scene with no bicycle"),
    ("P10", "a surreal composition where a bicycle dissolves into flying birds", 28, 1110,
     "unusual composition that is nonetheless coherent"),
    ("P11", "a bicycle in a garden", 1, 1111,
     "DEGRADED ON PURPOSE (1 sampling step): example of severe rendering failure"),
    ("P12", "a city street with bicycles", 2, 1112,
     "DEGRADED ON PURPOSE (2 sampling steps): example of severe rendering failure"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    items = [i for i in ITEMS if (not a.only or i[0] in a.only)]

    # hard pre-flight against the ceilings
    ledger.enforce(need_gpu_hours=0.3, need_images=len(items))

    from diffusers import FluxPipeline
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    load_s = time.time() - t0

    recs, decoded, t_gen0 = [], 0, time.time()
    try:
        for pid, prompt, steps, seed, purpose in items:
            g = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                img = pipe(prompt, height=1024, width=1024, num_inference_steps=steps,
                           guidance_scale=3.5, max_sequence_length=512, generator=g).images[0]
            fp = os.path.join(OUT, f"{pid}.png")
            img.save(fp)
            decoded += 1
            recs.append({"practice_id": pid, "prompt": prompt, "steps": steps, "seed": seed,
                         "purpose": purpose, "file": f"{pid}.png",
                         "sha256": sha(fp), "bytes": os.path.getsize(fp)})
            print(f"  {pid} steps={steps:2d} -> {fp}", flush=True)
    finally:
        wall = time.time() - t_gen0 + load_s
        ledger.job("gen_practice", "generation", 1, wall, images_decoded=decoded,
                   ok=(decoded == len(items)), bucket="practice",
                   note=f"FLUX.1-dev base only, no erasure; {decoded}/{len(items)} decoded; "
                        f"load {load_s:.1f}s")

    meta = {"base_model": "black-forest-labs/FLUX.1-dev",
            "revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
            "settings": {"height": 1024, "width": 1024, "guidance_scale": 3.5,
                         "max_sequence_length": 512, "dtype": "bfloat16",
                         "generator": "torch.Generator(device='cpu').manual_seed(seed)"},
            "practice_object": "bicycle",
            "status_note": ("NON-STUDY instructional aids. Not study cases, not empirical "
                            "labels, not a reviewer-accuracy gate. The practice object is "
                            "deliberately outside any study target set."),
            "items": recs}
    json.dump(meta, open(f"{OUT}/practice_manifest.json", "w"), indent=1)
    print(f"\nwrote {len(recs)} practice images + manifest")


if __name__ == "__main__":
    main()
