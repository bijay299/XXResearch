#!/usr/bin/env python3
"""Create the per-concept `instance_data_dir` training images.

The author repo ships config/disentangle/config_frida.yaml pointing at
`image/disentangle/frida/`, but NO image directories are included in the repository, so the
training set must be created. Following the author's own utils/generate_dataset.py, images
are generated from the config's `instance_prompt` with the author's inference settings
(FLUX.1-dev, 512x512, 28 steps, max_sequence_length=256, bf16).

N_IMAGES=10 per concept is OUR choice (the author specifies none); it is frozen here.
Seeds are derived deterministically from the concept name, so the training set is
reproducible and is identical across all arms and training seeds.

Evaluation prompts are NOT used here: the only text is the config `instance_prompt`.
"""
import argparse, hashlib, json, os, pathlib, time
import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import EA_STATE

N_IMAGES = 10

ap = argparse.ArgumentParser()
ap.add_argument("--concept", required=True)
ap.add_argument("--instance-prompt", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

EA_STATE["explicit_attn"] = True; EA_STATE["maps_only_last"] = False  # author-faithful
rev = os.environ["FLUX_DEV_REVISION"]
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", revision=rev,
                                    torch_dtype=torch.bfloat16).to(a.device)
pipe.set_progress_bar_config(disable=True)
out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
seeds = []
for i in range(N_IMAGES):
    s = int(hashlib.sha256(f"ea-train-img|{a.concept}|{i}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
    seeds.append(s)
    f = out / f"{i:02d}.png"
    if f.exists():
        continue
    g = torch.Generator(device="cpu").manual_seed(s)
    pipe(prompt=a.instance_prompt, generator=g, height=512, width=512,
         num_inference_steps=28, max_sequence_length=256, guidance_scale=3.5).images[0].save(f)
    print(f"  {f.name} seed={s}", flush=True)
(out / "MANIFEST.json").write_text(json.dumps(
    {"concept": a.concept, "instance_prompt": a.instance_prompt, "n_images": N_IMAGES,
     "seeds": seeds, "settings": {"res": 512, "steps": 28, "max_seq_len": 256,
                                  "guidance_scale": 3.5, "dtype": "bf16",
                                  "attn": "author-faithful explicit"},
     "revision": rev, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
print("done", out)
