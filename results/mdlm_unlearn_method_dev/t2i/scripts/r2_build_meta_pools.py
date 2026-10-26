"""Build Meta-Unlearning's three data pools per the author protocol (src/gen_images.py).

  hrm = the unlearned concept            -> church
  tgt = RETAINED concepts RELATED to it  -> cathedral / chapel / basilica
  irt = RETAINED concepts UNRELATED      -> sampled from the author's coco_object_retain.csv

30 train + 30 test per pool, the author's exact negative prompt, metadata.jsonl per directory.
"""
import argparse, json, os, random, torch, pandas as pd

NEG = ("bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,"
       "low quality,lowres,error,blurry,text,cropped,username")
P = "/data/bijaypandey/t2i_unlearn"

def gen(pipe, text, n, seed0):
    out = []
    for i in range(n):
        g = torch.Generator(device="cuda").manual_seed(seed0 + i)
        im = pipe(prompt=text, negative_prompt=NEG, guidance_scale=7.5, generator=g).images[0]
        out.append((im, text))
    return out

def save(data, d):
    os.makedirs(d, exist_ok=True)
    meta = []
    for i, (im, txt) in enumerate(data):
        fn = f"{i}.png"; im.save(os.path.join(d, fn))
        meta.append({"file_name": fn, "text": txt})
    with open(os.path.join(d, "metadata.jsonl"), "w") as f:
        for m in meta: f.write(json.dumps(m) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=",1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float16,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to("cuda")
    coco = pd.read_csv(f"{P}/upstream/Meta-Unlearning/data/coco_object_retain.csv")["prompt"].tolist()
    random.Random(0).shuffle(coco)
    pools = {
        "hrm": ["a photo of a church"],                       # the UNLEARNED concept
        "tgt": ["a photo of a cathedral", "a photo of a chapel", "a photo of a basilica"],
        "irt": coco[:6],                                      # UNRELATED retained objects
    }
    for name, texts in pools.items():
        for split, s0 in (("train", 1000), ("test", 5000)):
            d = os.path.join(a.root, name, split)
            if os.path.exists(os.path.join(d, "metadata.jsonl")):
                print(f"  {name}/{split} exists, skip", flush=True); continue
            per = max(1, a.n // len(texts)); data = []
            for j, t in enumerate(texts):
                data += gen(pipe, t, per, s0 + j * 100)
            save(data, d); print(f"  {name}/{split}: {len(data)} images from {texts}", flush=True)
    print("[r2pools] done", flush=True)
