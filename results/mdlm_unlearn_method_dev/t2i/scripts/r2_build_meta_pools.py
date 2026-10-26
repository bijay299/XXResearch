"""Build Meta-Unlearning's three data pools for upstream src/esd_meta.py.

Pool meanings, verbatim from upstream/Meta-Unlearning/README.md L18:
  hrm = the UNLEARNED concept
  irt = retained concepts UNRELATED to it
  tgt = retained concepts RELATED to it
meta.py / esd_meta.py load them with `load_dataset("imagefolder", data_dir="./dataset/<pool>")`
(L671-673), so each pool needs train/ and test/ subdirectories each carrying a metadata.jsonl with
{"file_name", "text"}.

AUDIT REPAIRS 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md B-6, B-7):

1. SIZING. upstream src/gen_images.py L90-92 generates `num_images=30` train AND 30 test PER
   PROMPT, so a 3-prompt pool is 90 images per split. The previous version of this file allocated
   30 per POOL *split across* prompts (10/prompt for tgt, 5/prompt for irt) while its docstring
   claimed to follow gen_images.py. Both sizings are now selectable and the choice is RECORDED:
     --per-prompt N     upstream semantics (DEFAULT, N=30)
     --total-per-pool N the historical compute adaptation (N images per pool per split)
   `--total-per-pool` is a legitimate adaptation; it is simply not "the author protocol" and
   pools.json now says which was used.

2. SEMANTICS. The legacy scheme erases "church" while RETAINING "cathedral/chapel/basilica" as the
   RELATED-RETAIN pool. Those are hyponyms of church, not neighbours of it -- a cathedral IS a
   church -- so that objective can only be satisfied by suppressing the TOKEN "church", not the
   CONCEPT. Upstream's own related-retain is a superordinate/sibling category (gen_images.py
   L62-66: hrm="Thomas Kinkade" with paint="a painting"), never a hyponym. Two schemes:
     --pool-scheme legacy    the existing pools, reproduced bit-for-bit (same prompts, same seeds)
     --pool-scheme coherent  hrm=church paraphrases; tgt=co-hyponyms+superordinate (mosque,
                             synagogue, temple, castle, town hall, stone building); and a FOURTH
                             pool `fgen` holding cathedral/chapel/basilica, to be scored as FORGET
                             (forget-generalisation), never as retain.
   Whichever is used, report the legacy scheme as PROMPT-SPECIFIC TOKEN SUPPRESSION and the
   coherent scheme as CONCEPT ERASURE. They are different claims.

3. FAIL-FAST. Blank (NSFW-filtered) frames are rejected and regenerated; counts are validated after
   writing; a pre-existing directory is no longer silently skipped (--resume is now explicit); any
   shortfall exits non-zero.
"""
import argparse, csv, hashlib, json, os, sys

NEG = ("bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,"
       "low quality,lowres,error,blurry,text,cropped,username")   # gen_images.py L46, verbatim
P = "/data/bijaypandey/t2i_unlearn"
CFG = 7.5                                                          # gen_images.py L18
UPSTREAM_PER_PROMPT = 30                                           # gen_images.py L91-92

SCHEMES = {
    # name -> {pool: [prompts]}, plus "forget_side" listing pools scored as FORGET
    "legacy": {
        "pools": {
            "hrm": ["a photo of a church"],
            "tgt": ["a photo of a cathedral", "a photo of a chapel", "a photo of a basilica"],
            "irt": None,                       # sampled from coco_object_retain.csv
        },
        "forget_side": ["hrm"],
        "claim": ("PROMPT-SPECIFIC TOKEN SUPPRESSION: cathedral/chapel/basilica are hyponyms of "
                  "church and are held in the RETAIN pool, so a model that merely stops responding "
                  "to the string 'church' satisfies this objective. Do NOT report results from "
                  "this scheme as 'church concept erasure'."),
    },
    "coherent": {
        "pools": {
            "hrm": ["a photo of a church", "a photo of a church building",
                    "a photograph of a church exterior"],
            "fgen": ["a photo of a cathedral", "a photo of a chapel", "a photo of a basilica"],
            "tgt": ["a photo of a mosque", "a photo of a synagogue", "a photo of a temple",
                    "a photo of a castle", "a photo of a town hall", "a photo of a stone building"],
            "irt": None,
        },
        "forget_side": ["hrm", "fgen"],
        "claim": ("CONCEPT ERASURE: hyponyms (cathedral/chapel/basilica) sit in `fgen` and are "
                  "scored as FORGET; the related-retain pool holds co-hyponyms and the "
                  "superordinate, which share structure with the target without being instances "
                  "of it. NOTE: upstream esd_meta.py reads only ./dataset/{hrm,irt,tgt}; `fgen` is "
                  "an EVALUATION pool and is not consumed by the trainer."),
    },
}
# train/test seed bases, per pool. Disjoint by construction so the splits never share a sample.
SEED_BASE = {"train": 1000, "test": 5000}


def die(msg):
    print(f"[r2pools] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def gen_one(pipe, text, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt=text, negative_prompt=NEG, guidance_scale=CFG, generator=g).images[0]


def is_blank(im):
    import numpy as np
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    return bool(a.max() == 0 or a.std() < 1.0)


def gen(pipe, text, n, seed0, max_retries=20):
    """n usable images for `text`. Blank (safety-checker-blanked) frames are regenerated on a
    fresh seed rather than silently written, which is the defect found in the IMMA data path."""
    out, seed, rejects = [], seed0, 0
    while len(out) < n:
        if seed - seed0 >= n + max_retries:
            die(f"could not get {n} usable images for {text!r}: {rejects} blank frames rejected")
        im = gen_one(pipe, text, seed)
        seed += 1
        if is_blank(im):
            rejects += 1
            continue
        out.append((im, text))
    if rejects:
        print(f"    rejected {rejects} blank frame(s) for {text!r}", flush=True)
    return out, rejects


def save(data, d):
    os.makedirs(d, exist_ok=True)
    meta = []
    for i, (im, txt) in enumerate(data):
        fn = f"{i}.png"
        im.save(os.path.join(d, fn))
        meta.append({"file_name": fn, "text": txt})
    with open(os.path.join(d, "metadata.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    return meta


def validate(d, expect):
    """Hard post-write validation: counts, metadata agreement, no blanks."""
    import glob
    from PIL import Image
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    mp = os.path.join(d, "metadata.jsonl")
    if not os.path.isfile(mp):
        die(f"{d}: no metadata.jsonl")
    rows = [json.loads(l) for l in open(mp) if l.strip()]
    names = set(r["file_name"] for r in rows)
    disk = set(os.path.basename(f) for f in files)
    if len(files) != expect:
        die(f"{d}: {len(files)} png on disk, expected {expect}")
    if len(rows) != expect:
        die(f"{d}: {len(rows)} metadata rows, expected {expect}")
    if names != disk:
        die(f"{d}: metadata/disk mismatch: {sorted(names ^ disk)[:5]}")
    blank = [os.path.basename(f) for f in files if is_blank(Image.open(f))]
    if blank:
        die(f"{d}: {len(blank)} blank image(s) survived: {blank[:5]}")
    return len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pool root; esd_meta.py expects it as ./dataset")
    ap.add_argument("--pool-scheme", choices=sorted(SCHEMES), default="legacy",
                    help="legacy = the historical church/cathedral pools (token suppression); "
                         "coherent = the recommended forget/forget-generalisation/related split")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-prompt", type=int, default=None,
                   help=f"images per PROMPT per split, upstream semantics "
                        f"(gen_images.py uses {UPSTREAM_PER_PROMPT})")
    g.add_argument("--total-per-pool", type=int, default=None,
                   help="images per POOL per split, split across that pool's prompts "
                        "(the historical compute adaptation, NOT the author protocol)")
    ap.add_argument("--n-irt-prompts", type=int, default=6,
                    help="how many unrelated-retain prompts to draw from coco_object_retain.csv")
    ap.add_argument("--resume", action="store_true",
                    help="keep any split that already validates instead of failing")
    a = ap.parse_args()

    if a.per_prompt is None and a.total_per_pool is None:
        a.per_prompt = UPSTREAM_PER_PROMPT
    sizing = "per_prompt" if a.per_prompt is not None else "total_per_pool"

    scheme = SCHEMES[a.pool_scheme]
    coco_path = f"{P}/upstream/Meta-Unlearning/data/coco_object_retain.csv"
    if not os.path.isfile(coco_path):
        die(f"missing {coco_path}")
    import random
    coco = [r["prompt"] for r in csv.DictReader(open(coco_path))]
    if len(coco) < a.n_irt_prompts:
        die(f"coco_object_retain.csv has only {len(coco)} prompts")
    random.Random(0).shuffle(coco)
    pools = {k: (list(v) if v is not None else coco[:a.n_irt_prompts])
             for k, v in scheme["pools"].items()}

    plan = {}
    for name, texts in pools.items():
        if sizing == "per_prompt":
            per = a.per_prompt
        else:
            per = max(1, a.total_per_pool // len(texts))
        plan[name] = {"prompts": texts, "per_prompt": per, "per_split": per * len(texts)}
    print(f"[r2pools] scheme={a.pool_scheme} sizing={sizing}", flush=True)
    for name, p in plan.items():
        up = UPSTREAM_PER_PROMPT * len(p["prompts"])
        flag = "" if p["per_split"] == up else f"  <-- upstream would be {up}/split"
        print(f"  {name}: {len(p['prompts'])} prompt(s) x {p['per_prompt']} "
              f"= {p['per_split']}/split{flag}", flush=True)

    todo = []
    for name, p in plan.items():
        for split in ("train", "test"):
            d = os.path.join(a.root, name, split)
            if os.path.isfile(os.path.join(d, "metadata.jsonl")):
                if not a.resume:
                    die(f"{d} already exists; pass --resume to keep it or remove it first")
                validate(d, p["per_split"])
                print(f"  {name}/{split} exists and validates, keeping", flush=True)
                continue
            todo.append((name, split, d, p))

    manifest = {
        "script": "scripts/r2_build_meta_pools.py",
        "root": os.path.abspath(a.root),
        "pool_scheme": a.pool_scheme,
        "scheme_claim": scheme["claim"],
        "forget_side_pools": scheme["forget_side"],
        "sizing": sizing,
        "per_prompt_requested": a.per_prompt,
        "total_per_pool_requested": a.total_per_pool,
        "upstream_per_prompt": UPSTREAM_PER_PROMPT,
        "upstream_reference": "upstream/Meta-Unlearning/src/gen_images.py L90-92",
        "negative_prompt": NEG,
        "guidance_scale": CFG,
        "seed_base": SEED_BASE,
        "trainer_pools": ["hrm", "irt", "tgt"],
        "pools": {},
    }

    if todo:
        import torch
        from diffusers import StableDiffusionPipeline
        pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float16,
            variant="fp16", safety_checker=None, requires_safety_checker=False).to("cuda")
        manifest["base_model"] = "CompVis/stable-diffusion-v1-4"
        manifest["base_revision"] = pin["revision"]
        for name, split, d, p in todo:
            data, rejects = [], 0
            for j, t in enumerate(p["prompts"]):
                got, rej = gen(pipe, t, p["per_prompt"], SEED_BASE[split] + j * 100)
                data += got
                rejects += rej
            save(data, d)
            n = validate(d, p["per_split"])
            print(f"  {name}/{split}: {n} images, {rejects} blank rejected, from {p['prompts']}",
                  flush=True)

    for name, p in plan.items():
        counts = {}
        for split in ("train", "test"):
            counts[split] = validate(os.path.join(a.root, name, split), p["per_split"])
        manifest["pools"][name] = {
            "prompts": p["prompts"], "per_prompt": p["per_prompt"],
            "counts": counts,
            "scored_as": "forget" if name in scheme["forget_side"] else "retain",
            "consumed_by_trainer": name in ("hrm", "irt", "tgt"),
        }
    mp = os.path.join(a.root, "pools.json")
    os.makedirs(a.root, exist_ok=True)
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[r2pools] all pools validated; manifest -> {mp}", flush=True)
    print(f"[r2pools] CLAIM SCOPE: {scheme['claim']}", flush=True)
