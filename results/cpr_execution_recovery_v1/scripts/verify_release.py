#!/usr/bin/env python3
"""Fresh-area release verification (CPU only; no GPU, no generation). Writes artifacts/verify/release_verification.json."""
import glob, hashlib, json, os, sys, tarfile, urllib.request

ROOT = os.environ["CPRX_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "code"))
import torch
from PIL import Image
from safetensors.torch import load_file

VER = os.path.join(ROOT, "verify")
OUT = os.path.join(ROOT, "artifacts", "verify")
os.makedirs(OUT, exist_ok=True)
os.makedirs(VER, exist_ok=True)
assert not os.listdir(VER), "verification area must start empty"
TAG = "cpr-execution-recovery-v1"
BASE = f"https://github.com/bijay299/XXResearch/releases/download/{TAG}/"
ASSETS = json.load(open(os.path.join(ROOT, "RELEASE_ASSETS.json")))["assets"]
R = {"checks": {}, "counts": {"adapter_loads_cpu": 0, "png_decodes_cpu": 0, "model_generated_images": 0, "gpu_used": False}}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


# 1. download via public URLs, compare size and hash
for a in ASSETS:
    dst = os.path.join(VER, a["name"])
    urllib.request.urlretrieve(BASE + a["name"], dst)
    ok = os.path.getsize(dst) == a["bytes"] and (a["sha256"] is None or sha(dst) == a["sha256"])
    R["checks"]["asset::" + a["name"]] = {"ok": ok, "bytes": os.path.getsize(dst), "sha256": sha(dst)}
sums = {l.split()[1]: l.split()[0] for l in open(os.path.join(VER, "SHA256SUMS"))}
R["checks"]["SHA256SUMS_matches_downloads"] = {"ok": all(sha(os.path.join(VER, n)) == h for n, h in sums.items()), "n": len(sums)}

# 2. extract
for t in sorted(os.listdir(VER)):
    if t.endswith(".tar"):
        d = os.path.join(VER, t[:-4]); os.makedirs(d)
        with tarfile.open(os.path.join(VER, t)) as tf:
            tf.extractall(d)

# 3. PNG counts + hashes + decodes against the per-job generation records
exp = {}
for f in sorted(glob.glob(os.path.join(ROOT, "artifacts", "gen_records", "*.json"))):
    for s in json.load(open(f))["sets"]:
        for im in s["images"]:
            exp[im["path"]] = im["png_sha256"]
for tarname in ("images_heldout_primary", "images_dev_primary"):
    base = os.path.join(VER, tarname)
    found = {}
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(".png"):
                p = os.path.join(r, f)
                found[os.path.relpath(p, base)] = p
    bad, missing_in_rec = [], []
    for rel, p in found.items():
        e = exp.get(rel)
        if e is None:
            missing_in_rec.append(rel)
        elif sha(p) != e:
            bad.append(rel)
        with Image.open(p) as im:
            im.load(); assert im.size == (512, 512)
        R["counts"]["png_decodes_cpu"] += 1
    prefix = ("images/m0/ho", "images/ho") if "heldout" in tarname else ("images/m0/dev", "images/dev")
    want = {k for k in exp if k.startswith(prefix)}
    R["checks"]["pngs::" + tarname] = {"ok": not bad and not missing_in_rec and set(found) == want,
                                       "n_png": len(found), "n_expected": len(want), "n_hash_mismatch": len(bad),
                                       "n_not_in_records": len(missing_in_rec)}

# 4. adapter restoration on a meta-device PEFT skeleton + bitwise state-hash recompute
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
snap = os.environ["CPRD_BASE_MODEL_PATH__FLUX"]
cfg = FluxTransformer2DModel.load_config(os.path.join(snap, "transformer"))
dirs = sorted(r for r, _, fs in os.walk(VER) if "adapter_model.safetensors" in fs)
a_any = json.load(open(os.path.join(dirs[0], "adapter_meta.json")))
tm = a_any["peft_config"]["target_modules"]
tm = sorted({x.split("transformer.", 1)[-1] if x.startswith("transformer.") else x for x in (tm.split("|") if isinstance(tm, str) else tm)})
with torch.device("meta"):
    skel = FluxTransformer2DModel.from_config(cfg)
pm = get_peft_model(skel, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=tm))
skel_shapes = {n: tuple(p.shape) for n, p in pm.named_parameters() if "lora_" in n}
R["skeleton_lora_tensors"] = len(skel_shapes)
R["skeleton_trainable_params"] = int(sum(torch.Size(s).numel() for s in skel_shapes.values()))


def tail(k):
    k = k.split("transformer.", 1)[-1] if "transformer." in k else k
    k = k.replace("base_model.model.", "", 1).replace(".default", "")
    return k


sk = {tail(k): s for k, s in skel_shapes.items()}
for d in dirs:
    rel = os.path.relpath(d, VER)
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    R["counts"]["adapter_loads_cpu"] += 1
    meta = json.load(open(os.path.join(d, "adapter_meta.json")))
    sd32 = {k: v.to(torch.float32).cpu() for k, v in sd.items()}
    sh = state_hash(sd32)
    pub = {tail(k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")): tuple(v.shape) for k, v in sd.items()}
    shapes_ok = sk == pub
    R["checks"]["restore::" + rel] = {
        "ok": bool(sh == meta["tensor_state_sha256"] and sha(os.path.join(d, "adapter_model.safetensors")) == meta["file_sha256"]
                   and len(sd) == 152 and shapes_ok and all(torch.isfinite(v).all() for v in sd32.values())),
        "tensor_state_sha256": sh, "matches_meta": sh == meta["tensor_state_sha256"], "n_tensors": len(sd),
        "n_params": int(sum(v.numel() for v in sd.values())), "skeleton_key_shape_match": shapes_ok}
ch = json.load(open(os.path.join(ROOT, "artifacts", "analysis", "checkpoint_hashes.json")))
pref = os.path.join(VER, "adapters_native_peft", "adapters") + os.sep
rels = [d[len(pref):] for d in dirs]
R["checks"]["checkpoint_hash_table_matches"] = {"ok": all(r in ch and ch[r]["file_sha256"] == json.load(open(os.path.join(pref, r, "adapter_meta.json")))["file_sha256"] for r in rels),
                                                "n_table": len(ch), "n_published": len(rels)}
R["counts"]["adapter_dirs"] = len(dirs)
R["counts"]["release_assets"] = len(ASSETS)
R["all_ok"] = all(v.get("ok", True) for v in R["checks"].values())
json.dump(R, open(os.path.join(OUT, "release_verification.json"), "w"), indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in R.items() if k != "checks"}, indent=1), "ALL_OK" if R["all_ok"] else "NOT_OK")
for k, v in R["checks"].items():
    print(k, v.get("ok"), v.get("skeleton_key_shape_match", ""))
