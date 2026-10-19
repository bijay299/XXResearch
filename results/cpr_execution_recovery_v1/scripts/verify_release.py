#!/usr/bin/env python3
"""Fresh-area release verification (CPU only; no GPU, no generation). Writes artifacts/verify/*.json."""
import hashlib, json, os, sys, tarfile, urllib.request

ROOT = os.environ["CPRX_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "code"))
import torch
from safetensors.torch import load_file

VER = os.path.join(ROOT, "verify")
OUT = os.path.join(ROOT, "artifacts", "verify")
os.makedirs(OUT, exist_ok=True)
assert not os.listdir(VER), "verification area must start empty"
TAG = "cpr-direct-agreement-controls-v1"
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


# 1. download via public URLs, compare hash and size
for a in ASSETS:
    dst = os.path.join(VER, a["name"])
    urllib.request.urlretrieve(BASE + a["name"], dst)
    ok = os.path.getsize(dst) == a["bytes"] and (a["sha256"] is None or sha(dst) == a["sha256"])
    R["checks"]["asset::" + a["name"]] = {"ok": ok, "bytes": os.path.getsize(dst), "sha256": sha(dst)}
sums = {l.split()[1]: l.split()[0] for l in open(os.path.join(VER, "SHA256SUMS"))}
R["checks"]["SHA256SUMS_matches_downloads"] = {"ok": all(sha(os.path.join(VER, n)) == h for n, h in sums.items()), "n": len(sums)}

# 2. extract and restore
for t in ("adapters_gate_run2.tar", "adapters_gate_run1_superseded.tar"):
    d = os.path.join(VER, t[:-4])
    os.makedirs(d)
    with tarfile.open(os.path.join(VER, t)) as tf:
        tf.extractall(d)

# PEFT skeleton on the meta device (no weights loaded): same key set and shapes as the published adapters
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
snap = os.environ["CPRD_BASE_MODEL_PATH__FLUX"]
cfg = FluxTransformer2DModel.load_config(os.path.join(snap, "transformer"))
with torch.device("meta"):
    skel = FluxTransformer2DModel.from_config(cfg)
a_any = json.load(open(os.path.join(VER, "adapters_gate_run2", "dev.native_gem.lr_1e-3", "terminal", "adapter_meta.json")))
targets = sorted({n.split("transformer.", 1)[-1] for n in a_any["peft_config"]["target_modules"]}) if "target_modules" in a_any["peft_config"] else None
R["skeleton_target_modules"] = len(targets or [])
lc = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=targets)
pm = get_peft_model(skel, lc)
skel_shapes = {n: tuple(p.shape) for n, p in pm.named_parameters() if "lora_" in n}
R["skeleton_lora_tensors"] = len(skel_shapes)
R["skeleton_trainable_params"] = sum(int(torch.Size(s).numel()) for s in skel_shapes.values())

dirs = []
for root, ds, fs in os.walk(VER):
    if "adapter_model.safetensors" in fs:
        dirs.append(root)
for d in sorted(dirs):
    rel = os.path.relpath(d, VER)
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    R["counts"]["adapter_loads_cpu"] += 1
    meta = json.load(open(os.path.join(d, "adapter_meta.json")))
    sd32 = {k: v.to(torch.float32).cpu() for k, v in sd.items()}
    sh = state_hash(sd32)
    norm = {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): tuple(v.shape) for k, v in sd.items()}
    def tail(k):
        return k.split("transformer.", 1)[-1] if "transformer." in k else k
    sk = {tail(k).replace("base_model.model.", "", 1): s for k, s in skel_shapes.items()}
    pub = {tail(k).replace("base_model.model.", "", 1): s for k, s in norm.items()}
    shapes_ok = sk == pub
    R["checks"]["restore::" + rel] = {
        "ok": bool(sh == meta["tensor_state_sha256"] and sha(os.path.join(d, "adapter_model.safetensors")) == meta["file_sha256"]
                   and len(sd) == 152 and all(torch.isfinite(v).all() for v in sd32.values())),
        "tensor_state_sha256": sh, "matches_meta": sh == meta["tensor_state_sha256"], "n_tensors": len(sd),
        "n_params": int(sum(v.numel() for v in sd.values())), "skeleton_key_shape_match": shapes_ok}
R["counts"]["adapter_dirs"] = len(dirs)
R["counts"]["release_assets"] = len(ASSETS)
R["all_ok"] = all(v.get("ok", True) for v in R["checks"].values())
json.dump(R, open(os.path.join(OUT, "release_verification.json"), "w"), indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in R.items() if k != "checks"}, indent=1), "ALL_OK" if R["all_ok"] else "NOT_OK")
for k, v in R["checks"].items():
    print(k, v.get("ok"), v.get("skeleton_key_shape_match", ""))
