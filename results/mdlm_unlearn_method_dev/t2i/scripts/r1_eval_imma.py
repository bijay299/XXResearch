"""Compose the OFFICIAL IMMA checkpoint into an evaluable UNet, STRICTLY.

upstream saves ONLY its defence scope S (`imma_unet_xatten_layer.pt`), so the evaluable model is:
base SD-1.4 -> load the ESD delta -> overlay IMMA's S tensors. That is exactly what upstream's own
code does, in two places:
  eval/text2image.py L65-69            (--defense_ckpt)
  train/defend_text_to_image_lora.py   L458-462 (--imma_ckpt)
both: model_dict = unet.state_dict(); model_dict.update(torch.load(ckpt)); load_state_dict(...).

AUDIT 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md A-5). The previous version of this script
matched keys fuzzily (`k + ".weight"`) and force-fitted tensors with `.reshape(target.shape)`.
That path cannot be trusted: a reshape that succeeds on element count alone can place a tensor in
the wrong module and report success. It is therefore GONE. Every IMMA tensor must now match a UNet
key EXACTLY and have an EXACTLY equal shape, or this script exits non-zero.

(For the record, the historical REF_imma_vangogh composition was re-audited offline and was clean:
160 exact matches, 0 fuzzy, 0 reshapes, 0 skips. The strictness below is to keep it that way.
CAVEAT: logs/r1c_eval.log:13 does not match this file's current print strings, so the on-disk
adapters/REF_imma_vangogh/unet.pt was produced by the PRE-AUDIT fuzzy version, not by this one.)

REPAIR v2 2026-10-26 (docs/REPAIR_BASELINES_v2.md B-4):

INITIAL CHECKPOINT IDENTITY is now recorded explicitly, because two different initialisations are
in play and conflating them produces exactly the kind of false evidence this repair exists to stop:

  * TRAINING (upstream/IMMA/train/defend_text_to_image_lora.py:439-454) loads the base with
    `revision=args.revision` and NO variant -- the script has no `variant` argument at all -- so it
    reads the fp32 `main` weights. Our historical run passed no --revision either, so it floated on
    `main` and matched the pinned revision only because the local cache holds exactly one snapshot.
  * COMPOSITION (below) references the base with variant="fp16".

For the COMPOSED UNET this difference is immaterial, and the script proves it rather than asserting
it: the ESD checkpoint is a COMPLETE 686-tensor UNet that overwrites every key, and IMMA's 160
tensors overwrite a subset of those, so every output tensor is bit-identical to one of the two
inputs (verified at the `unexplained` check). The base contributes NOTHING to the output.

But it is NOT immaterial to the REPORTED DELTAS. `rel_delta_vs_base` is measured against the base,
and main-vs-fp16-variant alone differs in ALL 686 tensors at a relative Frobenius of ~2.07e-04
(the fp16 variant is bit-exactly main.half()). Any per-class delta at or near that floor is a
dtype artifact, not a model difference. So the deltas are now reported against BOTH base forms
with the floor stated alongside, and the floor is written into compose.json. This is the same trap
that made "all 686 tensors differ from our base" look like evidence about erasure scope when it
is not: see docs/REPAIR_BASELINES_v2.md section C.

Writes:
  <out>/unet.pt      fp16 full UNet state dict, consumable by gen_eval.py / attack_relearn.py
  <out>/compose.json provenance: counts, per-class delta norms AND the fp16 floor they must beat,
                     initial-checkpoint identity for both training and composition, sha256 of
                     every input
"""
import argparse, hashlib, json, os, sys
import torch

P = "/data/bijaypandey/t2i_unlearn"


def die(msg):
    print(f"[r1eval] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def cls_of(k):
    return "attn2" if ".attn2." in k else ("attn1" if ".attn1." in k else "other")


ap = argparse.ArgumentParser()
ap.add_argument("--esd", required=True, help="ESD-erased full UNet state dict (.pt)")
ap.add_argument("--imma", default=None,
                help="IMMA defence scope S (.pt). Omit together with --esd-only for the control.")
ap.add_argument("--out", required=True)
ap.add_argument("--esd-only", action="store_true",
                help="Compose base+ESD ONLY, no IMMA overlay. This is the control arm: it isolates "
                     "the direction of the ESD delta from anything IMMA did.")
a = ap.parse_args()

if a.esd_only and a.imma:
    die("--esd-only and --imma are mutually exclusive")
if not a.esd_only and not a.imma:
    die("either --imma or --esd-only is required")
for pth in [a.esd] + ([a.imma] if a.imma else []):
    if not os.path.isfile(pth):
        die(f"missing input: {pth}")

from diffusers import StableDiffusionPipeline

pin = dict(l.split("=", 1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
    variant="fp16", safety_checker=None, requires_safety_checker=False)
unet = pipe.unet
base_sd = {k: v.detach().clone() for k, v in unet.state_dict().items()}

# ---- 0. the OTHER base form, so every reported delta can be judged against the dtype floor.
#         This is reference material only; it never enters the composition.
from diffusers import UNet2DConditionModel

base_main = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", revision=pin["revision"],
    torch_dtype=torch.float32).state_dict()
_fd = sum((base_sd[k] - base_main[k]).pow(2).sum() for k in base_sd).sqrt()
_fw = sum(base_main[k].pow(2).sum() for k in base_sd).sqrt()
VARIANT_FLOOR = (_fd / _fw).item()
_floor_cls = {"attn1": [], "attn2": [], "other": []}
print(f"[r1eval] base forms: fp16-variant vs main differ in "
      f"{sum(1 for k in base_sd if not torch.equal(base_sd[k], base_main[k]))}/{len(base_sd)} "
      f"tensors at rel={VARIANT_FLOOR:.3e} -- deltas at this magnitude are dtype, not model",
      flush=True)

# ---- 1. the ESD delta must be a COMPLETE UNet state dict, no more and no less
esd = torch.load(a.esd, map_location="cpu")
if not isinstance(esd, dict):
    die(f"--esd did not deserialise to a dict, got {type(esd)}")
esd = esd.get("unet", esd)
missing = sorted(set(base_sd) - set(esd))
extra = sorted(set(esd) - set(base_sd))
if missing or extra:
    die(f"ESD key set != UNet key set. missing={len(missing)} extra={len(extra)}; "
        f"first missing={missing[:3]} first extra={extra[:3]}")
for k, v in esd.items():
    if tuple(v.shape) != tuple(base_sd[k].shape):
        die(f"ESD shape mismatch on {k}: ckpt {tuple(v.shape)} vs unet {tuple(base_sd[k].shape)}")
unet.load_state_dict({k: v.float() for k, v in esd.items()})
print(f"[r1eval] ESD loaded: {len(esd)} tensors, "
      f"{sum(v.numel() for v in esd.values())/1e6:.3f}M params", flush=True)

# ---- 2. overlay S, STRICTLY: exact key, exact shape, no reshape, no skip
sd = unet.state_dict()
n_imma = 0
imma = {}
if a.imma:
    imma = torch.load(a.imma, map_location="cpu")
    if not isinstance(imma, dict):
        die(f"--imma did not deserialise to a dict, got {type(imma)}")
    imma = imma.get("unet", imma)
    absent = [k for k in imma if k not in sd]
    if absent:
        die(f"{len(absent)} IMMA keys are not UNet parameter names (no fuzzy fallback is "
            f"permitted): {absent[:5]}")
    bad = [(k, tuple(v.shape), tuple(sd[k].shape)) for k, v in imma.items()
           if tuple(v.shape) != tuple(sd[k].shape)]
    if bad:
        die(f"{len(bad)} IMMA tensors have the wrong shape for their module (no reshape is "
            f"permitted): {bad[:5]}")
    for k, v in imma.items():
        sd[k] = v.float()
        n_imma += 1
    if n_imma != len(imma):
        die(f"overlaid {n_imma} of {len(imma)} IMMA tensors")
    unet.load_state_dict(sd)
    print(f"[r1eval] IMMA overlaid: {n_imma}/{len(imma)} tensors, exact keys, exact shapes, "
          f"0 fuzzy, 0 reshaped, 0 skipped", flush=True)

# ---- 3. the result must be finite and fully explained by its two inputs
final = unet.state_dict()
nonfinite = [k for k, v in final.items() if not torch.isfinite(v).all()]
if nonfinite:
    die(f"{len(nonfinite)} composed tensors contain NaN/Inf: {nonfinite[:5]}")

stats = {"attn1": [], "attn2": [], "other": []}
stats_main = {"attn1": [], "attn2": [], "other": []}
unexplained = []
for k, v in final.items():
    src = imma[k] if k in imma else esd[k]
    if not torch.allclose(v, src.float(), rtol=0, atol=0):
        unexplained.append(k)
    c = cls_of(k)
    stats[c].append((v - base_sd[k]).norm().item() / (base_sd[k].norm().item() + 1e-12))
    stats_main[c].append((v - base_main[k]).norm().item() / (base_main[k].norm().item() + 1e-12))
    _floor_cls[c].append((base_sd[k] - base_main[k]).norm().item()
                         / (base_main[k].norm().item() + 1e-12))
if unexplained:
    die(f"{len(unexplained)} composed tensors match neither their ESD nor their IMMA source: "
        f"{unexplained[:5]}")
# Every output tensor came from ESD or IMMA, so the base contributed nothing to the model. The
# choice of base form therefore cannot change the composed UNet -- only the deltas reported below.


def summarise(v):
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "median": float(s[len(s) // 2]), "max": float(s[-1]),
            "min": float(s[0])}

os.makedirs(a.out, exist_ok=True)
torch.save({k: v.half() for k, v in final.items()}, f"{a.out}/unet.pt")

prov = {
    "script": "scripts/r1_eval_imma.py",
    "mode": "esd_only" if a.esd_only else "esd_plus_imma",
    "initial_checkpoint_identity": {
        "composition_reference": {
            "model": "CompVis/stable-diffusion-v1-4", "revision": pin["revision"],
            "variant": "fp16",
            "role": "REFERENCE ONLY. Every composed tensor comes from the ESD checkpoint or from "
                    "IMMA's 160 tensors (verified above), so the base contributes nothing to the "
                    "output. It is used solely as the denominator of the deltas below."},
        "training_initialisation": {
            "model": "CompVis/stable-diffusion-v1-4",
            "revision": "not pinned at the command level by the historical run (--revision "
                        "absent, default None -> `main`); it matched the pinned revision only "
                        "because the local cache holds exactly one snapshot",
            "variant": "none -- defend_text_to_image_lora.py has NO variant argument, so it "
                       "loaded the fp32 `main` weights",
            "evidence": "upstream/IMMA/train/defend_text_to_image_lora.py:439-454",
            "note": "the UNet is then fully overwritten by the ESD checkpoint at :452-454, so "
                    "only text_encoder/vae/scheduler actually come from the base"},
        "divergence_is_immaterial_to_the_model": True,
        "divergence_is_material_to_the_reported_deltas": True},
    "fp16_variant_floor": {
        "global": VARIANT_FLOOR,
        "per_class": {c: summarise(v) for c, v in _floor_cls.items()},
        "meaning": "main-vs-fp16-variant alone differs in ALL 686 tensors at this magnitude (the "
                   "fp16 variant is bit-exactly main.half()). Any delta at or near this floor is "
                   "a dtype artifact and is NOT evidence of a model difference."},
    "base_model": "CompVis/stable-diffusion-v1-4",
    "base_revision": pin["revision"],
    "base_variant": "fp16",
    "esd_path": a.esd, "esd_sha256": sha256(a.esd), "esd_tensors": len(esd),
    "imma_path": a.imma, "imma_sha256": sha256(a.imma) if a.imma else None,
    "imma_tensors": len(imma), "imma_overlaid": n_imma,
    "imma_fuzzy_matched": 0, "imma_reshaped": 0, "imma_skipped": 0,
    "composed_tensors": len(final),
    "composed_params_M": sum(v.numel() for v in final.values()) / 1e6,
    "rel_delta_vs_base_fp16variant": {c: summarise(v) for c, v in stats.items()},
    "rel_delta_vs_base_main": {c: summarise(v) for c, v in stats_main.items()},
    # kept under its historical name so older readers do not silently get nothing
    "rel_delta_vs_base": {c: summarise(v) for c, v in stats.items()},
    "upstream_equivalence": ("identical to eval/text2image.py L65-69 and "
                             "train/defend_text_to_image_lora.py L458-462"),
    "interpretation_warning": (
        "An IMMA-immunised checkpoint is NOT meant to be evaluated standalone. The IMMA paper "
        "evaluates only the POST-ADAPTATION model (README L146-151). Standalone suppression or "
        "utility numbers from mode=esd_plus_imma are not IMMA results. See "
        "docs/AGENT2_BASELINES_FINDINGS.md A-6."),
}
with open(f"{a.out}/compose.json", "w") as f:
    json.dump(prov, f, indent=1)
print(f"[r1eval] wrote {a.out}/unet.pt ({prov['composed_params_M']:.3f}M params) "
      f"+ compose.json [mode={prov['mode']}]", flush=True)
