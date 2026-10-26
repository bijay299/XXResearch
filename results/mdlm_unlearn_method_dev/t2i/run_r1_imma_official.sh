#!/bin/bash
# R1: dataset preparation + a minimal runtime check for the OFFICIAL IMMA pipeline.
# upstream/IMMA @ 2f55041141705031d503134dec22f674f09e21bf
#
# ============================== STATUS ==============================
# The project is FROZEN. GPU work is gated by scripts/gpu_reserve.sh (ALLOW_FROZEN_RUN=1 plus an
# explicit --allow-frozen, lead only). R1_MODE=cpu runs the blank-image audit over the dataset
# already on disk and needs no GPU; it has been executed. See docs/REPAIR_BASELINES_v2.md.
# ====================================================================
#
# Concept = Van Gogh, not church. Reason, recorded: erasing.baulab.info hosts ESD checkpoints for
# ART concepts ONLY (VanGogh, KellyMcKernan, KilianEng, ThomasKinkade, TylerEdlin, AjinDemiHuman);
# /object/ returns 404. The README's own example is VanGogh. van_gogh IS one of our four evaluated
# concepts, so the resulting checkpoint is still evaluable under our common protocol.
#
# ESD CHECKPOINT PROVENANCE (repair C). Obtaining it was correct AND documented: the URL is
# supplied verbatim by the pinned README at upstream/IMMA/README.md:44
#   wget https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt
# and README:43 itself labels it "Only for illustration purposes". That does NOT establish the
# downloaded bytes' training provenance. Measured facts (CPU, offline):
#   sha256 8881d70b1bd376c5532e933f60c5a78fc76eb63e153edfbfaaf6d649106b167e, 3,438,307,109 bytes
#   bare state dict, 686 fp32 tensors, 859,520,964 params, key set EXACTLY equal to the SD-1.4 UNet
# The earlier inference that "all 686 tensors differ, therefore it is not a cross-attention-only
# erasure" was WITHDRAWN and remains withdrawn. What the measurements now establish is narrower and
# stronger: of the benign explanations offered for that observation, the fp16-variant/main
# distinction, dtype round-tripping and diffusers-format conversion are all RULED OUT (the fp16
# variant is bit-exactly main.half(); the rounding floor is ~2.07e-04 while non-attn2 deltas median
# 4.09e-03; key sets are identical with zero LDM-style markers). SD-1.5 as the base is also ruled
# out. EMA-vs-non-EMA remains an untested HYPOTHESIS. Upstream consumes the file as a COMPLETE
# UNet regardless of the "delta" in its argument name (defend_text_to_image_lora.py:452-454
# `unet.load_state_dict(torch.load(args.delta_ckpt))`, a strict full load). Full detail, including
# the two surviving hypotheses and the decisive experiment, in docs/REPAIR_BASELINES_v2.md.
#
# DATA QUALITY (repair B-3) -- CONFIRMED DEFECT IN THE PREVIOUS VERSION OF THIS FILE.
# Upstream eval/text2image.py builds its pipeline WITHOUT safety_checker=None, so SD-1.4's checker
# blanks flagged samples to pure black. 3 of the 100 generated images are blank: 7.png, 29.png,
# 41.png. The previous version of this script contained a filter that was supposed to exclude them
# -- but the whole block was nested inside `if [ ! -f metadata.csv ]`, and metadata.csv already
# existed, SO THE FILTER NEVER RAN. The dataset on disk still carries all 100 rows including the
# three blanks, and run_r1b_imma_train.sh's header claimed otherwise. Both are corrected.
# The repair deliberately does NOT rewrite metadata.csv: it is the input the historical
# REF_imma_vangogh artifact was produced from, and overwriting it would make that artifact
# unreproducible. Instead the audit ALWAYS runs and writes, beside it:
#   blank_audit.json       every image, its max pixel, and the blank list
#   metadata_noblank.csv   the same metadata with blank rows removed, for opt-in use
# For the README-faithful 20-sample run this is moot: the resolved subset excludes all three
# blanks (see run_r1b_imma_train.sh stage 2).
#
# Env: venv_imma (torch 2.4.1+cu124, diffusers 0.24.0, huggingface_hub 0.25.2, accelerate 1.1.1).
# diffusers MUST be <=0.25: LoRAAttnProcessor lost its `hidden_size` kwarg afterwards, which
# get_lora_procs (defend_text_to_image_lora.py L509-513) requires.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

R=upstream_runs/imma
L=logs/r1_imma.log
U=/data/bijaypandey/t2i_unlearn/upstream/IMMA
UPSTREAM_PIN=2f55041141705031d503134dec22f674f09e21bf
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python
MODE="${R1_MODE:-cpu}"            # cpu = audit only | prep = generate data if absent + smoke
case "$MODE" in cpu|prep) ;; *) echo "R1_MODE must be cpu|prep" >&2; exit 2 ;; esac
MIN_IMAGES="${IMMA_MIN_IMAGES:-90}"       # abort prep if fewer usable (non-blank) images than this
NUM_IMAGES="${IMMA_NUM_IMAGES:-100}"      # README L40

mkdir -p logs "$R"
say() { printf '%s\n' "$*" | tee -a "$L"; }
die() { printf 'R1 FAILED: %s\n' "$*" | tee -a "$L" >&2; exit 1; }

say "=== R1 $(date -Is) mode=$MODE"
GOT_PIN=$(git -C "$U" rev-parse HEAD 2>/dev/null || echo NO-GIT)
[ "$GOT_PIN" = "$UPSTREAM_PIN" ] || die "upstream/IMMA is at $GOT_PIN, expected $UPSTREAM_PIN"
say "    upstream pin OK: $GOT_PIN"
[ -x "$V" ]  || die "missing venv_imma"
[ -x "$VT" ] || die "missing venv_t2i"
ESD="$R/diffusers-VanGogh-ESDx1-UNET.pt"
[ -f "$ESD" ] || die "missing the ESD checkpoint $ESD (README.md:44 supplies the URL)"

export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_DATASETS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE"

# ================= STAGE 1: record the ESD checkpoint's identity ===============================
# No sha256 was captured at download time, so this pins the bytes we HOLD. It cannot prove they
# match what the URL served; that limitation is recorded rather than glossed.
say "--- stage 1: ESD checkpoint identity"
$VT - "$ESD" "$R/esd_identity.json" <<'PY' 2>&1 | tee -a "$L" || die "stage 1 (ESD identity)"
import collections, hashlib, json, os, sys, torch
p, out = sys.argv[1], sys.argv[2]
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
sd = torch.load(p, map_location="cpu")
wrapped = [k for k in ("state_dict", "ema", "model_ema", "model", "module") if k in sd]
inner = sd.get("unet", sd)
dt = collections.Counter(str(v.dtype) for v in inner.values())
print(f"    sha256={h.hexdigest()}")
print(f"    size={os.path.getsize(p)} tensors={len(inner)} dtypes={dict(dt)} "
      f"params={sum(v.numel() for v in inner.values())}")
print(f"    wrapper keys present: {wrapped or 'NONE (bare state dict)'}")
json.dump({"path": p, "sha256": h.hexdigest(), "bytes": os.path.getsize(p),
           "tensors": len(inner), "dtypes": dict(dt),
           "params": sum(v.numel() for v in inner.values()),
           "wrapper_keys": wrapped,
           "documented_source": "upstream/IMMA/README.md:44 "
                                "https://erasing.baulab.info/weights/esd_models/art/"
                                "diffusers-VanGogh-ESDx1-UNET.pt",
           "upstream_self_description": "README.md:43 'Only for illustration purposes.'",
           "consumed_as": "a COMPLETE UNet state dict, strict load "
                          "(defend_text_to_image_lora.py:452-454), despite the 'delta' naming",
           "provenance_status": ("UNRESOLVED HYPOTHESIS. Ruled out by measurement: "
                                 "fp16-variant-vs-main, dtype round-trip, format conversion, "
                                 "SD-1.5 base. Surviving hypotheses: (H-A) trained with a scope "
                                 "wider than cross-attention; (H-B) cross-attention-only but from "
                                 "a different SD-1.4 instantiation, EMA vs non-EMA being the "
                                 "leading candidate -- UNTESTED, the non-EMA weights are not "
                                 "cached and the host is offline."),
           "hash_caveat": "no sha256 was captured at download time; this pins the bytes we hold, "
                          "not what the URL served"},
          open(out, "w"), indent=1)
print(f"    -> {out}")
PY

# ================= STAGE 2: generate the training data, if absent (GPU) ========================
if [ ! -d "$R/data/vangogh" ] || [ -z "$(ls -A "$R/data/vangogh"/*.png 2>/dev/null)" ]; then
  if [ "$MODE" = "cpu" ]; then
    die "no training images under $R/data/vangogh and generating them needs a GPU; run with
  R1_MODE=prep (lead only, with the reservation override while the freeze is active)."
  fi
  say "--- stage 2: generating $NUM_IMAGES training images (GPU)"
  [ -f scripts/gpu_reserve.sh ] || die "scripts/gpu_reserve.sh is missing"
  # The reservation API is CALLED, never reimplemented. The retired campaign_free.sh selection
  # (`G=$(bash scripts/campaign_free.sh|head -1)`) was check-then-select with no lock held
  # (register D-001) and is gone. CONTRACT, CPU-verified: --allow-frozen must be the THIRD
  # argument to gpu_reserve_hold; second binds it to wait_seconds and the override is lost.
  FROZEN_ARG=()
  [ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
  # shellcheck source=scripts/gpu_reserve.sh
  source scripts/gpu_reserve.sh
  gpu_reserve_hold "r1_imma_prep" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null \
    || die "no GPU reservation granted (see gpu_reserve stderr above)"
  [ -n "${GPU_RESERVED:-}" ] || die "gpu_reserve_hold returned 0 but set no GPU_RESERVED"
  export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  say "    reserved GPU $GPU_RESERVED"
  mkdir -p "$R/data/vangogh"
  rc=0
  ( cd "$U" && $V eval/text2image.py --prompt 'An artwork by Van Gogh' \
      --num_images "$NUM_IMAGES" \
      --output_dir "/data/bijaypandey/t2i_unlearn/$R/data/vangogh" ) >>"$L" 2>&1 || rc=$?
  say "    text2image rc=$rc"
  [ "$rc" -eq 0 ] || die "data generation exited $rc"
else
  say "--- stage 2: training images already present, not regenerating"
fi
D="$R/data/vangogh"
N_PNG=$(ls -1 "$D"/*.png 2>/dev/null | wc -l)
[ "$N_PNG" -gt 0 ] || die "no png under $D"

# ================= STAGE 3: the blank-image audit -- ALWAYS RUNS ===============================
# This is the repair. The audit is unconditional and idempotent; it never rewrites metadata.csv.
say "--- stage 3: blank-image audit ($N_PNG images on disk)"
$VT - "$D" "$MIN_IMAGES" <<'PY' 2>&1 | tee -a "$L" || die "stage 3: data-quality failure"
import csv, glob, json, os, sys
import numpy as np
from PIL import Image
d, min_images = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(d, "*.png")), key=lambda p: os.path.basename(p))
rows, blanks = [], []
for f in files:
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
    b = os.path.basename(f)
    rec = {"file_name": b, "max": float(a.max()), "std": float(a.std()),
           "blank": bool(a.max() == 0)}
    rows.append(rec)
    if rec["blank"]: blanks.append(b)
near = sorted((r for r in rows if not r["blank"]), key=lambda r: r["std"])[:3]
usable = len(files) - len(blanks)
print(f"    on disk={len(files)} blank(max==0)={len(blanks)} {blanks} usable={usable}")
print(f"    lowest-std non-blank: " + ", ".join(f"{r['file_name']}(std={r['std']:.1f})" for r in near)
      + "  -- no borderline cases if these are far from 0")
# metadata.csv is the HISTORICAL input; it is read and reported, never rewritten.
mp = os.path.join(d, "metadata.csv")
meta = None
if os.path.isfile(mp):
    with open(mp, newline="") as fh:
        meta = list(csv.DictReader(fh))
    names = [r["file_name"] for r in meta]
    in_meta = sorted(set(blanks) & set(names))
    print(f"    metadata.csv: {len(meta)} rows, {len(set(names))} distinct, "
          f"prompts={sorted(set(r['prompt'] for r in meta))}")
    print(f"    blank images PRESENT in metadata.csv: {in_meta or 'NONE'}")
    if in_meta:
        print(f"    NOT rewriting metadata.csv: it is the input REF_imma_vangogh was produced "
              f"from. Writing metadata_noblank.csv alongside it instead.")
        out = os.path.join(d, "metadata_noblank.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file_name", "prompt"]); w.writeheader()
            for r in meta:
                if r["file_name"] not in set(blanks):
                    w.writerow({"file_name": r["file_name"], "prompt": r["prompt"]})
        print(f"    -> {out} ({len(meta) - len(in_meta)} rows)")
else:
    # First-time prep: there is no historical input to protect, so write a clean metadata.csv.
    with open(mp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file_name", "prompt"])
        for r in rows:
            if not r["blank"]: w.writerow([r["file_name"], "An artwork by Van Gogh"])
    print(f"    wrote a fresh metadata.csv with {usable} non-blank rows")
json.dump({"dir": d, "images": len(files), "blank": blanks, "usable": usable,
           "per_image": rows,
           "root_cause": "upstream eval/text2image.py builds DiffusionPipeline without "
                         "safety_checker=None, so flagged samples are blanked to pure black",
           "trainer_filters_blanks": False,
           "metadata_rows": None if meta is None else len(meta),
           "metadata_policy": "metadata.csv is the historical input and is never rewritten; "
                              "metadata_noblank.csv is the opt-in filtered copy"},
          open(os.path.join(d, "blank_audit.json"), "w"), indent=1)
print(f"    -> {os.path.join(d, 'blank_audit.json')}")
if usable < min_images:
    print(f"    DATA QUALITY FAILURE: only {usable} usable images, need >= {min_images}")
    sys.exit(1)
PY

if [ "$MODE" = "cpu" ]; then
  say "R1 CPU AUDIT COMPLETE $(date -Is) -- no GPU was requested."
  exit 0
fi

# ================= STAGE 4: minimal runtime check (2 steps) ====================================
# Before spending the real budget: confirm the trainer runs end to end and emits the expected
# 160-tensor artifact. File existence and a successful import are not sufficient.
say "--- stage 4: minimal runtime check (2 steps)"
PIN_REVISION=$(sed -n 's/^revision=//p' MODEL_PIN.txt | head -1)
[ -n "$PIN_REVISION" ] || die "could not read revision= from MODEL_PIN.txt"
# RESERVATION IS UNCONDITIONAL (fixed 2026-10-26, audit defect D-049).
#
# This block used to be wrapped in `if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]`, which meant that on any
# re-run in a shell where CUDA_VISIBLE_DEVICES was already exported -- including a shell left over
# from an earlier stage of this very script -- the whole reservation was SKIPPED and the smoke job
# ran UNRESERVED on whatever device the inherited value named. A reservation that an inherited
# environment variable can switch off is not a reservation.
#
# An inherited CUDA_VISIBLE_DEVICES is treated as an ERROR rather than as a shortcut: it means
# someone pre-selected a device outside the reservation system, which is exactly the condition
# D-001 arose from (three runners each independently picking GPU 0). Clear it and re-run, or set
# R1_ACK_INHERITED_CVD=1 to acknowledge deliberately -- and note that even then we reserve, and the
# reservation's device wins over the inherited value.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${R1_ACK_INHERITED_CVD:-0}" != "1" ]; then
    die "CUDA_VISIBLE_DEVICES is already set to '${CUDA_VISIBLE_DEVICES}'. Refusing: a pre-selected
    device bypasses the reservation system. Unset it and re-run, or set R1_ACK_INHERITED_CVD=1 to
    proceed (the reservation's device will still override it)."
  fi
  say "    WARNING: inherited CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' acknowledged; it will"
  say "             be OVERRIDDEN by the reservation below, not honoured."
fi
FROZEN_ARG=()
[ "${ALLOW_FROZEN_RUN:-0}" = "1" ] && FROZEN_ARG=(--allow-frozen)
# shellcheck source=scripts/gpu_reserve.sh
source scripts/gpu_reserve.sh
gpu_reserve_hold "r1_imma_smoke" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null \
  || die "no GPU reservation granted for the smoke check (rc=${GPU_RESERVE_RC:-?})"
export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "    reserved GPU $GPU_RESERVED (uuid ${GPU_RESERVED_UUID:-?})"
rm -rf "$R/smoke"
rc=0
( cd "$U" && timeout 1800 $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --revision="$PIN_REVISION" \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$D" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size=1 --max_train_steps=2 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed=42 --report_to=tensorboard --inner_loop_steps=1 --outer_loop_steps=1 \
    --max_train_samples 20 \
    --delta_ckpt="/data/bijaypandey/t2i_unlearn/$ESD" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/smoke" ) >>"$L" 2>&1 || rc=$?
say "    smoke rc=$rc"
[ "$rc" -eq 0 ] || die "the smoke run exited $rc"
CK="$R/smoke/imma_unet_xatten_layer.pt"
[ -s "$CK" ] || die "the smoke run produced no checkpoint"
$VT - "$CK" <<'PY' 2>&1 | tee -a "$L" || die "stage 4: unexpected smoke artifact"
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
n = sum(v.numel() for v in sd.values())
a1 = sum(1 for k in sd if ".attn1." in k); a2 = sum(1 for k in sd if ".attn2." in k)
print(f"    smoke ckpt: {len(sd)} tensors {n/1e6:.3f}M params attn1={a1} attn2={a2}")
# The defence scope is a SUBSTRING match over to_q/to_k/to_v/to_out across ALL named_parameters
# (defend_text_to_image_lora.py:563-567), so it covers SELF-attention too: 160 tensors /
# 93.536640M. The "xatten" in the artifact filename is a misnomer.
assert len(sd) == 160 and a1 == 80 and a2 == 80, "unexpected defence scope"
assert n == 93_536_640, f"unexpected param count {n}"
assert all(torch.isfinite(v).all() for v in sd.values()), "non-finite tensors in the checkpoint"
print("    scope verified: 160 tensors / 93.536640M = 80 attn1 (SELF) + 80 attn2 (CROSS)")
PY
say "R1 PREP DONE $(date -Is) -- next: run_r1b_imma_train.sh (R1_MODE=train)"
