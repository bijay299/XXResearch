#!/bin/bash
# R1: the OFFICIAL IMMA pipeline, author-documented setting, CONCEPT-MATCHED. Prep + smoke only.
#
# Concept = Van Gogh, not church. Reason, recorded: erasing.baulab.info hosts ESD checkpoints for
# ART concepts ONLY (VanGogh, KellyMcKernan, KilianEng, ThomasKinkade, TylerEdlin, AjinDemiHuman);
# /object/ returns 404. The README's own example is VanGogh. Substituting a church ESD checkpoint
# does not exist, and substituting the VanGogh checkpoint for a church run would be the
# concept-mismatch we are told not to do. van_gogh IS one of our four evaluated concepts, so the
# resulting checkpoint is still evaluable under our common protocol.
#
# Author protocol, verbatim from upstream/IMMA/README.md L38-66:
#   python eval/text2image.py --prompt 'An artwork by Van Gogh' --num_images 100 --output_dir ...
#   accelerate launch train/defend_text_to_image_lora.py --mixed_precision="no" ... \
#     --train_batch_size=1 --num_train_epochs=50 --learning_rate_lora=1e-04 --learning_rate=1e-05 \
#     --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 --report_to="tensorboard" \
#     --max_train_samples 20 --inner_loop_steps=1 --outer_loop_steps=1
# ALL of those are now passed explicitly below. The pre-audit version silently omitted
# --max_train_samples, which is NOT a no-op: see docs/AGENT2_BASELINES_FINDINGS.md A-1.
#
# ESD checkpoint: downloaded from the README's URL. Verified on CPU: 686 tensors / 859.520964M
# params / key set exactly equal to the SD-1.4 UNet. CAVEAT (findings A-2): despite the "ESDx1"
# name, ALL 686 tensors differ from base SD-1.4 (resnets up to 0.20 relative), so do not describe
# it as a cross-attention-only erasure.
#
# DATA-QUALITY REPAIR (findings A-3): upstream eval/text2image.py builds its pipeline WITHOUT
# safety_checker=None, so SD-1.4's checker blanks flagged samples to pure black. The historical
# run trained on 3 such images out of 100. metadata.csv is now written only over images that are
# not blank, and the run aborts if too few survive.
#
# Env: venv_imma (torch 2.4.1+cu124, diffusers 0.24.0, huggingface_hub 0.25.2, accelerate 1.1.1).
# diffusers MUST be <=0.25: LoRAAttnProcessor lost its `hidden_size` kwarg afterwards, which
# get_lora_procs (defend_text_to_image_lora.py L509-513) requires.
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

if [ -f FREEZE_ACTIVE ] && [ "${R1_FREEZE_OVERRIDE:-0}" != "1" ]; then
  echo "FREEZE_ACTIVE is present; refusing to launch GPU work. Set R1_FREEZE_OVERRIDE=1 to run." >&2
  exit 1
fi

R=upstream_runs/imma
L=logs/r1_imma.log
MAX_SAMPLES="${IMMA_MAX_SAMPLES:-20}"      # README value. 0 or "all" = use every image.
MIN_IMAGES="${IMMA_MIN_IMAGES:-90}"        # abort if fewer usable (non-blank) images than this
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python
[ -x "$V" ]  || { echo "missing venv_imma" >&2; exit 1; }
[ -x "$VT" ] || { echo "missing venv_t2i" >&2; exit 1; }
[ -f "$R/diffusers-VanGogh-ESDx1-UNET.pt" ] || { echo "missing ESD ckpt $R/..." >&2; exit 1; }

while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R1 on GPU $G $(date -Is) max_train_samples=$MAX_SAMPLES" >>"$L"

# ---- 1. author-specified training data (100 images), THEN the metadata.csv their loader needs
if [ ! -f "$R/data/vangogh/metadata.csv" ]; then
  mkdir -p "$R/data/vangogh"
  rc=0
  ( cd upstream/IMMA && $V eval/text2image.py --prompt 'An artwork by Van Gogh' \
      --num_images 100 --output_dir "/data/bijaypandey/t2i_unlearn/$R/data/vangogh" ) >>"$L" 2>&1 || rc=$?
  echo "  text2image rc=$rc" >>"$L"
  [ "$rc" -eq 0 ] || { echo "DATA GENERATION FAILED rc=$rc" >>"$L"; exit 1; }

  # ---- 1b. reject NSFW-blanked (all-black) images before they reach the trainer
  $VT - "$R/data/vangogh" "$MIN_IMAGES" <<'PY' >>"$L" 2>&1
import csv, glob, os, sys
import numpy as np
from PIL import Image
d, min_images = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(d, "*.png")))
keep, drop = [], []
for f in files:
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
    (drop if (a.max() == 0 or a.std() < 1.0) else keep).append(os.path.basename(f))
print(f"  images on disk={len(files)} usable={len(keep)} rejected_blank={len(drop)} {drop}")
if len(keep) < min_images:
    print(f"  DATA QUALITY FAILURE: only {len(keep)} usable images, need >= {min_images}")
    sys.exit(1)
with open(os.path.join(d, "metadata.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file_name", "prompt"])
    for n in keep: w.writerow([n, "An artwork by Van Gogh"])
print(f"  wrote metadata.csv with {len(keep)} rows (blank images excluded from training)")
PY
  [ -f "$R/data/vangogh/metadata.csv" ] || { echo "METADATA WRITE FAILED" >>"$L"; exit 1; }
fi
echo "  dataset rows=$(( $(wc -l < "$R/data/vangogh/metadata.csv") - 1 ))" >>"$L"

# ---- 2. MINIMAL RUNTIME CHECK before spending real GPU time: 2 steps, verify the artifact appears
rm -rf "$R/smoke"
echo "--- smoke: 2 steps ---" >>"$L"
MS_ARG=()
case "$MAX_SAMPLES" in 0|all|ALL) ;; *) MS_ARG=(--max_train_samples "$MAX_SAMPLES") ;; esac
rc=0
( cd upstream/IMMA && timeout 1800 $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$R/data/vangogh" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size=1 --max_train_steps=2 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed=42 --report_to=tensorboard --inner_loop_steps=1 --outer_loop_steps=1 "${MS_ARG[@]}" \
    --delta_ckpt="/data/bijaypandey/t2i_unlearn/$R/diffusers-VanGogh-ESDx1-UNET.pt" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/smoke" ) >>"$L" 2>&1 || rc=$?
echo "  smoke rc=$rc" >>"$L"
[ "$rc" -eq 0 ] || { echo "SMOKE FAILED rc=$rc" >>"$L"; exit 1; }
CK="$R/smoke/imma_unet_xatten_layer.pt"
[ -s "$CK" ] || { echo "SMOKE PRODUCED NO CHECKPOINT" >>"$L"; exit 1; }
$VT - "$CK" <<'PY' >>"$L" 2>&1
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
n = sum(v.numel() for v in sd.values())
a1 = sum(1 for k in sd if ".attn1." in k); a2 = sum(1 for k in sd if ".attn2." in k)
print(f"  smoke ckpt: {len(sd)} tensors {n/1e6:.3f}M params attn1={a1} attn2={a2}")
# the defence scope is a SUBSTRING match over to_q/to_k/to_v/to_out across ALL named_parameters,
# so it covers self-attention too: 160 tensors / 93.536640M. "xatten" in the filename is a misnomer.
assert len(sd) == 160 and a1 == 80 and a2 == 80, "unexpected defence scope"
assert abs(n - 93_536_640) == 0, f"unexpected param count {n}"
assert all(torch.isfinite(v).all() for v in sd.values()), "non-finite tensors in checkpoint"
print("  smoke ckpt scope verified against the reference (160 tensors / 93.536640M)")
PY
echo "R1 PREP DONE $(date -Is)" >>"$L"
