#!/bin/bash
# R1: the OFFICIAL IMMA pipeline, author-documented setting, CONCEPT-MATCHED.
#
# Concept = Van Gogh, not church. Reason, recorded: erasing.baulab.info hosts ESD checkpoints for
# ART concepts ONLY (VanGogh, KellyMcKernan, KilianEng, ThomasKinkade, TylerEdlin, AjinDemiHuman);
# /object/ returns 404. The README's own example is VanGogh. Substituting a church ESD checkpoint
# does not exist, and substituting the VanGogh checkpoint for a church run would be the
# concept-mismatch we are told not to do. van_gogh IS one of our four evaluated concepts, so the
# resulting checkpoint is still evaluable under our common protocol.
#
# Author protocol, verbatim from upstream/IMMA/README.md:
#   python eval/text2image.py --prompt 'An artwork by Van Gogh' --num_images 100 --output_dir ...
#   accelerate launch train/defend_text_to_image_lora.py --mixed_precision="no" ... \
#     --train_batch_size=1 --num_train_epochs=50 --learning_rate_lora=1e-04 --learning_rate=1e-05 \
#     --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42
#
# ESD checkpoint: downloaded from the README's URL; verified 859.52M params / 686 tensors / 80 attn2.
# Env: venv_imma (isolated copy; torch 2.4.1, torchvision 0.19.1, datasets 5.0.1, diffusers 0.31.0).
set -u
cd /data/bijaypandey/t2i_unlearn
R=upstream_runs/imma
L=logs/r1_imma.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R1 on GPU $G $(date -Is)" >>"$L"
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python

# ---- 1. author-specified training data (100 images), THEN the metadata.csv their loader needs
if [ ! -f "$R/data/vangogh/metadata.csv" ]; then
  mkdir -p "$R/data/vangogh"
  ( cd upstream/IMMA && $V eval/text2image.py --prompt 'An artwork by Van Gogh' \
      --num_images 100 --output_dir "/data/bijaypandey/t2i_unlearn/$R/data/vangogh" ) >>"$L" 2>&1
  N=$(ls "$R/data/vangogh"/*.png 2>/dev/null | wc -l)
  echo "  generated $N images" >>"$L"
  if [ "$N" -gt 0 ]; then
    { echo "file_name,prompt"; for f in "$R/data/vangogh"/*.png; do
        echo "$(basename "$f"),An artwork by Van Gogh"; done; } > "$R/data/vangogh/metadata.csv"
    echo "  wrote metadata.csv ($(wc -l < "$R/data/vangogh/metadata.csv") lines)" >>"$L"
  else
    echo "  DATA GENERATION FAILED -- see above" >>"$L"; exit 1
  fi
fi

# ---- 2. MINIMAL RUNTIME CHECK before spending real GPU time: 2 steps, verify params actually move
echo "--- smoke: 2 steps, check trainable params move ---" >>"$L"
( cd upstream/IMMA && timeout 1800 accelerate launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$R/data/vangogh" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size=1 --max_train_steps=2 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed=42 --delta_ckpt="/data/bijaypandey/t2i_unlearn/$R/diffusers-VanGogh-ESDx1-UNET.pt" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/smoke" ) >>"$L" 2>&1
echo "  smoke rc=$?" >>"$L"
ls -la "$R/smoke" >>"$L" 2>&1
echo "R1 PREP DONE $(date -Is)" >>"$L"
