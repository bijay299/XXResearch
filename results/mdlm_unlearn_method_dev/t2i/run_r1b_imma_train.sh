#!/bin/bash
# R1b: the FULL official IMMA defence training, author-documented settings (README verbatim):
#   --train_batch_size=1 --num_train_epochs=50 --learning_rate_lora=1e-04 --learning_rate=1e-05
#   --lr_scheduler=constant --lr_warmup_steps=0 --seed=42 --mixed_precision=no --resolution=512
# Concept: Van Gogh (concept-MATCHED to the only ESD checkpoints baulab publishes; /object/ 404s).
# Data: 100 images from their eval/text2image.py at 'An artwork by Van Gogh'.
# Env: venv_imma, now diffusers 0.24.0 + hub 0.25.2 -- required because LoRAAttnProcessor's
#      signature changed in later diffusers (imports fine, but drops `hidden_size`).
# Runtime check already PASSED: 2 steps produced imma_unet_xatten_layer.pt, 374 MB = 93.5M fp32
# params, matching the reference's to_q/k/v/out scope exactly.
set -u
cd /data/bijaypandey/t2i_unlearn
R=upstream_runs/imma
L=logs/r1b_imma_train.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python
echo "=== R1b on GPU $G $(date -Is)" >>"$L"
( cd upstream/IMMA && $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$R/data/vangogh" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed=42 --delta_ckpt="/data/bijaypandey/t2i_unlearn/$R/diffusers-VanGogh-ESDx1-UNET.pt" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/REF_imma_vangogh" ) >>"$L" 2>&1
echo "  rc=$? $(date -Is)" >>"$L"
ls -la "$R/REF_imma_vangogh" >>"$L" 2>&1
echo "R1b DONE $(date -Is)" >>"$L"
