#!/bin/bash
# Completes the frozen three-point grid on uce_sd15_church_lam1000 (as-is target 0.2000 /
# retain8 0.9500 -- the usable SD-1.5 UCE arm the properly scaled lambda sweep recovered).
# Seed 0 claimed a label during the earlier run but produced no output; seeds 1 and 6 are done.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export BASE_MODEL=runwayml/stable-diffusion-v1-5 BASE_VARIANT=fp16
export BASE_REV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
export CUDA_VISIBLE_DEVICES=0
L=uce_sd15_church_lam1000; K=H89L_${L}_sd0_lr1e-5
mkdir -p ".labels/$K"
venv_t2i/bin/python scripts/attack_relearn.py --concept church \
  --unet "adapters/$L/unet.pt" --seed 0 --lr 1e-5 --label "$K" >>logs/uce_cm_atk.log 2>&1
echo "$K done rc=$? $(date -Is)" >>logs/uce_cm.log
