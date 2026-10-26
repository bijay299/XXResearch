#!/bin/bash
# Calibration anchors for E6. Both write JSONs only -- no checkpoints.
#  (a) SD-1.5 base on FINAL  -> lets the two SD-1.5 arms be normalised
#  (b) SD-1.4 base on DEV    -> the missing cell: without it the DEV->FINAL gap cannot be
#      attributed to prompt difficulty vs defence overfitting
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
venv_t2i/bin/python scripts/gen_eval_final.py --only BASE_sd15_anchor >>logs/e6_final.log 2>&1
venv_t2i/bin/python scripts/gen_eval.py --label E6cal_BASE_sd14_on_DEV --target church \
  >>logs/e6_final.log 2>&1
echo "E6 ANCHORS DONE rc=$? $(date -Is)" >>logs/e6_final.log
