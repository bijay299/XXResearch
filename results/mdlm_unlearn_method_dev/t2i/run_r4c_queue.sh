#!/bin/bash
# R4c: actual UCE checkpoints + matched-prompt/matched-seed generations (user item 3).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4c.log
while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R4c on GPU $G $(date -Is)" >>"$L"
for lam in 0.5 0.1; do
  venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py --concept church --lamb "$lam" \
    --erase-scale 1.0 --preserve-scale 1.0 >>"$L" 2>&1
done
echo "R4c DONE $(date -Is)" >>"$L"
