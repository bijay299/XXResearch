#!/bin/bash
# R4d: upstream UCE at ITS OWN DEFAULT -- NO preserve concepts.
# R4c supplied our port's 10 preserve concepts to both sides (matched inputs). But upstream's
# __main__ sets `preserve_concepts = []` when the flag is absent (uce_sd_erase.py:148-149), so
# 1 erase term against 10 preserve terms is NOT how upstream is meant to be run. This measures
# upstream as intended.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4d.log
while :; do G=$(bash scripts/campaign_free.sh|head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R4d on GPU $G $(date -Is)" >>"$L"
for lam in 0.5 0.1; do
  venv_uce/bin/python scripts/r4c_uce_checkpoints_and_images.py --concept church --lamb "$lam" \
    --no-preserve >>"$L" 2>&1
done
echo "R4d DONE $(date -Is)" >>"$L"
