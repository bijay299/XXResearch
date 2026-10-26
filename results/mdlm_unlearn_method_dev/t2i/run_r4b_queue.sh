#!/bin/bash
# R4b queue. Waits on CAMPAIGN OWNERSHIP (scripts/campaign_free.sh), not on a memory dip.
# Runs the official default (lamb 0.5) and the two representative historical settings actually
# found in our artifacts: lamb 0.1 (13 of 22 arms) and lamb 1000 (the SD-1.5 arm that cleared K1).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/r4b_uce.log
while :; do
  G=$(bash scripts/campaign_free.sh)
  [ -n "$G" ] && break
  sleep 120
done
echo "=== R4b starting on GPU $G $(date -Is)" >>"$L"
export CUDA_VISIBLE_DEVICES=$G
for lam in 0.5 0.1 1000; do
  echo "--- lamb=$lam ---" >>"$L"
  venv_uce/bin/python scripts/r4b_uce_upstream_vs_port.py --concept church --lamb "$lam" \
    --erase-scale 1.0 --preserve-scale 1.0 >>"$L" 2>&1
done
echo "R4b DONE $(date -Is)" >>"$L"
