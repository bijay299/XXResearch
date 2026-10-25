#!/bin/bash
# FINDING58 S3a has LoRA and TI only on beta=1.0. Filling them in on the deeper usable arms (1.3,
# 1.35) completes the adversary matrix ACROSS the frontier rather than at one point, so "every usable
# arm falls to every adversary family" is measured rather than extrapolated from the shallowest arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
lora() { K="H81LF_$1_sd$2"; mkdir ".labels/$K" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_lora.py --concept church \
    --unet "adapters/$1/unet.pt" --seed "$2" --label "$K" >>logs/gift_matrix.log 2>&1; }
( lora gift_church_b1.3 0 0; lora gift_church_b1.35 0 0; echo "MF A DONE $(date -Is)" >>logs/gift_matrix.log ) &
( lora gift_church_b1.3 1 1; lora gift_church_b1.35 1 1; echo "MF B DONE $(date -Is)" >>logs/gift_matrix.log ) &
wait
