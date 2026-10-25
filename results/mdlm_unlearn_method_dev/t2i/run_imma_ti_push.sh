#!/bin/bash
# Textual Inversion is the ONLY family still below the ceiling on the IMMA arm: 0.400 at 2000 steps,
# against 1.000 for LoRA (900 steps), full-UNet (8/8) and DreamBooth. Under FINDING35 S4 that 0.400
# is provisional and the ratchet obligation is live on TI specifically. The LoRA lesson is the
# guide: 3x the budget took LoRA from 0.867 to 1.000, so TI gets the same treatment -- 4000 and
# 6000 steps, and a higher lr, across seeds.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
A=imma_church_frp_a1e-6_s1500
ti() { K="H82TP_${A}_s$1_lr$2_sd$3"; mkdir ".labels/$K" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$4
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --unet "adapters/$A/unet.pt" --steps "$1" --lr "$2" --seed "$3" --label "$K" \
    >>logs/imma_ti_push.log 2>&1; }
( ti 4000 1e-2 1 0; ti 6000 1e-2 0 0; echo "TIP A DONE $(date -Is)" >>logs/imma_ti_push.log ) &
( ti 4000 2e-2 6 1; ti 2000 2e-2 1 1; echo "TIP B DONE $(date -Is)" >>logs/imma_ti_push.log ) &
wait
