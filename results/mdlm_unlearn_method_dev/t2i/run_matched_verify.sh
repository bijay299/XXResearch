#!/bin/bash
# Re-verify FINDING48's headline table with BISECTED matched points instead of nearest-bracket ones.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/matched_removal.py --arm _base_ref \
    --attacked adapters/_base_ref/unet.pt --concept church --label base_church >>logs/mr.log 2>&1
  venv_t2i/bin/python scripts/matched_removal.py --arm frp_church_r3_l05 \
    --attacked attacked/DIAG_H66_frp_church_sd0_lr1e-5_keepckpt/unet.pt --concept church \
    --label frp_church >>logs/mr.log 2>&1
  echo "MR A DONE $(date -Is)" >>logs/mr.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/matched_removal.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_church_sd0_lr1e-5/unet.pt --concept church \
    --label advu_church >>logs/mr2.log 2>&1
  venv_t2i/bin/python scripts/matched_removal.py --arm h68v2_rf_a0.3_church \
    --attacked attacked/KEEP_h68rf_church_sd0_lr1e-5/unet.pt --concept church \
    --label h68rf_church >>logs/mr2.log 2>&1
  echo "MR B DONE $(date -Is)" >>logs/mr2.log ) &
wait
