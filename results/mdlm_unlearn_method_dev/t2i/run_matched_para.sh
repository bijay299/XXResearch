#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/matched_removal.py --arm _base_ref \
    --attacked adapters/_base_ref/unet.pt --concept parachute --label base_para >>logs/mrp.log 2>&1
  venv_t2i/bin/python scripts/matched_removal.py --arm frp_parachute_r5_l05 \
    --attacked attacked/DIAG_H66_frp_parachute_sd1_lr1e-5_keepckpt/unet.pt --concept parachute \
    --label frp_para >>logs/mrp.log 2>&1
  echo "MRP A DONE $(date -Is)" >>logs/mrp.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/matched_removal.py --arm _base_ref \
    --attacked attacked/KEEP_advunlearn_parachute_sd0_lr1e-5/unet.pt --concept parachute \
    --label advu_para >>logs/mrp2.log 2>&1
  venv_t2i/bin/python scripts/matched_removal.py --arm h68v2_rf_a0.3_parachute \
    --attacked attacked/KEEP_h68rf_parachute_sd0_lr1e-5/unet.pt --concept parachute \
    --label h68rf_para >>logs/mrp2.log 2>&1
  echo "MRP B DONE $(date -Is)" >>logs/mrp2.log ) &
wait
