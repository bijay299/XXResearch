#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked attacked/DIAG_H66_frp_church_sd0_lr1e-5_keepckpt/unet.pt --concept church \
    >>logs/redun_church.log 2>&1; echo "REDUN CHURCH DONE $(date -Is)" >>logs/redun_church.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_parachute_r5_l05 \
    --attacked attacked/DIAG_H66_frp_parachute_sd1_lr1e-5_keepckpt/unet.pt --concept parachute \
    >>logs/redun_parachute.log 2>&1; echo "REDUN PARA DONE $(date -Is)" >>logs/redun_parachute.log ) &
wait
