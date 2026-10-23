#!/bin/bash
# Is FRP-arm entrenchment reproducible across attack seeds, or a single-run artifact?
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  L=KEEP_frp_church_sd1_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed 1 --n-images 16 \
    --keep-checkpoint --label "$L" >>logs/harden_rep_church.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked "attacked/$L/unet.pt" --concept church --state attacked_frp_sd1 \
    >>logs/harden_rep_church.log 2>&1
  echo "REP CHURCH DONE $(date -Is)" >>logs/harden_rep_church.log ) &
( export CUDA_VISIBLE_DEVICES=1
  L=KEEP_frp_church_sd6_lr1e-5
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed 6 --n-images 16 \
    --keep-checkpoint --label "$L" >>logs/harden_rep_church6.log 2>&1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked "attacked/$L/unet.pt" --concept church --state attacked_frp_sd6 \
    >>logs/harden_rep_church6.log 2>&1
  echo "REP CHURCH6 DONE $(date -Is)" >>logs/harden_rep_church6.log ) &
wait
