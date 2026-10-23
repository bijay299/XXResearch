#!/bin/bash
# Third state for FINDING46: the ERASED-but-UNATTACKED FRP arm, swept identically. With base and
# attacked already measured, this separates "erasure changes removal-robustness" from "the ATTACK
# changes it".
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
    --attacked adapters/frp_church_r3_l05/unet.pt --concept church \
    >>logs/redun_erased_church.log 2>&1
  echo "ERASED CHURCH DONE $(date -Is)" >>logs/redun_erased_church.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_parachute_r5_l05 \
    --attacked adapters/frp_parachute_r5_l05/unet.pt --concept parachute \
    >>logs/redun_erased_parachute.log 2>&1
  echo "ERASED PARA DONE $(date -Is)" >>logs/redun_erased_parachute.log ) &
wait
