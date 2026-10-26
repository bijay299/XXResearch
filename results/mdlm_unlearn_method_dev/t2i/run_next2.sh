#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/rezero_control.py --arm frp_church_r3_l05 \
    --attacked attacked/DIAG_H66_frp_church_sd0_lr1e-5_keepckpt/unet.pt --label church_r3 \
    >>logs/rezero_ctl.log 2>&1
  venv_t2i/bin/python scripts/gen_eval.py --label REZEROCTL_frp_church_r3 \
    --unet adapters/_rezeroctl_church_r3/unet.pt --target church >>logs/rezero_ctl.log 2>&1
  echo "CTL DONE $(date -Is)" >>logs/rezero_ctl.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for spec in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $spec; SD=$1; LR=$2
    L="H70_h68rf_parachute_sd${SD}_lr${LR}"
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_relearn.py --concept parachute \
      --unet adapters/h68v2_rf_a0.3_parachute/unet.pt --steps 300 --lr "$LR" --seed "$SD" \
      --n-images 16 --label "$L" >>logs/h70_h68rf_parachute.log 2>&1
  done
  echo "H70 PARA DONE $(date -Is)" >>logs/h70_h68rf_parachute.log ) &
wait
