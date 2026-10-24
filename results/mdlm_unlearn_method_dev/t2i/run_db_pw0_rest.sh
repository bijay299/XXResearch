#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TEV=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/vangogh_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1; do
    L=H73Z_db_advu_vangogh_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept van_gogh --text-encoder "$TEV" \
      --steps 300 --lr 1e-5 --seed $SD --prior-weight 0.0 --label "$L" >>logs/h73zr_a.log 2>&1
  done
  echo "ZR A DONE $(date -Is)" >>logs/h73zr_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H73Z_db_frp_church_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --label "$L" >>logs/h73zr_b.log 2>&1
  done
  echo "ZR B DONE $(date -Is)" >>logs/h73zr_b.log ) &
wait
