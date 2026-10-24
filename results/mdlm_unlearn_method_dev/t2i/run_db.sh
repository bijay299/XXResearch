#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1; do
    L=H73_db_advunlearn_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church --text-encoder "$TE" \
      --steps 300 --lr 1e-5 --seed $SD --label "$L" >>logs/h73_a.log 2>&1
  done
  echo "DB A DONE $(date -Is)" >>logs/h73_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H73_db_frp_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed $SD --label "$L" \
      >>logs/h73_b.log 2>&1
  done
  echo "DB B DONE $(date -Is)" >>logs/h73_b.log ) &
wait
