#!/bin/bash
# Textual Inversion adversary: 768 trainable scalars (0.00009% of the UNet). Third adversary class.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1; do
    L=H72_ti_advunlearn_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church --text-encoder "$TE" \
      --steps 1000 --lr 5e-3 --seed $SD --label "$L" >>logs/h72_a.log 2>&1
  done
  echo "TI A DONE $(date -Is)" >>logs/h72_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H72_ti_frp_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h72_b.log 2>&1
  done
  echo "TI B DONE $(date -Is)" >>logs/h72_b.log ) &
wait
