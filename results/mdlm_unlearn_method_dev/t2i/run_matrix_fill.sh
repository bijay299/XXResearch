#!/bin/bash
# Fill the remaining adversary-matrix cells so the report's table has no estimated entries.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  # TI on the damage+distil arm, parachute (church already done)
  for SD in 0 1; do
    L=H74_ti_h68rf_parachute_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/h68v2_rf_a0.3_parachute/unet.pt --steps 1000 --lr 5e-3 --seed $SD \
      --label "$L" >>logs/h74_a.log 2>&1
  done
  echo "MF A DONE $(date -Is)" >>logs/h74_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  # DreamBooth pw=0 on damage+distil and on FRP parachute
  for SD in 0 1; do
    L=H74_db_h68rf_church_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --label "$L" >>logs/h74_b.log 2>&1
  done
  for SD in 0 1; do
    L=H74_db_frp_parachute_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --label "$L" >>logs/h74_b.log 2>&1
  done
  echo "MF B DONE $(date -Is)" >>logs/h74_b.log ) &
wait
