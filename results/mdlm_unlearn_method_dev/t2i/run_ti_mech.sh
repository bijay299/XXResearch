#!/bin/bash
# TWO things before any claim:
#  (a) MECHANISM TEST. If the erasure SITE predicts vulnerability to embedding-space attacks, then
#      the damage+distil arm (UNet-space, like FRP) should resist TI as FRP does. If it falls to
#      1.000 like AdvUnlearn, the site explanation is wrong.
#  (b) FRP parachute gave 0 of 2 VALID attacks. Per FINDING20 that is NOT resistance. Strengthen TI
#      (more steps, higher lr) to either obtain a valid attack or document that it cannot be made
#      valid on that arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1 6; do
    L=H72M_ti_h68rf_church_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 1000 --lr 5e-3 --seed $SD --label "$L" \
      >>logs/h72mech_a.log 2>&1
  done
  echo "MECH A DONE $(date -Is)" >>logs/h72mech_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for cfg in "2000 5e-3 6" "2000 1e-2 0" "3000 1e-2 1"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H72S_ti_frp_parachute_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h72strong_b.log 2>&1
  done
  echo "STRONG TI DONE $(date -Is)" >>logs/h72strong_b.log ) &
wait
