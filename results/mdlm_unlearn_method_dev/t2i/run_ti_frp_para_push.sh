#!/bin/bash
# FRP-church's 0.667 cap broke at 0.933 under a stronger TI. FRP-parachute still reads 0.667 with
# only 3 of 8 valid. Push it the same way, so the one remaining sub-ceiling number is either broken
# or established under a comparable adversary.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for cfg in "3000 1e-2 2" "3000 1e-2 3" "5000 1e-2 0"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H72P_ti_frp_parachute_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h72p_a.log 2>&1
  done
  echo "P A DONE $(date -Is)" >>logs/h72p_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for cfg in "5000 1e-2 6" "3000 2e-2 2" "2000 1e-2 4"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H72P_ti_frp_parachute_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h72p_b.log 2>&1
  done
  echo "P B DONE $(date -Is)" >>logs/h72p_b.log ) &
wait
