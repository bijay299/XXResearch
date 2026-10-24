#!/bin/bash
# FRP is the ONLY arm TI does not fully reverse (0.667 on both concepts). It is the only partial
# resistance measured anywhere in this phase, so it gets more seeds before it is characterised.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for cfg in "3000 1e-2 0" "3000 1e-2 6" "2000 1e-2 2"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H72F_ti_frp_church_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h72f_a.log 2>&1
  done
  echo "F A DONE $(date -Is)" >>logs/h72f_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for cfg in "3000 1e-2 0" "3000 1e-2 6" "4000 2e-2 1"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H72F_ti_frp_parachute_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept parachute \
      --unet adapters/frp_parachute_r5_l05/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h72f_b.log 2>&1
  done
  echo "F B DONE $(date -Is)" >>logs/h72f_b.log ) &
wait
