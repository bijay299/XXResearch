#!/bin/bash
# frp_garbage_truck_r3_l05 is FRP's best operating point in the phase: target 0.1333 at retain
# 0.9667 (church was 0.1333/0.9417). It clears the mandate's 0.85 utility floor, so its robustness
# is the question that matters. Frozen six-point grid plus the two cheap adversaries.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for spec in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $spec; SD=$1; LR=$2
    L=H77_frpgt_sd${SD}_lr${LR}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_relearn.py --concept garbage_truck \
      --unet adapters/frp_garbage_truck_r3_l05/unet.pt --steps 300 --lr "$LR" --seed "$SD" \
      --n-images 16 --label "$L" >>logs/h77_a.log 2>&1
  done
  echo "H77 A DONE $(date -Is)" >>logs/h77_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1 6; do
    L=H77_ti_frpgt_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept garbage_truck \
      --unet adapters/frp_garbage_truck_r3_l05/unet.pt --steps 1000 --lr 5e-3 --seed $SD \
      --label "$L" >>logs/h77_b.log 2>&1
  done
  for SD in 0 1; do
    L=H77_db_frpgt_pw0_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_dreambooth.py --concept garbage_truck \
      --unet adapters/frp_garbage_truck_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed $SD \
      --prior-weight 0.0 --measure-utility --label "$L" >>logs/h77_b.log 2>&1
  done
  echo "H77 B DONE $(date -Is)" >>logs/h77_b.log ) &
wait
