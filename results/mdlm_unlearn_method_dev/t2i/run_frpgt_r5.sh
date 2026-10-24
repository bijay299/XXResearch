#!/bin/bash
# frp_garbage_truck_r5_l05 is the only arm in the phase with COMPLETE erasure (target 0.0000) at
# non-trivial utility (retain 0.7750). It fails the 0.85 floor so it is not a candidate, but its
# robustness completes the record.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for spec in "0 1e-5" "1 1e-5" "6 1e-5"; do
    set -- $spec; SD=$1; LR=$2
    L=H78_frpgt5_sd${SD}_lr${LR}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_relearn.py --concept garbage_truck \
      --unet adapters/frp_garbage_truck_r5_l05/unet.pt --steps 300 --lr "$LR" --seed "$SD" \
      --n-images 16 --label "$L" >>logs/h78_a.log 2>&1
  done
  echo "H78 A DONE $(date -Is)" >>logs/h78_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H78_ti_frpgt5_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept garbage_truck \
      --unet adapters/frp_garbage_truck_r5_l05/unet.pt --steps 1000 --lr 5e-3 --seed $SD \
      --label "$L" >>logs/h78_b.log 2>&1
  done
  echo "H78 B DONE $(date -Is)" >>logs/h78_b.log ) &
wait
