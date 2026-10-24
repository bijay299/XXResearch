#!/bin/bash
# Does entrenchment keep rising past 300 steps? If it is a gradient in BUDGET rather than a
# threshold at recovery-completion, it should -- H is already pinned at 1.000 and cannot move.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for S in 600; do
    L=DOSE_frp_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_ext_a.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
      --attacked "attacked/$L/unet.pt" --concept church --state dose_s${S} \
      --modes random --fracs 0.22,0.25,0.27,0.30 >>logs/dose_ext_a.log 2>&1
  done
  echo "EXT A DONE $(date -Is)" >>logs/dose_ext_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 600; do
    L=DOSE2_h68rf_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_ext_b.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
      --attacked "attacked/$L/unet.pt" --concept church --state dose2_s${S} \
      --modes random --fracs 0.17,0.20,0.23,0.26 >>logs/dose_ext_b.log 2>&1
  done
  echo "EXT B DONE $(date -Is)" >>logs/dose_ext_b.log ) &
wait
