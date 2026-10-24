#!/bin/bash
# The h68rf arm reaches H=1.000 by 50 steps yet shows NO entrenchment at 50 or 100, while its
# 300-step attack entrenches to 0.867. So complete recovery is not sufficient on this arm and the
# threshold sits somewhere in 100-300. Bracket it. Fracs widened to actually straddle retain8=0.45.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for S in 150 200; do
    L=DOSE2_h68rf_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose2_mid_a.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
      --attacked "attacked/$L/unet.pt" --concept church --state dose2_s${S} \
      --modes random --fracs 0.15,0.17,0.19,0.21 >>logs/dose2_mid_a.log 2>&1
  done
  echo "DOSE2 MID A DONE $(date -Is)" >>logs/dose2_mid_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  # re-sweep s50 and s100 with fracs that straddle retain8=0.45 on THIS arm, so the earlier
  # readings are not resting on a poorly-placed bracket
  for S in 50 100; do
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
      --attacked "attacked/DOSE2_h68rf_church_s${S}/unet.pt" --concept church \
      --state dose2b_s${S} --modes random --fracs 0.12,0.14,0.16,0.18 \
      >>logs/dose2_mid_b.log 2>&1
  done
  echo "DOSE2 MID B DONE $(date -Is)" >>logs/dose2_mid_b.log ) &
wait
