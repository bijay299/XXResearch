#!/bin/bash
# Complete the dose curve at the informative end: budgets small enough that recovery is PARTIAL.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for S in 5 10; do
    L=DOSE_frp_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_low_a.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
      --attacked "attacked/$L/unet.pt" --concept church --state dose_s${S} \
      --modes random --fracs 0.22,0.25,0.27 >>logs/dose_low_a.log 2>&1
  done
  echo "DOSE LOW A DONE $(date -Is)" >>logs/dose_low_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 20 50; do
    L=DOSE_frp_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_low_b.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
      --attacked "attacked/$L/unet.pt" --concept church --state dose_s${S} \
      --modes random --fracs 0.22,0.25,0.27 >>logs/dose_low_b.log 2>&1
  done
  echo "DOSE LOW B DONE $(date -Is)" >>logs/dose_low_b.log ) &
wait
