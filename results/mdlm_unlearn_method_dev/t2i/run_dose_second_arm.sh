#!/bin/bash
# Does the entrenchment THRESHOLD replicate on a second arm? Learned the hard way today: run the
# generality check BEFORE publishing the general form.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for S in 10 20; do
    L=DOSE2_h68rf_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose2_a.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
      --attacked "attacked/$L/unet.pt" --concept church --state dose2_s${S} \
      --modes random --fracs 0.15,0.17,0.20 >>logs/dose2_a.log 2>&1
  done
  echo "DOSE2 A DONE $(date -Is)" >>logs/dose2_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 50 100; do
    L=DOSE2_h68rf_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose2_b.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm h68v2_rf_a0.3_church \
      --attacked "attacked/$L/unet.pt" --concept church --state dose2_s${S} \
      --modes random --fracs 0.15,0.17,0.20 >>logs/dose2_b.log 2>&1
  done
  echo "DOSE2 B DONE $(date -Is)" >>logs/dose2_b.log ) &
wait
