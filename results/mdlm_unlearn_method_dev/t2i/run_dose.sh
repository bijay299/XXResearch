#!/bin/bash
# Dose-response hypothesis: does entrenchment scale with how completely the attack recovered the
# concept? AdvUnlearn sd1 recovered to H=0.733 and entrenched to 0.600, against sd0's H=1.000 and
# 0.867. One point is not a relation -- these are deliberately WEAKENED attacks on the FRP arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for S in 30 75; do
    L=DOSE_frp_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_a.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
      --attacked "attacked/$L/unet.pt" --concept church --state dose_s${S} \
      --modes random --fracs 0.22,0.25,0.27 >>logs/dose_a.log 2>&1
  done
  echo "DOSE A DONE $(date -Is)" >>logs/dose_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 150 300; do
    L=DOSE_frp_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/frp_church_r3_l05/unet.pt --steps $S --lr 1e-5 --seed 0 --n-images 16 \
      --keep-checkpoint --label "$L" >>logs/dose_b.log 2>&1
    venv_t2i/bin/python scripts/redundancy_sweep.py --arm frp_church_r3_l05 \
      --attacked "attacked/$L/unet.pt" --concept church --state dose_s${S} \
      --modes random --fracs 0.22,0.25,0.27 >>logs/dose_b.log 2>&1
  done
  echo "DOSE B DONE $(date -Is)" >>logs/dose_b.log ) &
wait
