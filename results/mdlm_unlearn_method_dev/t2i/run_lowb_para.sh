#!/bin/bash
# Second concept for FINDING52's attacker-cost curve.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for S in 25 50 100; do
    L=LOWBP_advunlearn_parachute_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept parachute --text-encoder "$TE" \
      --steps $S --lr 1e-5 --seed 0 --n-images 16 --label "$L" >>logs/lowbp_a.log 2>&1
  done
  echo "LOWBP A DONE $(date -Is)" >>logs/lowbp_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 150 200; do
    L=LOWBP_advunlearn_parachute_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept parachute --text-encoder "$TE" \
      --steps $S --lr 1e-5 --seed 0 --n-images 16 --label "$L" >>logs/lowbp_b.log 2>&1
  done
  echo "LOWBP B DONE $(date -Is)" >>logs/lowbp_b.log ) &
wait
