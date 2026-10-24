#!/bin/bash
# The AdvUnlearn cost curve is non-monotone at 150->200 (0.933 -> 0.867). Second seed to check
# whether that is quantisation noise before the curve is cited.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  for S in 50 100; do
    L=LOWB1_advunlearn_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church --text-encoder "$TE" \
      --steps $S --lr 1e-5 --seed 1 --n-images 16 --label "$L" >>logs/lowb_s1a.log 2>&1
  done
  echo "S1A DONE $(date -Is)" >>logs/lowb_s1a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for S in 150 200; do
    L=LOWB1_advunlearn_church_s${S}
    venv_t2i/bin/python scripts/attack_relearn.py --concept church --text-encoder "$TE" \
      --steps $S --lr 1e-5 --seed 1 --n-images 16 --label "$L" >>logs/lowb_s1b.log 2>&1
  done
  echo "S1B DONE $(date -Is)" >>logs/lowb_s1b.log ) &
wait
