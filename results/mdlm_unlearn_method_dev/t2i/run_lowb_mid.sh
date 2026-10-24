#!/bin/bash
# Complete the official-AdvUnlearn attacker-cost curve: 50 steps gives only H=0.533, 300 gives
# 1.000, so the completion point is in between. This makes the per-arm attacker cost comparable.
set -u
cd /data/bijaypandey/t2i_unlearn
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
for S in 150 200; do
  L=LOWB_advunlearn_church_s${S}
  [ -f "outputs/atk_${L}.json" ] && continue
  venv_t2i/bin/python scripts/attack_relearn.py --concept church --text-encoder "$TE" \
    --steps $S --lr 1e-5 --seed 0 --n-images 16 --label "$L" >>logs/lowb_mid.log 2>&1
done
echo "LOWB MID DONE $(date -Is)" >>logs/lowb_mid.log
