#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; shift
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
# H60: re-measure the remaining VOID text-encoder arms with the corrected harness, prioritising
# the load-bearing ones: the three remaining official concepts at the 300-step budget, 3 seeds each.
for spec in "$@"; do
  IFS='|' read -r conc tedir <<< "$spec"
  TE="$ROOT/$tedir"
  for sd in 0 1 6; do
    lab="H60_${conc}_sd${sd}_s300"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept $conc \
      --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h60_remeasure.txt
  done
done
echo "[h60] done"
