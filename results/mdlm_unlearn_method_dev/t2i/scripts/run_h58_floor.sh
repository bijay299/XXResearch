#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; CONC=$2; TEDIR=$3
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/$TEDIR')")
# H58: LOCATE THE FLOOR. FINDING37 found recovery saturated at every budget down to 1 image /
# 50 steps, so the threshold is at or BELOW the smallest point tested and the finding records it as
# "not located". This sweeps further down -- 2, 5, 10, 25 steps on a single image -- to find where
# recovery actually begins. Again the direction only WEAKENS the adversary, so it can only make the
# defence look better.
for nstep in 2 5 10 25; do
  lab="H58_${CONC}_n1_s${nstep}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept $CONC \
    --steps $nstep --lr 1e-5 --n-images 1 --eval-n 5 --seed 0 --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h58_floor.txt
done
echo "[h58-$CONC] done"
