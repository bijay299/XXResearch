#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; CONC=$2; TEDIR=$3
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/$TEDIR')")
# H59: RE-MEASURE every text-encoder arm with the corrected post-attack harness. The previous
# numbers for these arms are void -- H_after was read off a pipeline carrying the BASE text
# encoder, so it could only ever return the base model's value.
for spec in "0|300" "1|300" "6|300" "0|50" "0|2"; do
  IFS='|' read -r sd nstep <<< "$spec"
  lab="H59_${CONC}_sd${sd}_s${nstep}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept $CONC \
    --steps $nstep --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h59_remeasure.txt
done
echo "[h59-$CONC] done"
