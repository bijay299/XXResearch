#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; CONC=$2; TEDIR=$3
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/$TEDIR')")
# H57: HOW CHEAP IS THE ATTACK? Every result so far used a fixed 300 steps / 16 images, which
# establishes that erasure is reversible but not at what cost. That cost is the practically
# meaningful quantity: a defence undone by 16 images and 300 steps is weak, but one undone by
# 1 image and 50 steps is not a defence at all. Sweeping the attacker's budget DOWNWARD on the
# official AdvUnlearn arm converts "reversible" into "reversible at cost X".
# Note the direction: this WEAKENS the adversary, so it can only make the defence look better --
# the opposite of the ratchet the audit warned about.
for spec in "1|50" "1|150" "2|150" "4|50" "4|150" "8|150" "16|50" "1|300" "4|300"; do
  IFS='|' read -r nimg nstep <<< "$spec"
  lab="H57_${CONC}_n${nimg}_s${nstep}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept $CONC \
    --steps $nstep --lr 1e-5 --n-images $nimg --eval-n 5 --seed 0 --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h57_cost.txt
done
echo "[h57-$CONC] done"
