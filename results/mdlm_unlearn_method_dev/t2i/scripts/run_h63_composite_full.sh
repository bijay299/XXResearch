#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; CONC=$2; TEDIR=$3
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
TE="$ROOT/$TEDIR"
# H63: apply the lesson H62 just taught. Tench looked like a partial-resistance outlier at 0.733 on
# THREE seeds at one lr, and reached 1.000 once given 11 attacks across three lrs. The composite
# arms' numbers (church 0.933, van_gogh 1.000) also rest on three seeds at a single lr, so church's
# 0.933 is probably an underestimate for exactly the same reason.
# Full budget here: 5 seeds at 1e-5 plus 3 at 5e-5 plus 1 at 1e-4 = 9 attacks per arm.
arm=h63_${CONC}_a0.3
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept $CONC --alpha 0.3 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for spec in "0|1e-5" "1|1e-5" "2|1e-5" "3|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "6|5e-5" "0|1e-4"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="H63_${CONC}_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept $CONC --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h63_composite.txt
done
rm -f $o/unet.pt
echo "[h63-$CONC] done"
