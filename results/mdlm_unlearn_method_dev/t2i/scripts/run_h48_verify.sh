#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H48: H47's alpha=0.3 arm (retain 0.867, church 0.000) gave max-over-VALID H = 0.533 on ONE
# valid attack of three -- above its declared 0.50 bar, and resting on n=1. H43 already showed
# gate outcomes flip with seed, and three claims in this project have weakened when the adversary
# was strengthened AFTER the fact. So before ANY verdict: rebuild the arm (its checkpoint was
# deleted for storage) and attack it with more seeds AND higher learning rates.
# The checkpoint is KEPT this time so the verdict does not rest on a rebuild.
arm=h48_adv_a0.3
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept church --alpha 0.3 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
# more seeds at the reference lr, then ADVERSARY STRENGTHENING at higher lr
for spec in "4|1e-5" "5|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "0|1e-4"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="${arm}_church_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept church --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h48] done"
