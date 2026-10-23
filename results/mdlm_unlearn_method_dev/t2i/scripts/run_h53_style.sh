#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
TE="$ROOT/vangogh_unlearned"
# H53: GENERALITY TO A STYLE. Every claim in this phase rests on OBJECT concepts (church,
# parachute). Van Gogh is a STYLE -- diffuse, not object-localised -- and is the setting where
# ESD/UCE report their own method behaves differently (ESD-x vs ESD-u exists precisely because
# styles and objects need different parameter subsets). So this is a qualitatively different test
# of PREREG_H40's wall, not a third sample of the same thing.
#   1. verify the official AdvUnlearn style erasure works
#   2. attack it under the pre-declared SYMMETRIC 6-attack budget
#   3. the composite at alpha=0.3, same budget
[ -f outputs/gen_advunlearn_vangogh.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py \
  --label advunlearn_vangogh --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 \
  > outputs/log_advunlearn_vangogh.txt 2>&1
for spec in "0|1e-5" "1|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "0|1e-4"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="H53_advunlearn_vangogh_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept van_gogh \
    --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
arm=h53_vg_a0.3
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept van_gogh --alpha 0.3 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for spec in "0|1e-5" "1|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "0|1e-4"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="${arm}_vangogh_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept van_gogh --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[h53] done"
