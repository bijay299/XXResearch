#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H51: CONCEPT-TARGETED DAMAGE -- the sharpest available test of PREREG_H40's wall.
# FINDING35: no arm at retain >= 0.78 has EVER resisted (6/6 at exactly H 1.000). Every one of
# those arms damaged all gradient paths roughly uniformly, so utility and the concept's route died
# together. forget_fisher concentrates the SAME mean damage on the parameters that matter most to
# the erased concept, leaving the rest intact -- the one mechanism that could, in principle, give
# robustness without the utility cost.
#
# Learning from H48: attacks are run at THREE learning rates and multiple seeds from the start,
# because a robustness number at one lr with few valid attacks has never survived in this project.
for al in 0.3 0.5; do
  arm=h51_ff_a${al}
  o=adapters/$arm
  [ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
    --teacher-te "$TE" --concept church --alpha $al --beta 0.1 --protect forget_fisher \
    --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
    > outputs/log_$arm.txt 2>&1 || continue
  [ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
    --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
  for spec in "0|1e-5" "1|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "0|1e-4"; do
    IFS='|' read -r sd lr <<< "$spec"
    lab="${arm}_church_sd${sd}_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
      --concept church --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
  rm -f $o/unet.pt
  echo "[h51] $arm done" >> outputs/log_h51_progress.txt
done
echo "[h51] complete"
