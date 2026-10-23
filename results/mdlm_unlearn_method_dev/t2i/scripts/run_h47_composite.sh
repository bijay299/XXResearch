#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H47: COMPOSITE DEFENCE -- official AdvUnlearn text-encoder erasure + damage/distil UNet
# robustification. Motivated by two measurements rather than hope:
#   (a) H46: AdvUnlearn alone is fully reversed (H 0.400 -> 1.000, gate PASS, 2/2 seeds so far)
#   (b) H45: damage+distil is 2.1x more robust than UCE AT MATCHED UTILITY
# and by the diagnosis of why H38 failed: the UCE teacher sat at retain 0.850 and damage dropped
# it to 0.317. AdvUnlearn's teacher sits at retain 0.992, so the same damage should land higher.
# alpha=0.3 and 0.5 bracket the ladder; alpha=0 is the no-damage control.
for al in 0.0 0.3 0.5; do
  arm=h47_adv_a${al}
  o=adapters/$arm
  if [ ! -f outputs/undo_${arm}.json ]; then
    CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py --teacher-te "$TE" --concept church \
      --alpha $al --beta 0.1 --protect retain_fisher --steps 6000 --log-every 2000 \
      --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
      > outputs/log_$arm.txt 2>&1 || continue
  fi
  [ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
    --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 \
    >> outputs/log_$arm.txt 2>&1
  for sd in 0 1 3; do
    lab="${arm}_church_sd${sd}_lr1e-5"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
      --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
      > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
  rm -f $o/unet.pt
  echo "[h47] $arm done" >> outputs/log_h47_progress.txt
done
echo "[h47] complete"
