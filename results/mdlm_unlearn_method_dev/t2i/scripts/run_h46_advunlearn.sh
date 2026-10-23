#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H46: the DECISIVE test of PREREG_H40 against a WORKING robust baseline -- the official
# AdvUnlearn (2405.15234) church text encoder, which erases church 1.000 -> 0.400 at retain 0.992.
# My own three adversarial re-implementations failed to erase, so this replaces them entirely.
#
# TWO ADVERSARIES, because AdvUnlearn's defence lives in the TEXT ENCODER:
#   (a) UNet-only  -- identical protocol to every other arm, so the comparison is matched
#   (b) UNet + text encoder -- a weight-access adversary would attack where the defence IS.
#       Reporting only (a) would be a needlessly weak adversary against this specific defence.
for sd in 0 1 3 4; do
  lab="H46_advunlearn_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept church \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
for sd in 0 1; do
  lab="H46TE_advunlearn_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --attack-te \
    --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h46] done"
