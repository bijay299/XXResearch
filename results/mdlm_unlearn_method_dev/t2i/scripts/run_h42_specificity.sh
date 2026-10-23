#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# H42: is the concept-specific resistance real, or does it vanish when the adversary is
# STRENGTHENED? At alpha=0.5 the targeted concept (church) will not relearn (T rise +0.125) while
# an untargeted one (tench) does (+0.542) on the SAME arm. That contradicts FINDING32 S5's
# conclusion. Three of this project's claims have weakened when the adversary was strengthened
# after publication, so the correction is tested at higher lr BEFORE it is written up.
o=adapters/undo_a0.5_big
for lr in 1e-4 3e-4; do
  for c in church tench; do
    pre=""; [ "$c" = "tench" ] && pre="TRAIN_"
    lab="${pre}H42_a0.5_big_${c}_s300_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept $c \
      --steps 300 --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
done
echo "[h42] done"
