#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# PREREG_H38 S4's "real bar": a method BUILT to resist attack. If it falls to the same
# 150-step/16-image attack, that is strong evidence for PREREG_H40. If it holds at high utility,
# H40 is REFUTED and this is the method the mandate asks for.
arm=adv_church
o=adapters/$arm
[ -f outputs/erase_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_adv.py \
  --concept church --mode esd_x --steps 400 --log-every 100 --adv-steps 3 --adv-eps 0.05 \
  --lam-preserve 1.0 --seed 0 --label $arm --out $o > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --n-per-prompt 5 --steps 25 --seed 0 --save-grid >> outputs/log_$arm.txt 2>&1
for cfg in "300|1e-5" "300|5e-5" "150|1e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="${arm}_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept church \
    --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[adv] done"
