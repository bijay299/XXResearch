#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# THE DECIDING RUN for H38. The alpha ladder shows NO alpha that both preserves utility and
# buys robustness: alpha=0.3 keeps retain 0.458 and is still fully reversed (1.000), alpha=0.5
# resists partially at retain 0.092, alpha=0.7 is dead. Before declaring H38 false, test whether
# that is a REPAIR BUDGET artifact -- the MDLM phase's H19, where 4x repair moved retain
# 0.457 -> 0.67+. Here: 4x steps (6000) and 3x pool (144 images) at the same alpha=0.5.
arm=undo_a0.5_big
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher adapters/uce_church/unet.pt --concept church --alpha 0.5 --beta 0.1 \
  --protect retain_fisher --steps 6000 --log-every 1500 --n-images 144 --forget-frac 0.34 \
  --seed 0 --label $arm --out $o > outputs/log_$arm.txt 2>&1
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
# TRAINABILITY CONTROL on the same arm: attack a concept that was NEVER an erasure target.
for cfg in "300|1e-5" "300|5e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="TRAIN_${arm}_tench_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept tench \
    --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h38-decide] done"
