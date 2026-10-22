#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# Rebuild alpha=0.3 (its checkpoint was deleted by the sweep) and run the trainability control
# on it. alpha=0.3 is the key control point: it has NONZERO retain (0.458) yet is still fully
# reversed, so if it is already measurably harder to fine-tune, trainability loss sets in BEFORE
# any robustness is bought -- which would mean the two cannot be separated on this ladder.
arm=undo_a0.3_rebuild
o=adapters/$arm
[ -f $o/unet.pt ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher adapters/uce_church/unet.pt --concept church --alpha 0.3 --beta 0.1 \
  --protect retain_fisher --steps 1500 --log-every 1500 --n-images 48 --forget-frac 0.34 \
  --seed 0 --label $arm --out $o > outputs/log_$arm.txt 2>&1
for cfg in "300|1e-5" "300|5e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="TRAIN_undo_a0.3_tench_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept tench \
    --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[train-control-dmg] done"
