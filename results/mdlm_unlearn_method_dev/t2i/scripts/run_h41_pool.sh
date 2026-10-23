#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# H41: test FINDING33 S3's prediction. Pool DIVERSITY, not more steps: 300 images over ~52
# retain prompts instead of 48-144 over ~20. If repair can restore utility, the prediction is
# that the arm becomes trainable again AND shows fact-specific robustness.
arm=undo_a0.5_pool
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher adapters/uce_church/unet.pt --concept church --alpha 0.5 --beta 0.1 \
  --protect retain_fisher --steps 6000 --log-every 1500 --n-images 300 --forget-frac 0.20 \
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
lab="TRAIN_${arm}_tench_s300_lr1e-5"
[ -f outputs/atk_${lab}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py \
  --unet $o/unet.pt --concept tench --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --label $lab \
  > outputs/log_atk_$lab.txt 2>&1
grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
rm -f $o/unet.pt
echo "[h41] done"
