#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# RETRY. The first adversarial-erasure run DID NOT ERASE (church stayed 1.000, retain 1.000), so
# it was evidence about nothing -- a broken baseline would have let the attack look powerful for
# the wrong reason. Cause: esd_x trains only 44M cross-attention params and I gave it 400 steps at
# lr 1e-5 WITH a preservation penalty, while the ESD run that actually erased used esd_u (815M)
# for 500 steps. Matching the working configuration and weakening the preservation term.
arm=adv2_church
o=adapters/$arm
[ -f outputs/erase_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_adv.py \
  --concept church --mode esd_u --steps 500 --log-every 100 --lr 1e-5 \
  --adv-steps 2 --adv-eps 0.05 --lam-preserve 0.5 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --n-per-prompt 5 --steps 25 --seed 0 --save-grid >> outputs/log_$arm.txt 2>&1
# GATE THE COMPARISON: only attack it if it actually erased. Reported either way.
ch=$($PY -c "import json;print(json.load(open('outputs/gen_${arm}.json'))['per_concept']['church']['clip_acc'])" 2>/dev/null || echo 1.0)
echo "[adv2] church after erasure = $ch" >> outputs/log_attacks_progress.txt
for cfg in "300|1e-5" "300|5e-5" "150|1e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="${arm}_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept church \
    --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[adv2] done"
