#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# THIRD attempt, and diagnostic rather than blind tuning. Attempt 1 (esd_x, 400 steps) left church
# at 1.000; attempt 2 (esd_u, 500 steps, lam_preserve 0.5) reached only 0.933, while plain ESD at
# the SAME esd_u/500 config reached 0.467. The two differences are the preservation term and the
# adversarial term, so this isolates them: lam_preserve=0 removes the preservation term entirely.
# If it then erases, preservation was the blocker. If it still does not, the adversarial term is.
arm=adv3_church
o=adapters/$arm
[ -f outputs/erase_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/erase_adv.py \
  --concept church --mode esd_u --steps 500 --log-every 100 --lr 1e-5 \
  --adv-steps 2 --adv-eps 0.05 --lam-preserve 0.0 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
ch=$($PY -c "import json;print(json.load(open('outputs/gen_${arm}.json'))['per_concept']['church']['clip_acc'])" 2>/dev/null || echo 1.0)
echo "[adv3] church after erasure = $ch (ESD reference 0.467, UCE 0.067)" >> outputs/log_attacks_progress.txt
for cfg in "300|1e-5" "300|5e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="${arm}_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --concept church \
    --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[adv3] done"
