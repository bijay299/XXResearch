#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# ESD: erasure quality first, then the gated attack grid
[ -f outputs/gen_esd_church.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py \
  --label esd_church --unet adapters/esd_church/unet.pt --n-per-prompt 5 --steps 25 --seed 0 \
  --save-grid > outputs/log_gen_esd.txt 2>&1
for cfg in "300|1e-5" "300|5e-5" "150|1e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  lab="esd_church_s${st}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet adapters/esd_church/unet.pt \
    --concept church --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[esd-attacks] done"
