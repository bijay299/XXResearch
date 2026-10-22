#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; shift
# max-over-attacks at a MATCHED attacker budget, with the validity gate on every run
for spec in "$@"; do
  IFS='|' read -r arm unet <<< "$spec"
  for cfg in "300|1e-5" "300|5e-5" "150|1e-5"; do
    IFS='|' read -r st lr <<< "$cfg"
    lab="${arm}_s${st}_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    u=""; [ "$unet" != "base" ] && u="--unet $unet"
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py $u --concept church \
      --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
done
echo "[attacks] done"
