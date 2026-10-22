#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
# Trainability-control REFERENCE: how fast can the UNDAMAGED teacher learn tench?
# The damaged arm's tench number is only interpretable against this.
for cfg in "300|1e-5" "300|5e-5"; do
  IFS='|' read -r st lr <<< "$cfg"
  for spec in "uce_church|adapters/uce_church/unet.pt" "base|base"; do
    IFS='|' read -r nm unet <<< "$spec"
    lab="TRAIN_${nm}_tench_s${st}_lr${lr}"
    [ -f outputs/atk_${lab}.json ] && continue
    u=""; [ "$unet" != "base" ] && u="--unet $unet"
    CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py $u --concept tench \
      --steps $st --lr $lr --n-images 16 --eval-n 5 --label $lab > outputs/log_atk_$lab.txt 2>&1
    grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
  done
done
echo "[train-control-ref] done"
