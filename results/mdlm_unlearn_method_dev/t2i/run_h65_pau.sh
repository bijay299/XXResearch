#!/bin/bash
# H65 -- post-attack utility (ISSUE 7) measured under PROTOCOL_FROZEN_v1.
#
# The grid is the frozen six-point one, applied identically to both arms, and is NOT extended
# after the numbers are seen: seeds {0,1,6} at lr 1e-5, {0,1} at 5e-5, {0} at 1e-4; 300 steps;
# 16 attacker images. New labels (H65_pau_*) -- no historical record is overwritten.
#
# Usage: run_h65_pau.sh <gpu> <concept>
set -u
cd /data/bijaypandey/t2i_unlearn
GPU=$1; CONCEPT=$2
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/${CONCEPT}_unlearned
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/bijaypandey/hf_home
LOG=logs/h65_pau_${CONCEPT}.log
mkdir -p logs
for spec in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
  set -- $spec; SD=$1; LR=$2
  L="H65_pau_${CONCEPT}_sd${SD}_lr${LR}"
  if [ -f "outputs/atk_${L}.json" ]; then echo "[skip] $L exists" >>$LOG; continue; fi
  echo "=== $L $(date -Is)" >>$LOG
  venv_t2i/bin/python scripts/attack_relearn.py \
    --concept "$CONCEPT" --text-encoder "$TE" \
    --steps 300 --lr "$LR" --seed "$SD" --n-images 16 \
    --label "$L" --save-grids >>$LOG 2>&1
  echo "=== rc=$? $(date -Is)" >>$LOG
done
echo "ALL DONE $(date -Is)" >>$LOG
