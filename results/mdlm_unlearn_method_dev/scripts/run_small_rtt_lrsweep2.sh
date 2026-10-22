#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# The MDU-teacher arms only had lr 1e-4, so their recoveries are LOWER BOUNDS and not
# comparable to the AFC-teacher arms' max-over-lr figures. lr 3e-5 was adversarially optimal
# for every AFC arm, so it must be run here too.
i=0
for arm in su_teacher_mdu undo_m_a0p0 undo_m_a0p3; do
  for lr in 3e-5 3e-4 1e-3; do
    lab="rtt_${arm}_lr${lr}"
    [ -f "outputs/rtt_$lab.json" ] && continue
    [ -f "adapters/$arm/model.pt" ] || { echo "[lrsweep2] no ckpt $arm"; continue; }
    g=$(( i % 2 + 2 )); i=$((i+1))
    echo "[lrsweep2] $arm lr=$lr gpu$g"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$arm/model.pt" \
      --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
echo "[lrsweep2] done"
