#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
for arm in su_teacher undo_s_a0p3 undo_s_a0p5; do
  for lr in 3e-5 3e-4 1e-3; do
    lab="rtt_${arm}_lr${lr}"
    [ -f "outputs/rtt_$lab.json" ] && continue
    [ -f "adapters/$arm/model.pt" ] || { echo "[lrsweep] no ckpt $arm"; continue; }
    echo "[lrsweep] $arm lr=$lr"
    CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_rtt.py --arm "adapters/$arm/model.pt" \
      --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 \
      || echo "[lrsweep] FAILED $arm lr=$lr"
  done
done
echo "[lrsweep] done"
