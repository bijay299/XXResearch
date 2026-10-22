#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
GPU="${GPU:-2}"
for spec in $ARMS; do
  lab="rtt_$spec"
  ck="adapters/$spec/model.pt"
  [ -f "$ck" ] || { echo "[rtt] no ckpt $spec"; continue; }
  [ -f "outputs/rtt_$lab.json" ] && continue
  echo "[rtt] $spec"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_small_rtt.py --arm "$ck" \
    --steps 600 --lr 1e-4 --label "$lab" > "outputs/log_$lab.txt" 2>&1 \
    || echo "[rtt] FAILED $spec"
done
echo "[rtt] done"
