#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# The baseline frontier recoveries are n=1. This project has twice had single-seed claims
# overturned, so they get replicated before the table is relied on.
run () {
  local arm=$1
  local lf=$2
  local s=$3
  local lab="base_${arm}_f${lf/./p}_s${s}"
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_unlearn.py \
      --arm "$arm" --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 1500 --seed "$s" \
      --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
for s in 1 2; do
  run afc 1.0 $s; run mdu 0.5 $s
  wait
  run emdu 1.0 $s; run ga 0.5 $s
  wait
done
echo "[baseline_seeds] done"
