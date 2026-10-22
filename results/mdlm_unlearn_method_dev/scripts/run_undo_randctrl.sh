#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
T=adapters/su_teacher/model.pt; i=0
for s in 0 1 2; do
  for al in 0.5; do
    lab="undo_r_a${al/./p}$([ $s -gt 0 ] && echo _s$s)"
    g=$(( i % 2 + 2 )); i=$((i+1))
    (
      [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
        --teacher "$T" --alpha "$al" --beta 0.1 --protect random --seed "$s" \
        --steps 3000 --log-every 3000 --out "adapters/$lab" \
        > "outputs/log_$lab.txt" 2>&1 || exit 0
      CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
        --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
        --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
      rm -f "adapters/$lab/model.pt"
    ) &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
echo "[randctrl] done"
