#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# undo_s_a0p0 / a0p1 have only lr 1e-4, so their recoveries are LOWER BOUNDS and not comparable
# to the max-over-lr figures. Without them the alpha=0 control on the AFC line is not sound.
i=0
for arm in undo_s_a0p0 undo_s_a0p1; do
  for lr in 3e-5 3e-4 1e-3; do
    lab="rtt_${arm}_lr${lr}"
    [ -f "outputs/rtt_$lab.json" ] && continue
    [ -f "adapters/$arm/model.pt" ] || { echo "[lr3] no ckpt $arm"; continue; }
    g=$(( i % 2 + 2 )); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$arm/model.pt" \
      --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
# alpha=0.5 on the MDU teacher, to match the AFC line's strongest point
T=adapters/su_teacher_mdu/model.pt
if [ -f "$T" ] && [ ! -f adapters/undo_m_a0p5/model.pt ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py --teacher "$T" --alpha 0.5 --beta 0.1 \
    --steps 3000 --log-every 1500 --out adapters/undo_m_a0p5 \
    > outputs/log_undo_m_a0p5.txt 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY scripts/rank_probe_small.py --ckpt adapters/undo_m_a0p5/model.pt \
    --label undo_m_a0p5 > outputs/log_rank_undo_m_a0p5.txt 2>&1
  j=0
  for lr in 1e-4 3e-5 3e-4; do
    g=$(( j % 2 + 2 )); j=$((j+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py \
      --arm adapters/undo_m_a0p5/model.pt --steps 600 --lr "$lr" \
      --label "rtt_undo_m_a0p5_lr${lr}" > "outputs/log_rtt_undo_m_a0p5_lr${lr}.txt" 2>&1 &
    [ $((j % 2)) -eq 0 ] && wait
  done
  wait
fi
echo "[lr3] done"
