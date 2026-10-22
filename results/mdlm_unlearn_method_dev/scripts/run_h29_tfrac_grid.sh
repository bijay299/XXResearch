#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
T=adapters/su_teacher/model.pt
A=adapters/undo_tafc_a06_tf
[ -f "$A/model.pt" ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py --teacher "$T" \
  --alpha 0.6 --beta 0.1 --protect retain_fisher --steps 12000 --log-every 12000 \
  --retain-oversample 3 --seed 0 --out "$A" > outputs/log_undo_tafc_a06_tf.txt 2>&1
i=0
# complete max-over-lr at the STRONGER attacker budget, and add an intermediate budget, so the
# robustness claim is stated as a function of attacker data rather than at one point
for lr in 1e-4 3e-4 1e-3; do
  lab="rtt_tafc_a06_tf08_lr${lr}"
  [ -f "outputs/rtt_${lab}.json" ] && continue
  g=$(( i % 2 + 2 )); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$A/model.pt" --t-frac 0.8 \
    --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
for lr in 3e-5 1e-5; do
  lab="rtt_tafc_a06_tf065_lr${lr}"
  [ -f "outputs/rtt_${lab}.json" ] && continue
  g=$(( i % 2 + 2 )); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$A/model.pt" --t-frac 0.65 \
    --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
rm -f "$A/model.pt"
echo "[h29] done"
