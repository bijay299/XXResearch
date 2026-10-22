#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# Every robustness number so far uses ONE attacker data budget (t_frac 0.5). A stronger attacker
# with more of the forget set is a different attack surface, and the claim should not rest on a
# single one. Rebuild the selected arm and a baseline, attack both at t_frac 0.8.
T=adapters/su_teacher/model.pt
A=adapters/undo_tafc_a06_tfrac
[ -f "$A/model.pt" ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py --teacher "$T" \
  --alpha 0.6 --beta 0.1 --protect retain_fisher --steps 12000 --log-every 12000 \
  --retain-oversample 3 --seed 0 --out "$A" > outputs/log_undo_tafc_a06_tfrac.txt 2>&1
B=adapters/base_mdu_f0p5_tfrac
[ -f "$B/model.pt" ] || CUDA_VISIBLE_DEVICES=3 $PY scripts/train_small_unlearn.py --arm mdu \
  --lam-forget 0.5 --lam-retain 1.0 --steps 1500 --log-every 1500 --out "$B" \
  > outputs/log_base_mdu_tfrac.txt 2>&1
i=0
for spec in "$A undo_tafc_a06" "$B base_mdu_f0p5"; do
  set -- $spec
  for lr in 3e-5 1e-5; do
    lab="rtt_$2_tf08_lr${lr}"
    [ -f "outputs/rtt_${lab}.json" ] && continue
    g=$(( i % 2 + 2 )); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$1/model.pt" --t-frac 0.8 \
      --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
rm -f "$A/model.pt" "$B/model.pt"
echo "[h28] done"
