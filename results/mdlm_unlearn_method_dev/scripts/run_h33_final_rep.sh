#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
TR=adapters/small_ref/model.pt
[ -f adapters/sel_final/model.pt ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py \
  --teacher adapters/su_teacher/model.pt --alpha 0.6 --beta 0.1 --protect retain_fisher \
  --steps 12000 --log-every 12000 --retain-oversample 3 --seed 5 --out adapters/sel_final \
  > outputs/log_sel_final.txt 2>&1
# pin the conservative (max-over-ALL-attacks) number at the binding budget for the selected point:
# it currently rests on a single seed at +0.2299 while four replicates maxed at +0.1142
i=0
for s in 3 4 5; do
  for lr in 3e-5 1e-5 1e-4; do
    lab="rtt_selfin_tf08_lr${lr}_s${s}"
    [ -f "outputs/rtt_${lab}.json" ] && continue
    g=$(( i % 2 + 2 )); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm adapters/sel_final/model.pt \
      --t-frac 0.8 --steps 600 --lr "$lr" --seed "$s" --label "$lab" \
      > "outputs/log_$lab.txt" 2>&1 &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
rm -f adapters/sel_final/model.pt
echo "[h33] done"
