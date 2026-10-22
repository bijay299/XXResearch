#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# The headline comparison (H31b +0.0559 vs selected point +0.2299 at t_frac 0.8) is n=1 per lr.
# Replicate the binding budget on both arms before the frontier claim is made.
# both arms' weights were reclaimed after evaluation; rebuild them
TR=adapters/small_ref/model.pt
[ -f adapters/h31b_comp_s0/model.pt ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py \
  --teacher "$TR" --teacher-forget adapters/base_afc_f8p0_s0/model.pt --alpha 0.6 --beta 0.1 \
  --protect retain_fisher --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 \
  --out adapters/h31b_comp_s0 > outputs/log_rebuild_h31b.txt 2>&1
[ -f adapters/undo_tafc_a06_tf/model.pt ] || CUDA_VISIBLE_DEVICES=3 $PY scripts/train_small_undo.py \
  --teacher adapters/su_teacher/model.pt --alpha 0.6 --beta 0.1 --protect retain_fisher \
  --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 \
  --out adapters/undo_tafc_a06_tf > outputs/log_rebuild_sel.txt 2>&1

i=0
atk () {
  local arm=$1 lab=$2 lr=$3 s=$4 g
  g=$(( i % 2 + 2 )); i=$((i+1))
  [ -f "outputs/rtt_${lab}.json" ] && return
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$arm" --t-frac 0.8 \
    --steps 600 --lr "$lr" --seed "$s" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
}
for s in 1 2; do
  for lr in 3e-5 1e-5; do
    atk adapters/h31b_comp_s0/model.pt "rtt_h31b_tf08_lr${lr}_s${s}" "$lr" "$s"
    atk adapters/undo_tafc_a06_tf/model.pt "rtt_sel_tf08_lr${lr}_s${s}" "$lr" "$s"
  done
done
wait
rm -f adapters/h31b_comp_s0/model.pt adapters/undo_tafc_a06_tf/model.pt
echo "[h32] done"
