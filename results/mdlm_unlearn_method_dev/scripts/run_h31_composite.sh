#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
TR=adapters/small_ref/model.pt          # retain+general teacher: the model being unlearned
TF=adapters/su_teacher_ga/model.pt      # forget teacher: GA, rank-forget 0.0850 (below chance)
i=0
run () {
  local lab=$1
  local s=$2
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/arm_${lab}.json" ] && [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$TR" --teacher-forget "$TF" --alpha 0.6 --beta 0.1 --protect retain_fisher \
      --steps 12000 --log-every 12000 --retain-oversample 3 --seed "$s" --out "adapters/$lab" \
      > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" --infill --ref-profile outputs/arm_h18_oracle.json \
      > "outputs/log_part_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1 ) &
  [ $((i % 2)) -eq 0 ] && wait
}
for s in 0 1 2; do run h31_comp_s$s $s; done
wait
# full lr grid + both attacker budgets on seed 0, per the pre-registration
A=adapters/h31_comp_s0
j=0
for tf in 0.5 0.8; do
  for lr in 1e-5 1e-4 3e-4 1e-3; do
    tag=$(echo $tf | tr -d '.')
    lab="rtt_h31_tf${tag}_lr${lr}"
    [ -f "outputs/rtt_${lab}.json" ] && continue
    g=$(( j % 2 + 2 )); j=$((j+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$A/model.pt" --t-frac "$tf" \
      --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
    [ $((j % 2)) -eq 0 ] && wait
  done
done
wait
for lr in 3e-5; do
  lab="rtt_h31_tf08_lr${lr}"
  [ -f "outputs/rtt_${lab}.json" ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_rtt.py \
    --arm "$A/model.pt" --t-frac 0.8 --steps 600 --lr "$lr" --label "$lab" \
    > "outputs/log_$lab.txt" 2>&1
done
rm -f adapters/h31_comp_s*/model.pt
echo "[h31] done"
