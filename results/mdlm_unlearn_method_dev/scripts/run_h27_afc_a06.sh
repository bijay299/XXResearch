#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
T=adapters/su_teacher/model.pt
i=0
# The AFC teacher at alpha=0.6 dominates EMDU on all three axes (retain 0.6525 vs 0.5850,
# recovery 0.0139 vs 0.0287, TV 0.0738 vs 0.3225). FINDING26 named the EMDU point as best and
# attached the lr sweep to it, so both the seeds and the sweep must be redone on THIS arm.
run () {
  local lab=$1
  local s=$2
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && [ -f "outputs/arm_${lab}.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$T" --alpha 0.6 --beta 0.1 --protect retain_fisher --steps 12000 \
      --log-every 12000 --retain-oversample 3 --seed "$s" --out "adapters/$lab" \
      > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" --infill --ref-profile outputs/arm_h18_oracle.json \
      > "outputs/log_part_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
for s in 2 3 4; do run undo_tafc_a06_s$s $s; done
wait
# adversarial lr sweep on THIS arm -- the published -0.0096 belongs to the wrong teacher
A=adapters/undo_tafc_a06_lrsweep
if [ ! -f "$A/model.pt" ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_undo.py --teacher "$T" --alpha 0.6 --beta 0.1 \
    --protect retain_fisher --steps 12000 --log-every 12000 --retain-oversample 3 --seed 0 \
    --out "$A" > outputs/log_undo_tafc_a06_lrsweep.txt 2>&1
fi
j=0
for lr in 1e-5 1e-4 3e-4 1e-3; do
  lab="rtt_tafc_a06_lr${lr}"
  [ -f "outputs/rtt_${lab}.json" ] && continue
  g=$(( j % 2 + 2 )); j=$((j+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$A/model.pt" \
    --steps 600 --lr "$lr" --label "$lab" > "outputs/log_$lab.txt" 2>&1 &
  [ $((j % 2)) -eq 0 ] && wait
done
wait
rm -f "$A/model.pt"
echo "[h27] done"
