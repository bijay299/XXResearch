#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
run () {
  local lab=$1
  local teach=$2
  local extra=$3
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$teach" --alpha 0.5 --beta 0.1 --protect retain_fisher \
      --steps 12000 --log-every 3000 $extra --out "adapters/$lab" \
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
# A: the test. B: retain-oversampled pool. C: CEILING, oracle teacher -- never a method.
run undo_h19_long adapters/su_teacher/model.pt     ""
run undo_h19_ret  adapters/su_teacher/model.pt     "--retain-oversample 3"
wait
run undo_h19_ceil adapters/small_oracle/model.pt   ""
wait
echo "[h19] done"
