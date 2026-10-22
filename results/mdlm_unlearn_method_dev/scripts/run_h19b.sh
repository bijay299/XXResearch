#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
run () {
  local lab=$1
  local steps=$2
  local extra=$3
  local s=$4
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher adapters/su_teacher/model.pt --alpha 0.5 --beta 0.1 --protect retain_fisher \
      --steps "$steps" --log-every "$steps" --seed "$s" $extra --out "adapters/$lab" \
      > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
# map the repair-budget curve: we have 3000 and 12000, fill 1500 and 6000
run undo_b1500 1500 "" 0 ; run undo_b6000 6000 "" 0 ; wait
# seeds for the two 12k arms, which are currently n=1 and carry the headline
for s in 1 2; do run undo_h19_ret_s$s 12000 "--retain-oversample 3" $s; done; wait
for s in 1 2; do run undo_h19_long_s$s 12000 "" $s; done; wait
echo "[h19b] done"
