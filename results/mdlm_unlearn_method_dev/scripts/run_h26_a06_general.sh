#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# alpha=0.6 survives five attack lrs on the EMDU teacher. Two things remain before it can be
# called the selected point: its BEHAVIOURAL profile (TV->oracle, six-cell) and whether the
# result is teacher-general or specific to EMDU.
run () {
  local lab=$1
  local teach=$2
  local s=$3
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/arm_${lab}.json" ] && [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$teach" --alpha 0.6 --beta 0.1 --protect retain_fisher --steps 12000 \
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
# behavioural profile for the EMDU-teacher point + teacher generality on AFC and MDU
run undo_temd_a06_tv  adapters/su_teacher_emdu/model.pt 0
run undo_tafc_a06_s0  adapters/su_teacher/model.pt      0
wait
run undo_tafc_a06_s1  adapters/su_teacher/model.pt      1
run undo_tmdu_a06_s0  adapters/su_teacher_mdu/model.pt  0
wait
echo "[h26] done"
