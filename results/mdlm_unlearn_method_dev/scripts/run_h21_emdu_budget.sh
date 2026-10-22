#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# The EMDU teacher is the best found (retain 0.6908, recovery 0.1161 at 12k). The AFC-teacher
# budget curve showed recovery rises with repair, so a SHORTER budget on this teacher may sit at a
# better point. Sweep 3k and 6k, n=2, before claiming 12k is the right operating point.
run () {
  local lab=$1
  local steps=$2
  local s=$3
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher adapters/su_teacher_emdu/model.pt --alpha 0.5 --beta 0.1 --protect retain_fisher \
      --steps "$steps" --log-every "$steps" --retain-oversample 3 --seed "$s" \
      --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || exit 0
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
for s in 0 1; do run undo_temd_6k_s$s 6000 $s; done; wait
for s in 0 1; do run undo_temd_3k_s$s 3000 $s; done; wait
echo "[h21] done"
