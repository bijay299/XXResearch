#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# build the low-residual teachers if absent
for spec in "ga 0.5 su_teacher_ga" "emdu 1.0 su_teacher_emdu"; do
  set -- $spec
  [ -f "adapters/$3/model.pt" ] || CUDA_VISIBLE_DEVICES=2 $PY scripts/train_small_unlearn.py \
    --arm "$1" --lam-forget "$2" --lam-retain 1.0 --steps 1500 --log-every 1500 \
    --out "adapters/$3" > "outputs/log_$3.txt" 2>&1
done
run () {
  local lab=$1
  local teach=$2
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$teach" --alpha 0.5 --beta 0.1 --protect retain_fisher --steps 12000 \
      --log-every 12000 --retain-oversample 3 --out "adapters/$lab" \
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
run undo_tga_12k  adapters/su_teacher_ga/model.pt
run undo_temd_12k adapters/su_teacher_emdu/model.pt
wait
echo "[h20] done"
