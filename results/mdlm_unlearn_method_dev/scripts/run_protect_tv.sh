#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
for lab in undo_p_a0p3 undo_p_a0p5 undo_r_a0p3 undo_r_a0p5 undo_pm_a0p5 undo_mr_a0p5 undo_mp_a0p5; do
  [ -f "outputs/arm_$lab.json" ] && continue
  ck="adapters/$lab/model.pt"
  if [ ! -f "$ck" ]; then echo "[tv] $lab weights reclaimed, retraining"; 
    case $lab in
      undo_p_*) T=adapters/su_teacher/model.pt; PR=retain_fisher;;
      undo_r_*) T=adapters/su_teacher/model.pt; PR=random;;
      undo_pm_*) T=adapters/su_teacher_mdu/model.pt; PR=retain_fisher;;
      undo_mr_*) T=adapters/su_teacher_mdu/model.pt; PR=random;;
      undo_mp_*) T=adapters/su_teacher_mdu/model.pt; PR=none;;
    esac
    AL=0.3; [[ $lab == *a0p5 ]] && AL=0.5
    g=$(( i % 2 + 2 )); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py --teacher "$T" --alpha $AL --beta 0.1 \
      --protect "$PR" --steps 3000 --log-every 3000 --out "adapters/$lab" \
      > "outputs/log_${lab}_tv.txt" 2>&1 || continue
  fi
  g=$(( i % 2 + 2 )); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
    --label "$lab" --infill --ref-profile outputs/arm_h18_oracle.json \
    > "outputs/log_part_$lab.txt" 2>&1 || echo "[tv] EVAL FAIL $lab"
  rm -f "adapters/$lab/model.pt"
done
echo "[tv] done"
