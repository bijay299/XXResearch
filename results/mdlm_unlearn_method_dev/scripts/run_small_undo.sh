#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
GPU="${GPU:-3}"; ORACLE=outputs/arm_h18_oracle.json
T=adapters/su_teacher/model.pt
if [ ! -f "$T" ]; then
  echo "[undo] retraining teacher (afc lam=2, best static arm)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_small_unlearn.py --arm afc --lam-forget 2.0 \
    --lam-retain 1.0 --steps 1500 --log-every 750 --out adapters/su_teacher \
    > outputs/log_su_teacher.txt 2>&1 || { echo "[undo] TEACHER FAILED"; exit 1; }
fi
for al in 0.0 0.1 0.3 0.5; do
  lab="undo_s_a${al/./p}"
  if [ ! -f "outputs/rank_$lab.json" ]; then
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_small_undo.py \
      --teacher "$T" --alpha "$al" --beta 0.1 --steps 3000 --log-every 1000 \
      --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || { echo "[undo] FAIL $lab"; continue; }
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1 || echo "[undo] RANK FAIL $lab"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" --infill --ref-profile "$ORACLE" \
      > "outputs/log_part_$lab.txt" 2>&1 || echo "[undo] EVAL FAIL $lab"
  fi
done
echo "[undo] done"
