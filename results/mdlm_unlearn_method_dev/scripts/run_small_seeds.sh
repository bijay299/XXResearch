#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
GPU="${GPU:-2}"; ORACLE=outputs/arm_h18_oracle.json
for s in 1 2 3 4; do
  for spec in "afc 1.0" "mdu 0.5"; do
    set -- $spec; arm=$1; lf=$2
    lab="su_${arm}_f${lf/./p}_s${s}"
    [ -f "outputs/arm_$lab.json" ] && continue
    echo "[seeds] $lab"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_small_unlearn.py --arm "$arm" \
      --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 750 --seed "$s" \
      --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || { echo "[seeds] TRAIN FAIL $lab"; continue; }
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1 || echo "[seeds] RANK FAIL $lab"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" --infill --ref-profile "$ORACLE" \
      > "outputs/log_part_$lab.txt" 2>&1 || echo "[seeds] EVAL FAIL $lab"
    rm -f "adapters/$lab/model.pt"
  done
done
echo "[seeds] done"
