#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
T=adapters/su_teacher/model.pt
# The headline robustness numbers are SINGLE-SEED. n=3 seeds already produced a retracted
# stability claim on 8B (RESULTS.md), so the central claim gets replicated before it is relied
# on. RTT at lr 3e-5, which the sweep showed is adversarially optimal for every arm.
i=0
for s in 1 2; do
  for al in 0.3 0.5; do
    lab="undo_s_a${al/./p}_s${s}"
    g=$(( i % 2 + 2 )); i=$((i+1))
    (
      if [ ! -f "adapters/$lab/model.pt" ]; then
        CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py --teacher "$T" --alpha "$al" \
          --beta 0.1 --steps 3000 --log-every 3000 --seed "$s" --out "adapters/$lab" \
          > "outputs/log_$lab.txt" 2>&1 || exit 0
      fi
      CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
        --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
        --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" \
        > "outputs/log_rtt_$lab.txt" 2>&1
      rm -f "adapters/$lab/model.pt"
    ) &
    [ $((i % 2)) -eq 0 ] && wait
  done
done
wait
# retain numbers for the MDU-teacher arms, so the robustness/retain frontier is complete
for lab in undo_m_a0p0 undo_m_a0p3 undo_m_a0p5 su_teacher_mdu; do
  [ -f "outputs/rank_$lab.json" ] && continue
  [ -f "adapters/$lab/model.pt" ] || continue
  CUDA_VISIBLE_DEVICES=2 $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
    --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
done
echo "[undo_seeds] done"
