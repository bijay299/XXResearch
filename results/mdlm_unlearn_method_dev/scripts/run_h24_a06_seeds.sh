#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# wait for the budget sweep to vacate the GPUs
# NOT pgrep: the shell that creates this file via heredoc carries the pattern in its own
# command line, so pgrep matched the launcher and the loop spun forever. Check GPU compute
# apps instead, which cannot self-match.
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ]; do sleep 20; done
i=0
# alpha=0.7 collapsed earlier (retain 0.13-0.14, all nine attacks REJECTED) -- but that was at
# 3000 repair steps. The budget curve since showed retain is budget-bound, so that collapse may
# itself have been an under-repair artifact. Retest alpha=0.7 with 4x the repair and the best
# teacher. If retain recovers, a HIGHER-damage/longer-repair point may dominate everything found.
run () {
  local lab=$1
  local al=$2
  local s=$3
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher adapters/su_teacher_emdu/model.pt --alpha "$al" --beta 0.1 \
      --protect retain_fisher --steps 12000 --log-every 12000 --retain-oversample 3 --seed "$s" \
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

for s in 2 3 4; do run undo_temd_a06_12k_s$s 0.6 $s; done; wait
echo "[a06seeds] done"
