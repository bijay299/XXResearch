#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
run () { local lab=$1 pr=$2 s=$3 g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher adapters/su_teacher_mdu/model.pt --alpha 0.3 --beta 0.1 --protect "$pr" \
      --seed "$s" --steps 3000 --log-every 3000 --out "adapters/$lab" \
      > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
# completes the teacher x alpha grid: is "non-uniformity alone suffices at alpha=0.3"
# teacher-general, or specific to the AFC teacher?
for s in 0 1 2; do run "undo_mp_a0p3$([ $s -gt 0 ] && echo _s$s)" none   $s; done; wait
for s in 0 1 2; do run "undo_mr_a0p3$([ $s -gt 0 ] && echo _s$s)" random $s; done; wait
for s in 0 1 2; do run "undo_pm_a0p3$([ $s -gt 0 ] && echo _s$s)" retain_fisher $s; done; wait
echo "[gen3] done"
