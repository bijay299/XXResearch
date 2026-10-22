#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
run () { local T=$1 lab=$2 al=$3 pr=$4 s=$5 g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_undo.py \
      --teacher "$T" --alpha "$al" --beta 0.1 --protect "$pr" --seed "$s" --steps 3000 \
      --log-every 3000 --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
# plain + shuffled on the MDU teacher at a0.5, so that line has the SAME three-way comparison
for s in 0 1 2; do run adapters/su_teacher_mdu/model.pt "undo_mp_a0p5$([ $s -gt 0 ] && echo _s$s)" 0.5 none $s; done
wait
for s in 0 1 2; do run adapters/su_teacher_mdu/model.pt "undo_mr_a0p5$([ $s -gt 0 ] && echo _s$s)" 0.5 random $s; done
wait
run adapters/su_teacher_mdu/model.pt "undo_pm_a0p5_s2" 0.5 retain_fisher 2
wait
echo "[gen2] done"
