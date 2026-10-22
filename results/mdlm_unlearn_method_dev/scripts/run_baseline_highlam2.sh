#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# FINDING25 says every anchor family is non-robust, but its own limitation is that lambda was
# swept over only {0.5,1,2}. If a HIGH lambda makes an anchor method robust, FINDING25 is WRONG.
# This is the test of our own claim, so it runs before the claim is relied on further.
run () {
  local arm=$1
  local lf=$2
  local s=$3
  local lab="base_${arm}_f${lf/./p}_s${s}"
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}_lr3e-5.json" ] && exit 0
    [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_unlearn.py \
      --arm "$arm" --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 1500 --seed "$s" \
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
for s in 0 1 2; do run emdu 4.0 $s; run ga 4.0 $s; wait; done
for s in 0 1 2; do run emdu 8.0 $s; run ga 8.0 $s; wait; done
# complete mdu_fam to n=3, the one n=1 cell in FINDING25
for s in 0 1 2; do run mdu_fam 8.0 $s; done
wait
echo "[highlam] done"
