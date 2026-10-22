#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
# Only MDU lam0.5 has been attacked at the STRONG budget. The headline comparison -- selected
# point +0.2299 vs the anchor band -- mixes t_frac 0.5 baselines with a t_frac 0.8 treatment arm.
# Attack the remaining families at t_frac 0.8 so the comparison is budget-matched.
i=0
run () {
  local arm=$1
  local lf=$2
  local lab="base_${arm}_f${lf/./p}_tf08_s${S:-0}"
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "outputs/rtt_rtt_${lab}.json" ] && exit 0
    D=adapters/$lab
    [ -f "$D/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_unlearn.py \
      --arm "$arm" --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 1500 --seed $S \
      --out "$D" > "outputs/log_$lab.txt" 2>&1 || exit 0
    for lr in 3e-5 1e-5; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "$D/model.pt" --t-frac 0.8 \
        --steps 600 --lr "$lr" --label "rtt_${lab}_lr${lr}" \
        > "outputs/log_rtt_${lab}_lr${lr}.txt" 2>&1
    done
    rm -f "$D/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
for S in 1 2; do export S; run afc 1.0; run emdu 1.0; wait; done
wait
for S in 1 2; do export S; run ga 0.5; run mdu_fam 1.0; wait; done
wait
echo "[h34b] done"
