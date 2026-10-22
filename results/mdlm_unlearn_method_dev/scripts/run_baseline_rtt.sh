#!/bin/bash
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
i=0
# RTT so far covers only the two TEACHERS. Any frontier claim against published methods needs
# each anchor family attacked under the same gated, lr-3e-5 protocol. Weights were reclaimed, so
# retrain (90s) then attack.
run () {
  # separate `local` statements: bash expands all words of a single `local` BEFORE assigning,
  # so ${arm} inside the same statement is unbound under `set -u`.
  local arm=$1
  local lf=$2
  local lab="base_${arm}_f${lf/./p}"
  local g
  g=$(( i % 2 + 2 )); i=$((i+1))
  ( [ -f "adapters/$lab/model.pt" ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_unlearn.py \
      --arm "$arm" --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 1500 \
      --out "adapters/$lab" > "outputs/log_$lab.txt" 2>&1 || exit 0
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rank_probe_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" > "outputs/log_rank_$lab.txt" 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY scripts/train_small_rtt.py --arm "adapters/$lab/model.pt" \
      --steps 600 --lr 3e-5 --label "rtt_${lab}_lr3e-5" > "outputs/log_rtt_$lab.txt" 2>&1
    rm -f "adapters/$lab/model.pt" ) &
  [ $((i % 2)) -eq 0 ] && wait
}
run afc 1.0; run mdu 0.5
wait
run emdu 1.0; run mdu_fam 1.0
wait
run ga 0.5
wait
echo "[baseline_rtt] done"
