#!/bin/bash
# THE MANDATE-CRITICAL EXPERIMENT. FINDING_four_point_frontier S4 names this as the gap that keeps
# the contribution claim from being fully defensible:
#
#   "AFC's own lambda sweep is NOT at n=12 -- only lam=1 is. A fair frontier-vs-frontier comparison
#    would require it, and it is not claimed here."
#
# Right now the result compares ONE AFC point (lam=1, n=12) against MDU's CURVE (lam=1/2/3/4, n=12).
# That supports "AFC sits outside MDU's frontier" but not "AFC's frontier dominates MDU's frontier",
# which is what the mandate actually asks for: outperforming the strongest existing method on a
# clearly defined forgetting-utility trade-off.
#
# This builds AFC at lam_f = 2 and 4, twelve seeds each, so the comparison becomes curve-vs-curve.
# The existing single-seed AFC points (afc_f2 0.2338/0.5496, afc_f4 0.2589/0.5410) are n=1 and are
# deliberately NOT used -- FINDING62's retraction was caused by exactly that kind of single build,
# and the whole point of this run is to not repeat it.
#
# Pre-committed reading, before any arm exists:
#   AFC's curve lies below/left of MDU's at matched retention  -> the mandate's bar is met on this
#       trade-off and the contribution is defensible as a frontier result;
#   the curves overlap or cross                                -> AFC is a point on MDU's frontier,
#       not an improvement to it, and the claim narrows to the single matched-disclosure retention
#       win that already survives.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
mkdir -p .labels logs
# stay inside the two-GPU cap: wait for the lam=2 MDU gap run to finish first
while pgrep -f "mdu_l2_s[0-9]*_fr" >/dev/null; do sleep 120; done
build() { local LAM=$1 S=$2 G=$3 L="afc_l${LAM}_s${S}_fr"
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$G
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_h16.py --mode none --lam-forget "$LAM" \
    --lam-retain 1 --seed "$S" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
  echo "$L done $(date -Is)" >>logs/afc_frontier.log; }
( for S in 0 1 2 3 4 5; do build 2 "$S" 0; done
  for S in 0 1 2 3 4 5; do build 4 "$S" 0; done
  echo "AFCF A DONE $(date -Is)" >>logs/afc_frontier.log ) &
( for S in 6 7 8 9 10 11; do build 2 "$S" 1; done
  for S in 6 7 8 9 10 11; do build 4 "$S" 1; done
  echo "AFCF B DONE $(date -Is)" >>logs/afc_frontier.log ) &
wait
