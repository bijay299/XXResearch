#!/bin/bash
# FRONTIER FILL -- the honest threat to the retention claim, queued before it can be overlooked.
#
# The matched-retention slice is landing in AFC's favour: MDU lam=1 at 6/12 seeds sits at retain
# 0.5044 (matching AFC's 0.5153) but TV 0.4382, against AFC's 0.2983. Combined with lam=3 (TV 0.3223
# at retain 0.4057), AFC would sit OUTSIDE a two-point MDU frontier.
#
# But a two-point frontier is far too sparse to support "above it", and an EXISTING SINGLE-SEED point
# contradicts the picture: cv_mdu_l4 (MDU lam=4) measured TV 0.2325 at retain 0.4565 -- BETTER TV
# than AFC's 0.2983 at only modestly lower retain. If that holds at n=12, AFC does NOT dominate the
# frontier and the retention claim is the ONLY thing left again. cv_mdu_l2 (TV 0.3492, retain 0.4025)
# is the other gap.
#
# So: MDU lam=2 and lam=4 at n=12 under the explicit recorded settings. Pre-committed reading:
#   lam=4 reproduces ~0.23 TV at ~0.46 retain  -> MDU's frontier passes above/through AFC on the
#        forgetting axis at a modest retention cost, and NO frontier claim can be made for AFC;
#   lam=4 regresses toward 0.30+                -> the single-seed point was a lucky seed, and AFC's
#        position outside the frontier stands on lam=1/2/3/4 at n=12.
# AFC's own lam=2 and lam=4 single-seed points (TV 0.2338/0.2589 at retain 0.550/0.541) are also only
# n=1 and are NOT used in any comparison here.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
mkdir -p .labels
claim() { mkdir ".labels/$1" 2>/dev/null; }

build() {  # $1 lam  $2 seed  $3 gpu
  L="mdu_l$1_s$2_fr"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget "$1" \
    --lam-retain 1 --seed "$2" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
}

# never exceed two of my GPUs: wait for the matched-retention sweep to finish first
while pgrep -f "mdu_l1_s[0-9]*_mr" >/dev/null; do sleep 120; done
# lam=4 first on both workers -- it is the point that could overturn the claim
( for S in 0 1 2 3 4 5; do build 4 "$S" 0; done
  for S in 0 1 2 3 4 5; do build 2 "$S" 0; done
  echo "FR A DONE $(date -Is)" >>logs/frontier.log ) &
( for S in 6 7 8 9 10 11; do build 4 "$S" 1; done
  for S in 6 7 8 9 10 11; do build 2 "$S" 1; done
  echo "FR B DONE $(date -Is)" >>logs/frontier.log ) &
wait
