#!/bin/bash
# FINDING63 S4 names exactly two unfinished things, both blocked on disk at the time. /data has
# recovered to ~30 GB free, so both are now runnable:
#
#   (a) "K's effect is therefore UNMEASURED" -- the K=4 arm died mid-build. Tested here at K=4 AND
#       K=16, both at the usable lambda=0.3, so the unroll depth is probed in both directions. If
#       K=16 also lands at H=1.000, depth is ruled out as the missing ingredient rather than left
#       open -- which is the honest way to close my own hypothesis.
#   (b) "the retain floor was already missed before the attacks (0.7333 < 0.85)" -- lambda=0.15
#       halves the immunization pressure, aiming to clear 0.85 so an arm exists that COULD meet the
#       mandate's bar if it resisted.
#
# Each build writes a ~1.7 GB adapter. A disk guard refuses to start an arm below 8 GB free rather
# than risk the shared volume -- /data hit 100% earlier today and killed two builds mid-run.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
build() {  # $1 lam  $2 K  $3 gpu
  local L="utrap_church_l$1k$2"
  mkdir ".labels/$L" 2>/dev/null || return 0
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 8 ]; then
    echo "$L SKIPPED -- only ${FREE}G free on /data, refusing to risk the shared volume" >>logs/utrap2.log
    return 0
  fi
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_utrap.py --concept church \
    --teacher adapters/frp_church_r3_l05/unet.pt --lam "$1" --inner-k "$2" \
    --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/utrap2.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/utrap2.log
}
( build 0.3 16 0; build 0.15 8 0; echo "U2 A DONE $(date -Is)" >>logs/utrap2.log ) &
( build 0.3 4 1;                  echo "U2 B DONE $(date -Is)" >>logs/utrap2.log ) &
wait
