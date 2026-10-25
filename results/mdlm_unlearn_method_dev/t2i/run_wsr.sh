#!/bin/bash
# Weight-space-regularised erasure: the one thing FINDING60 left untested (their METHOD, not their
# predictor). lam_reg sweep -- the anchor strength is the whole method, and no value is transferable
# from a vision-classifier paper, so it is swept and the full sweep reported.
#   lam_reg = 0    is the ABLATION: plain erase+retain with no anchor. Required, or any effect of
#                  the anchor is unattributable.
# Arms clearing K1 (retain8 >= 0.60) are attacked on the frozen grid.
#
# Honest prior, recorded before the run: FINDING60 found GIFT at L2 = 0.18 -- all but identical to
# base -- reversed on 22 valid attacks. So weight-space proximity alone is already known not to
# confer resistance here, and I expect this to be reversed too. It is run because "their predictor
# fails" and "their method fails" are different claims and only one has been measured.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }
build() {  # $1 lam_reg  $2 gpu
  local L="wsr_church_lr$1"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$2
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_wsr.py --concept church \
    --lam-reg "$1" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/wsr.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/wsr.log
}
( build 0 0;    build 10 0;  echo "WSR A DONE $(date -Is)" >>logs/wsr.log ) &
( build 1 1;    build 100 1; echo "WSR B DONE $(date -Is)" >>logs/wsr.log ) &
wait
# attack every arm that clears K1
for LR in 0 1 10 100; do
  A="wsr_church_lr${LR}"
  [ -f "outputs/gen_$A.json" ] || { echo "$A NO EVAL OUTPUT -- no K1 verdict" >>logs/wsr.log; continue; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$A retain8=$R below K1 -- not attacked" >>logs/wsr.log; continue; }
  i=0
  for cfg in "0 1e-5" "1 1e-5" "6 1e-5"; do
    set -- $cfg; SD=$1; L=$2
    K="H83_${A}_sd${SD}_lr${L}"
    claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=$((i % 2)); i=$((i+1))
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$L" --label "$K" >>logs/wsr_atk.log 2>&1
  done
done
echo "WSR ATK DONE $(date -Is)" >>logs/wsr.log
