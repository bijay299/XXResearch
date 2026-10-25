#!/bin/bash
# UTRAP per PREREG_H84. lam sweep with K=8 primary, plus a K=4 point to test whether the unroll
# depth is what matters (FINDING59 identified one-step-vs-multi-step as the operative difference).
# K5 of the prereg: any as-is point gets >= 3 seeds or is labelled a single build.
# K4: if max-over-valid H lands BELOW the ceiling the result is PROVISIONAL and the full escalation
# (LoRA, TI, DreamBooth, 3x budget on the strongest family) runs BEFORE any headline.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }
build() {  # $1 lam  $2 K  $3 gpu
  local L="utrap_church_l$1k$2"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_utrap.py --concept church \
    --teacher adapters/frp_church_r3_l05/unet.pt --lam "$1" --inner-k "$2" \
    --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/utrap.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/utrap.log
}
( build 1.0 8 0; build 3.0 8 0; echo "UTRAP A DONE $(date -Is)" >>logs/utrap.log ) &
( build 0.3 8 1; build 1.0 4 1; echo "UTRAP B DONE $(date -Is)" >>logs/utrap.log ) &
wait
# attack every arm clearing K1 on the FROZEN SIX-POINT GRID -- unchanged protocol, no scoring changes
for A in utrap_church_l1.0k8 utrap_church_l0.3k8 utrap_church_l3.0k8 utrap_church_l1.0k4; do
  [ -f "outputs/gen_$A.json" ] || { echo "$A NO EVAL OUTPUT -- no K1 verdict" >>logs/utrap.log; continue; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$A retain8=$R below K1 -- destroys the model, not attacked" >>logs/utrap.log; continue; }
  echo "$A retain8=$R clears K1 -- frozen grid" >>logs/utrap.log
  i=0
  for cfg in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $cfg; SD=$1; LR=$2
    K="H84_${A}_sd${SD}_lr${LR}"
    claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=$((i % 2)); i=$((i+1))
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" >>logs/utrap_atk.log 2>&1
  done
done
echo "UTRAP ATK DONE $(date -Is)" >>logs/utrap.log
