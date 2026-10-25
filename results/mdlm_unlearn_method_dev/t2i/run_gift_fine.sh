#!/bin/bash
# FINDING58 section 2: GIFT's usable window is a single point. beta=1.0 gives target 0.4667 at
# retain8 0.7833; beta=10.0 destroys the model (retain 0.0333); beta=0.1 does nothing. So the whole
# erasure/utility trade lives between 1 and 10, unmeasured -- and the H80 finer sweep is a direct
# precedent for why that matters: it found lambda=0.7 at target 0.0000 / retain 0.7000, a strictly
# better operating point the coarse grid missed, and retention there was NON-MONOTONE.
#
# If a beta in (1,10) reaches deeper erasure while still clearing K1, GIFT gets its fairest shot at
# the mandate's bar before the finding is treated as settled. Any arm clearing K1 is then attacked
# on the frozen six-point grid.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }

build() {  # $1 beta  $2 gpu
  local B=$1 L="gift_church_b$1"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$2
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_gift.py --concept church \
    --beta "$B" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/gift_fine.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
}

( build 2.0 0; build 5.0 0; echo "FINE A DONE $(date -Is)" >>logs/gift_fine.log ) &
( build 3.0 1;              echo "FINE B DONE $(date -Is)" >>logs/gift_fine.log ) &
wait

for B in 2.0 3.0 5.0; do
  A="gift_church_b${B}"
  [ -f "outputs/gen_$A.json" ] || { echo "$A NO EVAL OUTPUT -- no K1 verdict possible" >>logs/gift_fine.log; continue; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['target_suppression_acc'])")
  echo "$A as-is: target=$T retain8=$R" >>logs/gift_fine.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$A retain8=$R MEASURED below K1 -- not attacked" >>logs/gift_fine.log; continue; }
  i=0
  for cfg in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $cfg; SD=$1; LR=$2
    K="H81F_${A}_sd${SD}_lr${LR}"
    claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=$((i % 2)); i=$((i+1))
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py \
      --concept church --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" \
      >>logs/gift_fine_atk.log 2>&1
  done
done
echo "FINE ATK DONE $(date -Is)" >>logs/gift_fine.log
