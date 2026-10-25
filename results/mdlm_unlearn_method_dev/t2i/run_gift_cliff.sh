#!/bin/bash
# The finer sweep (beta 2/3/5/10) found NOTHING usable -- beta=2.0 already collapses retention to
# 0.2750 (K1 FAIL) while beta=1.0 holds 0.7833. So unlike H80, where the finer sweep found a better
# arm at lambda=0.7, GIFT's window really is a single point and the trade-off has a CLIFF between
# beta 1 and 2. This narrows it: if any beta in (1,2) reaches deeper erasure while still clearing
# K1, that is GIFT's best shot at the mandate's bar and it should be measured before FINDING58's
# "usable window is one point" is treated as final.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
mkdir -p .labels logs
for B in 1.5 1.2; do
  L="gift_church_b${B}"
  mkdir ".labels/$L" 2>/dev/null || continue
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_gift.py --concept church \
    --beta "$B" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result" >>logs/gift_cliff.log; continue
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/gift_cliff.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/gift_cliff.log; continue; }
  for SD in 0 1 6; do
    K="H81C_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py \
      --concept church --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" \
      >>logs/gift_cliff_atk.log 2>&1
  done
done
echo "GIFT CLIFF DONE $(date -Is)" >>logs/gift_cliff.log
