#!/bin/bash
# GIFT primary arm (beta=1.0 per PREREG_H81) on GPU 0 alone. run_gift.sh waits for BOTH GPUs 0 and 1
# and GPU 1 is holding the H80 lambda_meta attacks, so GPU 0 would otherwise sit idle. Labels are
# claimed with mkdir, so whichever script reaches an arm first owns it and the other skips.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p .labels logs
for B in 1.0 0.1 10.0; do
  L="gift_church_b${B}"
  mkdir ".labels/$L" 2>/dev/null || continue
  if [ ! -f "adapters/$L/unet.pt" ]; then
    venv_t2i/bin/python scripts/erase_gift.py --concept church --beta "$B" \
      --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  fi
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- no unet.pt, NOT a kill-rule result; see logs/$L.log" >>logs/gift.log
    continue
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
    --label "$L" --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo FAIL)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo FAIL)
  echo "$L as-is: target=$T retain8=$R" >>logs/gift.log
done
echo "GIFT GPU0 DONE $(date -Is)" >>logs/gift.log
