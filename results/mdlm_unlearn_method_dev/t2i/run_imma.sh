#!/bin/bash
# IMMA per PREREG_H82. Two arms: the primary on frp_church_r3_l05 (same base as H80/H81, so the
# three bi-level defences are directly comparable) and a secondary on ESD-church, which matches
# IMMA's own S5.1 setting exactly and is the one to believe if the two disagree.
# PRIMARY measurement is LoRA, not full-UNet -- LoRA is the adversary IMMA is immunized against.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
build() {  # $1 label  $2 teacher  $3 gpu
  mkdir ".labels/$1" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "adapters/$1/unet.pt" ] || venv_t2i/bin/python scripts/erase_imma.py --concept church \
    --teacher "$2" --label "$1" --out "adapters/$1" >>"logs/$1.log" 2>&1
  if [ ! -f "adapters/$1/unet.pt" ]; then
    echo "$1 BUILD FAILED -- NOT a kill-rule result; see logs/$1.log" >>logs/imma.log; return 1
  fi
  [ -f "outputs/gen_$1.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$1" \
    --unet "adapters/$1/unet.pt" --target church >>"logs/$1.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$1.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$1.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$1 as-is: target=$T retain8=$R" >>logs/imma.log
}
ESD=$(ls adapters/esd_church/unet.pt 2>/dev/null || ls adapters/esd_church*/unet.pt 2>/dev/null | head -1)
( build imma_church_frp adapters/frp_church_r3_l05/unet.pt 0; echo "IMMA A DONE $(date -Is)" >>logs/imma.log ) &
( [ -n "$ESD" ] && build imma_church_esd "$ESD" 1 || echo "no ESD arm on disk -- secondary skipped" >>logs/imma.log
  echo "IMMA B DONE $(date -Is)" >>logs/imma.log ) &
wait
