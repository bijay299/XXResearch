#!/bin/bash
# GIFT (arXiv 2507.13598) per PREREG_H81: beta sweep 0.1 / 1.0 / 10.0 with 1.0 pre-declared as the
# primary, then evaluate, then attack every arm clearing K1 (retain8 >= 0.60) on the frozen grid.
# Build failures are reported as BUILD FAILED, never as a measured kill-rule result -- the bug that
# produced logs/t2i_next_FALSE_K2_VERDICTS.log.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }
gpu_free() { local u; u=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ -n "$u" ] && [ "$u" -lt 1000 ]; }
wait_for_gpu() { while ! gpu_free "$1"; do sleep 60; done; }

build() {  # $1 beta  $2 gpu
  local B=$1 GPU=$2 L
  L="gift_church_b${B}"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$GPU
  if [ ! -f "adapters/$L/unet.pt" ]; then
    venv_t2i/bin/python scripts/erase_gift.py --concept church --beta "$B" \
      --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  fi
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- no unet.pt, NOT a kill-rule result; see logs/$L.log" >>logs/gift.log
    return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
    --label "$L" --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
}

wait_for_gpu 0; wait_for_gpu 1
( build 1.0 0; build 10.0 0; echo "GIFT A DONE $(date -Is)" >>logs/gift.log ) &
( build 0.1 1;               echo "GIFT B DONE $(date -Is)" >>logs/gift.log ) &
wait

for B in 1.0 0.1 10.0; do
  A="gift_church_b${B}"
  [ -f "outputs/gen_$A.json" ] || { echo "$A NO EVAL OUTPUT -- no K1 verdict possible" >>logs/gift.log; continue; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['target_suppression_acc'])")
  echo "$A as-is: target=$T retain8=$R" >>logs/gift.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$A retain8=$R MEASURED below K1 0.60 -- destroys the model, not attacked" >>logs/gift.log; continue; }
  for cfg in "0 1e-5 0" "1 1e-5 0" "6 1e-5 0" "0 5e-5 1" "1 5e-5 1" "0 1e-4 1"; do
    set -- $cfg; SD=$1; LR=$2; G=$3
    K="H81_${A}_sd${SD}_lr${LR}"
    claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=$G
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py \
      --concept church --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" \
      >>logs/gift_atk.log 2>&1
  done
done
echo "GIFT ATK DONE $(date -Is)" >>logs/gift.log
