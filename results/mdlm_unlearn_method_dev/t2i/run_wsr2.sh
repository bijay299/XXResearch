#!/bin/bash
# FINDING61 S4's corrected retry: anchor an ALREADY-ERASED model rather than erasing from scratch.
# Start AT frp_church_r3_l05 (target 0.1333, retain8 0.9417 -- usable), hold its erasure with a
# BOUNDED distillation term, and anchor toward the BASE weights. Same teacher as the H80 and H82
# arms, so this is directly comparable to Meta-Unlearning and IMMA.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }
build() { local L="wsr2_church_lr$1"
  claim "$L" || return 0; export CUDA_VISIBLE_DEVICES=$2
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_wsr.py --concept church \
    --teacher adapters/frp_church_r3_l05/unet.pt --lam-reg "$1" --label "$L" --out "adapters/$L" \
    >>"logs/$L.log" 2>&1
  [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED -- NOT a kill-rule result" >>logs/wsr2.log; return 1; }
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/wsr2.log; }
( build 1 0;   build 100 0; echo "WSR2 A DONE $(date -Is)" >>logs/wsr2.log ) &
( build 10 1;              echo "WSR2 B DONE $(date -Is)" >>logs/wsr2.log ) &
wait
for LR in 1 10 100; do
  A="wsr2_church_lr${LR}"
  [ -f "outputs/gen_$A.json" ] || { echo "$A NO EVAL OUTPUT" >>logs/wsr2.log; continue; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$A retain8=$R below K1 -- not attacked" >>logs/wsr2.log; continue; }
  i=0
  for SD in 0 1 6; do
    K="H83B_${A}_sd${SD}_lr1e-5"; claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=$((i % 2)); i=$((i+1))
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/wsr2_atk.log 2>&1
  done
done
echo "WSR2 ATK DONE $(date -Is)" >>logs/wsr2.log
