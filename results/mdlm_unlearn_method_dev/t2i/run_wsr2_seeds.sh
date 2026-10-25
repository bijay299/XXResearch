#!/bin/bash
# FINDING62 S4 flags its own weakest point: "the as-is claim rests on a single build per lam_reg and
# would need seeds before being called a stable operating point. It is reported as the best point
# MEASURED, not as a reproducible method."
#
# This builds lam_reg=100 at two more seeds. The claim at stake is specific: target 0.0667 at
# retain8 0.9500, which beats our FRP port on BOTH axes and is the best operating point in the
# phase. If seeds 1 and 2 reproduce it, "best measured point" becomes "reproducible operating
# point". If they scatter the way AFC's seeds did in the MDLM line (TV sd 0.005 at n=3 -> 0.158 at
# n=5), the claim is downgraded to one lucky build and FINDING62 S1 is corrected.
#
# Attacks are NOT re-run: H is already 1.000 on this arm, the ratchet cannot be raised, and the open
# question is about the AS-IS point only.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
build() { local L="wsr2_church_lr100_sd$1"
  mkdir ".labels/$L" 2>/dev/null || return 0; export CUDA_VISIBLE_DEVICES=$2
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_wsr.py --concept church \
    --teacher adapters/frp_church_r3_l05/unet.pt --lam-reg 100 --seed "$1" \
    --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED" >>logs/wsr2_seeds.log; return 1; }
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R   (seed 0 was 0.0667 / 0.9500)" >>logs/wsr2_seeds.log; }
( build 1 0; echo "SEEDS A DONE $(date -Is)" >>logs/wsr2_seeds.log ) &
( build 2 1; echo "SEEDS B DONE $(date -Is)" >>logs/wsr2_seeds.log ) &
wait
