#!/bin/bash
# THE DECISIVE ARM. utrap_church_l0.15k8: target 0.0000 at retain8 0.8667 -- the FIRST arm in this
# entire phase to clear the mandate's retain floor (>= 0.85) with the target fully suppressed.
# The mandate's bar is retain8 >= 0.85 AND max-over-valid H <= 0.40. The retain half is met. This
# grid decides the other half, and therefore the mandate.
#
# Unchanged protocol: PROTOCOL_FROZEN_v1's six points, the identical grid all twelve previous
# defences were measured under. No scoring, alias or success-definition change.
# Sequential, ~1.7 GB transient each, with a disk guard.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p .labels logs
A=utrap_church_l0.15k8
for cfg in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
  set -- $cfg; SD=$1; LR=$2
  K="H84M_${A}_sd${SD}_lr${LR}"
  mkdir ".labels/$K" 2>/dev/null || continue
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 4 ] && { echo "$K SKIPPED -- ${FREE}G free" >>logs/utrap_l015_atk.log; continue; }
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" >>logs/utrap_l015_atk.log 2>&1
  echo "$K done $(date -Is)" >>logs/utrap_l015_atk.log
done
echo "L015 ATK DONE $(date -Is)" >>logs/utrap_l015_atk.log
