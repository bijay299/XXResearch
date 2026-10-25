#!/bin/bash
# K now shows a clean monotone dose-response on retention at identical target suppression:
#   K=4   target 0.0000  retain8 0.6583
#   K=8   target 0.0000  retain8 0.7333
#   K=16  target 0.0000  retain8 0.7583
# Deeper unrolling buys RETENTION, not erasure depth. The open question FINDING63 S4 named is
# whether it buys RESISTANCE -- K=8 was reversed 6/6 at H=1.000, so this attacks the K=16 arm on the
# same frozen grid. If K=16 is also 1.000, unroll depth is ruled out as the missing ingredient and my
# own hypothesis closes cleanly rather than staying open as an untested excuse.
# Sequential: each attack needs ~1.7 GB transient on a volume still nominally at 100%.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
mkdir -p .labels logs
A=utrap_church_l0.3k16
for cfg in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
  set -- $cfg; SD=$1; LR=$2
  K="H84K_${A}_sd${SD}_lr${LR}"
  mkdir ".labels/$K" 2>/dev/null || continue
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 4 ] && { echo "$K SKIPPED -- ${FREE}G free" >>logs/utrap_k16_atk.log; continue; }
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" >>logs/utrap_k16_atk.log 2>&1
  echo "$K done $(date -Is)" >>logs/utrap_k16_atk.log
done
echo "K16 ATK DONE $(date -Is)" >>logs/utrap_k16_atk.log
