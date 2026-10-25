#!/bin/bash
# /data is at 100% with ~6 GB free (dominant consumer is cuig_pilot at 386 GB, not this project).
# attack_relearn.py writes a ~1.7 GB temporary attacked checkpoint and deletes it after measuring,
# so attacks are feasible ONE AT A TIME but not in parallel. Running the frozen six-point grid
# sequentially on the single UTRAP arm that cleared K1.
#
# utrap_church_l0.3k8: target 0.0000 at retain8 0.7333 -- full target suppression, clears the K1
# floor. This is the measurement PREREG_H84 exists for.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p .labels logs
A=utrap_church_l0.3k8
for cfg in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
  set -- $cfg; SD=$1; LR=$2
  K="H84_${A}_sd${SD}_lr${LR}"
  mkdir ".labels/$K" 2>/dev/null || continue
  # bail out rather than wedge the shared volume if headroom disappears
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 4 ]; then
    echo "$K SKIPPED -- only ${FREE}G free on /data, refusing to risk the shared volume" >>logs/utrap_atk.log
    continue
  fi
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --label "$K" >>logs/utrap_atk.log 2>&1
  echo "$K done $(date -Is) free=$(df --output=avail -BG /data|tail -1|tr -dc '0-9')G" >>logs/utrap_atk.log
done
echo "UTRAP SEQ ATK DONE $(date -Is)" >>logs/utrap_atk.log
