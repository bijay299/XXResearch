#!/bin/bash
# meta_bksdm at lam=0.5 came in at retain8 0.4833 -- BELOW K1, so it was correctly not attacked, and
# that leaves FINDING56 S5a's generalisation question UNANSWERED on the pruned architecture. The
# lambda was tuned on SD-1.4 (where 0.5 gives retain 0.6333); on a half-size UNet the same pressure
# costs more utility. lam=0.3 and 0.15 should clear the floor and let the TI question be asked.
#
# This is a transfer-of-hyperparameter issue, not a kill-rule result about the method -- recorded as
# such, the same way FINDING61 attributed its null to my own objective rather than the paper's idea.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=nota-ai/bk-sdm-small BASE_VARIANT="" BASE_REV=572238db7ed3
mkdir -p .labels logs
TEACH=adapters/frp_bksdm_church_r3_l05/unet.pt
for LAM in 0.3 0.15; do
  L="meta_bksdm_church_lam${LAM}"
  mkdir ".labels/$L" 2>/dev/null || continue
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 10 ] && { echo "$L SKIPPED -- ${FREE}G free" >>logs/meta_cm.log; continue; }
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/meta_unlearn.py --concept church \
    --teacher "$TEACH" --lam-meta "$LAM" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED -- NOT a kill-rule result" >>logs/meta_cm.log; continue; }
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/meta_cm.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/meta_cm.log; continue; }
  for cfg in "2000 1e-2 0" "3000 2e-2 1"; do
    set -- $cfg; K="H88T_${L}_s$1_lr$2_sd$3"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet "adapters/$L/unet.pt" --steps "$1" --lr "$2" --seed "$3" --label "$K" >>logs/meta_cm_atk.log 2>&1
  done
  for SD in 0; do
    K="H88F_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/meta_cm_atk.log 2>&1
  done
  echo "$L both adversaries done $(date -Is)" >>logs/meta_cm.log
  break   # first arm that clears K1 is enough to answer the question
done
echo "META BKSDM LOW DONE $(date -Is)" >>logs/meta_cm.log
