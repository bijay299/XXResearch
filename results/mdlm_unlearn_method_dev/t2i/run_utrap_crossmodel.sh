#!/bin/bash
# The most mandate-relevant remaining gap: UTRAP -- OUR OWN method, the one PREREG_H84 designed from
# all eleven prior failure modes -- has only ever been measured on SD-1.4. FINDING63 therefore reads
# "every defence WE could design from the failure modes also fails" on one checkpoint only.
#
# This runs it on both new models, using the FRP arms built there as teachers (frp_sd15_*,
# frp_bksdm_*), exactly as the SD-1.4 UTRAP arms used frp_church_r3_l05. lam=0.15 is the setting that
# cleared the mandate's retain floor on SD-1.4 (0.8667), so it is the right one to carry over.
#
# If UTRAP is reversed on a HALF-SIZE architecture too, FINDING63's claim stops being
# checkpoint-specific and the phase's negative result is complete for our own method as well.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p .labels logs
run_one() {  # $1 tag  $2 model  $3 variant  $4 gpu  $5 revision
  local TAG=$1 M=$2 VAR=$3 G=$4 REV=$5
  local TEACH="adapters/frp_${TAG}_church_r3_l05/unet.pt"
  local L="utrap_${TAG}_church_l0.15k8"
  [ -f "$TEACH" ] || { echo "$L SKIPPED -- teacher $TEACH missing" >>logs/utrap_cm.log; return 0; }
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$G BASE_MODEL="$M" BASE_VARIANT="$VAR" BASE_REV="$REV"
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 10 ] && { echo "$L SKIPPED -- ${FREE}G free" >>logs/utrap_cm.log; return 0; }
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_utrap.py --concept church \
    --teacher "$TEACH" --lam 0.15 --inner-k 8 --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/utrap_cm.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L ($M) as-is: target=$T retain8=$R" >>logs/utrap_cm.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/utrap_cm.log; return 0; }
  for SD in 0 1 6; do
    local K="H87_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/utrap_cm_atk.log 2>&1
  done
  echo "$L grid done $(date -Is)" >>logs/utrap_cm.log
}
( run_one bksdm nota-ai/bk-sdm-small "" 0 572238db7ed3; echo "UCM A DONE $(date -Is)" >>logs/utrap_cm.log ) &
( run_one sd15 runwayml/stable-diffusion-v1-5 fp16 1 451f4fe16113bff5a5d2269ed5ad43b0592e9a14; echo "UCM B DONE $(date -Is)" >>logs/utrap_cm.log ) &
wait
