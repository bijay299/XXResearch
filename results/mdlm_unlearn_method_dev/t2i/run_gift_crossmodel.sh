#!/bin/bash
# FINDING64 S3 records what it did NOT measure: "Only FRP ... was run on the new models. The other
# defences' cross-model behaviour is INFERRED, not measured." This measures the most informative one.
#
# GIFT is the right second choice: it edits CROSS-ATTENTION (psi), and on BK-SDM-small the pruned
# UNet makes cross-attention a DIFFERENT FRACTION of the model. FINDING64 S2 already narrowed the
# parameter-ratio explanation; running the subspace-editing defence on the pruned architecture tests
# the subspace half of that argument on new ground rather than assuming it.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# all four cache vars: HF_HUB_CACHE et al. are set host-wide to another project's cache and override
# HF_HOME (FINDING64 S4 bug 3)
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p .labels logs
run_one() {  # $1 tag  $2 model  $3 variant  $4 gpu  $5 revision
  local TAG=$1 M=$2 VAR=$3 G=$4 REV=$5
  local L="gift_${TAG}_church_b1.0"
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$G BASE_MODEL="$M" BASE_VARIANT="$VAR" BASE_REV="$REV"
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 10 ] && { echo "$L SKIPPED -- ${FREE}G free" >>logs/gift_cm.log; return 0; }
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_gift.py --concept church \
    --beta 1.0 --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/gift_cm.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L ($M) as-is: target=$T retain8=$R" >>logs/gift_cm.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/gift_cm.log; return 0; }
  for SD in 0 1 6; do
    local K="H86_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/gift_cm_atk.log 2>&1
  done
  echo "$L grid done $(date -Is)" >>logs/gift_cm.log
}
( run_one bksdm nota-ai/bk-sdm-small "" 0 572238db7ed3; echo "GCM A DONE $(date -Is)" >>logs/gift_cm.log ) &
( run_one sd15 runwayml/stable-diffusion-v1-5 fp16 1 451f4fe16113bff5a5d2269ed5ad43b0592e9a14; echo "GCM B DONE $(date -Is)" >>logs/gift_cm.log ) &
wait
